"""Grid search for stage-wise rotation weights in two-stage bias fitting.

This utility iterates over combinations of stage1 rotation weights (used while
estimating rotation biases) and stage2 rotation weights (used during the second
fitting stage after biases are fixed). For each pair it runs the two-stage
solver and reports key error metrics, making it easy to identify a good weight
configuration.

Example:

    uv run python -m models.bridge_fitting.search_rotation_weights \
        --data data/augmented/dt_24hours_data_new_rsensor0501_noise5_biased.csv \
        --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
        --stage1-grid 10 20 30 --stage2-grid 0.5 1 2 \
        --stage1-span-grid 10 --stage2-span-grid 0.5 \
        --use-span-rotations --temp-basis fourier --fourier-harmonics 8

The script prints a summary table sorted by EI factor RMSE. Optionally it can
write the results to CSV for further analysis.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from models.bridge_fitting import (
    MeasurementConfig,
    VectorizedMultiCaseFitter,
    load_multi_case_data,
    setup_measurement_matrices,
)
from models.bridge_fitting.config import ColumnNames
from models.bridge_fitting.results import extract_true_parameters
from models.bridge_fitting.temperature_models import FourierTemperatureBasis, RawTemperatureBasis


def _parse_float_list(values: Sequence[str]) -> list[float]:
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"无法解析浮点数: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("至少需要一个权重")
    return parsed


@dataclass
class RunMetrics:
    stage1_weight: float
    stage2_weight: float
    stage1_span_weight: float | None
    stage2_span_weight: float | None
    stage1_bias_rmse: float
    stage1_bias_mae: float
    stage1_span_bias_rmse: float | None
    stage1_span_bias_mae: float | None
    stage2_cost: float
    stage2_success: bool
    settlement_rmse: float
    ei_rmse: float
    temperature_rmse: float
    message: str


def _build_temperature_basis(
    df,
    temp_basis: str,
    temp_segments: int,
    harmonics: int,
    include_bias: bool,
    fourier_period: float | None,
    fourier_time_column: str | None,
    fourier_coeff_lower: float | None,
    fourier_coeff_upper: float | None,
) -> tuple[np.ndarray, int, object]:
    if temp_basis == "fourier":
        if fourier_time_column:
            if fourier_time_column not in df.columns:
                raise SystemExit(f"数据中不存在时间列: {fourier_time_column}")
            time_values = df[fourier_time_column].to_numpy(dtype=float)
        else:
            time_values = np.arange(len(df), dtype=float)
        period = fourier_period if fourier_period is not None else float(len(df))
        basis = FourierTemperatureBasis(
            n_cases=len(df),
            temp_segments=temp_segments,
            harmonics=max(0, harmonics),
            include_bias=include_bias,
            time_values=time_values,
            fundamental_period=period,
            coeff_lower=fourier_coeff_lower,
            coeff_upper=fourier_coeff_upper,
        )
        return basis, temp_segments, time_values

    if temp_basis != "raw":
        raise SystemExit(f"未知的温度基: {temp_basis}")

    basis = RawTemperatureBasis(n_cases=len(df), temp_segments=temp_segments)
    return basis, temp_segments, None


def _compute_bias_errors(fitted: np.ndarray, true_bias: np.ndarray) -> tuple[float, float]:
    diff = fitted[: len(true_bias)] - true_bias
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


def _compute_struct_metrics(result: dict, const_params: dict, true_temps: np.ndarray) -> tuple[float, float, float]:
    true_settlements, true_ei, _ = extract_true_parameters(const_params)
    fitted_sp = result['struct_params']
    fitted_settlements = np.asarray(fitted_sp.settlements, dtype=float)
    fitted_ei = np.asarray(fitted_sp.ei_factors, dtype=float)

    settlement_rmse = float(np.sqrt(np.mean((fitted_settlements - true_settlements) ** 2)))
    ei_rmse = float(np.sqrt(np.mean((fitted_ei - true_ei) ** 2)))

    fitted_temps = np.array([ts.dT_spans for ts in result['thermal_states']], dtype=float)
    temp_rmse = float(np.sqrt(np.mean((fitted_temps - true_temps) ** 2)))

    return settlement_rmse, ei_rmse, temp_rmse


def _prepare_true_bias(df, use_span_rotations: bool) -> tuple[np.ndarray, np.ndarray | None]:
    support_cols = [
        'theta_A_rad_bias',
        'theta_B_rad_bias',
        'theta_C_rad_bias',
        'theta_D_rad_bias',
    ]
    if not all(col in df.columns for col in support_cols):
        raise SystemExit("数据中缺少支座转角 *_rad_bias 列，无法比较真值偏差")
    support_bias = df[support_cols].iloc[0].to_numpy(dtype=float)

    span_bias = None
    if use_span_rotations:
        span_cols = [
            'theta_S1-5_6L_rad_bias',
            'theta_S2-4_6L_rad_bias',
            'theta_S3-5_6L_rad_bias',
        ]
        if not all(col in df.columns for col in span_cols):
            raise SystemExit("数据中缺少跨中转角 *_rad_bias 列，无法比较真值偏差")
        span_bias = df[span_cols].iloc[0].to_numpy(dtype=float)

    return support_bias, span_bias


def _describe_combo(metrics: RunMetrics) -> str:
    span_part = (
        f"span(w1={metrics.stage1_span_weight:.3f}, w2={metrics.stage2_span_weight:.3f}), "
        if metrics.stage1_span_weight is not None
        else ""
    )
    span_bias_part = (
        f"Span Bias RMSE={metrics.stage1_span_bias_rmse:.4e}, "
        if metrics.stage1_span_bias_rmse is not None
        else ""
    )
    return (
        f"stage1={metrics.stage1_weight:.3f}, stage2={metrics.stage2_weight:.3f}, "
        + span_part
        + f"EI RMSE={metrics.ei_rmse:.4f}, Sett RMSE={metrics.settlement_rmse:.3f} mm, "
        + f"Temp RMSE={metrics.temperature_rmse:.3f} °C, Bias RMSE={metrics.stage1_bias_rmse:.4e}, "
        + span_bias_part
        + f"success={metrics.stage2_success}"
    )


def run_combo(
    data_paths: Sequence[Path],
    stage1_weight: float,
    stage2_weight: float,
    stage1_span_weight: float | None,
    stage2_span_weight: float | None,
    use_span_rotations: bool,
    temp_basis_name: str,
    temp_segments: int,
    harmonics: int,
    include_bias: bool,
    fourier_period: float | None,
    fourier_time_column: str | None,
    fourier_coeff_lower: float | None,
    fourier_coeff_upper: float | None,
    fixed_kv: tuple[float, float, float, float] | None,
    fix_first_settlement: bool,
    maxiter_stage1: int,
    maxiter_stage2: int,
    no_fit_struct: bool,
    auto_normalize: bool,
) -> RunMetrics:
    stage1_span_weight_value = stage1_span_weight if stage1_span_weight is not None else stage1_weight
    measurement_config_stage1 = MeasurementConfig(
        use_reactions=True,
        use_displacements=False,
        use_rotations=True,
        use_span_rotations=use_span_rotations,
        rotation_weight=stage1_weight,
        span_rotation_weight=stage1_span_weight_value,
        auto_normalize=auto_normalize,
    )

    df, const_params, _ = load_multi_case_data(data_paths, None, measurement_config_stage1)

    true_bias_support, true_bias_span = _prepare_true_bias(df, use_span_rotations)

    (
        reactions_matrix,
        _,
        rotations_matrix_with_bias,
        span_rotations_matrix_with_bias,
        measurement_config_stage1,
        rotation_column_labels,
        span_rotation_column_labels,
    ) = setup_measurement_matrices(
        df,
        measurement_config_stage1,
        use_rotation_bias_columns=True,
    )

    # Determine temperature basis and true temperature array (same logic as CLI)
    if temp_segments == 3:
        if all(c in df.columns for c in ColumnNames.TEMPS_3_SPAN):
            true_temps = df[ColumnNames.TEMPS_3_SPAN].to_numpy()
            actual_segments = 3
        elif all(c in df.columns for c in ColumnNames.TEMPS_2_SEG):
            true_temps = df[ColumnNames.TEMPS_2_SEG].to_numpy()
            actual_segments = 2
        else:
            raise SystemExit('缺少温度列：需要 dT_s1_C/dT_s2_C/dT_s3_C 或 dT_left_C/dT_right_C')
    else:
        if all(c in df.columns for c in ColumnNames.TEMPS_2_SEG):
            true_temps = df[ColumnNames.TEMPS_2_SEG].to_numpy()
            actual_segments = 2
        elif all(c in df.columns for c in ColumnNames.TEMPS_3_SPAN):
            true_temps = df[ColumnNames.TEMPS_3_SPAN].to_numpy()
            actual_segments = 3
        else:
            raise SystemExit('缺少温度列：需要 dT_left_C/dT_right_C 或 dT_s1_C/dT_s2_C/dT_s3_C')

    basis, actual_segments, _ = _build_temperature_basis(
        df,
        temp_basis=temp_basis_name,
        temp_segments=actual_segments,
        harmonics=harmonics,
        include_bias=include_bias,
        fourier_period=fourier_period,
        fourier_time_column=fourier_time_column,
        fourier_coeff_lower=fourier_coeff_lower,
        fourier_coeff_upper=fourier_coeff_upper,
    )

    uniform_load = float(df[ColumnNames.UNIFORM_LOAD].iloc[0])

    fitter_stage1 = VectorizedMultiCaseFitter(
        reactions_matrix=reactions_matrix,
        uniform_load=uniform_load,
        temp_segments=actual_segments,
        measurement_config=measurement_config_stage1,
        rotations_matrix=rotations_matrix_with_bias,
        span_rotations_matrix=span_rotations_matrix_with_bias,
        temperature_basis=basis,
        fixed_kv_factors=fixed_kv,
        fit_rotation_bias=True,
        rotation_column_labels=rotation_column_labels,
        span_rotation_column_labels=span_rotation_column_labels,
        fix_first_settlement=fix_first_settlement,
    )

    result_stage1 = fitter_stage1.fit(
        fit_struct=not no_fit_struct,
        maxiter=maxiter_stage1,
        verbose=0,
    )

    fitted_bias = result_stage1.get('rotation_biases')
    if fitted_bias is None:
        raise RuntimeError("Stage1 未返回 rotation_biases")

    fitted_bias = np.asarray(fitted_bias, dtype=float)
    bias_rmse, bias_mae = _compute_bias_errors(
        fitted_bias[: fitter_stage1._rotation_bias_support_count], true_bias_support
    )

    span_bias_rmse = span_bias_mae = None

    rotations_corrected = rotations_matrix_with_bias - fitted_bias[: fitter_stage1._rotation_bias_support_count][np.newaxis, :]

    span_corrected = None
    if use_span_rotations and span_rotations_matrix_with_bias is not None and true_bias_span is not None:
        span_bias = fitted_bias[fitter_stage1._rotation_bias_support_count:]
        span_corrected = span_rotations_matrix_with_bias - span_bias[np.newaxis, :]
        span_bias_rmse, span_bias_mae = _compute_bias_errors(span_bias, true_bias_span)

    stage2_span_weight_value = stage2_span_weight if stage2_span_weight is not None else stage2_weight
    measurement_config_stage2 = MeasurementConfig(
        use_reactions=True,
        use_displacements=False,
        use_rotations=True,
        use_span_rotations=use_span_rotations,
        rotation_weight=stage2_weight,
        span_rotation_weight=stage2_span_weight_value,
        auto_normalize=auto_normalize,
    )

    fitter_stage2 = VectorizedMultiCaseFitter(
        reactions_matrix=reactions_matrix,
        uniform_load=uniform_load,
        temp_segments=actual_segments,
        measurement_config=measurement_config_stage2,
        rotations_matrix=rotations_corrected,
        span_rotations_matrix=span_corrected,
        temperature_basis=basis,
        fixed_kv_factors=fixed_kv,
        fit_rotation_bias=False,
        rotation_column_labels=rotation_column_labels,
        span_rotation_column_labels=span_rotation_column_labels,
        fix_first_settlement=fix_first_settlement,
    )

    result_stage2 = fitter_stage2.fit(
        fit_struct=not no_fit_struct,
        maxiter=maxiter_stage2,
        verbose=0,
    )

    settlement_rmse, ei_rmse, temp_rmse = _compute_struct_metrics(result_stage2, const_params, true_temps)

    metrics = RunMetrics(
        stage1_weight=stage1_weight,
        stage2_weight=stage2_weight,
        stage1_span_weight=stage1_span_weight if use_span_rotations else None,
        stage2_span_weight=stage2_span_weight if use_span_rotations else None,
        stage1_bias_rmse=bias_rmse,
        stage1_bias_mae=bias_mae,
        stage1_span_bias_rmse=span_bias_rmse,
        stage1_span_bias_mae=span_bias_mae,
        stage2_cost=float(result_stage2['cost']),
        stage2_success=bool(result_stage2['success']),
        settlement_rmse=settlement_rmse,
        ei_rmse=ei_rmse,
        temperature_rmse=temp_rmse,
        message=result_stage2.get('message', ''),
    )

    return metrics


def _write_csv(path: Path, rows: Iterable[RunMetrics]) -> None:
    fieldnames = list(asdict(next(iter(rows := list(rows)))).keys())
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for rotation weights in two-stage bias fitting")
    parser.add_argument('--data', type=Path, action='append', required=True, help='CSV数据文件 (可多次指定)')
    parser.add_argument('--fixed-kv', type=float, nargs=4, metavar=('KVA', 'KVB', 'KVC', 'KVD'))
    parser.add_argument('--no-fit-struct', action='store_true', help='仅拟合温度 (不拟合结构参数)')
    parser.add_argument('--maxiter-stage1', type=int, default=100)
    parser.add_argument('--maxiter-stage2', type=int, default=100)
    parser.add_argument('--stage1-grid', type=str, nargs='+', required=True, help='Stage1 rotation weight 列表 (空格分隔)')
    parser.add_argument('--stage2-grid', type=str, nargs='+', required=True, help='Stage2 rotation weight 列表 (空格分隔)')
    parser.add_argument('--stage1-span-grid', type=str, nargs='+', help='Stage1 跨中转角权重列表 (空格分隔)')
    parser.add_argument('--stage2-span-grid', type=str, nargs='+', help='Stage2 跨中转角权重列表 (空格分隔)')
    parser.add_argument('--use-span-rotations', action='store_true', help='同时使用跨中转角及其偏差')
    parser.add_argument('--temp-basis', choices=['raw', 'fourier'], default='fourier')
    parser.add_argument('--temp-segments', type=int, default=3, choices=[2, 3])
    parser.add_argument('--fourier-harmonics', type=int, default=8)
    parser.add_argument('--fourier-period', type=float)
    parser.add_argument('--fourier-time-column', type=str)
    parser.add_argument('--no-fourier-bias', action='store_true')
    parser.add_argument('--fourier-coeff-lower', type=float)
    parser.add_argument('--fourier-coeff-upper', type=float)
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--no-auto-normalize', action='store_true')

    args = parser.parse_args()

    stage1_weights = _parse_float_list(args.stage1_grid)
    stage2_weights = _parse_float_list(args.stage2_grid)

    if args.use_span_rotations:
        if args.stage1_span_grid:
            stage1_span_weights = _parse_float_list(args.stage1_span_grid)
        else:
            stage1_span_weights = stage1_weights
        if args.stage2_span_grid:
            stage2_span_weights = _parse_float_list(args.stage2_span_grid)
        else:
            stage2_span_weights = stage2_weights
    else:
        stage1_span_weights = [None]
        stage2_span_weights = [None]

    fixed_kv = tuple(args.fixed_kv) if args.fixed_kv is not None else None

    combos = list(itertools.product(stage1_weights, stage2_weights, stage1_span_weights, stage2_span_weights))
    results: list[RunMetrics] = []

    for s1, s2, s1_span, s2_span in combos:
        print(f"\n{'='*60}\n开始组合: stage1_weight={s1}, stage2_weight={s2}" + (f", span_weights=({s1_span}, {s2_span})" if args.use_span_rotations else "") + f"\n{'='*60}")
        metrics = run_combo(
            data_paths=args.data,
            stage1_weight=s1,
            stage2_weight=s2,
            stage1_span_weight=s1_span,
            stage2_span_weight=s2_span,
            use_span_rotations=args.use_span_rotations,
            temp_basis_name=args.temp_basis,
            temp_segments=args.temp_segments,
            harmonics=args.fourier_harmonics,
            include_bias=not args.no_fourier_bias,
            fourier_period=args.fourier_period,
            fourier_time_column=args.fourier_time_column,
            fourier_coeff_lower=args.fourier_coeff_lower,
            fourier_coeff_upper=args.fourier_coeff_upper,
            fixed_kv=fixed_kv,
            fix_first_settlement=True,
            maxiter_stage1=args.maxiter_stage1,
            maxiter_stage2=args.maxiter_stage2,
            no_fit_struct=args.no_fit_struct,
            auto_normalize=not args.no_auto_normalize,
        )
        results.append(metrics)
        print(_describe_combo(metrics))

    results.sort(key=lambda m: (m.ei_rmse, m.settlement_rmse))

    print("\n最佳组合 (按 EI RMSE 排序):")
    for metrics in results[: min(len(results), 10)]:
        print("  " + _describe_combo(metrics))

    if args.output_csv:
        _write_csv(args.output_csv, results)
        print(f"\n结果已写入 {args.output_csv}")


if __name__ == '__main__':
    main()
