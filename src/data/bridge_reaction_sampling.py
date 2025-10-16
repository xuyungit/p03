# -*- coding: utf-8 -*-
"""
批量生成三跨连续梁支座反力（LHS采样）

可控变量与范围：
1) 支座沉降：以 A 为零点，B/C/D 为相对高度（mm），范围 [-10, 10]
2) 支座竖向刚度系数：A/B/C/D 的系数 ∈ [0.3, 1.5]（相对 Kv0）
3) 梁 EI 系数：每跨的系数 ∈ [0.3, 1.5]
4) 温度梯度：每跨 ΔT ∈ [-10, 20] ℃，相邻跨差值 |ΔT(i+1) - ΔT(i)| ≤ 5 ℃
5) 采样数量可配置，结果输出到 data/ 下 CSV

实现说明：
- 使用内置的拉丁超立方（LHS）实现，无 SciPy 依赖。
- 有限元模型与传热换算逻辑统一复用 ``bridge.mechanics``，保证训练
  与数据生成阶段的一致性。
- 温差线性梯度对应恒定曲率 κ = α·ΔT/h，施加等效固定端弯矩 ±EIκ。

用法示例：
- 生成 200 条样本（默认 data/augmented/bridge_lhs_samples.csv）：
  uv run python data/bridge_reaction_sampling.py --n 200

- 指定随机种子和输出文件：
  uv run python data/bridge_reaction_sampling.py --n 1000 --seed 123 --out data/augmented/bridge_lhs_1000.csv

- 指定三跨长度（单位 m）：
  uv run python data/bridge_reaction_sampling.py --span-lengths 30:60:30

- 使用外部参数 CSV（禁用 LHS，逐行生成）：
  uv run python data/bridge_reaction_sampling.py --params data/custom_params.csv --out data/augmented/bridge_custom.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

from bridge.mechanics import (
    Q_UNIFORM,
    StructuralParams,
    ThermalState,
    get_span_lengths_mm,
    sample_19pt_deflection_rotation,
    set_span_lengths_mm as mechanics_set_span_lengths_mm,
    solve_case,
    total_length_mm,
)


# ----------------- LHS 采样工具（无 SciPy 依赖） -----------------
def _lhs_builtin(n_samples: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """简易 LHS：[0,1] 超立方体，逐维分层并打乱。"""
    H = np.zeros((n_samples, dims), dtype=float)
    for j in range(dims):
        cut = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(cut)
        H[:, j] = cut
    return H


def _lhs_scipy(n_samples: int, dims: int, seed: int | None) -> np.ndarray:
    """若可用，使用 SciPy qmc.LatinHypercube 生成 LHS。"""
    try:
        from scipy.stats import qmc  # type: ignore
    except Exception:
        return None  # type: ignore
    try:
        sampler = qmc.LatinHypercube(d=dims, centered=True, optimization="random-cd", seed=seed)
    except TypeError:
        try:
            sampler = qmc.LatinHypercube(d=dims, seed=seed)
        except Exception:
            return None  # type: ignore
    return sampler.random(n=n_samples)


def lhs_unit_hypercube(
    n_samples: int,
    dims: int,
    rng: np.random.Generator,
    seed: int | None,
    backend: str = "auto",
) -> np.ndarray:
    """LHS 生成：优先 SciPy，失败则回退到内置实现。"""
    if backend in ("auto", "scipy"):
        H = _lhs_scipy(n_samples, dims, seed)
        if H is not None:
            return H
        if backend == "scipy":
            raise RuntimeError("SciPy qmc.LatinHypercube not available.")
    return _lhs_builtin(n_samples, dims, rng)


def scale(u01: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * u01


# ----------------- 参数容器 -----------------
@dataclass
class BridgeCase:
    structural: StructuralParams
    thermal: ThermalState


def parse_span_lengths_arg(text: str) -> List[float]:
    tokens = [tok for tok in re.split(r"[:,;\s]+", text.strip()) if tok]
    if len(tokens) != 3:
        raise ValueError("需要三个跨度长度，例如 '40:40:40'")
    try:
        return [float(tok) for tok in tokens]
    except ValueError as exc:  # pragma: no cover - CLI 级别错误
        raise ValueError("跨度长度必须为数字") from exc


# ----------------- 采样与批处理 -----------------
def sample_cases_lhs(
    n: int,
    seed: int | None = None,
    lhs_backend: str = "auto",
    temp_segments: int = 3,
) -> List[BridgeCase]:
    rng = np.random.default_rng(seed)

    if temp_segments == 3:
        dims = 13  # t1, dt12, dt23
    elif temp_segments == 2:
        dims = 12  # t_left, dt_lr
    else:
        raise ValueError("temp_segments must be 2 or 3")
    U = lhs_unit_hypercube(n, dims=dims, rng=rng, seed=seed, backend=lhs_backend)

    sett_bcd = scale(U[:, 0:3], -10.0, 10.0)
    settlements = np.zeros((n, 4), dtype=float)
    settlements[:, 1:4] = sett_bcd

    kv_factors = scale(U[:, 3:7], 0.3, 1.5)
    ei_factors = scale(U[:, 7:10], 0.3, 1.5)

    if temp_segments == 3:
        t1 = scale(U[:, 10], -10.0, 20.0)
        u12 = U[:, 11]
        u23 = U[:, 12]
        t2 = np.zeros_like(t1)
        t3 = np.zeros_like(t1)
        for i in range(n):
            low1 = max(-5.0, -10.0 - t1[i])
            high1 = min(5.0, 20.0 - t1[i])
            s1 = low1 + (high1 - low1) * u12[i]
            t2[i] = t1[i] + s1

            low2 = max(-5.0, -10.0 - t2[i])
            high2 = min(5.0, 20.0 - t2[i])
            s2 = low2 + (high2 - low2) * u23[i]
            t3[i] = t2[i] + s2
        dT_spans = np.stack([t1, t2, t3], axis=1)
    else:
        tL = scale(U[:, 10], -10.0, 20.0)
        uLR = U[:, 11]
        tR = np.zeros_like(tL)
        for i in range(n):
            low = max(-5.0, -10.0 - tL[i])
            high = min(5.0, 20.0 - tL[i])
            s = low + (high - low) * uLR[i]
            tR[i] = tL[i] + s
        dT_spans = np.stack([tL, tR], axis=1)

    cases: List[BridgeCase] = []
    for i in range(n):
        struct = StructuralParams(
            settlements=tuple(map(float, settlements[i, :])),
            kv_factors=tuple(map(float, kv_factors[i, :])),
            ei_factors=tuple(map(float, ei_factors[i, :])),
            uniform_load=float(Q_UNIFORM),
        )
        thermal = ThermalState(dT_spans=tuple(map(float, dT_spans[i, :])))
        cases.append(BridgeCase(structural=struct, thermal=thermal))
    return cases


PARAMETER_CSV_REQUIRED_COLS_3 = [
    "settlement_b_mm",
    "settlement_c_mm",
    "settlement_d_mm",
    "kv_factor_a",
    "kv_factor_b",
    "kv_factor_c",
    "kv_factor_d",
    "ei_factor_s1",
    "ei_factor_s2",
    "ei_factor_s3",
    "dT_s1_C",
    "dT_s2_C",
    "dT_s3_C",
]
PARAMETER_CSV_REQUIRED_COLS_2 = [
    "settlement_b_mm",
    "settlement_c_mm",
    "settlement_d_mm",
    "kv_factor_a",
    "kv_factor_b",
    "kv_factor_c",
    "kv_factor_d",
    "ei_factor_s1",
    "ei_factor_s2",
    "ei_factor_s3",
    "dT_left_C",
    "dT_right_C",
]

PARAMETER_CSV_OPTIONAL_SETTLEMENT_A = "settlement_a_mm"
PARAMETER_CSV_Q_CANDIDATES = ("q", "q_N_per_mm", "uniform_load_N_per_mm")
PARAMETER_CSV_NE_CANDIDATES = ("ne_per_span", "elements_per_span")


def _safe_float(value: Any, default: float) -> float:
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _safe_int(value: Any, default: int) -> int:
    if value is None or pd.isna(value):
        return int(default)
    return int(value)


def load_cases_from_parameter_csv(param_csv: Path) -> Tuple[List[BridgeCase], List[Any] | None]:
    df = pd.read_csv(param_csv)

    has_3 = all(c in df.columns for c in PARAMETER_CSV_REQUIRED_COLS_3)
    has_2 = all(c in df.columns for c in PARAMETER_CSV_REQUIRED_COLS_2)
    if not (has_3 or has_2):
        raise ValueError(
            "参数 CSV 缺少必需列: 需要 {dT_s1_C,dT_s2_C,dT_s3_C} 或 {dT_left_C,dT_right_C}"
        )

    has_sample_id = "sample_id" in df.columns
    sample_ids: List[Any] = []
    cases: List[BridgeCase] = []

    for record in df.to_dict(orient="records"):
        settlements = (
            _safe_float(record.get(PARAMETER_CSV_OPTIONAL_SETTLEMENT_A, 0.0), 0.0),
            _safe_float(record["settlement_b_mm"], 0.0),
            _safe_float(record["settlement_c_mm"], 0.0),
            _safe_float(record["settlement_d_mm"], 0.0),
        )

        kv_factors = (
            _safe_float(record["kv_factor_a"], 1.0),
            _safe_float(record["kv_factor_b"], 1.0),
            _safe_float(record["kv_factor_c"], 1.0),
            _safe_float(record["kv_factor_d"], 1.0),
        )

        ei_factors = (
            _safe_float(record["ei_factor_s1"], 1.0),
            _safe_float(record["ei_factor_s2"], 1.0),
            _safe_float(record["ei_factor_s3"], 1.0),
        )

        if has_3:
            dT_spans = (
                _safe_float(record["dT_s1_C"], 0.0),
                _safe_float(record["dT_s2_C"], 0.0),
                _safe_float(record["dT_s3_C"], 0.0),
            )
        else:
            dT_spans = (
                _safe_float(record["dT_left_C"], 0.0),
                _safe_float(record["dT_right_C"], 0.0),
            )

        q_value = None
        for col in PARAMETER_CSV_Q_CANDIDATES:
            if col in record and not pd.isna(record[col]):
                q_value = float(record[col])
                break
        if q_value is None:
            q_value = Q_UNIFORM

        ne_value = None
        for col in PARAMETER_CSV_NE_CANDIDATES:
            if col in record and not pd.isna(record[col]):
                ne_value = _safe_int(record[col], 64)
                break
        if ne_value is None:
            ne_value = 64

        struct = StructuralParams(
            settlements=tuple(float(v) for v in settlements),
            kv_factors=tuple(float(v) for v in kv_factors),
            ei_factors=tuple(float(v) for v in ei_factors),
            uniform_load=float(q_value),
            ne_per_span=int(ne_value),
        )
        thermal = ThermalState(dT_spans=tuple(float(v) for v in dT_spans))
        cases.append(BridgeCase(structural=struct, thermal=thermal))

        if has_sample_id:
            sample_ids.append(record.get("sample_id"))

    return cases, (sample_ids if has_sample_id else None)


def generate_dataset(
    cases: Sequence[BridgeCase],
    out_path: Path,
    ne_per_span: int | None = None,
    sample_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_ids is not None and len(sample_ids) != len(cases):
        raise ValueError("sample_ids 长度与案例数量不一致")

    bridge_length_mm = total_length_mm()
    rows = []

    for idx, case in enumerate(cases):
        struct = case.structural
        thermal = case.thermal

        if ne_per_span is not None:
            struct.ne_per_span = ne_per_span

        if sample_ids is not None:
            sid = sample_ids[idx]
            sample_id = idx if pd.isna(sid) else sid
        else:
            sample_id = idx

        load_kN = -struct.uniform_load * bridge_length_mm / 1_000.0

        U, reactions = solve_case(struct, thermal)
        reactions_list = [float(r) for r in reactions]

        sum_R = float(reactions.sum())
        eq_err = sum_R - load_kN

        sensor_samples = sample_19pt_deflection_rotation(struct.ne_per_span, U)

        span_lengths_mm = get_span_lengths_mm()

        base_row = {
            "sample_id": sample_id,
            "settlement_a_mm": struct.settlements[0],
            "settlement_b_mm": struct.settlements[1],
            "settlement_c_mm": struct.settlements[2],
            "settlement_d_mm": struct.settlements[3],
            "kv_factor_a": struct.kv_factors[0],
            "kv_factor_b": struct.kv_factors[1],
            "kv_factor_c": struct.kv_factors[2],
            "kv_factor_d": struct.kv_factors[3],
            "ei_factor_s1": struct.ei_factors[0],
            "ei_factor_s2": struct.ei_factors[1],
            "ei_factor_s3": struct.ei_factors[2],
            "uniform_load_N_per_mm": struct.uniform_load,
            "R_a_kN": reactions_list[0],
            "R_b_kN": reactions_list[1],
            "R_c_kN": reactions_list[2],
            "R_d_kN": reactions_list[3],
            "sum_R_kN": sum_R,
            "total_udl_kN": load_kN,
            "equilibrium_err_kN": eq_err,
            "has_negative_reaction": bool(np.any(reactions < 0.0)),
            "span_length_s1_mm": span_lengths_mm[0],
            "span_length_s2_mm": span_lengths_mm[1],
            "span_length_s3_mm": span_lengths_mm[2],
        }

        if len(thermal.dT_spans) == 3:
            base_row.update(
                {
                    "dT_s1_C": thermal.dT_spans[0],
                    "dT_s2_C": thermal.dT_spans[1],
                    "dT_s3_C": thermal.dT_spans[2],
                }
            )
        else:
            base_row.update(
                {
                    "dT_left_C": thermal.dT_spans[0],
                    "dT_right_C": thermal.dT_spans[1],
                }
            )

        base_row.update(sensor_samples)
        rows.append(base_row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


# ----------------- 主函数 -----------------
def main() -> None:
    parser = argparse.ArgumentParser(description="三跨连续梁支座反力 LHS 采样生成器")
    parser.add_argument("--n", type=int, default=200, help="样本数量（仅 LHS 模式）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（仅 LHS 模式）")
    parser.add_argument("--ne", type=int, default=None, help="每跨单元数量，默认 64")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data/augmented/bridge_lhs_samples.csv")),
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="参数 CSV 路径；提供后跳过 LHS，直接使用文件中的参数",
    )
    parser.add_argument(
        "--lhs",
        type=str,
        default="auto",
        choices=["auto", "scipy", "builtin"],
        help="LHS 后端：auto优先SciPy，否则回退；scipy强制SciPy；builtin内置实现",
    )
    parser.add_argument(
        "--temp-segments",
        type=int,
        default=3,
        choices=[2, 3],
        help="温度分段数：3=每跨；2=按全长50/50",
    )
    parser.add_argument(
        "--span-lengths",
        type=str,
        default=None,
        help="三跨长度（米），格式如 '40:40:40' 或 '30:60:30'",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    ne_override = args.ne

    if args.span_lengths:
        try:
            spans_m = parse_span_lengths_arg(args.span_lengths)
        except ValueError as exc:
            parser.error(str(exc))
        mechanics_set_span_lengths_mm([L * 1_000.0 for L in spans_m])

    if args.params:
        param_path = Path(args.params)
        cases, sample_ids = load_cases_from_parameter_csv(param_path)
        print(f"从参数 CSV 加载 {len(cases)} 条样本: {param_path}")
    else:
        cases = sample_cases_lhs(
            n=args.n,
            seed=args.seed,
            lhs_backend=args.lhs,
            temp_segments=int(args.temp_segments),
        )
        sample_ids = None

    df = generate_dataset(
        cases=cases,
        out_path=out_path,
        ne_per_span=ne_override,
        sample_ids=sample_ids,
    )

    print(f"生成完成: {out_path} ({len(df)} 行)")
    neg = int(df["has_negative_reaction"].sum())
    print(f"含负反力样本: {neg}/{len(df)}")
    print(
        f"平衡误差 |mean| = {df['equilibrium_err_kN'].abs().mean():.3f} kN, max = {df['equilibrium_err_kN'].abs().max():.3f} kN"
    )


if __name__ == "__main__":
    main()
