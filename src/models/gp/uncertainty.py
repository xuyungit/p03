"""Uncertainty utilities shared across GP model entrypoints."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch


def _normal_quantile(prob: float) -> float:
    if not 0.0 < prob < 1.0:
        raise ValueError(f"Quantile probability must be in (0, 1); received {prob}")
    tensor = torch.tensor(float(prob), dtype=torch.float64)
    distribution = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
    )
    return float(distribution.icdf(tensor).item())


def _gaussian_nll_matrix(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    eps = 1e-9
    std = np.maximum(y_std, eps)
    var = np.square(std)
    residual = y_true - y_pred
    return 0.5 * (np.log(2.0 * np.pi * var) + np.square(residual) / var)


def _coverage_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    level: float,
) -> np.ndarray:
    alpha = 0.5 + level / 2.0
    margin = _normal_quantile(alpha) * y_std
    lower = y_pred - margin
    upper = y_pred + margin
    return (y_true >= lower) & (y_true <= upper)


def _ensure_2d(array: np.ndarray) -> tuple[np.ndarray, bool]:
    data = np.asarray(array, dtype=np.float64)
    if data.ndim == 1:
        return data[None, :], True
    if data.ndim != 2:
        raise ValueError("Expected a 1D or 2D array")
    return data, False


def scale_standard_deviation(
    std: np.ndarray,
    *,
    global_tau: Optional[float] = None,
    per_group_tau: Optional[Mapping[str, float]] = None,
    group_indices: Optional[Mapping[str, Sequence[int]]] = None,
) -> np.ndarray:
    """Apply global and/or per-group tau scaling to a std matrix."""

    std_arr, squeezed = _ensure_2d(std)
    scaled = std_arr.copy()

    if global_tau is not None:
        scaled *= float(global_tau)

    if per_group_tau and group_indices:
        for group, indices in group_indices.items():
            if not indices:
                continue
            tau_value = per_group_tau.get(group)
            if tau_value is None:
                continue
            scaled[..., list(indices)] *= float(tau_value)

    if squeezed:
        return scaled[0]
    return scaled


QuantileKey = Union[str, float]


def _normalize_level_mapping(mapping: Mapping[QuantileKey, float]) -> Dict[float, float]:
    normalized: Dict[float, float] = {}
    for key, value in mapping.items():
        try:
            level = float(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Quantile level keys must be convertible to float: got {key!r}") from exc
        normalized[level] = float(value)
    return normalized


def _summarize_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    coverage_levels: Sequence[float],
    *,
    target_names: Sequence[str],
    group_indices: Optional[Mapping[str, Sequence[int]]] = None,
) -> Dict[str, object]:
    nll_matrix = _gaussian_nll_matrix(y_true, y_pred, y_std)
    nll_summary: Dict[str, object] = {
        "overall": float(np.mean(nll_matrix)),
        "per_target": {
            name: float(np.mean(nll_matrix[:, idx])) for idx, name in enumerate(target_names)
        },
    }
    if group_indices:
        nll_summary["per_group"] = {
            group: float(np.mean(nll_matrix[:, list(indices)]))
            for group, indices in group_indices.items()
            if indices
        }

    coverage_summary: Dict[str, Dict[str, object]] = {}
    for level in coverage_levels:
        mask = _coverage_mask(y_true, y_pred, y_std, level)
        coverage_entry: Dict[str, object] = {
            "overall": float(np.mean(mask)),
            "per_target": {
                name: float(np.mean(mask[:, idx])) for idx, name in enumerate(target_names)
            },
        }
        if group_indices:
            coverage_entry["per_group"] = {
                group: float(np.mean(mask[:, list(indices)]))
                for group, indices in group_indices.items()
                if indices
            }
        coverage_summary[str(level)] = coverage_entry

    return {"nll": nll_summary, "coverage": coverage_summary}


def _summarize_quantile_tau(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    tau_map: Mapping[float, float],
    *,
    coverage_levels: Sequence[float],
    target_names: Sequence[str],
    group_indices: Optional[Mapping[str, Sequence[int]]] = None,
    level_weights: Optional[Mapping[float, float]] = None,
) -> Dict[str, object]:
    if not tau_map:
        return {}

    coverage_payload: Dict[str, object] = {}
    nll_payload: Dict[str, object] = {}

    errors: list[float] = []
    weights: list[float] = []

    weight_lookup: Dict[float, float] = {}
    if level_weights is not None:
        weight_lookup = {float(level): float(weight) for level, weight in level_weights.items()}

    ordered_levels: list[float] = []
    seen: set[float] = set()
    for level in coverage_levels:
        level_f = float(level)
        if not 0.0 < level_f < 1.0:
            continue
        if level_f in tau_map and level_f not in seen:
            ordered_levels.append(level_f)
            seen.add(level_f)
    for level in tau_map.keys():
        level_f = float(level)
        if not 0.0 < level_f < 1.0:
            continue
        if level_f not in seen:
            ordered_levels.append(level_f)
            seen.add(level_f)

    for level in ordered_levels:
        tau_value = tau_map[level]
        scaled = scale_standard_deviation(y_std, global_tau=tau_value)
        summary = _summarize_uncertainty(
            y_true,
            y_pred,
            scaled,
            [level],
            target_names=target_names,
            group_indices=group_indices,
        )
        level_key = f"{level:g}"
        coverage_payload[level_key] = summary["coverage"][str(level)]
        nll_payload[level_key] = summary["nll"]
        overall_coverage = float(summary["coverage"][str(level)]["overall"])
        errors.append(overall_coverage - level)
        weights.append(weight_lookup.get(level, 1.0))

    metrics: Dict[str, float] = {}
    if errors:
        weights_arr = np.asarray(weights, dtype=np.float64)
        errors_arr = np.asarray(errors, dtype=np.float64)
        total_weight = float(np.sum(weights_arr))
        if total_weight <= 0.0:
            weights_arr = np.ones_like(errors_arr)
            total_weight = float(len(errors_arr))
        mae = float(np.sum(np.abs(errors_arr) * weights_arr) / total_weight)
        rmse = float(np.sqrt(np.sum((errors_arr ** 2) * weights_arr) / total_weight))
        bias = float(np.sum(errors_arr * weights_arr) / total_weight)
        max_abs = float(np.max(np.abs(errors_arr)))
        metrics = {
            "overall_mae": mae,
            "overall_rmse": rmse,
            "overall_bias": bias,
            "max_abs_error": max_abs,
        }

    payload: Dict[str, object] = {
        "values": {f"{level:g}": float(tau_map[level]) for level in ordered_levels},
        "coverage": coverage_payload,
        "nll": nll_payload,
    }
    if metrics:
        payload["metrics"] = metrics
    if weight_lookup:
        payload["weights"] = {f"{level:g}": float(weight_lookup.get(level, 1.0)) for level in ordered_levels}

    # Provide quick access to the objective deltas for reference / downstream use.
    if errors:
        payload["target_levels"] = [float(level) for level in ordered_levels]
        payload["coverage_deltas"] = [float(err) for err in errors]

    return payload


def build_uncertainty_summary(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    coverage_levels: Sequence[float],
    *,
    global_tau: Optional[float] = None,
    per_group_tau: Optional[Mapping[str, float]] = None,
    group_indices: Optional[Mapping[str, Sequence[int]]] = None,
    quantile_tau: Optional[Mapping[QuantileKey, float]] = None,
    quantile_weights: Optional[Mapping[QuantileKey, float]] = None,
) -> Optional[Dict[str, object]]:
    """Assemble uncertainty diagnostics for raw and scaled predictions."""

    if std_df is None or std_df.empty:
        return None

    y_true = true_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = pred_df.to_numpy(dtype=np.float64, copy=False)
    y_std = std_df.to_numpy(dtype=np.float64, copy=False)
    targets = list(true_df.columns)

    payload: Dict[str, object] = {"coverage_levels": [float(level) for level in coverage_levels]}
    payload["raw"] = _summarize_uncertainty(
        y_true,
        y_pred,
        y_std,
        coverage_levels,
        target_names=targets,
        group_indices=group_indices,
    )

    if global_tau is not None:
        scaled = scale_standard_deviation(y_std, global_tau=global_tau)
        payload["tau_scaled"] = {
            "value": float(global_tau),
            **_summarize_uncertainty(
                y_true,
                y_pred,
                scaled,
                coverage_levels,
                target_names=targets,
                group_indices=group_indices,
            ),
        }

    if per_group_tau:
        if not group_indices:
            raise ValueError("group_indices must be provided when per_group_tau is set")
        scaled = scale_standard_deviation(
            y_std, per_group_tau=per_group_tau, group_indices=group_indices
        )
        payload["per_group_tau_scaled"] = {
            "values": {group: float(value) for group, value in per_group_tau.items()},
            **_summarize_uncertainty(
                y_true,
                y_pred,
                scaled,
                coverage_levels,
                target_names=targets,
                group_indices=group_indices,
            ),
        }

    if quantile_tau:
        tau_map = _normalize_level_mapping(quantile_tau)
        weights_map = (
            _normalize_level_mapping(quantile_weights)
            if quantile_weights is not None
            else None
        )
        quantile_payload = _summarize_quantile_tau(
            y_true,
            y_pred,
            y_std,
            tau_map,
            coverage_levels=coverage_levels,
            target_names=targets,
            group_indices=group_indices,
            level_weights=weights_map,
        )
        if quantile_payload:
            payload["quantile_tau_scaled"] = quantile_payload

    return payload


def _calibrate_tau_core(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    coverage_levels: Sequence[float],
    *,
    level_weights: Optional[Sequence[float]] = None,
) -> Optional[float]:
    levels: list[float] = []
    weights: list[float] = []

    if level_weights is not None and len(level_weights) != len(coverage_levels):
        raise ValueError(
            "level_weights (if provided) must have the same length as coverage_levels"
        )

    for idx, level in enumerate(coverage_levels):
        if not 0.0 < level < 1.0:
            continue
        weight = 1.0
        if level_weights is not None:
            weight = float(level_weights[idx])
            if weight <= 0.0:
                continue
        levels.append(float(level))
        weights.append(weight)

    if not levels:
        return None

    weights_arr = np.asarray(weights, dtype=np.float64)
    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0.0:
        weights_arr = np.ones(len(levels), dtype=np.float64) / float(len(levels))
    else:
        weights_arr = weights_arr / total_weight

    def objective(tau_value: float) -> float:
        scaled = scale_standard_deviation(y_std, global_tau=tau_value)
        error = 0.0
        for level, weight in zip(levels, weights_arr):
            mask = _coverage_mask(y_true, y_pred, scaled, level)
            coverage = float(np.mean(mask))
            diff = coverage - level
            error += weight * diff * diff
        return error

    search_grid = np.logspace(-1, 1, num=60)
    best_tau = 1.0
    best_error = float("inf")
    for candidate in search_grid:
        err = objective(candidate)
        if err < best_error:
            best_error = err
            best_tau = candidate

    fine_lower = max(best_tau * 0.25, 1e-3)
    fine_upper = best_tau * 4.0
    for candidate in np.linspace(fine_lower, fine_upper, num=80):
        if candidate <= 0.0:
            continue
        err = objective(candidate)
        if err < best_error:
            best_error = err
            best_tau = candidate

    return float(best_tau)


def calibrate_global_tau(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    coverage_levels: Sequence[float],
    *,
    level_weights: Optional[Sequence[float]] = None,
) -> Optional[float]:
    if std_df is None or std_df.empty:
        return None
    y_true = true_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = pred_df.to_numpy(dtype=np.float64, copy=False)
    y_std = std_df.to_numpy(dtype=np.float64, copy=False)
    return _calibrate_tau_core(
        y_true,
        y_pred,
        y_std,
        coverage_levels,
        level_weights=level_weights,
    )


def calibrate_groupwise_tau(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    coverage_levels: Sequence[float],
    group_indices: Mapping[str, Sequence[int]],
    *,
    level_weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    if std_df is None or std_df.empty:
        return {}

    values: Dict[str, float] = {}
    for group, indices in group_indices.items():
        if not indices:
            continue
        sub_true = true_df.iloc[:, list(indices)]
        sub_pred = pred_df.iloc[:, list(indices)]
        sub_std = std_df.iloc[:, list(indices)]
        tau = calibrate_global_tau(
            sub_true,
            sub_pred,
            sub_std,
            coverage_levels,
            level_weights=level_weights,
        )
        if tau is not None:
            values[group] = tau
    return values


def calibrate_quantile_tau_map(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    coverage_levels: Sequence[float],
    *,
    level_weights: Optional[Sequence[float]] = None,
) -> Dict[float, float]:
    if std_df is None or std_df.empty:
        return {}

    y_true = true_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = pred_df.to_numpy(dtype=np.float64, copy=False)
    y_std = std_df.to_numpy(dtype=np.float64, copy=False)

    if level_weights is not None and len(level_weights) != len(coverage_levels):
        raise ValueError(
            "level_weights (if provided) must have the same length as coverage_levels"
        )

    tau_values: Dict[float, float] = {}
    for idx, level in enumerate(coverage_levels):
        if not 0.0 < level < 1.0:
            continue
        if level_weights is not None and float(level_weights[idx]) <= 0.0:
            continue
        tau = _calibrate_tau_core(
            y_true,
            y_pred,
            y_std,
            [float(level)],
            level_weights=[1.0],
        )
        if tau is not None:
            tau_values[float(level)] = tau
    return tau_values
