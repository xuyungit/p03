"""Uncertainty utilities shared across GP model entrypoints."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

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


def build_uncertainty_summary(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    coverage_levels: Sequence[float],
    *,
    global_tau: Optional[float] = None,
    per_group_tau: Optional[Mapping[str, float]] = None,
    group_indices: Optional[Mapping[str, Sequence[int]]] = None,
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

    return payload


def _calibrate_tau_core(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    coverage_levels: Sequence[float],
) -> Optional[float]:
    levels = [level for level in coverage_levels if 0.0 < level < 1.0]
    if not levels:
        return None

    def objective(tau_value: float) -> float:
        scaled = scale_standard_deviation(y_std, global_tau=tau_value)
        error = 0.0
        for level in levels:
            mask = _coverage_mask(y_true, y_pred, scaled, level)
            coverage = float(np.mean(mask))
            error += (coverage - level) ** 2
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
) -> Optional[float]:
    if std_df is None or std_df.empty:
        return None
    y_true = true_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = pred_df.to_numpy(dtype=np.float64, copy=False)
    y_std = std_df.to_numpy(dtype=np.float64, copy=False)
    return _calibrate_tau_core(y_true, y_pred, y_std, coverage_levels)


def calibrate_groupwise_tau(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    coverage_levels: Sequence[float],
    group_indices: Mapping[str, Sequence[int]],
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
        tau = calibrate_global_tau(sub_true, sub_pred, sub_std, coverage_levels)
        if tau is not None:
            values[group] = tau
    return values
