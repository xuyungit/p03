"""
Permutation feature importance for multi-output regression models.

Computes decrease in performance (e.g., increase in RMSE/MAE) when permuting each input column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class PermutationImportanceResult:
    feature_names: Sequence[str]
    importances: np.ndarray  # shape [D]
    stds: np.ndarray         # shape [D]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": list(self.feature_names),
            "importance": self.importances,
            "std": self.stds,
        }).sort_values("importance", ascending=False)


@torch.no_grad()
def _predict_original_units(
    model: nn.Module,
    X_s: np.ndarray,
    scaler_y,
    device: torch.device,
) -> np.ndarray:
    xt = torch.tensor(X_s, dtype=torch.float32, device=device)
    pred_s = model(xt).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s)
    return pred


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_permutation_importance(
    *,
    model: nn.Module,
    X_s: np.ndarray,
    y_true: np.ndarray,  # original units
    scaler_y,
    feature_names: Sequence[str],
    device: torch.device,
    metric: str = "rmse",
    n_repeats: int = 5,
    random_state: int = 42,
) -> PermutationImportanceResult:
    """
    Compute permutation importance on standardized inputs, evaluating metrics in original units.

    Args:
      model: trained model
      X_s: standardized inputs [N, D]
      y_true: true targets in original units [N, T]
      scaler_y: fitted StandardScaler (to invert predictions)
      feature_names: list of input column names (length D)
      device: torch device
      metric: 'rmse' or 'mae' (averaged over outputs)
      n_repeats: number of permutation repeats per feature
      random_state: RNG seed for reproducibility
    """
    rng = np.random.default_rng(random_state)
    # Baseline predictions and score
    y_pred_base = _predict_original_units(model, X_s, scaler_y, device)
    if metric == "rmse":
        base = np.mean([_rmse(y_true[:, i], y_pred_base[:, i]) for i in range(y_true.shape[1])])
    elif metric == "mae":
        base = np.mean([_mae(y_true[:, i], y_pred_base[:, i]) for i in range(y_true.shape[1])])
    else:
        raise ValueError("metric must be 'rmse' or 'mae'")

    N, D = X_s.shape
    imps = np.zeros(D, dtype=float)
    stds = np.zeros(D, dtype=float)

    for j in range(D):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_s.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            y_pred_perm = _predict_original_units(model, X_perm, scaler_y, device)
            if metric == "rmse":
                score = np.mean([
                    _rmse(y_true[:, i], y_pred_perm[:, i]) for i in range(y_true.shape[1])
                ])
            else:
                score = np.mean([
                    _mae(y_true[:, i], y_pred_perm[:, i]) for i in range(y_true.shape[1])
                ])
            scores.append(score - base)  # increase in error
        imps[j] = float(np.mean(scores))
        stds[j] = float(np.std(scores))

    return PermutationImportanceResult(feature_names=feature_names, importances=imps, stds=stds)

