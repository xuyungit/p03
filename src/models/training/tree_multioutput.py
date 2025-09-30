"""Utility helpers for multi-target tree models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:  # pragma: no cover - dependency guard for tooling reuse
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "LightGBM is required for LightGBMMultiTargetRegressor. Install via `uv add lightgbm`."
    ) from exc


def _to_dataframe(X, columns: Optional[Sequence[str]]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        if columns is None:
            return X
        return X.loc[:, columns]
    columns_list = list(columns) if columns is not None else None
    X_arr = np.ascontiguousarray(X)
    return pd.DataFrame(X_arr, columns=columns_list)


def _to_numpy(y) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        return y.to_numpy(dtype=np.float32)
    if isinstance(y, pd.Series):
        return y.to_numpy(dtype=np.float32)[:, None]
    return np.asarray(y, dtype=np.float32)


@dataclass
class LightGBMTrialSummary:
    per_target_best_iteration: List[int]


class LightGBMMultiTargetRegressor:
    """Train one LightGBM regressor per target with optional early stopping."""

    def __init__(
        self,
        base_params: Dict,
        feature_names: Sequence[str],
        *,
        parallel_jobs: Optional[int] = None,
    ) -> None:
        self.base_params = dict(base_params)
        self.feature_names = list(feature_names)
        self.parallel_jobs = parallel_jobs if (parallel_jobs or 0) > 1 else None
        self.models: List[LGBMRegressor] = []
        self.summary: Optional[LightGBMTrialSummary] = None

    def fit(
        self,
        X,
        Y,
        *,
        X_val=None,
        Y_val=None,
        early_stopping_rounds: Optional[int] = None,
    ) -> LightGBMMultiTargetRegressor:
        X_df = _to_dataframe(X, self.feature_names)
        Y_np = _to_numpy(Y)
        if Y_np.ndim != 2:
            raise ValueError("Y must be 2D [n_samples, n_targets]")

        X_val_df = (
            _to_dataframe(X_val, self.feature_names) if X_val is not None else None
        )
        Y_val_np = _to_numpy(Y_val) if Y_val is not None else None
        n_targets = Y_np.shape[1]
        early_stop = (
            early_stopping_rounds
            if early_stopping_rounds and early_stopping_rounds > 0
            else None
        )
        if (
            early_stop is not None
            and str(self.base_params.get("boosting_type", "gbdt")) == "dart"
        ):
            early_stop = None

        def _fit_single(target_idx: int) -> LGBMRegressor:
            estimator = LGBMRegressor(**self.base_params)
            fit_kwargs = {}
            if early_stop is not None and X_val_df is not None and Y_val_np is not None:
                fit_kwargs["eval_set"] = [(X_val_df, Y_val_np[:, target_idx])]
                try:
                    import lightgbm as lgb

                    fit_kwargs["callbacks"] = [
                        lgb.early_stopping(int(early_stop), verbose=False),
                        lgb.log_evaluation(period=0),
                    ]
                except Exception:
                    pass
            estimator.fit(X_df, Y_np[:, target_idx], **fit_kwargs)
            return estimator

        if self.parallel_jobs:
            self.models = Parallel(n_jobs=self.parallel_jobs)(
                delayed(_fit_single)(i) for i in range(n_targets)
            )
        else:
            self.models = [_fit_single(i) for i in range(n_targets)]

        best_iters = []
        for model in self.models:
            best_iters.append(
                int(getattr(model, "best_iteration_", model.n_estimators))
            )
        self.summary = LightGBMTrialSummary(per_target_best_iteration=best_iters)
        return self

    def predict(self, X) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model is not fitted")
        X_df = _to_dataframe(X, self.feature_names)
        preds = [model.predict(X_df) for model in self.models]
        return np.column_stack(preds)

    def feature_importances(self) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model is not fitted")
        mat = np.vstack([model.feature_importances_ for model in self.models])
        return mat

    def get_feature_importance_summary(self) -> Dict[str, np.ndarray]:
        mat = self.feature_importances()
        return {"mean": mat.mean(axis=0), "std": mat.std(axis=0)}

    @property
    def n_targets(self) -> int:
        return len(self.models)

    def __getstate__(self):
        return {
            "base_params": self.base_params,
            "feature_names": self.feature_names,
            "parallel_jobs": self.parallel_jobs,
            "models": self.models,
            "summary": self.summary,
        }

    def __setstate__(self, state):
        self.base_params = state["base_params"]
        self.feature_names = state["feature_names"]
        self.parallel_jobs = state["parallel_jobs"]
        self.models = state["models"]
        self.summary = state.get("summary")
