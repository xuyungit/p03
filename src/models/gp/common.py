"""Shared scaffolding for Gaussian Process model variants."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.evaluation.report import EvaluationReport
from models.training.data import augment_with_flip, load_augmentor
from models.utils import ColumnSpec, resolve_from_csvs


@dataclass(frozen=True)
class PCAPipelineConfig:
    """Hyperparameters controlling PCA dimensionality reduction."""

    n_components: Optional[int] = None
    variance_threshold: float = 0.98
    max_components: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_components is not None and self.n_components <= 0:
            raise ValueError("n_components must be positive when provided")
        if not (0.0 < self.variance_threshold <= 1.0):
            raise ValueError("variance_threshold must be in (0, 1]")
        if self.max_components is not None and self.max_components <= 0:
            raise ValueError("max_components must be positive when provided")
        if self.n_components is not None and self.max_components is not None:
            if self.n_components > self.max_components:
                raise ValueError("n_components cannot exceed max_components")


class PCAPipeline:
    """PCA wrapper that keeps sklearn objects picklable and handy."""

    def __init__(self, cfg: PCAPipelineConfig) -> None:
        self.cfg = cfg
        self._pca: Optional[PCA] = None
        self._fitted = False

    def _require_fitted(self) -> PCA:
        if not self._fitted or self._pca is None:
            raise RuntimeError("PCA pipeline is not fitted yet")
        return self._pca

    def fit(self, y_std: np.ndarray) -> "PCAPipeline":
        data = np.asarray(y_std, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("PCA fit expects a 2D array")
        n_samples, n_features = data.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("Cannot fit PCA on empty data")
        max_components = self.cfg.max_components or min(n_samples, n_features)
        if self.cfg.n_components is not None:
            n_comp = min(self.cfg.n_components, max_components)
        else:
            probe = PCA(n_components=min(max_components, n_features), svd_solver="full")
            probe.fit(data)
            cumulative = np.cumsum(probe.explained_variance_ratio_)
            threshold = min(max(self.cfg.variance_threshold, 0.0), 1.0)
            idx = int(np.searchsorted(cumulative, threshold, side="right"))
            n_comp = max(idx + 1, 1)
            if self.cfg.max_components is not None:
                n_comp = min(n_comp, self.cfg.max_components)
        pca = PCA(n_components=n_comp, svd_solver="full")
        pca.fit(data)
        self._pca = pca
        self._fitted = True
        return self

    @property
    def n_components_(self) -> int:
        return self._require_fitted().n_components_

    @property
    def components_(self) -> np.ndarray:
        return self._require_fitted().components_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self._require_fitted().explained_variance_ratio_

    def transform(self, y_std: np.ndarray) -> np.ndarray:
        pca = self._require_fitted()
        data = np.asarray(y_std, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("PCA transform expects a 2D array")
        return pca.transform(data)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        pca = self._require_fitted()
        latent = np.asarray(z, dtype=np.float64)
        if latent.ndim != 2:
            raise ValueError("PCA inverse_transform expects a 2D array")
        return pca.inverse_transform(latent)

    def project_variance(
        self,
        latent_variance: np.ndarray,
        *,
        scale: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pca = self._require_fitted()
        latent = np.asarray(latent_variance, dtype=np.float64)
        if latent.ndim == 1:
            latent = latent[None, :]
            squeeze = True
        elif latent.ndim == 2:
            squeeze = False
        else:
            raise ValueError("latent_variance must be 1D or 2D array")
        if latent.shape[-1] != pca.n_components_:
            raise ValueError(
                f"Variance dimension {latent.shape[-1]} does not match PCA components {pca.n_components_}"
            )
        comp_sq = np.square(pca.components_)  # (n_components, n_features)
        cov_diag_std = latent @ comp_sq  # (n?, n_features)
        if scale is not None:
            scale_arr = np.asarray(scale, dtype=np.float64)
            if scale_arr.ndim != 1:
                raise ValueError("scale must be a 1D array")
            if scale_arr.shape[0] != comp_sq.shape[1]:
                raise ValueError(
                    f"Scale length {scale_arr.shape[0]} does not match output dim {comp_sq.shape[1]}"
                )
            cov_diag_std = cov_diag_std * np.square(scale_arr)
        if squeeze:
            return cov_diag_std[0]
        return cov_diag_std

    def save(self, path: Path) -> None:
        pca = self._require_fitted()
        dump({"config": self.cfg, "pca": pca}, Path(path))

    @classmethod
    def load(cls, path: Path) -> "PCAPipeline":
        payload = load(Path(path))
        cfg = payload["config"]
        pca_obj = payload["pca"]
        pipe = cls(cfg)
        pipe._pca = pca_obj
        pipe._fitted = True
        return pipe


@dataclass
class GPDataConfig:
    """Configuration for preparing data for GP models."""

    train_csv: Sequence[Path]
    test_csv: Sequence[Path]
    input_spec: ColumnSpec
    target_spec: ColumnSpec
    val_ratio: float = 0.15
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_seed: int = 42
    augment_flip: bool = True
    augment_profile: Optional[str] = None
    augment_config: Optional[Path] = None
    use_pca: bool = True
    pca: PCAPipelineConfig = field(default_factory=PCAPipelineConfig)

    def __post_init__(self) -> None:
        if not self.train_csv:
            raise ValueError("At least one training CSV must be provided")
        if not self.test_csv:
            raise ValueError("At least one test CSV must be provided")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError("val_ratio must be in [0, 1)")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not isinstance(self.input_spec, ColumnSpec):
            raise TypeError("input_spec must be a ColumnSpec")
        if not isinstance(self.target_spec, ColumnSpec):
            raise TypeError("target_spec must be a ColumnSpec")


class GPDataModule:
    """Prepares standardized (and optionally PCA-projected) datasets for GP training."""

    def __init__(self, cfg: GPDataConfig) -> None:
        self.cfg = cfg
        self.input_cols: List[str] = []
        self.target_cols: List[str] = []
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.pca: Optional[PCAPipeline] = None

        self._train_dataset: Optional[TensorDataset] = None
        self._val_dataset: Optional[TensorDataset] = None
        self._test_dataset: Optional[TensorDataset] = None

        self._train_targets_std: Optional[np.ndarray] = None
        self._val_targets_std: Optional[np.ndarray] = None
        self._test_targets_std: Optional[np.ndarray] = None

        self._train_targets_encoded: Optional[np.ndarray] = None
        self._val_targets_encoded: Optional[np.ndarray] = None
        self._test_targets_encoded: Optional[np.ndarray] = None

        self._train_inputs: Optional[np.ndarray] = None
        self._val_inputs: Optional[np.ndarray] = None
        self._test_inputs: Optional[np.ndarray] = None

        self._train_aug_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._test_dfs: Optional[List[pd.DataFrame]] = None

    @staticmethod
    def _ensure_paths(paths: Sequence[Path]) -> List[Path]:
        return [Path(p) for p in paths]

    def setup(self) -> None:
        train_paths = self._ensure_paths(self.cfg.train_csv)
        test_paths = self._ensure_paths(self.cfg.test_csv)
        resolved = resolve_from_csvs(train_paths, test_paths, self.cfg.input_spec, self.cfg.target_spec)
        self.input_cols = list(resolved["input"])
        self.target_cols = list(resolved["target"])

        train_frames = [pd.read_csv(path) for path in train_paths]
        test_frames = [pd.read_csv(path) for path in test_paths]
        train_df = pd.concat(train_frames, axis=0, ignore_index=True)
        test_df = pd.concat(test_frames, axis=0, ignore_index=True)
        self._test_df = test_df
        self._test_dfs = test_frames

        if self.cfg.augment_flip:
            augmentor = None
            if self.cfg.augment_config or self.cfg.augment_profile:
                augmentor = load_augmentor(self.cfg.augment_config, self.cfg.augment_profile)
            if augmentor is not None:
                try:
                    flipped = augmentor.apply_df(train_df)
                except Exception:
                    flipped = augment_with_flip(train_df)
            else:
                flipped = augment_with_flip(train_df)
            train_aug_df = pd.concat([train_df, flipped], axis=0, ignore_index=True)
        else:
            train_aug_df = train_df.copy()
        self._train_aug_df = train_aug_df

        X_all = train_aug_df[self.input_cols].to_numpy(dtype=np.float32, copy=True)
        Y_all = train_aug_df[self.target_cols].to_numpy(dtype=np.float32, copy=True)
        X_all_s = self.scaler_x.fit_transform(X_all)
        Y_all_s = self.scaler_y.fit_transform(Y_all)

        if self.cfg.val_ratio > 0.0:
            X_tr, X_val, Y_tr, Y_val = train_test_split(
                X_all_s,
                Y_all_s,
                test_size=self.cfg.val_ratio,
                random_state=self.cfg.shuffle_seed,
            )
        else:
            X_tr, Y_tr = X_all_s, Y_all_s
            X_val = None
            Y_val = None

        X_test = test_df[self.input_cols].to_numpy(dtype=np.float32, copy=True)
        Y_test = test_df[self.target_cols].to_numpy(dtype=np.float32, copy=True)
        X_test_s = self.scaler_x.transform(X_test)
        Y_test_s = self.scaler_y.transform(Y_test)

        if self.cfg.use_pca:
            self.pca = PCAPipeline(self.cfg.pca)
            self.pca.fit(Y_all_s)
            Y_tr_enc = self.pca.transform(Y_tr)
            Y_val_enc = self.pca.transform(Y_val) if Y_val is not None else None
            Y_test_enc = self.pca.transform(Y_test_s)
        else:
            Y_tr_enc = Y_tr
            Y_val_enc = Y_val
            Y_test_enc = Y_test_s

        self._train_inputs = X_tr.astype(np.float32, copy=False)
        self._train_targets_std = Y_tr.astype(np.float32, copy=False)
        self._train_targets_encoded = Y_tr_enc.astype(np.float32, copy=False)

        self._test_inputs = X_test_s.astype(np.float32, copy=False)
        self._test_targets_std = Y_test_s.astype(np.float32, copy=False)
        self._test_targets_encoded = Y_test_enc.astype(np.float32, copy=False)

        if X_val is not None and Y_val is not None and Y_val_enc is not None:
            self._val_inputs = X_val.astype(np.float32, copy=False)
            self._val_targets_std = Y_val.astype(np.float32, copy=False)
            self._val_targets_encoded = Y_val_enc.astype(np.float32, copy=False)
        else:
            self._val_inputs = None
            self._val_targets_std = None
            self._val_targets_encoded = None

        self._train_dataset = TensorDataset(
            torch.from_numpy(self._train_inputs),
            torch.from_numpy(self._train_targets_encoded),
        )
        if self._val_inputs is not None and self._val_targets_encoded is not None:
            self._val_dataset = TensorDataset(
                torch.from_numpy(self._val_inputs),
                torch.from_numpy(self._val_targets_encoded),
            )
        else:
            self._val_dataset = None
        self._test_dataset = TensorDataset(
            torch.from_numpy(self._test_inputs),
            torch.from_numpy(self._test_targets_encoded),
        )

    # DataLoader helpers -------------------------------------------------
    def _loader(self, ds: Optional[TensorDataset], *, shuffle: bool) -> Optional[DataLoader]:
        if ds is None:
            return None
        loader_args = {
            "batch_size": self.cfg.batch_size,
            "shuffle": shuffle,
            "drop_last": False,
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.cfg.pin_memory,
        }
        if self.cfg.num_workers > 0:
            loader_args["persistent_workers"] = True
            loader_args["prefetch_factor"] = 2
        return DataLoader(ds, **loader_args)

    def train_dataloader(self, *, shuffle: bool = True) -> DataLoader:
        loader = self._loader(self._train_dataset, shuffle=shuffle)
        if loader is None:
            raise RuntimeError("Training dataset is not prepared; call setup() first")
        return loader

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._loader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        loader = self._loader(self._test_dataset, shuffle=False)
        if loader is None:
            raise RuntimeError("Test dataset is not prepared; call setup() first")
        return loader

    # Accessors ----------------------------------------------------------
    @property
    def target_latent_dim(self) -> int:
        if self.pca is not None:
            return self.pca.n_components_
        return len(self.target_cols)

    def encode_targets(self, y: np.ndarray | torch.Tensor | pd.DataFrame) -> np.ndarray:
        data = _to_numpy(y)
        y_std = self.scaler_y.transform(data)
        if self.pca is not None:
            return self.pca.transform(y_std)
        return y_std

    def decode_targets(self, y_encoded: np.ndarray | torch.Tensor | pd.DataFrame) -> np.ndarray:
        latent = _to_numpy(y_encoded)
        if self.pca is not None:
            y_std = self.pca.inverse_transform(latent)
        else:
            y_std = latent
        return self.scaler_y.inverse_transform(y_std)

    def project_latent_variance(self, latent_var: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.pca is None:
            # Already in standardized space; scale back directly.
            var = _to_numpy(latent_var)
            scale_sq = np.square(self.scaler_y.scale_)
            if var.ndim == 1:
                return var * scale_sq
            return var * scale_sq[None, :]
        latent = _to_numpy(latent_var)
        return self.pca.project_variance(latent, scale=self.scaler_y.scale_)

    def save_preprocessors(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dump(self.scaler_x, directory / "scaler_x.pkl")
        dump(self.scaler_y, directory / "scaler_y.pkl")
        if self.pca is not None:
            self.pca.save(directory / "pca.pkl")

    def train_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._train_inputs is None or self._train_targets_encoded is None:
            raise RuntimeError("DataModule not set up")
        return self._train_inputs, self._train_targets_encoded

    def val_arrays(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._val_inputs is None or self._val_targets_encoded is None:
            return None
        return self._val_inputs, self._val_targets_encoded

    def test_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._test_inputs is None or self._test_targets_encoded is None:
            raise RuntimeError("DataModule not set up")
        return self._test_inputs, self._test_targets_encoded

    def original_train_dataframe(self) -> Optional[pd.DataFrame]:
        return self._train_aug_df

    def combined_test_dataframe(self) -> Optional[pd.DataFrame]:
        return self._test_df

    def individual_test_dataframes(self) -> Optional[List[pd.DataFrame]]:
        return self._test_dfs


def _to_numpy(data: np.ndarray | torch.Tensor | pd.DataFrame | Sequence[float]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float64, copy=False)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float64, copy=False)
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=np.float64, copy=False)
    return np.asarray(data, dtype=np.float64)


def make_evaluation_report(
    model_name: str,
    target_cols: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> EvaluationReport:
    true_df = pd.DataFrame(y_true, columns=target_cols)
    pred_df = pd.DataFrame(y_pred, columns=target_cols)
    report = EvaluationReport(model_name)
    report.evaluate(true_df, pred_df)
    return report


@dataclass(frozen=True)
class TargetGroupMapping:
    """Maps output targets to logical groups for calibration/analysis."""

    group_to_columns: Dict[str, List[str]]
    column_to_group: Dict[str, str]
    group_to_indices: Dict[str, List[int]]

    def to_serializable(self) -> Dict[str, List[str]]:
        """Return a JSON-serialisable copy of the group→columns map."""

        return {group: list(cols) for group, cols in self.group_to_columns.items()}


def _assign_membership(
    target_cols: Sequence[str],
    initial_membership: Mapping[str, str],
) -> TargetGroupMapping:
    target_list = list(target_cols)

    normalized: Dict[str, List[str]] = {}
    column_to_group: Dict[str, str] = {col: grp for col, grp in initial_membership.items()}
    group_to_indices: Dict[str, List[int]] = {}

    for idx, col in enumerate(target_list):
        group = column_to_group.get(col)
        if group is None:
            group = col
            column_to_group[col] = group
        normalized.setdefault(group, []).append(col)
        group_to_indices.setdefault(group, []).append(idx)

    return TargetGroupMapping(normalized, column_to_group, group_to_indices)


def _load_group_config(path: Path) -> Dict[str, Dict[str, List[str]]]:
    try:
        with Path(path).open("r") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Failed to parse tau group config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Tau group config must be a JSON object of group→patterns")

    config: Dict[str, Dict[str, List[str]]] = {}
    for group_name, patterns in payload.items():
        explicit_cols: List[str] = []
        regexes: List[str] = []
        if isinstance(patterns, str):
            regexes = [patterns]
        elif isinstance(patterns, list):
            if not all(isinstance(item, str) for item in patterns):
                raise ValueError(
                    f"Tau group config for '{group_name}' must contain only strings"
                )
            regexes = [str(item) for item in patterns]
        elif isinstance(patterns, dict):
            cols = patterns.get("columns", [])
            pats = patterns.get("patterns", [])
            if not all(isinstance(item, str) for item in cols):
                raise ValueError(
                    f"Tau group config for '{group_name}' has non-string entries under 'columns'"
                )
            if not all(isinstance(item, str) for item in pats):
                raise ValueError(
                    f"Tau group config for '{group_name}' has non-string entries under 'patterns'"
                )
            explicit_cols = [str(item) for item in cols]
            regexes = [str(item) for item in pats]
        else:
            raise ValueError(
                f"Unsupported entry for group '{group_name}'; expected string/list/dict"
            )
        if not explicit_cols and not regexes:
            raise ValueError(
                f"Group '{group_name}' in tau config must specify at least one column or pattern"
            )
        config[group_name] = {"columns": explicit_cols, "patterns": regexes}
    return config


def resolve_target_groups(
    target_cols: Sequence[str],
    *,
    mode: str = "auto",
    config_path: Optional[Path] = None,
) -> Optional[TargetGroupMapping]:
    """Determine grouping of output targets for tau scaling/calibration.

    Parameters
    ----------
    target_cols:
        Ordered list of output column names.
    mode:
        ``"none"`` skips grouping entirely. ``"auto"`` groups by leading
        alphabetical prefix (matching ``^[A-Za-z]+``). ``"config"`` uses a
        user-provided JSON mapping.
    config_path:
        Path to a JSON file mapping group names to regex patterns (or explicit
        column lists) when ``mode="config"``.
    """

    mode_normalized = (mode or "auto").lower()
    if mode_normalized not in {"none", "auto", "config"}:
        raise ValueError(f"Unsupported tau group mode '{mode}'; expected none/auto/config")
    if mode_normalized == "none":
        return None

    target_list = list(target_cols)
    if not target_list:
        return None

    assigned: Dict[str, str] = {}

    if mode_normalized == "config":
        if config_path is None:
            raise ValueError("per-group tau mode 'config' requires --per-group-tau-config")
        pattern_map = _load_group_config(config_path)
        used_columns: Dict[str, str] = {}
        for group_name, spec in pattern_map.items():
            matched_for_group: List[str] = []

            for col in spec.get("columns", []):
                if col not in target_list:
                    raise ValueError(
                        f"Column '{col}' listed under group '{group_name}' is not present in targets"
                    )
                previous = used_columns.get(col)
                if previous is not None and previous != group_name:
                    raise ValueError(
                        f"Column '{col}' assigned to multiple tau groups: '{previous}' and '{group_name}'"
                    )
                used_columns[col] = group_name
                if col not in matched_for_group:
                    matched_for_group.append(col)

            for pattern in spec.get("patterns", []):
                try:
                    regex = re.compile(pattern)
                except re.error as exc:  # pragma: no cover - defensive path
                    raise ValueError(
                        f"Invalid regex '{pattern}' for group '{group_name}': {exc}"
                    ) from exc
                for col in target_list:
                    if not regex.match(col):
                        continue
                    previous = used_columns.get(col)
                    if previous is not None and previous != group_name:
                        raise ValueError(
                            f"Column '{col}' matched by multiple tau groups: '{previous}' and '{group_name}'"
                        )
                    used_columns[col] = group_name
                    if col not in matched_for_group:
                        matched_for_group.append(col)

            if not matched_for_group:
                raise ValueError(
                    f"Group '{group_name}' in tau config did not match any target columns"
                )

        assigned = used_columns
    else:  # auto mode
        prefix_pattern = re.compile(r"^[A-Za-z]+")
        for col in target_list:
            match = prefix_pattern.match(col)
            if match:
                assigned[col] = match.group(0)

    return _assign_membership(target_list, assigned)
