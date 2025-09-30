"""
Data pipeline utilities for tabular regression.

Provides:
- Flip augmentation compatible with rev02 mapping
- StandardScaler fit/apply helpers
- Simple DataModule to produce DataLoaders for train/val/test

Note: MixUp is intentionally not implemented per current plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# New: pluggable flip augmentor
from .augment_config import FlipAugmentor, load_augmentor
from torch.utils.data import DataLoader, TensorDataset


def flip_df(df: pd.DataFrame) -> pd.DataFrame:
    """Left-right flip augmentation for specific domain columns.

    This mirrors the mapping used in rev02_rtd_nf_e3.py so results remain comparable.
    """
    flipped = df.copy()
    mapping = {
        'R1': 'R4', 'R2': 'R3', 'R3': 'R2', 'R4': 'R1',
        'N5': 'N8', 'N6': 'N7', 'N7': 'N6', 'N8': 'N5',
        'D_t4': 'D_t16', 'D_t10': 'D_t10', 'D_t16': 'D_t4',
        'F1': 'F3', 'F2': 'F2', 'F3': 'F1',
    }
    for src, dst in mapping.items():
        if src in flipped.columns and dst in df.columns:
            flipped[src] = df[dst].values
    if {'Ry_t1','Ry_t2','Ry_t3','Ry_t4'}.issubset(df.columns):
        flipped['Ry_t1'] = np.negative(df['Ry_t4'].to_numpy(dtype=float))
        flipped['Ry_t2'] = np.negative(df['Ry_t3'].to_numpy(dtype=float))
        flipped['Ry_t3'] = np.negative(df['Ry_t2'].to_numpy(dtype=float))
        flipped['Ry_t4'] = np.negative(df['Ry_t1'].to_numpy(dtype=float))
    return flipped


def augment_with_flip(train_df: pd.DataFrame) -> pd.DataFrame:
    df_in = train_df.copy()
    flipped = flip_df(df_in)
    aug = pd.concat([df_in, flipped], axis=0, ignore_index=True)
    return aug


@dataclass
class DataConfig:
    train_csv: Path | List[Path]
    test_csv: Path | List[Path]
    input_cols: Sequence[str]
    target_cols: Sequence[str]
    val_ratio: float = 0.1
    shuffle_seed: int = 42
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    augment_flip: bool = True
    augment_profile: Optional[str] = None  # e.g., "rev02_flip", "d3d01_flip", "none"
    augment_config: Optional[Path] = None  # explicit JSON path has priority


class TabularDataModule:
    """Simple DataModule for tabular regression.

    Reads train/test CSVs, applies optional flip augmentation to train, fits StandardScaler
    on train only, transforms all splits, and returns DataLoaders.
    """

    def __init__(self, cfg: DataConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        self._train_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._test_dfs: Optional[List[pd.DataFrame]] = None  # Individual test files
        self._flip_augmentor: Optional[FlipAugmentor] = None

    def setup(self) -> None:
        # Load CSVs
        if isinstance(self.cfg.train_csv, (list, tuple)):
            if len(self.cfg.train_csv) == 0:
                raise SystemExit("No training CSVs provided.")
            train_parts = []
            for p in self.cfg.train_csv:
                part = pd.read_csv(p)
                train_parts.append(part)
            train_df = pd.concat(train_parts, axis=0, ignore_index=True)
        else:
            train_df = pd.read_csv(self.cfg.train_csv)
        # Load test CSV(s)
        if isinstance(self.cfg.test_csv, (list, tuple)):
            if len(self.cfg.test_csv) == 0:
                raise SystemExit("No test CSVs provided.")
            test_parts = []
            self._test_dfs = []
            for p in self.cfg.test_csv:
                part = pd.read_csv(p)
                test_parts.append(part)
                self._test_dfs.append(part.copy())  # Keep individual files
            test_df = pd.concat(test_parts, axis=0, ignore_index=True)
        else:
            test_df = pd.read_csv(self.cfg.test_csv)
            self._test_dfs = [test_df.copy()]  # Single file as list

        # Optional flip augmentation on training set (before scaler fit)
        if self.cfg.augment_flip:
            # Load augmentor if config/profile provided; else fallback to legacy mapping
            if self.cfg.augment_config or self.cfg.augment_profile:
                self._flip_augmentor = load_augmentor(self.cfg.augment_config, self.cfg.augment_profile)
                if self._flip_augmentor is not None:
                    try:
                        flipped = self._flip_augmentor.apply_df(train_df)
                        train_df = pd.concat([train_df, flipped], axis=0, ignore_index=True)
                    except Exception:
                        # Fallback to legacy if config application fails
                        train_df = augment_with_flip(train_df)
                else:
                    # No-op if profile 'none'
                    pass
            else:
                train_df = augment_with_flip(train_df)

        # Fit scalers on the entire (augmented) training set to match rev02 behavior
        X_all = train_df[self.cfg.input_cols].values
        Y_all = train_df[self.cfg.target_cols].values
        X_all_s = self.scaler_x.fit_transform(X_all)
        Y_all_s = self.scaler_y.fit_transform(Y_all)

        # Split after scaling for consistent distributions
        X_tr_s, X_val_s, Y_tr_s, Y_val_s = train_test_split(
            X_all_s, Y_all_s, test_size=self.cfg.val_ratio, random_state=self.cfg.shuffle_seed
        )

        # Transform test using fitted scalers
        X_test_s = self.scaler_x.transform(test_df[self.cfg.input_cols].values)
        Y_test_s = self.scaler_y.transform(test_df[self.cfg.target_cols].values)

        # Create datasets/tensors
        tr_ds = TensorDataset(
            torch.tensor(X_tr_s, dtype=torch.float32, device=self.device),
            torch.tensor(Y_tr_s, dtype=torch.float32, device=self.device),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_s, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_s, dtype=torch.float32, device=self.device),
        )
        test_ds = TensorDataset(
            torch.tensor(X_test_s, dtype=torch.float32, device=self.device),
            torch.tensor(Y_test_s, dtype=torch.float32, device=self.device),
        )

        # DataLoaders
        loader_args = {
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.cfg.pin_memory,
        }
        if self.cfg.num_workers > 0:
            loader_args["persistent_workers"] = True
            loader_args["prefetch_factor"] = 2

        self.train_loader = DataLoader(
            tr_ds,
            shuffle=True,
            drop_last=False,
            **loader_args,
        )
        self.val_loader = DataLoader(
            val_ds,
            shuffle=False,
            drop_last=False,
            **loader_args,
        )
        self.test_loader = DataLoader(
            test_ds,
            shuffle=False,
            drop_last=False,
            **loader_args,
        )

        # Hold original (unscaled) splits for evaluation/CSV saving
        # Provide original (unscaled) splits for potential analysis/CSV needs (optional)
        # Recreate splits on original arrays using same random state to align indices
        X_tr, X_val, Y_tr, Y_val = train_test_split(
            X_all, Y_all, test_size=self.cfg.val_ratio, random_state=self.cfg.shuffle_seed
        )
        self._train_df = pd.DataFrame({
            **{c: X_tr[:, i] for i, c in enumerate(self.cfg.input_cols)},
            **{c: Y_tr[:, i] for i, c in enumerate(self.cfg.target_cols)},
        })
        self._val_df = pd.DataFrame({
            **{c: X_val[:, i] for i, c in enumerate(self.cfg.input_cols)},
            **{c: Y_val[:, i] for i, c in enumerate(self.cfg.target_cols)},
        })
        self._test_df = test_df.copy()

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        assert self.train_loader and self.val_loader and self.test_loader
        return self.train_loader, self.val_loader, self.test_loader

    def get_original_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert self._train_df is not None and self._val_df is not None and self._test_df is not None
        return self._train_df, self._val_df, self._test_df

    def get_scalers(self) -> Tuple[StandardScaler, StandardScaler]:
        return self.scaler_x, self.scaler_y

    def get_individual_test_dfs(self) -> List[pd.DataFrame]:
        """Get individual test DataFrames for separate evaluation."""
        assert self._test_dfs is not None
        return self._test_dfs
