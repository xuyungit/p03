"""
f_forward

Inference wrapper for a forward model f: (R, N, F) -> A.

This module is analogous to e3_forward.py but for a forward model.

Goals
- Load checkpoint and scalers once (global singleton) for fast repeated calls.
- Allow explicit override for input/target columns.
- Expose a simple `f` function that maps a DataFrame with the training input
  columns to a DataFrame with the training target columns.

Usage
- Required inputs: a pandas DataFrame containing exactly the model's training
  input columns (e.g., R, N, F).
- Output: a pandas DataFrame with the model's training target columns (e.g., A).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


from models.training.builders import ModelConfig, build_model_components, OptimConfig

@dataclass
class _State:
    device: torch.device
    model: torch.nn.Module
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    input_cols: Tuple[str, ...]
    target_cols: Tuple[str, ...]


class ForwardModel:
    """Self-contained wrapper around a trained forward model checkpoint."""

    def __init__(
        self,
        checkpoint: Union[str, Path],
        input_cols: Sequence[str],
        target_cols: Sequence[str],
    ) -> None:
        self.checkpoint_path = Path(checkpoint)
        self.input_cols = tuple(str(c) for c in input_cols)
        self.target_cols = tuple(str(c) for c in target_cols)
        self._state: Optional[_State] = None

    def _load_state(self) -> _State:
        if self._state is not None:
            return self._state

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pkg = torch.load(self.checkpoint_path, map_location=device, weights_only=False)

        scaler_x: StandardScaler = pkg["scaler_x"]
        scaler_y: StandardScaler = pkg["scaler_y"]
        hparams: dict = pkg.get("hyperparameters", {})

        in_dim = int(getattr(scaler_x, "mean_", np.zeros(0)).shape[0])
        out_dim = int(getattr(scaler_y, "mean_", np.zeros(0)).shape[0])

        if len(self.input_cols) != in_dim:
            raise ValueError(
                f"Input columns length {len(self.input_cols)} != model input dim {in_dim}"
            )
        if len(self.target_cols) != out_dim:
            raise ValueError(
                f"Target columns length {len(self.target_cols)} != model output dim {out_dim}"
            )

        model = _build_model_for_checkpoint(hparams, in_dim, out_dim, device)
        model.load_state_dict(pkg["model_state_dict"], strict=True)
        model.eval()

        self._state = _State(
            device=device,
            model=model,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
        )
        return self._state

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        st = self._load_state()
        missing = [c for c in st.input_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required input columns: {missing}")

        X = df.loc[:, list(st.input_cols)].to_numpy(dtype=np.float32, copy=False)
        X_s = st.scaler_x.transform(X)

        with torch.no_grad():
            X_t = torch.tensor(X_s, dtype=torch.float32, device=st.device)
            Y_s = st.model(X_t).cpu().numpy()

        Y = st.scaler_y.inverse_transform(Y_s)
        return pd.DataFrame(Y, columns=list(st.target_cols), index=df.index)

    def predict_numpy(self, X: np.ndarray, columns: Sequence[str]) -> Tuple[np.ndarray, Tuple[str, ...]]:
        st = self._load_state()
        idx_map = {name: i for i, name in enumerate(columns)}
        try:
            idx = [idx_map[name] for name in st.input_cols]
        except KeyError as exc:
            raise KeyError(f"X is missing required column '{exc.args[0]}'")
        X_ord = X[:, idx].astype(np.float32, copy=False)
        X_s = st.scaler_x.transform(X_ord)
        with torch.no_grad():
            X_t = torch.tensor(X_s, dtype=torch.float32, device=st.device)
            Y_s = st.model(X_t).cpu().numpy()
        Y = st.scaler_y.inverse_transform(Y_s)
        return Y, st.target_cols

    def get_state(self) -> dict:
        st = self._load_state()
        return {
            "device": st.device,
            "model": st.model,
            "scaler_x": st.scaler_x,
            "scaler_y": st.scaler_y,
            "input_cols": st.input_cols,
            "target_cols": st.target_cols,
        }


_CHECKPOINT_PATH: Optional[Path] = None
_OVERRIDE_COLUMNS: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
_DEFAULT_MODEL: Optional[ForwardModel] = None


def _build_model_for_checkpoint(hparams: dict, in_dim: int, out_dim: int, device: torch.device) -> torch.nn.Module:
    """Rebuild model using saved hyperparameters and known dims."""
    model_cfg = ModelConfig(
        in_dim=in_dim,
        out_dim=out_dim,
        num_layers=int(hparams.get("num_layers", 4)),
        hidden_dim=int(hparams.get("hidden_dim", 128)),
        act_hidden=str(hparams.get("act_hidden", "silu")),
        act_out=str(hparams.get("act_out", "identity")),
        dropout_p=float(hparams.get("dropout_p", 0.1)),
        model_type=str(hparams.get("model_type", "mlp")),
        norm=str(hparams.get("norm", "none")),
        init=str(hparams.get("init", "xavier")),
        loss=str(hparams.get("loss", "mse")),
        loss_weights=None,
    )
    optim_cfg = OptimConfig(lr=1e-3, weight_decay=0.0, scheduler="none")
    model, _, _, _, _ = build_model_components(model_cfg, optim_cfg, device)
    return model


def _get_default_model(
    explicit_input_cols: Optional[Sequence[str]] = None,
    explicit_target_cols: Optional[Sequence[str]] = None,
) -> ForwardModel:
    global _DEFAULT_MODEL

    if _CHECKPOINT_PATH is None:
        raise RuntimeError("Checkpoint path not set. Call set_checkpoint() first.")

    input_cols: Optional[Tuple[str, ...]] = None
    target_cols: Optional[Tuple[str, ...]] = None
    if _OVERRIDE_COLUMNS is not None:
        input_cols, target_cols = _OVERRIDE_COLUMNS
    if explicit_input_cols is not None:
        input_cols = tuple(str(c) for c in explicit_input_cols)
    if explicit_target_cols is not None:
        target_cols = tuple(str(c) for c in explicit_target_cols)

    if input_cols is None or target_cols is None:
        raise RuntimeError("Input/target columns not configured. Call configure_columns() first.")

    needs_refresh = (
        _DEFAULT_MODEL is None
        or _DEFAULT_MODEL.checkpoint_path != _CHECKPOINT_PATH
        or _DEFAULT_MODEL.input_cols != input_cols
        or _DEFAULT_MODEL.target_cols != target_cols
    )

    if needs_refresh:
        _DEFAULT_MODEL = ForwardModel(
            checkpoint=_CHECKPOINT_PATH,
            input_cols=input_cols,
            target_cols=target_cols,
        )

    return _DEFAULT_MODEL


def f(df: pd.DataFrame) -> pd.DataFrame:
    """Model mapping f: (R, N, F, ...) -> A."""
    model = _get_default_model()
    return model.predict(df)


def f_np(X: np.ndarray, columns: Sequence[str]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Numpy API for f."""
    model = _get_default_model()
    return model.predict_numpy(X, columns)


def get_state() -> dict:
    """Return internal state needed for advanced workflows."""
    model = _get_default_model()
    return model.get_state()


def configure_columns(input_cols: Sequence[str], target_cols: Sequence[str]) -> None:
    """Configure the column names to use for inference."""
    global _OVERRIDE_COLUMNS, _DEFAULT_MODEL
    ic = tuple(str(c) for c in input_cols)
    tc = tuple(str(c) for c in target_cols)
    _OVERRIDE_COLUMNS = (ic, tc)
    _DEFAULT_MODEL = None


def set_checkpoint(path: Union[str, Path]) -> None:
    """Override the checkpoint path and reset loaded state."""
    global _CHECKPOINT_PATH, _DEFAULT_MODEL
    _CHECKPOINT_PATH = Path(path)
    _DEFAULT_MODEL = None


def preload(
    input_cols: Sequence[str],
    target_cols: Sequence[str],
    checkpoint: Union[str, Path]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Eagerly load model and scalers once."""
    set_checkpoint(checkpoint)
    configure_columns(input_cols, target_cols)
    model = _get_default_model()
    model.get_state()
    return model.input_cols, model.target_cols


__all__ = [
    "f",
    "f_np",
    "get_state",
    "configure_columns",
    "set_checkpoint",
    "preload",
    "ForwardModel",
]
