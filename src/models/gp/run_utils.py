"""Utility helpers shared by GP training entrypoints."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from models.evaluation.report import EvaluationReport
from models.gp.common import GPDataModule


def module_base_name(module_file: str) -> str:
    """Return the stem of the module filename for naming experiment folders."""

    return Path(module_file).stem


def default_experiment_root(module_file: str) -> Path:
    """Default root under ``experiments/`` for a GP entrypoint."""

    return Path("experiments") / module_base_name(module_file)


def start_run_dir(root: Path) -> Path:
    """Create a timestamped run directory with checkpoint and plot subfolders."""

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / ts
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_matplotlib(run_dir: Path) -> None:
    """Configure Matplotlib to run headless and cache under the run directory."""

    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
    (run_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    try:  # pragma: no cover - Matplotlib configuration is best effort
        import matplotlib as mpl

        mpl.use("Agg")
    except Exception:  # pragma: no cover - importing mpl may fail in slim envs
        pass


def save_run_config(run_dir: Path, payload: Mapping[str, object]) -> None:
    """Persist configuration payload as ``config.json`` under ``run_dir``."""

    target = run_dir / "config.json"
    # Convert to mutable mapping to ensure JSON serialisation is safe.
    data: MutableMapping[str, object] = dict(payload)
    with target.open("w") as handle:
        json.dump(data, handle, indent=2)


def write_history(run_dir: Path, history_rows: Sequence[Mapping[str, float]]) -> None:
    """Write per-epoch training history to ``history.csv`` if any rows exist."""

    if not history_rows:
        return
    df = pd.DataFrame(history_rows)
    df.to_csv(run_dir / "history.csv", index=False)


def prepare_split_frames(
    inputs_std: np.ndarray,
    true_latent: np.ndarray,
    pred_latent: np.ndarray,
    pred_var_latent: Optional[np.ndarray],
    data_module: GPDataModule,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Convert latent predictions back to original feature spaces for reporting."""

    scaler_x = data_module.scaler_x
    inv_inputs = scaler_x.inverse_transform(inputs_std)
    input_df = pd.DataFrame(inv_inputs, columns=data_module.input_cols)

    true_orig = data_module.decode_targets(true_latent)
    pred_orig = data_module.decode_targets(pred_latent)
    true_df = pd.DataFrame(true_orig, columns=data_module.target_cols)
    pred_df = pd.DataFrame(pred_orig, columns=data_module.target_cols)

    std_df: Optional[pd.DataFrame] = None
    if pred_var_latent is not None:
        projected = data_module.project_latent_variance(pred_var_latent)
        projected = np.maximum(projected, 0.0)
        std_df = pd.DataFrame(np.sqrt(projected), columns=data_module.target_cols)

    return input_df, true_df, pred_df, std_df


def save_report(
    run_dir: Path,
    split_name: str,
    input_df: pd.DataFrame,
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    std_df: Optional[pd.DataFrame],
    error_feature_names: Optional[Sequence[str]] = None,
    model_name: str = "gp",
) -> None:
    """Generate :class:`EvaluationReport` artifacts for a given data split."""

    report = EvaluationReport(model_name=f"{model_name}_{split_name}")
    report.evaluate(true_df, pred_df)
    report.save_artifacts(
        str(run_dir),
        split_name=split_name,
        y_true_df=true_df,
        y_pred_df=pred_df,
        X_df=input_df,
        error_feature_names=list(error_feature_names) if error_feature_names else None,
    )
    if std_df is not None:
        std_df.to_csv(run_dir / f"{split_name}_pred_std.csv", index=False)


def write_uncertainty_metrics(
    run_dir: Path,
    split_name: str,
    payload: Optional[Mapping[str, object]],
) -> None:
    """Persist uncertainty summary (if provided) for a split."""

    if payload is None:
        return
    target_path = run_dir / f"{split_name}_uncertainty.json"
    with target_path.open("w") as handle:
        json.dump(dict(payload), handle, indent=2)


def init_inducing_points(
    train_inputs: np.ndarray,
    inducing: int,
    method: str,
    *,
    seed: int,
) -> torch.Tensor:
    """Initialise inducing points via k-means or random subsampling."""

    if inducing <= 0:
        raise ValueError("Number of inducing points must be positive")
    num_samples = train_inputs.shape[0]
    if inducing > num_samples:
        raise ValueError(
            f"Requested {inducing} inducing points but only {num_samples} training samples available"
        )

    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=inducing, batch_size=min(2048, num_samples), random_state=seed)
        km.fit(train_inputs)
        centers = km.cluster_centers_.astype(np.float32)
    elif method == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(num_samples, size=inducing, replace=False)
        centers = train_inputs[indices].astype(np.float32)
    else:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"Unsupported inducing init method '{method}'")
    return torch.from_numpy(centers)
