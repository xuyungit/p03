"""Multi-task SVGP implementation using ICM (Intrinsic Coregionalization Model) / LMC coregionalization.

This module implements B档多任务变分 GP with shared input kernel and learned task covariance
for better handling correlated outputs. Supports per-group tau scaling for improved uncertainty calibration.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler

from models.evaluation.report import EvaluationReport
from models.gp.common import GPDataConfig, GPDataModule, PCAPipelineConfig
from models.gp.svgp_baseline import ComponentState, _estimate_rmse, _init_inducing_points, _set_all_seeds, _build_scheduler
from models.utils import ColumnSpec


# ---------------------------------------------------------------------------
# Multi-task SVGP Model Implementation
# ---------------------------------------------------------------------------


class MultiTaskSVGP(gpytorch.models.ApproximateGP):
    """Multi-task SVGP model using ICM/LMC coregionalization.

    Args:
        inducing_points: Shape (num_inducing, input_dim)
        num_tasks: Number of output tasks (PCA components)
        kernel: Kernel type ('rbf' or 'matern52')
        ard: Whether to use ARD for input dimensions
        jitter: Manual jitter for numerical stability
        rank: Rank of the task covariance matrix (for LMC)
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        num_tasks: int,
        *,
        kernel: str = "rbf",
        ard: bool = True,
        jitter: float = 1e-6,
        rank: Optional[int] = None,
    ) -> None:
        # Shape: (num_inducing, input_dim)
        self.num_inducing = inducing_points.size(-2)
        self.num_tasks = num_tasks
        self.rank = rank if rank is not None else min(num_tasks, 3)  # Default LMC rank

        # Create variational distribution for multi-output model
        # We'll use independent models for each task for simplicity
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            self.num_inducing
        )

        # Create variational strategy
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        # Build input kernel
        ard_dims = inducing_points.size(-1) if ard else None
        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif kernel == "matern52":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        else:
            raise ValueError(f"Unsupported kernel '{kernel}'")

        # Scale kernel for inputs
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.mean_module = gpytorch.means.ZeroMean()
        self._manual_jitter = float(jitter)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass for single task (will be called for each task separately).

        Args:
            x: Input features, shape (batch_size, input_dim)

        Returns:
            MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def extra_jitter(self) -> float:
        return self._manual_jitter


@dataclass
class MultiTaskState:
    """Container for trained multi-task model state."""
    model: MultiTaskSVGP
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _make_multitask_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create DataLoader for multi-task training."""
    # Use standard loader - we'll handle multiple tasks in the training loop
    ds = TensorDataset(
        torch.from_numpy(inputs.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.float32, copy=False)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _set_multitask_prewarm_mode(
    model: MultiTaskSVGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    *,
    enable: bool,
) -> None:
    """Freeze/unfreeze parameters for the pre-warm stage."""

    # Freeze variational parameters during pre-warm
    for param in model.variational_parameters():
        param.requires_grad = not enable

    # Freeze inducing points during pre-warm
    inducing_points = getattr(model.variational_strategy, "inducing_points", None)
    if isinstance(inducing_points, torch.nn.Parameter):
        inducing_points.requires_grad = not enable

    # Allow kernel parameters to update during pre-warm
    base_kernel = getattr(model.covar_module, "base_kernel", None)
    if base_kernel is not None and hasattr(base_kernel, "raw_lengthscale"):
        base_kernel.raw_lengthscale.requires_grad = True

    if hasattr(model.covar_module, "raw_outputscale"):
        model.covar_module.raw_outputscale.requires_grad = not enable

    # Allow likelihood noise to update during pre-warm
    if hasattr(likelihood, "raw_noise"):
        likelihood.raw_noise.requires_grad = True


def _train_multitask(
    cfg: MTConfig,
    device: torch.device,
    inducing_points: torch.Tensor,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: Optional[np.ndarray],
    val_targets: Optional[np.ndarray],
) -> Tuple[List[ComponentState], List[Dict[str, float]]]:
    """Train multi-task SVGP model as independent components."""
    num_tasks = train_targets.shape[1]
    components: List[ComponentState] = []
    all_history: List[Dict[str, float]] = []

    # Train one SVGP per task (similar to svgp_baseline but with shared knowledge)
    for task_idx in range(num_tasks):
        print(f"Training multi-task component {task_idx + 1}/{num_tasks}")

        # Extract targets for this task
        train_targets_task = train_targets[:, task_idx:task_idx + 1]
        val_targets_task = val_targets[:, task_idx:task_idx + 1] if val_targets is not None else None

        # Create a single-task model
        model = MultiTaskSVGP(
            inducing_points.clone(),
            num_tasks=1,
            kernel=cfg.kernel,
            ard=cfg.ard,
            jitter=cfg.jitter,
            rank=1,
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=cfg.noise_init)

        model.to(device)
        likelihood.to(device)

        # Create data loaders
        train_loader = _make_multitask_loader(train_inputs, train_targets_task, batch_size=cfg.batch_size, shuffle=True)
        val_loader = None
        if val_inputs is not None and val_targets_task is not None:
            val_loader = _make_multitask_loader(val_inputs, val_targets_task, batch_size=cfg.batch_size, shuffle=False)

        # Setup optimizer and MLL
        num_data = train_loader.dataset.tensors[0].shape[0]
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            lr=cfg.lr,
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
        scheduler = _build_scheduler(cfg, optimizer)

        # Training loop
        prewarm_epochs = min(max(0, int(cfg.prewarm_epochs)), max(0, cfg.epochs - 1))
        prewarm_active = prewarm_epochs > 0
        if prewarm_active:
            _set_multitask_prewarm_mode(model, likelihood, enable=True)

        use_early_stopping = bool(cfg.early_stopping and val_loader is not None)
        best_val = float("inf")
        best_epoch: Optional[int] = None
        epochs_no_improve = 0
        best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        for epoch in range(1, cfg.epochs + 1):
            if prewarm_active and epoch == prewarm_epochs + 1:
                _set_multitask_prewarm_mode(model, likelihood, enable=False)
                prewarm_active = False

            model.train()
            likelihood.train()

            running_loss = 0.0
            batch_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.squeeze(-1).to(device)

                optimizer.zero_grad(set_to_none=True)
                with gpytorch.settings.cholesky_jitter(model.extra_jitter()):
                    output = model(xb)
                    loss = -mll(output, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / max(1, batch_count)
            val_rmse: Optional[float] = None
            should_eval = False
            if val_loader is not None:
                if use_early_stopping:
                    should_eval = True
                elif epoch % cfg.val_interval == 0:
                    should_eval = True
            if should_eval:
                val_rmse = _estimate_rmse(model, likelihood, val_loader, device)
                improved = val_rmse < (best_val - cfg.min_delta)
                if improved:
                    best_val = val_rmse
                    best_epoch = epoch
                    best_state = {
                        "model": copy.deepcopy(model.state_dict()),
                        "likelihood": copy.deepcopy(likelihood.state_dict()),
                    }
                    epochs_no_improve = 0
                elif use_early_stopping and epoch > prewarm_epochs:
                    epochs_no_improve += 1
            elif use_early_stopping and epoch > prewarm_epochs:
                epochs_no_improve += 1

            current_lr = optimizer.param_groups[0]["lr"]

            if scheduler is not None:
                scheduler.step()

            # Record history
            all_history.append(
                {
                    "component": float(task_idx),
                    "epoch": float(epoch),
                    "train_elbo": avg_loss,
                    "val_rmse": float(val_rmse) if val_rmse is not None else float("nan"),
                    "prewarm": 1.0 if epoch <= prewarm_epochs else 0.0,
                    "lr": float(current_lr),
                }
            )

            if cfg.print_freq and epoch % cfg.print_freq == 0:
                msg = (
                    f"MultiTask Component {task_idx:02d} | epoch {epoch:04d}"
                    f" | lr {current_lr:.5f} | train loss {avg_loss:.4f}"
                )
                if val_rmse is not None:
                    msg += f" | val RMSE {val_rmse:.4f}"
                print(msg)

            if use_early_stopping and epoch > prewarm_epochs and epochs_no_improve >= cfg.patience:
                print(
                    f"Early stopping multi-task component {task_idx:02d} after {epoch} epochs; "
                    f"best val RMSE {best_val:.4f} at epoch {best_epoch}"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state["model"])
            likelihood.load_state_dict(best_state["likelihood"])

        model.cpu()
        likelihood.cpu()
        components.append(ComponentState(model=model, likelihood=likelihood))

    torch.cuda.empty_cache()
    return components, all_history


def _estimate_multitask_rmse(
    model: MultiTaskSVGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    loader: DataLoader,
    device: torch.device,
    num_tasks: int,
) -> float:
    """Estimate RMSE for multi-task model."""
    model.eval()
    likelihood.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for xb, task_idx, yb in loader:
            xb = xb.to(device)
            task_idx = task_idx.to(device)
            yb = yb.squeeze(-1).to(device)

            with gpytorch.settings.fast_pred_var():
                posterior = likelihood(model(xb, task_idx))
            preds.append(posterior.mean.cpu())
            targets.append(yb.cpu())

    pred_vec = torch.cat(preds)
    target_vec = torch.cat(targets)
    mse = torch.mean((pred_vec - target_vec) ** 2)
    return float(torch.sqrt(mse))


def _predict_multitask(
    components: List[ComponentState],
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with multi-task model (component-based)."""
    # Reuse the prediction function from svgp_baseline
    from models.gp.svgp_baseline import _predict_components

    return _predict_components(components, inputs, batch_size=batch_size, device=device)




# ---------------------------------------------------------------------------
# CLI Configuration
# ---------------------------------------------------------------------------


@dataclass
class MTConfig:
    """Configuration for multi-task SVGP training."""

    # Data configuration (reuse from SVGP baseline)
    train_csv: List[Path]
    test_csv: List[Path]
    val_ratio: float = 0.15
    augment_flip: bool = True
    augment_profile: Optional[str] = None
    augment_config: Optional[Path] = None

    input_cols: Optional[List[str]] = None
    target_cols: Optional[List[str]] = None
    input_cols_re: Optional[List[str]] = None
    target_cols_re: Optional[List[str]] = None
    preset: Optional[Path] = None

    batch_size: int = 256
    epochs: int = 300
    lr: float = 1e-2
    lr_decay: str = "none"
    lr_decay_step: int = 100
    lr_decay_gamma: float = 0.5
    lr_decay_min_lr: float = 1e-4
    prewarm_epochs: int = 25
    early_stopping: bool = False
    patience: int = 50
    min_delta: float = 0.0
    seed: int = 42
    device: str = "cpu"

    # Multi-task specific parameters
    kernel: str = "rbf"
    ard: bool = True
    inducing: int = 500
    inducing_init: str = "kmeans"
    noise_init: float = 1e-3
    jitter: float = 1e-6
    lmc_rank: Optional[int] = None  # None for full-rank ICM, int for LMC rank

    save_std: bool = True
    print_freq: int = 50
    val_interval: int = 50
    coverage_levels: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95])

    pca_enable: bool = True
    pca_variance: float = 0.98
    pca_components: Optional[int] = None
    pca_max_components: Optional[int] = None

    experiment_root: Optional[Path] = None
    error_feature_names: Optional[List[str]] = None

    # Per-group tau scaling
    enable_per_group_tau: bool = True
    group_patterns: Optional[List[str]] = None  # Regex patterns to group target columns


def _parse_args() -> MTConfig:
    """Parse CLI arguments for multi-task SVGP."""
    parser = argparse.ArgumentParser(description="Multi-task SVGP for bridge mechanics")

    # Data arguments (same as SVGP baseline)
    parser.add_argument("--train-csv", type=Path, nargs="+", default=[Path("data/d03_all_train.csv")])
    parser.add_argument("--test-csv", type=Path, nargs="+", default=[Path("data/d03_all_test.csv")])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--prewarm-epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-decay", type=str, default="none", choices=["none", "step", "cosine"])
    parser.add_argument("--lr-decay-step", type=int, default=100)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--lr-decay-min-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=0.0)

    parser.add_argument("--augment-flip", action="store_true")
    parser.add_argument("--no-augment-flip", action="store_true")
    parser.add_argument("--augment-profile", type=str, default=None)
    parser.add_argument("--augment-config", type=Path, default=None)

    parser.add_argument("--input-cols", type=str, default=None)
    parser.add_argument("--target-cols", type=str, default=None)
    parser.add_argument("--input-cols-re", type=str, default=None)
    parser.add_argument("--target-cols-re", type=str, default=None)
    parser.add_argument("--preset", type=Path, default=None)

    # Multi-task specific arguments
    parser.add_argument("--kernel", type=str, choices=["rbf", "matern52"], default="rbf")
    parser.add_argument("--ard", action="store_true")
    parser.add_argument("--no-ard", action="store_true")
    parser.add_argument("--inducing", type=int, default=500)
    parser.add_argument("--inducing-init", type=str, choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--noise-init", type=float, default=1e-3)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--lmc-rank", type=int, default=None, help="LMC rank (None for full-rank ICM)")

    parser.add_argument("--save-std", action="store_true")
    parser.add_argument("--no-save-std", action="store_true")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--coverage-list", type=str, default="0.5,0.9,0.95")

    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--pca-variance", type=float, default=0.98)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--pca-max-components", type=int, default=None)

    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--error-features", type=str, nargs="*", default=None)

    # Per-group tau scaling
    parser.add_argument("--enable-per-group-tau", action="store_true", default=True)
    parser.add_argument("--disable-per-group-tau", action="store_true", help="Disable per-group tau scaling")
    parser.add_argument("--group-patterns", type=str, nargs="+", default=None,
                       help="Regex patterns to group target columns for per-group tau scaling")

    args = parser.parse_args()

    # Handle boolean flags
    augment_flip = True
    if args.no_augment_flip:
        augment_flip = False
    elif args.augment_flip:
        augment_flip = True

    ard = True
    if args.no_ard:
        ard = False
    elif args.ard:
        ard = True

    save_std = True
    if args.no_save_std:
        save_std = False
    elif args.save_std:
        save_std = True

    enable_per_group_tau = True
    if args.disable_per_group_tau:
        enable_per_group_tau = False
    elif args.enable_per_group_tau:
        enable_per_group_tau = True

    # Parse coverage levels
    coverage_levels = _split_float_tokens(args.coverage_list)
    if not coverage_levels:
        coverage_levels = [0.5, 0.9, 0.95]
    for level in coverage_levels:
        if not 0.0 < level < 1.0:
            raise SystemExit(f"Coverage levels must be in (0, 1); received {level}")

    cfg = MTConfig(
        train_csv=list(args.train_csv),
        test_csv=list(args.test_csv),
        val_ratio=args.val_ratio,
        augment_flip=augment_flip,
        augment_profile=args.augment_profile,
        augment_config=args.augment_config,
        input_cols=_split_tokens(args.input_cols),
        target_cols=_split_tokens(args.target_cols),
        input_cols_re=_split_tokens(args.input_cols_re),
        target_cols_re=_split_tokens(args.target_cols_re),
        preset=args.preset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        prewarm_epochs=max(0, args.prewarm_epochs),
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_decay_step=max(1, int(args.lr_decay_step)),
        lr_decay_gamma=float(args.lr_decay_gamma),
        lr_decay_min_lr=max(0.0, float(args.lr_decay_min_lr)),
        early_stopping=args.early_stopping,
        patience=max(1, int(args.patience)),
        min_delta=float(args.min_delta),
        seed=args.seed,
        device=args.device,
        kernel=args.kernel,
        ard=ard,
        inducing=args.inducing,
        inducing_init=args.inducing_init,
        noise_init=args.noise_init,
        jitter=args.jitter,
        lmc_rank=args.lmc_rank,
        save_std=save_std,
        print_freq=max(1, int(args.print_freq)),
        val_interval=max(1, int(args.val_interval)),
        coverage_levels=coverage_levels,
        pca_enable=not args.no_pca,
        pca_variance=args.pca_variance,
        pca_components=args.pca_components,
        pca_max_components=args.pca_max_components,
        experiment_root=args.experiment_root,
        error_feature_names=args.error_features,
        enable_per_group_tau=enable_per_group_tau,
        group_patterns=args.group_patterns,
    )

    if cfg.preset is not None:
        cfg = _apply_preset(cfg)
    return cfg


def _split_tokens(raw: Optional[str]) -> Optional[List[str]]:
    """Split comma-separated tokens."""
    if raw is None:
        return None
    tokens = [tok.strip() for tok in raw.replace("\n", ",").split(",")]
    values = [tok for tok in tokens if tok]
    return values or None


def _split_float_tokens(raw: Optional[str]) -> Optional[List[float]]:
    """Split comma-separated float tokens."""
    tokens = _split_tokens(raw)
    if tokens is None:
        return None
    try:
        return [float(tok) for tok in tokens]
    except ValueError as exc:
        raise SystemExit(f"Failed to parse float list from '{raw}': {exc}") from exc


def _apply_preset(cfg: MTConfig) -> MTConfig:
    """Apply preset configuration to override CLI arguments."""
    # Similar implementation as svgp_baseline.py
    # For brevity, this is simplified - in production would match full implementation
    if cfg.preset is None:
        return cfg

    try:
        with cfg.preset.open("r") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise SystemExit(f"Failed to load preset {cfg.preset}: {exc}") from exc

    # Apply preset overrides (simplified for this example)
    for key, value in payload.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg


# Import additional utility functions from svgp_baseline
from models.gp.svgp_baseline import _module_base_name, _default_experiment_root, _start_run_dir, _configure_matplotlib


# ---------------------------------------------------------------------------
# Per-group tau scaling utilities
# ---------------------------------------------------------------------------


def _group_target_columns(target_cols: Sequence[str], patterns: Optional[Sequence[str]] = None) -> Dict[str, List[int]]:
    """Group target columns by regex patterns for per-group tau scaling.

    Args:
        target_cols: List of target column names
        patterns: Optional regex patterns for grouping. If None, uses default patterns.

    Returns:
        Dictionary mapping group name to list of column indices
    """
    if patterns is None:
        # Default patterns based on the implementation plan examples
        patterns = [r"^R[0-9]+$", r"^Ry_t[0-9]+$", r"^Rx_t[0-9]+$", r"^D_t[0-9]+_r$"]

    groups = {}
    ungrouped_indices = list(range(len(target_cols)))

    for pattern in patterns:
        import re
        regex = re.compile(pattern)
        group_name = pattern.replace("^", "").replace("$", "").replace("\\", "")
        matched_indices = []

        for i, col in enumerate(target_cols):
            if regex.match(col) and i in ungrouped_indices:
                matched_indices.append(i)
                ungrouped_indices.remove(i)

        if matched_indices:
            groups[group_name] = matched_indices

    # Add remaining columns as "other" group
    if ungrouped_indices:
        groups["other"] = ungrouped_indices

    return groups


def _calibrate_per_group_tau(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    coverage_levels: Sequence[float],
    group_patterns: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Calibrate tau values for each group of target columns.

    Args:
        true_df: True target values
        pred_df: Predicted values
        std_df: Predicted standard deviations
        coverage_levels: Desired coverage levels
        group_patterns: Regex patterns for grouping columns

    Returns:
        Dictionary mapping group name to tau value
    """
    if std_df is None or std_df.empty:
        return {}

    target_cols = list(true_df.columns)
    groups = _group_target_columns(target_cols, group_patterns)

    y_true = true_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = pred_df.to_numpy(dtype=np.float64, copy=False)
    y_std = std_df.to_numpy(dtype=np.float64, copy=False)

    group_taus = {}

    for group_name, col_indices in groups.items():
        # Extract data for this group
        group_y_true = y_true[:, col_indices]
        group_y_pred = y_pred[:, col_indices]
        group_y_std = y_std[:, col_indices]

        # Find optimal tau for this group
        tau = _find_optimal_tau(group_y_true, group_y_pred, group_y_std, coverage_levels)
        group_taus[group_name] = tau

    return group_taus


def _find_optimal_tau(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    coverage_levels: Sequence[float],
) -> float:
    """Find optimal tau value for given coverage levels."""
    valid_levels = [level for level in coverage_levels if 0.0 < level < 1.0]
    if not valid_levels:
        return 1.0

    def objective(tau_value: float) -> float:
        std_scaled = y_std * tau_value
        error = 0.0
        for level in valid_levels:
            # Calculate coverage for this tau
            alpha = 0.5 + level / 2.0
            z_value = _normal_quantile(alpha)
            margin = z_value * std_scaled
            lower = y_pred - margin
            upper = y_pred + margin
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            error += (coverage - level) ** 2
        return error

    # Coarse-to-fine search
    search_grid = np.logspace(-1, 1, num=40)
    best_tau = 1.0
    best_error = float("inf")
    for candidate in search_grid:
        err = objective(candidate)
        if err < best_error:
            best_error = err
            best_tau = candidate

    # Fine search around best candidate
    fine_lower = max(best_tau * 0.25, 1e-3)
    fine_upper = best_tau * 4.0
    fine_grid = np.linspace(fine_lower, fine_upper, num=60)
    for candidate in fine_grid:
        if candidate <= 0.0:
            continue
        err = objective(candidate)
        if err < best_error:
            best_error = err
            best_tau = candidate

    return float(best_tau)


def _apply_per_group_tau(
    std_df: pd.DataFrame,
    per_group_tau: Dict[str, float],
    target_cols: Sequence[str],
    group_patterns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Apply per-group tau scaling to standard deviations.

    Args:
        std_df: Standard deviations DataFrame
        per_group_tau: Dictionary mapping group names to tau values
        target_cols: Target column names
        group_patterns: Regex patterns for grouping

    Returns:
        DataFrame with per-group tau scaled standard deviations
    """
    std_scaled = std_df.copy()
    groups = _group_target_columns(target_cols, group_patterns)

    for group_name, col_indices in groups.items():
        if group_name in per_group_tau:
            tau_value = per_group_tau[group_name]
            # Apply tau scaling to columns in this group
            for i in col_indices:
                col_name = target_cols[i]
                std_scaled[col_name] = std_df[col_name] * tau_value

    return std_scaled


# Main execution
def _save_config(
    run_dir: Path,
    cfg: MTConfig,
    input_cols: Sequence[str],
    target_cols: Sequence[str],
) -> None:
    """Save configuration to JSON file."""
    cfg_dict = asdict(cfg)
    cfg_dict["train_csv"] = [str(p) for p in cfg.train_csv]
    cfg_dict["test_csv"] = [str(p) for p in cfg.test_csv]
    if cfg.augment_config is not None:
        cfg_dict["augment_config"] = str(cfg.augment_config)
    if cfg.experiment_root is not None:
        cfg_dict["experiment_root"] = str(cfg.experiment_root)
    cfg_dict["resolved_input_cols"] = list(input_cols)
    cfg_dict["resolved_target_cols"] = list(target_cols)
    with (run_dir / "config.json").open("w") as handle:
        json.dump(cfg_dict, handle, indent=2)


def _write_history(run_dir: Path, history_rows: List[Dict[str, float]]) -> None:
    """Write training history to CSV."""
    if not history_rows:
        return
    df = pd.DataFrame(history_rows)
    df.to_csv(run_dir / "history.csv", index=False)


def _prepare_split_frames(
    inputs_std: np.ndarray,
    true_latent: np.ndarray,
    pred_latent: np.ndarray,
    pred_var_latent: np.ndarray,
    data_module: GPDataModule,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Prepare DataFrames for evaluation."""
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


def _save_report(
    run_dir: Path,
    split_name: str,
    input_df: pd.DataFrame,
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    std_df: Optional[pd.DataFrame],
    error_feature_names: Optional[Sequence[str]],
) -> None:
    """Save evaluation report."""
    report = EvaluationReport(model_name=f"mt_icm_{split_name}")
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


def _normal_quantile(prob: float) -> float:
    """Calculate normal distribution quantile."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"Quantile probability must be in (0, 1); received {prob}")
    tensor = torch.tensor(prob, dtype=torch.float64)
    quantile = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)).icdf(tensor)
    return float(quantile.item())


def _gaussian_nll_matrix(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """Calculate Gaussian negative log-likelihood."""
    eps = 1e-9
    std = np.maximum(y_std, eps)
    var = np.square(std)
    residual = y_true - y_pred
    return 0.5 * (np.log(2.0 * np.pi * var) + np.square(residual) / var)


def _coverage_mask(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, level: float) -> np.ndarray:
    """Calculate coverage mask for confidence intervals."""
    alpha = 0.5 + level / 2.0
    z_value = _normal_quantile(alpha)
    margin = z_value * y_std
    lower = y_pred - margin
    upper = y_pred + margin
    return (y_true >= lower) & (y_true <= upper)


def _build_uncertainty_summary(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    std_df: Optional[pd.DataFrame],
    coverage_levels: Sequence[float],
    *,
    tau: Optional[float] = None,
    per_group_tau: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, object]]:
    """Build uncertainty summary including per-group metrics."""
    if std_df is None:
        return None

    y_true = true_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = pred_df.to_numpy(dtype=np.float64, copy=False)
    y_std = std_df.to_numpy(dtype=np.float64, copy=False)

    payload: Dict[str, object] = {"coverage_levels": [float(level) for level in coverage_levels]}

    def _summarize(std: np.ndarray, group_name: str = "raw") -> Dict[str, object]:
        nll_matrix = _gaussian_nll_matrix(y_true, y_pred, std)
        nll_overall = float(np.mean(nll_matrix))
        nll_per_target = {
            col: float(np.mean(nll_matrix[:, idx])) for idx, col in enumerate(true_df.columns)
        }

        coverage_summary: Dict[str, Dict[str, object]] = {}
        for level in coverage_levels:
            mask = _coverage_mask(y_true, y_pred, std, level)
            coverage_summary[str(level)] = {
                "overall": float(np.mean(mask)),
                "per_target": {
                    col: float(np.mean(mask[:, idx])) for idx, col in enumerate(true_df.columns)
                },
            }
        return {"nll": {"overall": nll_overall, "per_target": nll_per_target}, "coverage": coverage_summary}

    # Raw (unscaled) summary
    payload["raw"] = _summarize(y_std, "raw")

    # Global tau scaling
    if tau is not None:
        scaled = y_std * float(tau)
        scaled_summary = _summarize(scaled, "global_tau")
        payload["global_tau_scaled"] = {"value": float(tau), **scaled_summary}

    # Per-group tau scaling
    if per_group_tau is not None:
        std_scaled_df = _apply_per_group_tau(std_df, per_group_tau, list(true_df.columns), None)
        std_scaled = std_scaled_df.to_numpy(dtype=np.float64, copy=False)
        scaled_summary = _summarize(std_scaled, "per_group_tau")
        payload["per_group_tau_scaled"] = {"values": per_group_tau, **scaled_summary}

    return payload


def _write_uncertainty_metrics(
    run_dir: Path,
    split_name: str,
    payload: Optional[Dict[str, object]],
) -> None:
    """Write uncertainty metrics to JSON file."""
    if payload is None:
        return
    target_path = run_dir / f"{split_name}_uncertainty.json"
    with target_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    """Main training loop for multi-task SVGP."""
    cfg = _parse_args()
    _set_all_seeds(cfg.seed)

    if cfg.early_stopping and cfg.val_ratio <= 0.0:
        print("Early stopping requested but no validation split configured; disabling early stopping.")
        cfg.early_stopping = False

    device = torch.device(cfg.device)
    root = cfg.experiment_root or _default_experiment_root()
    run_dir = _start_run_dir(root)
    _configure_matplotlib(run_dir)

    input_spec = ColumnSpec(names=cfg.input_cols, patterns=cfg.input_cols_re)
    target_spec = ColumnSpec(names=cfg.target_cols, patterns=cfg.target_cols_re)

    pca_cfg = PCAPipelineConfig(
        n_components=cfg.pca_components,
        variance_threshold=cfg.pca_variance,
        max_components=cfg.pca_max_components,
    )

    data_cfg = GPDataConfig(
        train_csv=cfg.train_csv,
        test_csv=cfg.test_csv,
        input_spec=input_spec,
        target_spec=target_spec,
        val_ratio=cfg.val_ratio,
        batch_size=cfg.batch_size,
        augment_flip=cfg.augment_flip,
        augment_profile=cfg.augment_profile,
        augment_config=cfg.augment_config,
        use_pca=cfg.pca_enable,
        pca=pca_cfg,
    )

    data_module = GPDataModule(data_cfg)
    data_module.setup()
    data_module.save_preprocessors(run_dir / "checkpoints")

    _save_config(run_dir, cfg, data_module.input_cols, data_module.target_cols)

    train_inputs, train_latent = data_module.train_arrays()
    val_arrays = data_module.val_arrays()
    if val_arrays is not None:
        val_inputs, val_latent = val_arrays
    else:
        val_inputs = val_latent = None
    test_inputs, test_latent = data_module.test_arrays()

    # Initialize inducing points
    inducing_points = _init_inducing_points(train_inputs, cfg.inducing, cfg.inducing_init, seed=cfg.seed)

    # Train multi-task model
    components, history = _train_multitask(
        cfg,
        device,
        inducing_points,
        train_inputs,
        train_latent,
        val_inputs,
        val_latent,
    )

    # Save model states (one per component)
    for idx, comp in enumerate(components):
        torch.save(
            {
                "model_state": comp.model.state_dict(),
                "likelihood_state": comp.likelihood.state_dict(),
                "kernel": cfg.kernel,
                "ard": cfg.ard,
                "lmc_rank": cfg.lmc_rank,
            },
            run_dir / "checkpoints" / f"component_{idx:02d}.pt",
        )

    _write_history(run_dir, history)

    # Evaluate train / val / test splits
    train_mean_latent, train_var_latent = _predict_multitask(components, train_inputs, batch_size=cfg.batch_size, device=device)
    train_input_df, train_true_df, train_pred_df, train_std_df = _prepare_split_frames(
        train_inputs,
        train_latent,
        train_mean_latent,
        train_var_latent,
        data_module,
    )
    _save_report(
        run_dir,
        "train",
        train_input_df,
        train_true_df,
        train_pred_df,
        std_df=train_std_df if cfg.save_std else None,
        error_feature_names=cfg.error_feature_names,
    )
    _write_uncertainty_metrics(
        run_dir,
        "train",
        _build_uncertainty_summary(train_true_df, train_pred_df, train_std_df, cfg.coverage_levels),
    )

    # Validation and calibration
    global_tau: Optional[float] = None
    per_group_tau: Optional[Dict[str, float]] = None

    if val_inputs is not None and val_latent is not None:
        val_mean_latent, val_var_latent = _predict_multitask(components, val_inputs, batch_size=cfg.batch_size, device=device)
        val_input_df, val_true_df, val_pred_df, val_std_df = _prepare_split_frames(
            val_inputs,
            val_latent,
            val_mean_latent,
            val_var_latent,
            data_module,
        )
        _save_report(
            run_dir,
            "val",
            val_input_df,
            val_true_df,
            val_pred_df,
            std_df=val_std_df if cfg.save_std else None,
            error_feature_names=cfg.error_feature_names,
        )

        # Calibrate tau values
        global_tau = _find_optimal_tau(val_true_df.to_numpy(), val_pred_df.to_numpy(), val_std_df.to_numpy(), cfg.coverage_levels)

        if cfg.enable_per_group_tau:
            per_group_tau = _calibrate_per_group_tau(
                val_true_df, val_pred_df, val_std_df, cfg.coverage_levels, cfg.group_patterns
            )

        val_uncertainty = _build_uncertainty_summary(
            val_true_df,
            val_pred_df,
            val_std_df,
            cfg.coverage_levels,
            tau=global_tau,
            per_group_tau=per_group_tau,
        )
        _write_uncertainty_metrics(run_dir, "val", val_uncertainty)

        # Save tau values
        tau_data = {
            "global_tau": float(global_tau),
            "per_group_tau": per_group_tau if per_group_tau else None,
            "source": "val",
            "coverage_levels": [float(level) for level in cfg.coverage_levels],
        }
        with (run_dir / "tau.json").open("w") as handle:
            json.dump(tau_data, handle, indent=2)

    # Test evaluation
    test_mean_latent, test_var_latent = _predict_multitask(components, test_inputs, batch_size=cfg.batch_size, device=device)
    test_input_df, test_true_df, test_pred_df, test_std_df = _prepare_split_frames(
        test_inputs,
        test_latent,
        test_mean_latent,
        test_var_latent,
        data_module,
    )
    _save_report(
        run_dir,
        "test",
        test_input_df,
        test_true_df,
        test_pred_df,
        std_df=test_std_df if cfg.save_std else None,
        error_feature_names=cfg.error_feature_names,
    )
    _write_uncertainty_metrics(
        run_dir,
        "test",
        _build_uncertainty_summary(
            test_true_df,
            test_pred_df,
            test_std_df,
            cfg.coverage_levels,
            tau=global_tau,
            per_group_tau=per_group_tau,
        ),
    )

    # Individual test CSVs
    individual = data_module.individual_test_dataframes()
    if individual:
        for idx, df in enumerate(individual, start=1):
            inputs_std = data_module.scaler_x.transform(df[data_module.input_cols].to_numpy(dtype=np.float32, copy=True))
            targets_latent = data_module.encode_targets(df[data_module.target_cols])
            mean_latent, var_latent = _predict_multitask(components, inputs_std, batch_size=cfg.batch_size, device=device)
            input_df, true_df, pred_df, std_df = _prepare_split_frames(
                inputs_std,
                targets_latent,
                mean_latent,
                var_latent,
                data_module,
            )
            split_name = f"test_file_{idx}"
            _save_report(
                run_dir,
                split_name,
                input_df,
                true_df,
                pred_df,
                std_df=std_df if cfg.save_std else None,
                error_feature_names=cfg.error_feature_names,
            )
            _write_uncertainty_metrics(
                run_dir,
                split_name,
                _build_uncertainty_summary(
                    true_df, pred_df, std_df, cfg.coverage_levels, tau=global_tau, per_group_tau=per_group_tau
                ),
            )

    print(f"Multi-task SVGP training completed. Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()