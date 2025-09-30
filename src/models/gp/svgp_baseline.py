"""SVGP baseline training script using :class:`GPDataModule` for data handling.

This CLI mirrors existing tabular model entrypoints (e.g. bridge_nn.py,
rev03_rtd_nf_e3_tree.py) so that experiment directories and evaluation
artifacts remain comparable across model families.
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
from models.utils import ColumnSpec


# ---------------------------------------------------------------------------
# CLI configuration & argument parsing
# ---------------------------------------------------------------------------


@dataclass
class CLIConfig:
    """Aggregated configuration parsed from CLI (optionally preset-backed)."""

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

    kernel: str = "rbf"
    ard: bool = True
    inducing: int = 500
    inducing_init: str = "kmeans"
    noise_init: float = 1e-3
    jitter: float = 1e-6

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


def _split_tokens(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    tokens = [tok.strip() for tok in raw.replace("\n", ",").split(",")]
    values = [tok for tok in tokens if tok]
    return values or None


def _split_float_tokens(raw: Optional[str]) -> Optional[List[float]]:
    tokens = _split_tokens(raw)
    if tokens is None:
        return None
    try:
        return [float(tok) for tok in tokens]
    except ValueError as exc:  # pragma: no cover - defensive path
        raise SystemExit(f"Failed to parse float list from '{raw}': {exc}") from exc


def _maybe_path_list(value: object) -> List[Path]:
    if isinstance(value, (list, tuple)):
        return [Path(v) for v in value]
    return [Path(value)]


def _parse_args() -> CLIConfig:
    parser = argparse.ArgumentParser(description="SVGP baseline for bridge mechanics")
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

    parser.add_argument("--kernel", type=str, choices=["rbf", "matern52"], default="rbf")
    parser.add_argument("--ard", action="store_true")
    parser.add_argument("--no-ard", action="store_true")
    parser.add_argument("--inducing", type=int, default=500)
    parser.add_argument("--inducing-init", type=str, choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--noise-init", type=float, default=1e-3)
    parser.add_argument("--jitter", type=float, default=1e-6)

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

    args = parser.parse_args()

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

    coverage_levels = _split_float_tokens(args.coverage_list)
    if not coverage_levels:
        coverage_levels = [0.5, 0.9, 0.95]
    for level in coverage_levels:
        if not 0.0 < level < 1.0:
            raise SystemExit(f"Coverage levels must be in (0, 1); received {level}")

    cfg = CLIConfig(
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
        lr_decay_step=max(1, args.lr_decay_step),
        lr_decay_gamma=float(args.lr_decay_gamma),
        lr_decay_min_lr=max(0.0, float(args.lr_decay_min_lr)),
        early_stopping=args.early_stopping,
        patience=max(1, args.patience),
        min_delta=float(args.min_delta),
        seed=args.seed,
        device=args.device,
        kernel=args.kernel,
        ard=ard,
        inducing=args.inducing,
        inducing_init=args.inducing_init,
        noise_init=args.noise_init,
        jitter=args.jitter,
        save_std=save_std,
        print_freq=max(1, args.print_freq),
        val_interval=max(1, args.val_interval),
        coverage_levels=coverage_levels,
        pca_enable=not args.no_pca,
        pca_variance=args.pca_variance,
        pca_components=args.pca_components,
        pca_max_components=args.pca_max_components,
        experiment_root=args.experiment_root,
        error_feature_names=args.error_features,
    )
    if cfg.preset is not None:
        cfg = _apply_preset(cfg)
    return cfg


def _apply_preset(cfg: CLIConfig) -> CLIConfig:
    try:
        with cfg.preset.open("r") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise SystemExit(f"Failed to load preset {cfg.preset}: {exc}") from exc

    updated = dataclass_replace(cfg)

    if "train_csv" in payload:
        updated.train_csv = _maybe_path_list(payload["train_csv"])
    if "test_csv" in payload:
        updated.test_csv = _maybe_path_list(payload["test_csv"])
    if "val_ratio" in payload:
        updated.val_ratio = float(payload["val_ratio"])
    if "augment_flip" in payload:
        updated.augment_flip = bool(payload["augment_flip"])
    if "augment_profile" in payload and cfg.augment_profile is None:
        updated.augment_profile = str(payload["augment_profile"])
    if "augment_config" in payload and cfg.augment_config is None:
        updated.augment_config = Path(payload["augment_config"])

    if updated.input_cols is None and "input_cols" in payload:
        updated.input_cols = [str(x) for x in payload["input_cols"]]
    if updated.target_cols is None and "target_cols" in payload:
        updated.target_cols = [str(x) for x in payload["target_cols"]]
    if updated.input_cols_re is None and payload.get("input_cols_re"):
        updated.input_cols_re = [str(x) for x in payload["input_cols_re"]]
    if updated.target_cols_re is None and payload.get("target_cols_re"):
        updated.target_cols_re = [str(x) for x in payload["target_cols_re"]]

    for key in (
        "batch_size",
        "epochs",
        "lr",
        "lr_decay",
        "lr_decay_step",
        "lr_decay_gamma",
        "lr_decay_min_lr",
        "seed",
        "kernel",
        "ard",
        "inducing",
        "inducing_init",
        "noise_init",
        "jitter",
        "pca_enable",
        "pca_variance",
        "pca_components",
        "pca_max_components",
    ):
        if key in payload:
            setattr(updated, key, payload[key])
    if "prewarm_epochs" in payload:
        updated.prewarm_epochs = max(0, int(payload["prewarm_epochs"]))
    if "device" in payload and cfg.device == "cpu":
        updated.device = str(payload["device"])
    if "save_std" in payload:
        updated.save_std = bool(payload["save_std"])
    if "print_freq" in payload:
        updated.print_freq = int(payload["print_freq"])
    if "val_interval" in payload:
        updated.val_interval = int(payload["val_interval"])
    if "early_stopping" in payload:
        updated.early_stopping = bool(payload["early_stopping"])
    if "patience" in payload:
        updated.patience = max(1, int(payload["patience"]))
    if "min_delta" in payload:
        updated.min_delta = float(payload["min_delta"])
    if "coverage_levels" in payload:
        raw_levels = payload["coverage_levels"]
        if isinstance(raw_levels, str):
            levels = _split_float_tokens(raw_levels)
        else:
            try:
                levels = [float(x) for x in raw_levels]
            except TypeError as exc:  # pragma: no cover - defensive path
                raise SystemExit(f"Invalid coverage_levels in preset: {raw_levels}") from exc
        if not levels:
            raise SystemExit("coverage_levels in preset must not be empty")
        for lvl in levels:
            if not 0.0 < lvl < 1.0:
                raise SystemExit(f"Preset coverage level {lvl} must be in (0, 1)")
        updated.coverage_levels = levels
    if "experiment_root" in payload and cfg.experiment_root is None:
        updated.experiment_root = Path(payload["experiment_root"])
    if updated.error_feature_names is None and payload.get("error_feature_names"):
        updated.error_feature_names = [str(x) for x in payload["error_feature_names"]]

    return updated


def dataclass_replace(cfg: CLIConfig) -> CLIConfig:
    return CLIConfig(**asdict(cfg))


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _module_base_name() -> str:
    return Path(__file__).stem


def _default_experiment_root() -> Path:
    return Path("experiments") / _module_base_name()


def _start_run_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / ts
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def _configure_matplotlib(run_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
    (run_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    try:  # pragma: no cover - Matplotlib configuration best effort
        import matplotlib as mpl

        mpl.use("Agg")
    except Exception:
        pass


def _init_inducing_points(
    train_inputs: np.ndarray,
    inducing: int,
    method: str,
    *,
    seed: int,
) -> torch.Tensor:
    if inducing <= 0:
        raise ValueError("Number of inducing points must be positive")
    num_samples = train_inputs.shape[0]
    if inducing > num_samples:
        raise ValueError(
            f"Requested {inducing} inducing points but only {num_samples} training samples available"
        )
    if method == "kmeans":
        km = MiniBatchKMeans(n_clusters=inducing, batch_size=min(2048, num_samples), random_state=seed)
        km.fit(train_inputs)
        centers = km.cluster_centers_.astype(np.float32)
    elif method == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(num_samples, size=inducing, replace=False)
        centers = train_inputs[indices].astype(np.float32)
    else:  # pragma: no cover - validated by argparse choices
        raise ValueError(f"Unsupported inducing init: {method}")
    return torch.from_numpy(centers)


def _set_prewarm_mode(
    model: SingleTaskSVGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    *,
    enable: bool,
) -> None:
    """Freeze/unfreeze parameters for the pre-warm stage.

    When ``enable`` is True we only allow the kernel lengthscales and noise term
    to update; all other parameters are temporarily frozen.
    """

    for param in model.variational_parameters():
        param.requires_grad = not enable

    inducing_points = getattr(model.variational_strategy, "inducing_points", None)
    if isinstance(inducing_points, torch.nn.Parameter):
        inducing_points.requires_grad = not enable

    base_kernel = getattr(model.covar_module, "base_kernel", None)
    if base_kernel is not None and hasattr(base_kernel, "raw_lengthscale"):
        base_kernel.raw_lengthscale.requires_grad = True

    if hasattr(model.covar_module, "raw_outputscale"):
        model.covar_module.raw_outputscale.requires_grad = not enable

    if hasattr(likelihood, "raw_noise"):
        likelihood.raw_noise.requires_grad = True


def _build_scheduler(cfg: CLIConfig, optimizer: torch.optim.Optimizer) -> Optional[lr_scheduler._LRScheduler]:
    if cfg.lr_decay == "none":
        return None
    if cfg.lr_decay == "step":
        step_size = max(1, int(cfg.lr_decay_step))
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=float(cfg.lr_decay_gamma))
    if cfg.lr_decay == "cosine":
        t_max = max(1, int(cfg.epochs))
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=float(cfg.lr_decay_min_lr))
    raise ValueError(f"Unsupported lr_decay mode '{cfg.lr_decay}'")


class SingleTaskSVGP(gpytorch.models.ApproximateGP):
    """SVGP model for a single latent dimension."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        *,
        kernel: str,
        ard: bool,
        jitter: float,
    ) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        ard_dims = inducing_points.size(-1) if ard else None
        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif kernel == "matern52":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        else:  # pragma: no cover - guarded by argparse
            raise ValueError(f"Unsupported kernel '{kernel}'")
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self._manual_jitter = float(jitter)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def extra_jitter(self) -> float:
        return self._manual_jitter


@dataclass
class ComponentState:
    model: SingleTaskSVGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood


def _make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(inputs.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.float32, copy=False)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _train_component(
    component_idx: int,
    cfg: CLIConfig,
    device: torch.device,
    inducing_points: torch.Tensor,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: Optional[np.ndarray],
    val_targets: Optional[np.ndarray],
) -> Tuple[ComponentState, List[Dict[str, float]]]:
    model = SingleTaskSVGP(inducing_points.clone(), kernel=cfg.kernel, ard=cfg.ard, jitter=cfg.jitter)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.initialize(noise=cfg.noise_init)

    model.to(device)
    likelihood.to(device)

    train_loader = _make_loader(train_inputs, train_targets, batch_size=cfg.batch_size, shuffle=True)
    val_loader = None
    if val_inputs is not None and val_targets is not None:
        val_loader = _make_loader(val_inputs, val_targets, batch_size=cfg.batch_size, shuffle=False)

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

    prewarm_epochs = min(max(0, int(cfg.prewarm_epochs)), max(0, cfg.epochs - 1))
    prewarm_active = prewarm_epochs > 0
    if prewarm_active:
        _set_prewarm_mode(model, likelihood, enable=True)

    use_early_stopping = bool(cfg.early_stopping and val_loader is not None)
    best_val = float("inf")
    best_epoch: Optional[int] = None
    epochs_no_improve = 0
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    history: List[Dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        if prewarm_active and epoch == prewarm_epochs + 1:
            _set_prewarm_mode(model, likelihood, enable=False)
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

        history.append(
            {
                "component": float(component_idx),
                "epoch": float(epoch),
                "train_elbo": avg_loss,
                "val_rmse": float(val_rmse) if val_rmse is not None else float("nan"),
                "prewarm": 1.0 if epoch <= prewarm_epochs else 0.0,
                "lr": float(current_lr),
            }
        )

        if cfg.print_freq and epoch % cfg.print_freq == 0:
            msg = (
                f"Component {component_idx:02d} | epoch {epoch:04d}"
                f" | lr {current_lr:.5f} | train loss {avg_loss:.4f}"
            )
            if val_rmse is not None:
                msg += f" | val RMSE {val_rmse:.4f}"
            print(msg)

        if use_early_stopping and epoch > prewarm_epochs and epochs_no_improve >= cfg.patience:
            print(
                f"Early stopping component {component_idx:02d} after {epoch} epochs; "
                f"best val RMSE {best_val:.4f} at epoch {best_epoch}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    model.cpu()
    likelihood.cpu()
    torch.cuda.empty_cache()
    return ComponentState(model=model, likelihood=likelihood), history


def _estimate_rmse(
    model: SingleTaskSVGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    likelihood.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.squeeze(-1).to(device)
            with gpytorch.settings.fast_pred_var():
                posterior = likelihood(model(xb))
            preds.append(posterior.mean.cpu())
            targets.append(yb.cpu())
    pred_vec = torch.cat(preds)
    target_vec = torch.cat(targets)
    mse = torch.mean((pred_vec - target_vec) ** 2)
    return float(torch.sqrt(mse))


def _predict_components(
    components: Sequence[ComponentState],
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    num_samples = inputs.shape[0]
    latent_dim = len(components)
    mean_out = np.zeros((num_samples, latent_dim), dtype=np.float64)
    var_out = np.zeros((num_samples, latent_dim), dtype=np.float64)

    loader = _make_loader(inputs, np.zeros((num_samples, 1), dtype=np.float32), batch_size=batch_size, shuffle=False)

    for comp_idx, comp in enumerate(components):
        model = comp.model.to(device)
        likelihood = comp.likelihood.to(device)
        model.eval()
        likelihood.eval()

        preds: List[torch.Tensor] = []
        vars_: List[torch.Tensor] = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(model.extra_jitter()):
                    posterior = likelihood(model(xb))
                preds.append(posterior.mean.detach().cpu())
                vars_.append(posterior.variance.detach().cpu())
        mean_out[:, comp_idx] = torch.cat(preds).numpy()
        var_out[:, comp_idx] = torch.cat(vars_).numpy()
        model.cpu()
        likelihood.cpu()
    torch.cuda.empty_cache()
    return mean_out.astype(np.float64), var_out.astype(np.float64)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _save_config(
    run_dir: Path,
    cfg: CLIConfig,
    input_cols: Sequence[str],
    target_cols: Sequence[str],
) -> None:
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
    report = EvaluationReport(model_name=f"svgp_{split_name}")
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
    if not 0.0 < prob < 1.0:
        raise ValueError(f"Quantile probability must be in (0, 1); received {prob}")
    tensor = torch.tensor(prob, dtype=torch.float64)
    quantile = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)).icdf(tensor)
    return float(quantile.item())


def _gaussian_nll_matrix(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    eps = 1e-9
    std = np.maximum(y_std, eps)
    var = np.square(std)
    residual = y_true - y_pred
    return 0.5 * (np.log(2.0 * np.pi * var) + np.square(residual) / var)


def _coverage_mask(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, level: float) -> np.ndarray:
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
) -> Optional[Dict[str, object]]:
    if std_df is None:
        return None

    y_true = true_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = pred_df.to_numpy(dtype=np.float64, copy=False)
    y_std = std_df.to_numpy(dtype=np.float64, copy=False)

    payload: Dict[str, object] = {"coverage_levels": [float(level) for level in coverage_levels]}

    def _summarize(std: np.ndarray) -> Dict[str, object]:
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

    payload["raw"] = _summarize(y_std)

    if tau is not None:
        scaled = y_std * float(tau)
        scaled_summary = _summarize(scaled)
        payload["tau_scaled"] = {"value": float(tau), **scaled_summary}

    return payload


def _calibrate_tau(
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

    valid_levels = [level for level in coverage_levels if 0.0 < level < 1.0]
    if not valid_levels:
        return None

    def objective(tau_value: float) -> float:
        std_scaled = y_std * tau_value
        error = 0.0
        for level in valid_levels:
            mask = _coverage_mask(y_true, y_pred, std_scaled, level)
            coverage = float(np.mean(mask))
            error += (coverage - level) ** 2
        return error

    # Coarse-to-fine search on a log scale
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
    fine_grid = np.linspace(fine_lower, fine_upper, num=80)
    for candidate in fine_grid:
        if candidate <= 0.0:
            continue
        err = objective(candidate)
        if err < best_error:
            best_error = err
            best_tau = candidate

    return float(best_tau)


def _write_uncertainty_metrics(
    run_dir: Path,
    split_name: str,
    payload: Optional[Dict[str, object]],
) -> None:
    if payload is None:
        return
    target_path = run_dir / f"{split_name}_uncertainty.json"
    with target_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------


def main() -> None:
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

    inducing_points = _init_inducing_points(train_inputs, cfg.inducing, cfg.inducing_init, seed=cfg.seed)
    components: List[ComponentState] = []
    history_rows: List[Dict[str, float]] = []
    latent_dim = train_latent.shape[1]

    for idx in range(latent_dim):
        state, history = _train_component(
            idx,
            cfg,
            device,
            inducing_points,
            train_inputs,
            train_latent[:, idx : idx + 1],
            val_inputs if val_inputs is not None else None,
            val_latent[:, idx : idx + 1] if val_latent is not None else None,
        )
        components.append(state)
        history_rows.extend(history)
        torch.save(
            {
                "model_state": state.model.state_dict(),
                "likelihood_state": state.likelihood.state_dict(),
                "kernel": cfg.kernel,
                "ard": cfg.ard,
            },
            run_dir / "checkpoints" / f"component_{idx:02d}.pt",
        )

    _write_history(run_dir, history_rows)

    # Evaluate train / val / test splits
    train_mean_latent, train_var_latent = _predict_components(components, train_inputs, batch_size=cfg.batch_size, device=device)
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

    tau_value: Optional[float] = None
    if val_inputs is not None and val_latent is not None:
        val_mean_latent, val_var_latent = _predict_components(components, val_inputs, batch_size=cfg.batch_size, device=device)
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
        tau_value = _calibrate_tau(val_true_df, val_pred_df, val_std_df, cfg.coverage_levels)
        val_uncertainty = _build_uncertainty_summary(
            val_true_df,
            val_pred_df,
            val_std_df,
            cfg.coverage_levels,
            tau=tau_value,
        )
        _write_uncertainty_metrics(run_dir, "val", val_uncertainty)
        if tau_value is not None:
            with (run_dir / "tau.json").open("w") as handle:
                json.dump(
                    {
                        "value": float(tau_value),
                        "source": "val",
                        "coverage_levels": [float(level) for level in cfg.coverage_levels],
                    },
                    handle,
                    indent=2,
                )

    test_mean_latent, test_var_latent = _predict_components(components, test_inputs, batch_size=cfg.batch_size, device=device)
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
            tau=tau_value,
        ),
    )

    # Individual test CSVs
    individual = data_module.individual_test_dataframes()
    if individual:
        for idx, df in enumerate(individual, start=1):
            inputs_std = data_module.scaler_x.transform(df[data_module.input_cols].to_numpy(dtype=np.float32, copy=True))
            targets_latent = data_module.encode_targets(df[data_module.target_cols])
            mean_latent, var_latent = _predict_components(components, inputs_std, batch_size=cfg.batch_size, device=device)
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
                _build_uncertainty_summary(true_df, pred_df, std_df, cfg.coverage_levels, tau=tau_value),
            )

    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
