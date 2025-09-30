"""Multitask sparse variational GP with ICM kernel for PCA latent targets."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gpytorch
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from models.gp.common import (
    GPDataConfig,
    GPDataModule,
    PCAPipelineConfig,
    TargetGroupMapping,
    resolve_target_groups,
)
from models.gp.run_utils import (
    configure_matplotlib,
    default_experiment_root,
    init_inducing_points,
    prepare_split_frames,
    save_report,
    save_run_config,
    start_run_dir,
    write_history,
    write_uncertainty_metrics,
)
from models.gp.uncertainty import (
    build_uncertainty_summary,
    calibrate_global_tau,
    calibrate_groupwise_tau,
    calibrate_quantile_tau_map,
)
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
    icm_rank: int = 2
    noise_init: float = 1e-3
    jitter: float = 1e-6

    save_std: bool = True
    print_freq: int = 50
    val_interval: int = 50
    coverage_levels: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95])
    per_group_tau_mode: str = "auto"
    per_group_tau_config: Optional[Path] = None

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
    parser = argparse.ArgumentParser(description="Multitask SVGP with ICM kernel")
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
    parser.add_argument("--icm-rank", type=int, default=2)
    parser.add_argument("--noise-init", type=float, default=1e-3)
    parser.add_argument("--jitter", type=float, default=1e-6)

    parser.add_argument("--save-std", action="store_true")
    parser.add_argument("--no-save-std", action="store_true")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--coverage-list", type=str, default="0.5,0.9,0.95")
    parser.add_argument(
        "--per-group-tau-mode",
        type=str,
        default="auto",
        choices=["none", "auto", "config"],
        help="Strategy for grouping targets when calibrating per-group tau scaling",
    )
    parser.add_argument(
        "--per-group-tau-config",
        type=Path,
        default=None,
        help="JSON mapping of group name to column patterns (required if mode=config)",
    )

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

    coverage_levels = _split_float_tokens(args.coverage_list) or [0.5, 0.9, 0.95]
    coverage_levels = [level for level in coverage_levels if 0.0 < level < 1.0]
    if not coverage_levels:
        raise SystemExit("coverage list must contain at least one level in (0, 1)")

    per_group_tau_mode = args.per_group_tau_mode
    per_group_tau_config = args.per_group_tau_config
    if per_group_tau_mode == "config" and per_group_tau_config is None:
        raise SystemExit("per-group tau mode 'config' requires --per-group-tau-config")

    cfg = CLIConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
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
        icm_rank=max(1, args.icm_rank),
        noise_init=args.noise_init,
        jitter=args.jitter,
        save_std=save_std,
        print_freq=max(1, args.print_freq),
        val_interval=max(1, args.val_interval),
        coverage_levels=coverage_levels,
        per_group_tau_mode=per_group_tau_mode,
        per_group_tau_config=per_group_tau_config,
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


def dataclass_replace(cfg: CLIConfig) -> CLIConfig:
    return CLIConfig(**asdict(cfg))


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
        "icm_rank",
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
            except Exception as exc:  # pragma: no cover - invalid preset
                raise SystemExit(f"Invalid coverage_levels in preset: {raw_levels}") from exc
        if not levels:
            raise SystemExit("coverage_levels in preset must not be empty")
        updated.coverage_levels = [lvl for lvl in levels if 0.0 < lvl < 1.0]
    if "per_group_tau_mode" in payload:
        updated.per_group_tau_mode = str(payload["per_group_tau_mode"]).lower()
    if (
        "per_group_tau_config" in payload
        and updated.per_group_tau_config is None
        and payload["per_group_tau_config"] is not None
    ):
        updated.per_group_tau_config = Path(payload["per_group_tau_config"])
    if "experiment_root" in payload and cfg.experiment_root is None:
        updated.experiment_root = Path(payload["experiment_root"])
    if updated.error_feature_names is None and payload.get("error_feature_names"):
        updated.error_feature_names = [str(x) for x in payload["error_feature_names"]]

    return updated


# ---------------------------------------------------------------------------
# Model definition and training helpers
# ---------------------------------------------------------------------------


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class MultitaskICMSVGP(gpytorch.models.ApproximateGP):
    """Shared-inducing multi-output SVGP with an ICM kernel."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        num_tasks: int,
        *,
        kernel: str,
        ard: bool,
        icm_rank: int,
        jitter: float,
    ) -> None:
        num_tasks = int(num_tasks)
        num_latents = max(1, int(icm_rank))
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=torch.Size([num_latents]),
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
            num_latents=num_latents,
        )
        super().__init__(variational_strategy)

        batch_shape = torch.Size([num_latents])
        ard_dims = inducing_points.size(-1) if ard else None
        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=ard_dims)
        elif kernel == "matern52":
            base_kernel = gpytorch.kernels.MaternKernel(batch_shape=batch_shape, nu=2.5, ard_num_dims=ard_dims)
        else:  # pragma: no cover - guarded by argparse
            raise ValueError(f"Unsupported kernel '{kernel}'")
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)
        self._manual_jitter = float(jitter)
        self.num_tasks = num_tasks
        self.num_latents = num_latents

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def extra_jitter(self) -> float:
        return self._manual_jitter


def _set_prewarm_mode(
    model: MultitaskICMSVGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    *,
    enable: bool,
) -> None:
    if enable:
        for param in model.parameters():
            param.requires_grad = False
        for param in likelihood.parameters():
            param.requires_grad = False
        base_vs = getattr(model.variational_strategy, "base_variational_strategy", None)
        if base_vs is None:
            base_vs = getattr(model.variational_strategy, "_variational_strategy", None)
        if base_vs is not None:
            inducing = getattr(base_vs, "inducing_points", None)
            if isinstance(inducing, torch.nn.Parameter):
                inducing.requires_grad = False
        if hasattr(model.variational_strategy, "lmc_coefficients"):
            model.variational_strategy.lmc_coefficients.requires_grad = False
        base_kernel = getattr(model.covar_module, "base_kernel", model.covar_module)
        if hasattr(base_kernel, "raw_lengthscale"):
            base_kernel.raw_lengthscale.requires_grad = True
        if hasattr(likelihood, "raw_noise"):
            likelihood.raw_noise.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
        for param in likelihood.parameters():
            param.requires_grad = True


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


def _train_model(
    cfg: CLIConfig,
    device: torch.device,
    inducing_points: torch.Tensor,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: Optional[np.ndarray],
    val_targets: Optional[np.ndarray],
) -> Tuple[MultitaskICMSVGP, gpytorch.likelihoods.MultitaskGaussianLikelihood, List[Dict[str, float]]]:
    num_tasks = train_targets.shape[1]
    model = MultitaskICMSVGP(
        inducing_points.clone(),
        num_tasks,
        kernel=cfg.kernel,
        ard=cfg.ard,
        icm_rank=cfg.icm_rank,
        jitter=cfg.jitter,
    )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    likelihood.initialize(noise=cfg.noise_init)

    model.to(device)
    likelihood.to(device)

    train_loader = _make_loader(train_inputs, train_targets, batch_size=cfg.batch_size, shuffle=True)
    val_loader = None
    if val_inputs is not None and val_targets is not None:
        val_loader = _make_loader(val_inputs, val_targets, batch_size=cfg.batch_size, shuffle=False)

    num_data = train_loader.dataset.tensors[0].shape[0]
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=cfg.lr)
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
            yb = yb.to(device)

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
                "component": -1.0,
                "epoch": float(epoch),
                "train_elbo": avg_loss,
                "val_rmse": float(val_rmse) if val_rmse is not None else float("nan"),
                "prewarm": 1.0 if epoch <= prewarm_epochs else 0.0,
                "lr": float(current_lr),
            }
        )

        if cfg.print_freq and epoch % cfg.print_freq == 0:
            msg = f"Epoch {epoch:04d} | lr {current_lr:.5f} | train loss {avg_loss:.4f}"
            if val_rmse is not None:
                msg += f" | val RMSE {val_rmse:.4f}"
            print(msg)

        if use_early_stopping and epoch > prewarm_epochs and epochs_no_improve >= cfg.patience:
            print(
                f"Early stopping after {epoch} epochs; best val RMSE {best_val:.4f} at epoch {best_epoch}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    model.cpu()
    likelihood.cpu()
    torch.cuda.empty_cache()
    return model, likelihood, history


def _estimate_rmse(
    model: MultitaskICMSVGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
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
            yb = yb.to(device)
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(model.extra_jitter()):
                posterior = likelihood(model(xb))
            preds.append(posterior.mean.cpu())
            targets.append(yb.cpu())
    pred_mat = torch.cat(preds)
    target_mat = torch.cat(targets)
    mse = torch.mean((pred_mat - target_mat) ** 2)
    return float(torch.sqrt(mse))


def _predict_mean_var(
    model: MultitaskICMSVGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    likelihood = likelihood.to(device)
    model.eval()
    likelihood.eval()

    num_samples = inputs.shape[0]
    num_tasks = model.num_tasks
    mean_out = np.zeros((num_samples, num_tasks), dtype=np.float64)
    var_out = np.zeros((num_samples, num_tasks), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            xb = torch.from_numpy(inputs[start:end].astype(np.float32, copy=False)).to(device)
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(model.extra_jitter()):
                posterior = likelihood(model(xb))
            mean_out[start:end] = posterior.mean.detach().cpu().numpy()
            var_out[start:end] = posterior.variance.detach().cpu().numpy()

    model.cpu()
    likelihood.cpu()
    torch.cuda.empty_cache()
    return mean_out, var_out


def _save_task_covariance(
    model: MultitaskICMSVGP,
    run_dir: Path,
    *,
    component_labels: Sequence[str],
) -> None:
    if len(component_labels) != model.num_tasks:
        component_labels = [f"{idx}" for idx in range(model.num_tasks)]
    coeff = model.variational_strategy.lmc_coefficients.detach().cpu().numpy()
    coeff = np.reshape(coeff, (model.num_latents, model.num_tasks))
    outputscale = model.covar_module.outputscale.detach().cpu().numpy()
    outputscale = np.asarray(outputscale, dtype=np.float64)
    if outputscale.ndim == 0:
        outputscale = np.repeat(outputscale, model.num_latents)
    outputscale = outputscale.reshape(model.num_latents)
    weights = coeff * np.sqrt(np.maximum(outputscale, 0.0))[:, None]
    cov = weights.T @ weights
    np.savetxt(run_dir / "task_covariance.csv", cov, delimiter=",")

    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib may be unavailable
        return

    fig, ax = plt.subplots(figsize=(max(6, len(component_labels) * 0.4), 5))
    im = ax.imshow(cov, cmap="viridis")
    ax.set_title("ICM Task Covariance (latent space)")
    ax.set_xlabel("Component")
    ax.set_ylabel("Component")
    ax.set_xticks(range(len(component_labels)))
    ax.set_yticks(range(len(component_labels)))
    ax.set_xticklabels(component_labels, rotation=45, ha="right")
    ax.set_yticklabels(component_labels)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    plt.savefig(run_dir / "plots" / "task_covariance_heatmap.png", dpi=200)
    plt.close(fig)


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
    root = cfg.experiment_root or default_experiment_root(__file__)
    run_dir = start_run_dir(root)
    configure_matplotlib(run_dir)

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
    try:
        group_mapping = resolve_target_groups(
            data_module.target_cols,
            mode=cfg.per_group_tau_mode,
            config_path=cfg.per_group_tau_config,
        )
    except ValueError as exc:
        raise SystemExit(f"Failed to resolve target groups: {exc}") from exc

    cfg_payload = asdict(cfg)
    cfg_payload["train_csv"] = [str(p) for p in cfg.train_csv]
    cfg_payload["test_csv"] = [str(p) for p in cfg.test_csv]
    if cfg.augment_config is not None:
        cfg_payload["augment_config"] = str(cfg.augment_config)
    if cfg.experiment_root is not None:
        cfg_payload["experiment_root"] = str(cfg.experiment_root)
    if cfg.per_group_tau_config is not None:
        cfg_payload["per_group_tau_config"] = str(cfg.per_group_tau_config)
    cfg_payload["resolved_input_cols"] = list(data_module.input_cols)
    cfg_payload["resolved_target_cols"] = list(data_module.target_cols)
    cfg_payload["latent_dim"] = data_module.target_latent_dim
    if group_mapping is not None:
        cfg_payload["target_group_map"] = group_mapping.to_serializable()
    save_run_config(run_dir, cfg_payload)

    group_indices = group_mapping.group_to_indices if group_mapping is not None else None

    train_inputs, train_latent = data_module.train_arrays()
    val_arrays = data_module.val_arrays()
    if val_arrays is not None:
        val_inputs, val_latent = val_arrays
    else:
        val_inputs = val_latent = None
    test_inputs, test_latent = data_module.test_arrays()

    inducing_points = init_inducing_points(train_inputs, cfg.inducing, cfg.inducing_init, seed=cfg.seed)
    model, likelihood, history_rows = _train_model(
        cfg,
        device,
        inducing_points,
        train_inputs,
        train_latent,
        val_inputs if val_inputs is not None else None,
        val_latent if val_latent is not None else None,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "likelihood_state": likelihood.state_dict(),
            "kernel": cfg.kernel,
            "ard": cfg.ard,
            "icm_rank": cfg.icm_rank,
        },
        run_dir / "checkpoints" / "model.pt",
    )
    write_history(run_dir, history_rows)

    latent_labels = [f"z_{idx:02d}" for idx in range(data_module.target_latent_dim)]
    _save_task_covariance(model, run_dir, component_labels=latent_labels)

    # Evaluate train / val / test splits
    train_mean_latent, train_var_latent = _predict_mean_var(
        model,
        likelihood,
        train_inputs,
        batch_size=cfg.batch_size,
        device=device,
    )
    train_input_df, train_true_df, train_pred_df, train_std_df = prepare_split_frames(
        train_inputs,
        train_latent,
        train_mean_latent,
        train_var_latent,
        data_module,
    )
    save_report(
        run_dir,
        "train",
        train_input_df,
        train_true_df,
        train_pred_df,
        std_df=train_std_df if cfg.save_std else None,
        error_feature_names=cfg.error_feature_names,
        model_name="mticm",
    )
    write_uncertainty_metrics(
        run_dir,
        "train",
        build_uncertainty_summary(
            train_true_df,
            train_pred_df,
            train_std_df,
            cfg.coverage_levels,
            group_indices=group_indices,
        ),
    )

    tau_value: Optional[float] = None
    per_group_tau: Dict[str, float] = {}
    quantile_tau: Dict[float, float] = {}
    if val_inputs is not None and val_latent is not None:
        val_mean_latent, val_var_latent = _predict_mean_var(
            model,
            likelihood,
            val_inputs,
            batch_size=cfg.batch_size,
            device=device,
        )
        val_input_df, val_true_df, val_pred_df, val_std_df = prepare_split_frames(
            val_inputs,
            val_latent,
            val_mean_latent,
            val_var_latent,
            data_module,
        )
        save_report(
            run_dir,
            "val",
            val_input_df,
            val_true_df,
            val_pred_df,
            std_df=val_std_df if cfg.save_std else None,
            error_feature_names=cfg.error_feature_names,
            model_name="mticm",
        )
        tau_value = calibrate_global_tau(
            val_true_df,
            val_pred_df,
            val_std_df,
            cfg.coverage_levels,
        )
        if group_indices and cfg.per_group_tau_mode != "none":
            per_group_tau = calibrate_groupwise_tau(
                val_true_df,
                val_pred_df,
                val_std_df,
                cfg.coverage_levels,
                group_indices,
            )
        quantile_tau = calibrate_quantile_tau_map(
            val_true_df,
            val_pred_df,
            val_std_df,
            cfg.coverage_levels,
        )
        val_uncertainty = build_uncertainty_summary(
            val_true_df,
            val_pred_df,
            val_std_df,
            cfg.coverage_levels,
            global_tau=tau_value,
            per_group_tau=per_group_tau,
            group_indices=group_indices,
            quantile_tau=quantile_tau,
        )
        write_uncertainty_metrics(run_dir, "val", val_uncertainty)
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
        if per_group_tau:
            with (run_dir / "per_group_tau.json").open("w") as handle:
                json.dump(
                    {
                        "values": {k: float(v) for k, v in per_group_tau.items()},
                        "source": "val",
                        "coverage_levels": [float(level) for level in cfg.coverage_levels],
                        "groups": group_mapping.to_serializable() if group_mapping is not None else {},
                        "mode": cfg.per_group_tau_mode,
                    },
                    handle,
                    indent=2,
                )
        if quantile_tau:
            with (run_dir / "quantile_tau.json").open("w") as handle:
                json.dump(
                    {
                        "values": {f"{level:g}": float(value) for level, value in quantile_tau.items()},
                        "source": "val",
                        "coverage_levels": [float(level) for level in cfg.coverage_levels],
                    },
                    handle,
                    indent=2,
                )

    test_mean_latent, test_var_latent = _predict_mean_var(
        model,
        likelihood,
        test_inputs,
        batch_size=cfg.batch_size,
        device=device,
    )
    test_input_df, test_true_df, test_pred_df, test_std_df = prepare_split_frames(
        test_inputs,
        test_latent,
        test_mean_latent,
        test_var_latent,
        data_module,
    )
    save_report(
        run_dir,
        "test",
        test_input_df,
        test_true_df,
        test_pred_df,
        std_df=test_std_df if cfg.save_std else None,
        error_feature_names=cfg.error_feature_names,
        model_name="mticm",
    )
    write_uncertainty_metrics(
        run_dir,
        "test",
        build_uncertainty_summary(
            test_true_df,
            test_pred_df,
            test_std_df,
            cfg.coverage_levels,
            global_tau=tau_value,
            per_group_tau=per_group_tau,
            group_indices=group_indices,
            quantile_tau=quantile_tau,
        ),
    )

    individual = data_module.individual_test_dataframes()
    if individual:
        for idx, df in enumerate(individual, start=1):
            inputs_std = data_module.scaler_x.transform(
                df[data_module.input_cols].to_numpy(dtype=np.float32, copy=True)
            )
            targets_latent = data_module.encode_targets(df[data_module.target_cols])
            mean_latent, var_latent = _predict_mean_var(
                model,
                likelihood,
                inputs_std,
                batch_size=cfg.batch_size,
                device=device,
            )
            input_df, true_df, pred_df, std_df = prepare_split_frames(
                inputs_std,
                targets_latent,
                mean_latent,
                var_latent,
                data_module,
            )
            split_name = f"test_file_{idx}"
            save_report(
                run_dir,
                split_name,
                input_df,
                true_df,
                pred_df,
                std_df=std_df if cfg.save_std else None,
                error_feature_names=cfg.error_feature_names,
                model_name="mticm",
            )
            write_uncertainty_metrics(
                run_dir,
                split_name,
                build_uncertainty_summary(
                    true_df,
                    pred_df,
                    std_df,
                    cfg.coverage_levels,
                    global_tau=tau_value,
                    per_group_tau=per_group_tau,
                    group_indices=group_indices,
                    quantile_tau=quantile_tau,
                ),
            )

    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
