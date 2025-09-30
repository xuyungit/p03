"""
rev03_rtd_nf_e3_enhanced

Phase 0 enhanced training script:
- Full-epoch training with DataLoaders
- Experiment directory per run with TensorBoard
- Best/last checkpoints and complete artifacts

Note: MixUp intentionally not implemented at this stage.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Optional, List
import re

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from models.evaluation.report import EvaluationReport
from models.training.builders import ModelConfig, OptimConfig, build_model_components
from models.training.data import DataConfig, TabularDataModule
from models.training.loop import EarlyStoppingConfig, TrainRunConfig, run_training_loop
from models.training.noise import BatchNoiseInjector
from models.training.augment import BatchFlipTransform
from models.training.augment_config import load_augmentor
from models.training.predict import mc_dropout_predict
from data.noise_utils import make_gaussian_snr


def set_all_seeds(seed: int = 42) -> None:
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class CLIConfig:
    # Data
    train_csv: List[Path]
    test_csv: List[Path]
    val_ratio: float = 0.1
    batch_size: int = 128
    num_workers: int = 0
    augment_flip: bool = True
    # Columns (optional; defaults provided by script)
    input_cols: Optional[List[str]] = None
    target_cols: Optional[List[str]] = None
    input_cols_re: Optional[List[str]] = None
    target_cols_re: Optional[List[str]] = None
    preset: Optional[Path] = None
    # Model
    num_layers: int = 4
    hidden_dim: int = 128
    act_hidden: str = "silu"
    act_out: str = "identity"
    dropout_p: float = 0.1
    model_type: str = "mlp"
    norm: str = "none"
    init: str = "xavier"
    loss: str = "mse"
    loss_weights: Optional[List[float]] = None
    # Optim
    epochs: int = 30000
    lr: float = 2e-3
    weight_decay: float = 1e-4
    scheduler: str = "step"
    step_size: int = 3000
    gamma: float = 0.5
    print_freq: int = 2000
    grad_clip: float = 1.0
    warmup_steps: Optional[int] = None
    ckpt_interval: int = 0
    amp: bool = False
    enable_tf32: bool = False
    val_interval: int = 1
    # Misc
    seed: int = 42
    experiment_root: Optional[Path] = None  # default resolves to experiments/{module_base}
    # Inference / analysis
    mc_samples: int = 0
    permute_importance: bool = False
    permute_repeats: int = 5
    error_feature_names: Optional[List[str]] = None
    # Early stopping
    early_stopping: Optional[bool] = None
    no_early_stopping: Optional[bool] = None
    patience: int = 2000
    min_delta: float = 0.0
    # Augmentation config
    augment_profile: Optional[str] = None
    augment_config: Optional[Path] = None


def _parse_args() -> CLIConfig:
    p = argparse.ArgumentParser(description="rev03 enhanced training for RD->FN model")
    # Allow one or more training CSV files
    p.add_argument("--train-csv", type=Path, nargs='+', default=[Path("data/d03_all_train.csv")])
    p.add_argument("--test-csv", type=Path, nargs='+', default=[Path("data/d03_all_test.csv")])
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--augment-flip", action="store_true")
    p.add_argument("--no-augment-flip", action="store_true")
    # Columns
    p.add_argument(
        "--input-cols",
        type=str,
        default=None,
        help="Comma-separated input feature names. Defaults to script's built-in list.",
    )
    p.add_argument(
        "--target-cols",
        type=str,
        default=None,
        help="Comma-separated target names. Defaults to script's built-in list.",
    )
    p.add_argument(
        "--input-cols-re",
        type=str,
        default=None,
        help="Comma-separated regex patterns to match input columns (applied to CSV headers).",
    )
    p.add_argument(
        "--target-cols-re",
        type=str,
        default=None,
        help="Comma-separated regex patterns to match target columns (applied to CSV headers).",
    )
    p.add_argument(
        "--preset",
        type=Path,
        default=None,
        help="Optional JSON preset file with fields like train_csv, test_csv, input_cols, target_cols, input_cols_re, target_cols_re and hyperparameters.",
    )
    # Model
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--act-hidden", type=str, default="silu")
    p.add_argument("--act-out", type=str, default="identity")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--model-type", type=str, default="mlp", choices=["mlp", "res_mlp"])
    p.add_argument("--norm", type=str, default="none", choices=["none", "batchnorm", "layernorm"])
    p.add_argument("--init", type=str, default="xavier", choices=["xavier", "kaiming"])
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "huber", "l1"])
    p.add_argument("--loss-weights", type=str, default=None, help="Comma-separated per-target loss weights (length must match outputs)")
    # Optim
    p.add_argument("--epochs", type=int, default=30000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="step", choices=["none", "step", "plateau", "cosine"])
    p.add_argument("--step-size", type=int, default=3000)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--print-freq", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps (epochs); default heuristic if unset")
    p.add_argument("--ckpt-interval", type=int, default=0, help="If >0, save last checkpoint every N epochs; 0 saves every epoch")
    p.add_argument("--amp", action="store_true", help="Enable AMP mixed precision on CUDA for faster training")
    p.add_argument("--enable-tf32", action="store_true", help="Enable TF32 matmul/cudnn for faster FP32 on Ampere+")
    p.add_argument("--val-interval", type=int, default=1, help="Validate and evaluate best checkpoint every N epochs (1 = each epoch)")
    # Train-time noise injection (default off)
    p.add_argument("--train-noise-snr", type=float, default=None, help="If set, inject Gaussian noise by SNR(dB) into inputs during training")
    p.add_argument("--train-noise-prob", type=float, default=0.5, help="Probability per batch to inject noise when train-noise-snr is set")
    # Train-time random flip (batch-level) default off
    p.add_argument("--train-flip-prob", type=float, default=0.0, help="Probability per batch to apply left-right flip to both X and Y in training")
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--patience", type=int, default=2000)
    p.add_argument("--min-delta", type=float, default=0.0)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment-root", type=Path, default=None, help="Optional experiments root directory. Defaults to experiments/{module_base} derived from script name.")
    # Inference / analysis
    p.add_argument("--mc-samples", type=int, default=0, help="If >0, perform MC Dropout prediction with N samples and save mean/std CSVs")
    p.add_argument("--permute-importance", action="store_true", help="Enable permutation feature importance on test split")
    p.add_argument("--permute-repeats", type=int, default=5, help="Permutation repeats per feature")
    p.add_argument("--error-features", type=str, nargs='*', default=None, help="Feature names to plot error-vs-feature scatter; defaults to a subset of input cols")
    # Augmentation config
    p.add_argument("--augment-profile", type=str, default=None, help="Named augment profile under models/augments (e.g., 'rev02_flip', 'd3d01_flip', 'none')")
    p.add_argument("--augment-config", type=Path, default=None, help="Path to augment config JSON (overrides profile if set)")

    args = p.parse_args()
    augment_flip = True
    if args.no_augment_flip:
        augment_flip = False
    elif args.augment_flip:
        augment_flip = True

    # Parse optional list fields
    loss_weights = None
    if args.loss_weights:
        try:
            loss_weights = [float(x) for x in str(args.loss_weights).split(',') if x.strip()]
        except Exception as e:
            raise SystemExit(f"Invalid --loss-weights: {e}")
    error_feature_names = args.error_features if args.error_features else None

    # Parse optional columns
    def _parse_cols(s: Optional[str]) -> Optional[List[str]]:
        if s is None:
            return None
        # Allow comma/space separated; ignore empties
        raw = [x.strip() for x in s.replace("\n", ",").replace(" ", ",").split(",")]
        cols = [c for c in raw if c]
        if not cols:
            return None
        return cols
    input_cols = _parse_cols(args.input_cols)
    target_cols = _parse_cols(args.target_cols)
    input_cols_re = _parse_cols(args.input_cols_re)
    target_cols_re = _parse_cols(args.target_cols_re)

    # Start building base config
    cfg = CLIConfig(
        train_csv=list(args.train_csv),
        test_csv=list(args.test_csv),
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_flip=augment_flip,
        augment_profile=args.augment_profile,
        augment_config=args.augment_config,
        input_cols=input_cols,
        target_cols=target_cols,
        input_cols_re=input_cols_re,
        target_cols_re=target_cols_re,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        act_hidden=args.act_hidden,
        act_out=args.act_out,
        dropout_p=args.dropout,
        model_type=args.model_type,
        norm=args.norm,
        init=args.init,
        loss=args.loss,
        loss_weights=loss_weights,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        print_freq=args.print_freq,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        ckpt_interval=int(args.ckpt_interval),
        amp=bool(args.amp),
        enable_tf32=bool(args.enable_tf32),
        val_interval=int(args.val_interval),
        seed=args.seed,
        experiment_root=args.experiment_root,
        mc_samples=args.mc_samples,
        permute_importance=bool(args.permute_importance),
        permute_repeats=int(args.permute_repeats),
        error_feature_names=error_feature_names,
        preset=args.preset,
        early_stopping=bool(args.early_stopping),
        no_early_stopping=bool(args.no_early_stopping),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
    )

    # Merge preset JSON if provided
    if args.preset is not None:
        try:
            with open(args.preset, "r") as f:
                preset = json.load(f)
            # Basic paths
            if "train_csv" in preset:
                trv = preset["train_csv"]
                if isinstance(trv, list):
                    cfg.train_csv = [Path(x) for x in trv]  # type: ignore[assignment]
                else:
                    cfg.train_csv = [Path(trv)]  # type: ignore[assignment]
            if "test_csv" in preset:
                tsv = preset["test_csv"]
                if isinstance(tsv, list):
                    cfg.test_csv = [Path(x) for x in tsv]  # type: ignore[assignment]
                else:
                    cfg.test_csv = [Path(tsv)]  # type: ignore[assignment]
            # Columns (only set if not specified via CLI flags)
            if cfg.input_cols is None and "input_cols" in preset:
                cfg.input_cols = list(map(str, preset["input_cols"]))
            if cfg.target_cols is None and "target_cols" in preset:
                cfg.target_cols = list(map(str, preset["target_cols"]))
            if cfg.input_cols_re is None and ("input_cols_re" in preset or "input_patterns" in preset):
                src = preset.get("input_cols_re", preset.get("input_patterns"))
                if isinstance(src, list):
                    cfg.input_cols_re = [str(x) for x in src]
            if cfg.target_cols_re is None and ("target_cols_re" in preset or "target_patterns" in preset):
                src = preset.get("target_cols_re", preset.get("target_patterns"))
                if isinstance(src, list):
                    cfg.target_cols_re = [str(x) for x in src]
            # Flip augment
            if "augment_flip" in preset:
                cfg.augment_flip = bool(preset["augment_flip"])  # type: ignore[assignment]
            if cfg.augment_profile is None and "augment_profile" in preset:
                cfg.augment_profile = str(preset["augment_profile"])  # type: ignore[assignment]
            if cfg.augment_config is None and "augment_config" in preset:
                try:
                    cfg.augment_config = Path(preset["augment_config"])  # type: ignore[assignment]
                except Exception:
                    pass
            # Hyperparameters (optional overrides)
            for k in [
                "num_layers","hidden_dim","act_hidden","act_out","dropout_p","model_type","norm","init",
                "loss","epochs","lr","weight_decay","scheduler","step_size","gamma","print_freq","grad_clip",
                "seed","mc_samples","permute_repeats","warmup_steps","patience","min_delta"
            ]:
                if k in preset:
                    setattr(cfg, k, preset[k])
            if "permute_importance" in preset:
                cfg.permute_importance = bool(preset["permute_importance"])  # type: ignore[assignment]
            if "early_stopping" in preset:
                cfg.early_stopping = bool(preset["early_stopping"])  # type: ignore[assignment]
            if "no_early_stopping" in preset:
                cfg.no_early_stopping = bool(preset["no_early_stopping"])  # type: ignore[assignment]
            if "loss_weights" in preset and cfg.loss_weights is None:
                try:
                    cfg.loss_weights = [float(x) for x in preset["loss_weights"]]
                except Exception:
                    pass
        except Exception as e:
            raise SystemExit(f"Failed to load preset {args.preset}: {e}")
    return cfg


def _module_base_name() -> str:
    # Use file stem and strip common suffix like _enhanced
    stem = Path(__file__).stem
    if stem.endswith("_enhanced"):
        stem = stem[: -len("_enhanced")]
    return stem


def _default_experiment_root() -> Path:
    return Path("experiments") / _module_base_name()


def _default_checkpoint_archive_dir() -> Path:
    return Path("models/checkpoints") / _module_base_name()


def _start_run_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / ts
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def _configure_matplotlib(run_dir: Path) -> None:
    """Set Matplotlib to a headless, English-safe config and writable cache.

    - Backend: Agg
    - Font: DejaVu Sans
    - Disable unicode minus issues
    - MPLCONFIGDIR: {run_dir}/.mplconfig
    """
    import os
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
        (run_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        import matplotlib as mpl
        mpl.use("Agg")
        mpl.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})
    except Exception:
        pass


def _save_config(run_dir: Path, cfg: CLIConfig, data_cfg: DataConfig, model_cfg: ModelConfig, optim_cfg: OptimConfig, train_cfg: TrainRunConfig) -> None:
    cli_dict = asdict(cfg)
    # Convert Path fields to str for JSON serialization
    for k in list(cli_dict.keys()):
        v = cli_dict[k]
        if isinstance(v, Path):
            cli_dict[k] = str(v)
        elif isinstance(v, list):
            cli_dict[k] = [str(x) if isinstance(x, Path) else x for x in v]

    # Serialize train_csv list if needed
    if isinstance(data_cfg.train_csv, (list, tuple)):
        train_csv_serialized = [str(p) for p in data_cfg.train_csv]
    else:
        train_csv_serialized = str(data_cfg.train_csv)
    
    # Serialize test_csv list if needed
    if isinstance(data_cfg.test_csv, (list, tuple)):
        test_csv_serialized = [str(p) for p in data_cfg.test_csv]
    else:
        test_csv_serialized = str(data_cfg.test_csv)

    merged: Dict = {
        "cli": cli_dict,
        "data": {
            **asdict(data_cfg),
            "train_csv": train_csv_serialized,
            "test_csv": test_csv_serialized,
        },
        "model": asdict(model_cfg),
        "optim": asdict(optim_cfg),
        "train": asdict(train_cfg),
        "columns": {
            "input": list(data_cfg.input_cols),
            "target": list(data_cfg.target_cols),
        },
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(merged, f, indent=2)


def evaluate_and_save(
    model: torch.nn.Module,
    X_s: np.ndarray,
    Y_s: np.ndarray,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    split_name: str,
    run_dir: Path,
    *,
    mc_samples: int = 0,
    error_feature_names: Optional[List[str]] = None,
    input_cols: Optional[Sequence[str]] = None,
    target_cols: Optional[Sequence[str]] = None,
) -> None:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_s, dtype=torch.float32, device=next(model.parameters()).device)
        Y_t = torch.tensor(Y_s, dtype=torch.float32, device=next(model.parameters()).device)
        Y_pred_s = model(X_t).cpu().numpy()

    Y_true = scaler_y.inverse_transform(Y_t.cpu().numpy())
    Y_pred = scaler_y.inverse_transform(Y_pred_s)
    # Recover original-unit inputs
    X_orig = scaler_x.inverse_transform(X_s)

    tgt_cols = list(target_cols)
    in_cols = list(input_cols) 
    y_true_df = pd.DataFrame(Y_true, columns=tgt_cols)
    y_pred_df = pd.DataFrame(Y_pred, columns=tgt_cols)
    X_df = pd.DataFrame(X_orig, columns=in_cols)

    report = EvaluationReport(model_name=f"rev03_{split_name}")
    report.evaluate(y_true_df, y_pred_df)
    report.save_artifacts(
        str(run_dir),
        split_name=split_name,
        y_true_df=y_true_df,
        y_pred_df=y_pred_df,
        X_df=X_df,
        error_feature_names=error_feature_names,
    )

    if split_name == "test":
        error_df = (y_pred_df - y_true_df).abs()
        print(f"\nTop 10 largest errors for {split_name} set:")
        for col in tgt_cols:
            top_10_errors = error_df[col].nlargest(10)
            print(f"--- Channel: {col} ---")
            print(top_10_errors.to_string())

    # Optional MC Dropout predictions (save mean/std in original units)
    if mc_samples and mc_samples > 0:
        with torch.no_grad():
            X_t = torch.tensor(X_s, dtype=torch.float32, device=next(model.parameters()).device)
            mean_s, std_s = mc_dropout_predict(model, X_t, n_samples=mc_samples)
            mean = scaler_y.inverse_transform(mean_s.cpu().numpy())
            # StandardScaler: y = y_s * scale + mean  => std_y = std_y_s * scale
            std = std_s.cpu().numpy() * scaler_y.scale_.reshape(1, -1)
        pd.DataFrame(mean, columns=tgt_cols).to_csv(run_dir / f"{split_name}_pred_mc_mean.csv", index=False)
        pd.DataFrame(std, columns=tgt_cols).to_csv(run_dir / f"{split_name}_pred_mc_std.csv", index=False)


def main() -> None:
    cfg = _parse_args()
    set_all_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional TF32 acceleration on Ampere+ GPUs
    if cfg.enable_tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            # PyTorch 2.x
            torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
        except Exception:
            pass

    exp_root = cfg.experiment_root if cfg.experiment_root is not None else _default_experiment_root()
    run_dir = _start_run_dir(exp_root)
    _configure_matplotlib(run_dir)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    # Data
    # Resolve columns using explicit lists or regex patterns against CSV headers
    def _resolve_by_patterns(headers: List[str], patterns: Optional[List[str]]) -> Optional[List[str]]:
        if not patterns:
            return None
        compiled = []
        for pat in patterns:
            try:
                compiled.append(re.compile(pat))
            except re.error as e:
                raise SystemExit(f"Invalid regex pattern '{pat}': {e}")
        out: List[str] = []
        for h in headers:
            if any(rx.search(h) for rx in compiled):
                out.append(h)
        # Deduplicate while preserving order
        seen = set()
        uniq = [x for x in out if not (x in seen or seen.add(x))]
        return uniq

    # Prepare training CSV path list
    train_paths: List[Path] = list(cfg.train_csv)
    if len(train_paths) == 0:
        raise SystemExit("No training CSVs provided via --train-csv")

    # Peek headers across all training CSVs (union, preserve order)
    tr_headers_all: List[List[str]] = [list(pd.read_csv(p, nrows=0).columns) for p in train_paths]
    seen_cols = set()
    tr_head_cols: List[str] = []
    for cols in tr_headers_all:
        for c in cols:
            if c not in seen_cols:
                seen_cols.add(c)
                tr_head_cols.append(c)
    # Validate test CSV headers - ensure all test files have the same columns
    te_headers_all: List[List[str]] = [list(pd.read_csv(p, nrows=0).columns) for p in cfg.test_csv]
    if len(te_headers_all) > 1:
        # Check if all test files have identical columns
        first_headers = set(te_headers_all[0])
        for i, headers in enumerate(te_headers_all[1:], 1):
            if set(headers) != first_headers:
                raise SystemExit(f"Test files have inconsistent columns. File 1: {sorted(first_headers)}, File {i+1}: {sorted(set(headers))}")
    te_head_cols = te_headers_all[0] if te_headers_all else []
    if cfg.input_cols is not None:
        chosen_input_cols = list(cfg.input_cols)
    else:
        chosen_input_cols = _resolve_by_patterns(tr_head_cols, cfg.input_cols_re)
    if cfg.target_cols is not None:
        chosen_target_cols = list(cfg.target_cols)
    else:
        chosen_target_cols = _resolve_by_patterns(tr_head_cols, cfg.target_cols_re) 
    # Validate requested columns exist in provided CSVs
    try:
        problems = []
        # Validate each train CSV individually to avoid silent NaNs after concat
        for p in train_paths:
            cols = set(pd.read_csv(p, nrows=0).columns)
            missing_tr_in = [c for c in chosen_input_cols if c not in cols]
            missing_tr_tg = [c for c in chosen_target_cols if c not in cols]
            if missing_tr_in or missing_tr_tg:
                msg_parts = []
                if missing_tr_in:
                    msg_parts.append(f"missing input cols: {missing_tr_in}")
                if missing_tr_tg:
                    msg_parts.append(f"missing target cols: {missing_tr_tg}")
                problems.append(f"Train file {p}: " + ", ".join(msg_parts))
        # Validate each test CSV individually
        for i, p in enumerate(cfg.test_csv):
            cols = set(pd.read_csv(p, nrows=0).columns)
            missing_te_in = [c for c in chosen_input_cols if c not in cols]
            missing_te_tg = [c for c in chosen_target_cols if c not in cols]
            if missing_te_in or missing_te_tg:
                msg_parts = []
                if missing_te_in:
                    msg_parts.append(f"missing input cols: {missing_te_in}")
                if missing_te_tg:
                    msg_parts.append(f"missing target cols: {missing_te_tg}")
                problems.append(f"Test file {i+1} ({p}): " + ", ".join(msg_parts))
        if problems:
            raise SystemExit("Column validation failed. " + " | ".join(problems))
    except Exception:
        # Surface clear message and stop early for user to fix columns
        raise

    # Print resolved columns before any training starts
    print("Resolved input columns ({}):".format(len(chosen_input_cols)))
    print(", ".join(chosen_input_cols))
    print("Resolved target columns ({}):".format(len(chosen_target_cols)))
    print(", ".join(chosen_target_cols))
    print(f"Augment flip: {bool(cfg.augment_flip)}")
    data_cfg = DataConfig(
        train_csv=train_paths,
        test_csv=list(cfg.test_csv),
        input_cols=chosen_input_cols,
        target_cols=chosen_target_cols,
        val_ratio=cfg.val_ratio,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment_flip=cfg.augment_flip,
        augment_profile=cfg.augment_profile,
        augment_config=cfg.augment_config,
    )
    dm = TabularDataModule(data_cfg, device)
    dm.setup()
    tr_loader, val_loader, te_loader = dm.get_loaders()
    tr_df, val_df, te_df = dm.get_original_dfs()
    individual_test_dfs = dm.get_individual_test_dfs()
    scaler_x, scaler_y = dm.get_scalers()

    # Model & optim
    model_cfg = ModelConfig(
        in_dim=len(chosen_input_cols),
        out_dim=len(chosen_target_cols),
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        act_hidden=cfg.act_hidden,
        act_out=cfg.act_out,
        dropout_p=cfg.dropout_p,
        model_type=cfg.model_type,
        norm=cfg.norm,
        init=cfg.init,
        loss=cfg.loss,
        loss_weights=tuple(cfg.loss_weights) if cfg.loss_weights else None,
    )
    warmup_steps = cfg.epochs // 30 if cfg.epochs >= 600 else (100 if cfg.epochs >= 500 else 0)
    if hasattr(cfg, "warmup_steps") and cfg.warmup_steps is not None:
        warmup_steps = int(cfg.warmup_steps)

    optim_cfg = OptimConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        optimizer="adamw",
        scheduler=cfg.scheduler,  # step/plateau/cosine/none
        step_size=cfg.step_size,
        gamma=cfg.gamma,
        cosine_tmax=max(10, cfg.epochs),
        warmup_steps=warmup_steps,
    )
    model, optimizer, criterion, warmup_sched, scheduler = build_model_components(model_cfg, optim_cfg, device)

    train_cfg = TrainRunConfig(
        epochs=cfg.epochs,
        print_freq=cfg.print_freq,
        grad_clip=cfg.grad_clip,
        ckpt_interval=getattr(cfg, "ckpt_interval", 0),
        amp=bool(getattr(cfg, "amp", False)),
        val_interval=int(getattr(cfg, "val_interval", 1)),
    )

    # Early stopping config
    use_es = True
    if hasattr(cfg, "no_early_stopping") and cfg.no_early_stopping:
        use_es = False
    elif hasattr(cfg, "early_stopping") and cfg.early_stopping:
        use_es = True
    es_cfg = EarlyStoppingConfig(
        enabled=use_es,
        patience=getattr(cfg, "patience", 2000),
        min_delta=getattr(cfg, "min_delta", 0.0),
    )

    # Save run config
    _save_config(run_dir, cfg, data_cfg, model_cfg, optim_cfg, train_cfg)

    # Train
    # Optional training-time noise injector
    batch_noise_fn = None
    batch_pair_transform = None
    if getattr(cfg, "train_noise_snr", None) is not None:
        batch_noise_fn = BatchNoiseInjector(
            scaler_x=scaler_x,
            input_cols=chosen_input_cols,
            scheme_fn=make_gaussian_snr,
            strength=float(cfg.train_noise_snr),
            prob=float(getattr(cfg, "train_noise_prob", 0.5)),
        )
    # Optional per-batch flip of both X and Y
    if getattr(cfg, "train_flip_prob", 0.0) and float(getattr(cfg, "train_flip_prob", 0.0)) > 0.0:
        flip_augmentor = load_augmentor(cfg.augment_config, cfg.augment_profile) if cfg.augment_flip else None
        batch_pair_transform = BatchFlipTransform(
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            input_cols=chosen_input_cols,
            target_cols=chosen_target_cols,
            prob=float(getattr(cfg, "train_flip_prob", 0.0)),
            augmentor=flip_augmentor,
        )

    history = run_training_loop(
        model,
        tr_loader,
        val_loader,
        optimizer,
        criterion,
        warmup_sched,
        scheduler,
        device,
        run_dir,
        writer,
        train_cfg,
        es_cfg,
        batch_noise_fn=batch_noise_fn,
        batch_pair_transform=batch_pair_transform,
    )

    # Save composite checkpoint with scalers & history
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "history": history,
            "hyperparameters": {
                "num_layers": cfg.num_layers,
                "hidden_dim": cfg.hidden_dim,
                "act_hidden": cfg.act_hidden,
                "act_out": cfg.act_out,
                "dropout_p": cfg.dropout_p,
                "model_type": cfg.model_type,
                "norm": cfg.norm,
                "init": cfg.init,
                "loss": cfg.loss,
            },
        },
        run_dir / "checkpoints" / "last_full.pth",
    )

    # Evaluate splits using standardized arrays (then inverse-transform Y)
    # Reconstruct the same standardized arrays used by loaders for consistent comparison
    # Train
    Xy_tr = tr_loader.dataset.tensors  # type: ignore[attr-defined]
    X_tr_s = Xy_tr[0].cpu().numpy()
    Y_tr_s = Xy_tr[1].cpu().numpy()
    evaluate_and_save(
        model,
        X_tr_s,
        Y_tr_s,
        scaler_x,
        scaler_y,
        "train",
        run_dir,
        mc_samples=cfg.mc_samples,
        error_feature_names=(cfg.error_feature_names or chosen_input_cols[:3]),
        input_cols=chosen_input_cols,
        target_cols=chosen_target_cols,
    )
    # Val
    Xy_va = val_loader.dataset.tensors  # type: ignore[attr-defined]
    X_va_s = Xy_va[0].cpu().numpy()
    Y_va_s = Xy_va[1].cpu().numpy()
    evaluate_and_save(
        model,
        X_va_s,
        Y_va_s,
        scaler_x,
        scaler_y,
        "val",
        run_dir,
        mc_samples=cfg.mc_samples,
        error_feature_names=(cfg.error_feature_names or chosen_input_cols[:3]),
        input_cols=chosen_input_cols,
        target_cols=chosen_target_cols,
    )
    # Test - evaluate combined test set
    Xy_te = te_loader.dataset.tensors  # type: ignore[attr-defined]
    X_te_s = Xy_te[0].cpu().numpy()
    Y_te_s = Xy_te[1].cpu().numpy()
    evaluate_and_save(
        model,
        X_te_s,
        Y_te_s,
        scaler_x,
        scaler_y,
        "test",
        run_dir,
        mc_samples=cfg.mc_samples,
        error_feature_names=(cfg.error_feature_names or chosen_input_cols[:3]),
        input_cols=chosen_input_cols,
        target_cols=chosen_target_cols,
    )

    # Individual test files evaluation
    for i, test_df in enumerate(individual_test_dfs):
        # Transform individual test file using fitted scalers
        X_test_s = scaler_x.transform(test_df[chosen_input_cols].values)
        Y_test_s = scaler_y.transform(test_df[chosen_target_cols].values)
        
        evaluate_and_save(
            model,
            X_test_s,
            Y_test_s,
            scaler_x,
            scaler_y,
            f"test_file_{i+1}",
            run_dir,
            mc_samples=cfg.mc_samples,
            error_feature_names=(cfg.error_feature_names or chosen_input_cols[:3]),
            input_cols=chosen_input_cols,
            target_cols=chosen_target_cols,
        )

    print(f"Artifacts written to: {run_dir}")

    # Create a full best checkpoint (with scalers) for evaluation convenience
    try:
        import shutil
        best_src = run_dir / "checkpoints" / "best.pth"
        if best_src.exists():
            # Load best weights and package with scalers & hyperparameters
            best_state = torch.load(best_src, map_location=model.device if hasattr(model, 'device') else 'cpu')
            best_full_path = run_dir / "checkpoints" / "best_full.pth"
            torch.save(
                {
                    "model_state_dict": best_state.get("model_state_dict", model.state_dict()),
                    "scaler_x": scaler_x,
                    "scaler_y": scaler_y,
                    "history": history,
                    "hyperparameters": {
                        "num_layers": cfg.num_layers,
                        "hidden_dim": cfg.hidden_dim,
                        "act_hidden": cfg.act_hidden,
                        "act_out": cfg.act_out,
                        "dropout_p": cfg.dropout_p,
                        "model_type": cfg.model_type,
                        "norm": cfg.norm,
                        "init": cfg.init,
                        "loss": cfg.loss,
                    },
                },
                best_full_path,
            )

            # Archive both lightweight and full best checkpoints
            archive_dir = _default_checkpoint_archive_dir()
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_src, archive_dir / "best.pth")
            shutil.copy2(best_full_path, archive_dir / "best_full.pth")
            print(f"Best checkpoints archived to: {archive_dir}/best.pth and best_full.pth")
    except Exception as e:
        print(f"[WARN] Failed to create/archive best_full checkpoint: {e}")

    # Optional: permutation feature importance on test split
    if cfg.permute_importance:
        try:
            from models.evaluation.permutation_importance import compute_permutation_importance
            te_imp = compute_permutation_importance(
                model=model,
                X_s=X_te_s,
                y_true=scaler_y.inverse_transform(Y_te_s),
                scaler_y=scaler_y,
                feature_names=chosen_input_cols,
                device=device,
                metric="rmse",
                n_repeats=cfg.permute_repeats,
                random_state=cfg.seed,
            )
            df_imp = te_imp.to_dataframe()
            df_imp.to_csv(run_dir / "permutation_importance_test.csv", index=False)
            # Simple bar plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.bar(df_imp["feature"], df_imp["importance"], yerr=df_imp["std"], alpha=0.8)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Importance (ΔRMSE)")
            plt.title("Permutation Feature Importance (test)")
            plt.tight_layout()
            plt.savefig(run_dir / "permutation_importance_test.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved permutation importance results.")
        except Exception as e:
            print(f"[WARN] Permutation importance failed: {e}")


if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility with multiprocessing
    # This is crucial when using num_workers > 0 in DataLoader
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
