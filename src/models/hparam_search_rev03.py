"""
Hyperparameter search harness for rev03_rtd_nf_e3_enhanced.

This script orchestrates multiple runs of `models/rev03_rtd_nf_e3_enhanced.py`
with different hyperparameters, collects validation/test metrics, and writes a
summary CSV. It keeps per-trial artifacts isolated and reproducible.

Usage example (mirroring user's current setup):

  uv run python models/hparam_search_rev03.py \
      --train-csv data/augmented/2d01_1500.csv \
      data/augmented/2d01_1500_round4_noise3_train.csv \
      data/augmented/2d01_1500_round4_noise5_train.csv \
      data/augmented/2d01_5000_round4_noise3_train.csv \
      data/augmented/2d01_5000_round4_noise5_train.csv \
      data/augmented/2d01_15000_round4_noise3_train.csv \
      data/augmented/2d01_15000_round4_noise5_train.csv \
      --test-csv data/augmented/2d01_1500_round4_noise3_test.csv \
      data/augmented/2d01_5000_round4_noise3_test.csv \
      data/augmented/2d01_15000_round4_noise3_test.csv \
      --input-cols-re '^R_.+_kN$, ^theta_.+rad$' \
      --target-cols-re '^settlement_[b-d]_mm$, ^ei_factor_s\\d+$, ^dT_s\\d+_C$' \
      --no-augment-flip \
      --epochs 2500 --print-freq 100 --early-stopping --patience 750 \
      --scheduler cosine --warmup-steps 50 \
      --model-type res_mlp \
      --grid hidden_dim=256,384,512 num_layers=4,6,8 lr=5e-5,1e-4,2e-4 dropout=0,0.05,0.1 batch_size=8192,16384

Outputs:
- Experiments under `experiments/rev03_hparam_search/<timestamp>/trial_xxx_*/<run_ts>/`
- Summary CSV at `experiments/rev03_hparam_search/<timestamp>/summary.csv`

Notes:
- Trials run sequentially to avoid GPU contention.
- Objective defaults to minimizing validation RMSE; falls back to test RMSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import time


# --------------------------- CLI and config types ---------------------------


@dataclass
class BaseArgs:
    train_csv: List[Path]
    test_csv: List[Path]
    input_cols: Optional[List[str]] = None
    target_cols: Optional[List[str]] = None
    input_cols_re: Optional[List[str]] = None
    target_cols_re: Optional[List[str]] = None
    augment_flip: bool = False
    epochs: int = 2500
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    step_size: int = 3000
    gamma: float = 0.5
    warmup_steps: Optional[int] = 50
    early_stopping: bool = True
    patience: int = 750
    min_delta: float = 0.0
    print_freq: int = 100
    seed: int = 42
    model_type: str = "res_mlp"
    num_layers: int = 6
    hidden_dim: int = 256
    act_hidden: str = "silu"
    act_out: str = "identity"
    dropout: float = 0.0
    norm: str = "none"
    init: str = "xavier"
    loss: str = "mse"
    loss_weights: Optional[List[float]] = None
    batch_size: int = 16384
    num_workers: int = 0
    ckpt_interval: int = 0
    amp: bool = False
    enable_tf32: bool = False
    val_interval: int = 1
    experiment_root: Optional[Path] = None


@dataclass
class SearchSpace:
    grid: Dict[str, List[str]] = field(default_factory=dict)


def _parse_csv_list(values: Sequence[Path]) -> List[Path]:
    return [Path(v) for v in values]


def _parse_cols(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    raw = [x.strip() for x in s.replace("\n", ",").replace(" ", ",").split(",")]
    cols = [c for c in raw if c]
    return cols or None


def _parse_loss_weights(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    out: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out or None


def _parse_grid(tokens: List[str]) -> Dict[str, List[str]]:
    """Parse grid specs like ["lr=1e-4,2e-4", "num_layers=4,6,8"]."""
    grid: Dict[str, List[str]] = {}
    for tok in tokens:
        if "=" not in tok:
            raise SystemExit(f"Invalid --grid item '{tok}', expected key=v1,v2")
        k, v = tok.split("=", 1)
        k = k.strip()
        vals = [x.strip() for x in v.split(",") if x.strip()]
        if not k or not vals:
            raise SystemExit(f"Invalid --grid item '{tok}', empty key or values")
        grid[k] = vals
    return grid


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hyperparameter search for rev03 enhanced trainer")
    # Data and columns
    p.add_argument("--train-csv", type=Path, nargs='+', required=True)
    p.add_argument("--test-csv", type=Path, nargs='+', required=True)
    p.add_argument("--input-cols", type=str, default=None)
    p.add_argument("--target-cols", type=str, default=None)
    p.add_argument("--input-cols-re", type=str, default=None)
    p.add_argument("--target-cols-re", type=str, default=None)
    p.add_argument("--augment-flip", action="store_true")
    p.add_argument("--no-augment-flip", action="store_true")

    # Base hyperparameters (defaults mirror user's example)
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "step", "plateau", "cosine"])
    p.add_argument("--step-size", type=int, default=3000)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--patience", type=int, default=750)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--print-freq", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--model-type", type=str, default="res_mlp", choices=["mlp", "res_mlp"])
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--act-hidden", type=str, default="silu")
    p.add_argument("--act-out", type=str, default="identity")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--norm", type=str, default="none", choices=["none", "batchnorm", "layernorm"])
    p.add_argument("--init", type=str, default="xavier", choices=["xavier", "kaiming"])
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "huber", "l1"])
    p.add_argument("--loss-weights", type=str, default=None, help="Comma-separated per-target weights")

    # Training system
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--ckpt-interval", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--enable-tf32", action="store_true")
    p.add_argument("--val-interval", type=int, default=1)

    # Search controls
    p.add_argument("--grid", nargs='*', default=[], help="Grid items like key=v1,v2 (e.g., lr=1e-4,2e-4 num_layers=4,6)")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of trials (first N combos)")
    p.add_argument("--objective", type=str, default="rmse", choices=["rmse", "mae", "mape", "r2"], help="Metric to optimize (overall)")
    p.add_argument("--maximize", action="store_true", help="Maximize objective instead of minimize (useful for r2)")
    p.add_argument("--search-root", type=Path, default=None, help="Root folder to store search outputs; default experiments/rev03_hparam_search/<ts>")

    return p


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _resolve_search_root(user_root: Optional[Path]) -> Path:
    base = Path("experiments") / "rev03_hparam_search"
    if user_root is not None:
        return user_root
    root = base / _now_ts()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _trial_exp_root(search_root: Path, trial_name: str) -> Path:
    p = search_root / trial_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _list_subdirs(p: Path) -> List[Path]:
    return sorted([d for d in p.iterdir() if d.is_dir()])


def _find_single_child_dir(root: Path) -> Path:
    subdirs = _list_subdirs(root)
    if not subdirs:
        raise FileNotFoundError(f"No run directory created under {root}")
    return subdirs[-1]


def _metric_from_json(metrics: Dict, objective: str) -> float:
    if objective == "r2":
        return float(metrics["r2"]["overall"])  # maximize typically
    return float(metrics[objective]["overall"])  # rmse/mae/mape


def _read_metrics(run_dir: Path, objective: str) -> Tuple[str, float, Dict[str, float]]:
    """Return (split_used, objective_value, extras)."""
    metrics_path = run_dir / "val_metrics.json"
    split = "val"
    if not metrics_path.exists():
        metrics_path = run_dir / "test_metrics.json"
        split = "test"
    with metrics_path.open("r") as f:
        metrics = json.load(f)
    obj_val = _metric_from_json(metrics, objective)
    extras = {
        "r2": float(metrics["r2"]["overall"]),
        "rmse": float(metrics["rmse"]["overall"]),
        "mae": float(metrics["mae"]["overall"]),
        "mape": float(metrics["mape"]["overall"]),
    }
    return split, obj_val, extras


def _combo_signature(overrides: Dict[str, str]) -> str:
    """Stable signature for a combo: sorted key=value joined by ';'."""
    items = [f"{k}={v}" for k, v in sorted(overrides.items())]
    return ";".join(items)


def _load_completed_signatures(summary_path: Path) -> List[Tuple[str, Dict[str, str], Dict[str, str]]]:
    """Return list of (signature, params_dict, row_dict)."""
    completed: List[Tuple[str, Dict[str, str], Dict[str, str]]] = []
    try:
        if not summary_path.exists():
            return completed
        import csv
        with summary_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    params = json.loads(row.get("params", "{}"))
                    if isinstance(params, dict):
                        completed.append((_combo_signature({k: str(v) for k, v in params.items()}), params, row))
                except Exception:
                    continue
    except Exception:
        pass
    return completed


def _build_train_cmd(base: BaseArgs, overrides: Dict[str, str], exp_root: Path) -> List[str]:
    script = str(Path("models") / "rev03_rtd_nf_e3_enhanced.py")
    cmd: List[str] = [sys.executable, script]

    # Data
    cmd += ["--train-csv", *[str(p) for p in base.train_csv]]
    cmd += ["--test-csv", *[str(p) for p in base.test_csv]]

    # Columns
    if base.input_cols:
        cmd += ["--input-cols", ",".join(base.input_cols)]
    if base.target_cols:
        cmd += ["--target-cols", ",".join(base.target_cols)]
    if base.input_cols_re:
        cmd += ["--input-cols-re", ",".join(base.input_cols_re)]
    if base.target_cols_re:
        cmd += ["--target-cols-re", ",".join(base.target_cols_re)]

    # Flags
    if base.augment_flip:
        cmd.append("--augment-flip")
    else:
        cmd.append("--no-augment-flip")

    # Base hparams
    cmd += [
        "--epochs", str(base.epochs),
        "--lr", str(base.lr),
        "--weight-decay", str(base.weight_decay),
        "--scheduler", str(base.scheduler),
        "--step-size", str(base.step_size),
        "--gamma", str(base.gamma),
        "--print-freq", str(base.print_freq),
        "--seed", str(base.seed),
        "--model-type", str(base.model_type),
        "--num-layers", str(base.num_layers),
        "--hidden-dim", str(base.hidden_dim),
        "--act-hidden", str(base.act_hidden),
        "--act-out", str(base.act_out),
        "--dropout", str(base.dropout),
        "--norm", str(base.norm),
        "--init", str(base.init),
        "--loss", str(base.loss),
        "--batch-size", str(base.batch_size),
        "--num-workers", str(base.num_workers),
        "--ckpt-interval", str(base.ckpt_interval),
        "--val-interval", str(base.val_interval),
        "--experiment-root", str(exp_root),
    ]
    if base.loss_weights:
        cmd += ["--loss-weights", ",".join(str(x) for x in base.loss_weights)]
    if base.warmup_steps is not None:
        cmd += ["--warmup-steps", str(base.warmup_steps)]
    if base.early_stopping:
        cmd.append("--early-stopping")
        cmd += ["--patience", str(base.patience), "--min-delta", str(base.min_delta)]
    if base.amp:
        cmd.append("--amp")
    if base.enable_tf32:
        cmd.append("--enable-tf32")

    # Apply per-trial overrides (string values already)
    for k, v in overrides.items():
        flag = f"--{k.replace('_', '-')}"
        if v == "true" and k in {"augment_flip", "early_stopping", "amp", "enable_tf32"}:
            cmd.append(flag)
        elif v == "false" and k in {"augment_flip"}:
            cmd.append("--no-augment-flip")
        else:
            cmd += [flag, v]

    return cmd


def _product(grid: Dict[str, List[str]]) -> Iterable[Dict[str, str]]:
    if not grid:
        yield {}
        return
    # Cartesian product
    keys = list(grid.keys())
    lists = [grid[k] for k in keys]
    def rec(i: int, cur: Dict[str, str]):
        if i == len(keys):
            yield dict(cur)
            return
        k = keys[i]
        for v in lists[i]:
            cur[k] = v
            yield from rec(i + 1, cur)
        cur.pop(k, None)
    yield from rec(0, {})


def _grid_size(grid: Dict[str, List[str]]) -> int:
    if not grid:
        return 1
    n = 1
    for v in grid.values():
        n *= max(1, len(v))
    return n


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Base args
    augment_flip = True
    if args.no_augment_flip:
        augment_flip = False
    elif args.augment_flip:
        augment_flip = True

    base = BaseArgs(
        train_csv=_parse_csv_list(args.train_csv),
        test_csv=_parse_csv_list(args.test_csv),
        input_cols=_parse_cols(args.input_cols),
        target_cols=_parse_cols(args.target_cols),
        input_cols_re=_parse_cols(args.input_cols_re),
        target_cols_re=_parse_cols(args.target_cols_re),
        augment_flip=augment_flip,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        warmup_steps=args.warmup_steps,
        early_stopping=bool(args.early_stopping),
        patience=args.patience,
        min_delta=args.min_delta,
        print_freq=args.print_freq,
        seed=args.seed,
        model_type=args.model_type,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        act_hidden=args.act_hidden,
        act_out=args.act_out,
        dropout=args.dropout,
        norm=args.norm,
        init=args.init,
        loss=args.loss,
        loss_weights=_parse_loss_weights(args.loss_weights),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ckpt_interval=args.ckpt_interval,
        amp=bool(args.amp),
        enable_tf32=bool(args.enable_tf32),
        val_interval=args.val_interval,
        experiment_root=None,
    )

    grid = _parse_grid(args.grid)
    # Reasonable defaults if user didn't pass a grid
    if not grid:
        grid = {
            "hidden-dim": [str(base.hidden_dim), "384", "512"],
            "num-layers": [str(base.num_layers), "4", "8"],
            "lr": [str(base.lr), "5e-5", "2e-4"],
            "dropout": [str(base.dropout), "0.05", "0.1"],
            "batch-size": [str(base.batch_size), "8192", "16384"],
            "weight-decay": [str(base.weight_decay), "1e-5", "1e-4"],
        }

    search_root = _resolve_search_root(args.search_root)
    print(f"Search root: {search_root}")

    # Summary CSV
    summary_path = search_root / "summary.csv"
    fieldnames = [
        "trial_id", "run_dir", "objective", "objective_value", "split",
        "r2", "rmse", "mae", "mape",
        "params"
    ]

    # Load already completed combos for resume
    completed_list = _load_completed_signatures(summary_path)
    completed_signatures = {sig for sig, _, _ in completed_list}

    # Open summary in append mode; create header if new
    new_file = not summary_path.exists()
    with summary_path.open("a", newline="") as fsum:
        writer = csv.DictWriter(fsum, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        # Initialize best from existing rows (matching objective)
        best_row = None
        best_val = None
        maximize = bool(args.maximize or (args.objective == "r2"))
        for sig, params, row in completed_list:
            try:
                if row.get("objective") != args.objective:
                    continue
                val = float(row.get("objective_value", "nan"))
                if best_val is None:
                    best_val = val
                    best_row = row
                else:
                    better = (val > best_val) if maximize else (val < best_val)
                    if better:
                        best_val = val
                        best_row = row
            except Exception:
                continue

        all_combos = list(_product(grid))
        planned_total = len(all_combos)
        already_done = sum(1 for c in all_combos if _combo_signature(c) in completed_signatures)
        remaining_total = planned_total - already_done
        if args.limit is not None:
            remaining_total = min(remaining_total, args.limit)
        print(f"Planned combos: {planned_total} | Completed: {already_done} | Remaining (to run now): {remaining_total}")

        launched = 0
        serial_idx = already_done  # continue trial numbering after completed ones

        for overrides in all_combos:
            sig = _combo_signature(overrides)
            if sig in completed_signatures:
                # Already done, skip
                continue
            if args.limit is not None and launched >= args.limit:
                break

            trial_name = f"trial_{serial_idx:03d}"
            serial_idx += 1
            launched += 1
            trial_root = _trial_exp_root(search_root, trial_name)

            # Build command
            cmd = _build_train_cmd(base, overrides, trial_root)
            print(f"\n[{trial_name}] Running ({launched}/{remaining_total}) : {' '.join(shlex.quote(c) for c in cmd)}")
            print(f"[{trial_name}] Overrides: {json.dumps(overrides)}")
            t0 = time.time()

            # Run subprocess
            try:
                # Stream child output directly for visibility
                proc = subprocess.run(cmd, check=False)
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
            except subprocess.CalledProcessError as e:
                print(f"[{trial_name}] Training failed (code {e.returncode}). Skipping.")
                continue

            # Resolve run dir (single timestamp child)
            try:
                run_dir = _find_single_child_dir(trial_root)
            except Exception as e:
                print(f"[{trial_name}] Failed to locate run dir under {trial_root}: {e}")
                continue

            # Read metrics
            try:
                split, obj_val, extras = _read_metrics(run_dir, args.objective)
            except Exception as e:
                print(f"[{trial_name}] Failed to read metrics: {e}")
                continue

            dt = time.time() - t0
            print(
                f"[{trial_name}] split={split} r2={extras['r2']:.4f} rmse={extras['rmse']:.6g} "
                f"mae={extras['mae']:.6g} mape={extras['mape']:.3f}% {args.objective}={obj_val:.6g} "
                f"duration={dt:.1f}s"
            )
            print(f"[{trial_name}] run_dir: {run_dir}")

            row = {
                "trial_id": trial_name,
                "run_dir": str(run_dir),
                "objective": args.objective,
                "objective_value": obj_val,
                "split": split,
                "r2": extras["r2"],
                "rmse": extras["rmse"],
                "mae": extras["mae"],
                "mape": extras["mape"],
                "params": json.dumps(overrides),
            }
            writer.writerow(row)
            fsum.flush()

            improved = False
            if best_val is None:
                improved = True
            else:
                improved = (obj_val > best_val) if maximize else (obj_val < best_val)
            if improved:
                best_val = obj_val
                best_row = row
                print(f"[{trial_name}] New best ({args.objective}{'↑' if maximize else '↓'}): {obj_val:.6g}")

    # Final best summary
    if summary_path.exists():
        print(f"\nSummary written to: {summary_path}")
    if best_row is not None:
        print("Best trial:")
        print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
