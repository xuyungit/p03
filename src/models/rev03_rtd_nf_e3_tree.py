"""
Tree-based baseline for rev03 tabular mechanics mappings.

This script mirrors the data handling and reporting pipeline of
``rev03_rtd_nf_e3_enhanced.py`` but trains a gradient-boosted tree model
instead of a neural network so the two approaches can be compared
side-by-side.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from models.evaluation.report import EvaluationReport
from models.training.augment_config import load_augmentor
from models.training.data import augment_with_flip
from models.training.tree_multioutput import LightGBMMultiTargetRegressor


@dataclass
class CLIConfig:
    train_csv: List[Path]
    test_csv: List[Path]
    val_ratio: float
    augment_flip: bool
    augment_profile: Optional[str]
    augment_config: Optional[Path]
    input_cols: Optional[List[str]]
    target_cols: Optional[List[str]]
    input_cols_re: Optional[List[str]]
    target_cols_re: Optional[List[str]]
    preset: Optional[Path]
    model_type: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    multioutput_n_jobs: Optional[int]
    booster: str
    objective: str
    early_stopping_rounds: int
    seed: int
    experiment_root: Optional[Path]
    permute_importance: bool
    permute_repeats: int
    error_feature_names: Optional[List[str]]


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _module_base_name() -> str:
    stem = Path(__file__).stem
    if stem.endswith("_enhanced"):
        stem = stem[: -len("_enhanced")]
    return stem


def _default_experiment_root() -> Path:
    return Path("experiments") / _module_base_name()


def _start_run_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / ts
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def _configure_matplotlib(run_dir: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
        (run_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    except Exception:
        pass


def _parse_cols(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    tokens = [
        tok.strip() for tok in raw.replace("\n", ",").replace(" ", ",").split(",")
    ]
    cols = [tok for tok in tokens if tok]
    return cols or None


def _parse_args() -> CLIConfig:
    parser = argparse.ArgumentParser(description="rev03 tree-based baseline")
    parser.add_argument(
        "--train-csv", type=Path, nargs="+", default=[Path("data/d03_all_train.csv")]
    )
    parser.add_argument(
        "--test-csv", type=Path, nargs="+", default=[Path("data/d03_all_test.csv")]
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--augment-flip", action="store_true")
    parser.add_argument("--no-augment-flip", action="store_true")

    # Column specification
    parser.add_argument("--input-cols", type=str, default=None)
    parser.add_argument("--target-cols", type=str, default=None)
    parser.add_argument("--input-cols-re", type=str, default=None)
    parser.add_argument("--target-cols-re", type=str, default=None)
    parser.add_argument("--preset", type=Path, default=None)

    # Model hyperparameters
    parser.add_argument(
        "--model-type",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost", "catboost", "random_forest"],
    )
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument(
        "--multioutput-n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for MultiOutputRegressor (defaults to sequential)",
    )
    parser.add_argument(
        "--booster",
        type=str,
        default="gbdt",
        choices=["gbdt", "dart", "goss", "rf"],
        help="LightGBM boosting type",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="regression",
        choices=["regression", "gamma", "regression_l1", "huber"],
        help="LightGBM objective (loss)",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=0,
        help="Enable LightGBM early stopping using validation split (0 disables)",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Alias for --experiment-root so outputs align with neural run conventions.",
    )
    parser.add_argument("--permute-importance", action="store_true")
    parser.add_argument("--permute-repeats", type=int, default=5)
    parser.add_argument("--error-features", type=str, nargs="*", default=None)
    parser.add_argument("--augment-profile", type=str, default=None)
    parser.add_argument("--augment-config", type=Path, default=None)

    args = parser.parse_args()

    augment_flip = True
    if args.no_augment_flip:
        augment_flip = False
    elif args.augment_flip:
        augment_flip = True

    cfg = CLIConfig(
        train_csv=list(args.train_csv),
        test_csv=list(args.test_csv),
        val_ratio=float(args.val_ratio),
        augment_flip=augment_flip,
        augment_profile=args.augment_profile,
        augment_config=args.augment_config,
        input_cols=_parse_cols(args.input_cols),
        target_cols=_parse_cols(args.target_cols),
        input_cols_re=_parse_cols(args.input_cols_re),
        target_cols_re=_parse_cols(args.target_cols_re),
        preset=args.preset,
        model_type=args.model_type,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_alpha=float(args.reg_alpha),
        reg_lambda=float(args.reg_lambda),
        n_jobs=int(args.n_jobs),
        multioutput_n_jobs=int(args.multioutput_n_jobs)
        if args.multioutput_n_jobs is not None
        else None,
        booster=str(args.booster),
        objective=str(args.objective),
        early_stopping_rounds=int(args.early_stopping_rounds),
        seed=int(args.seed),
        experiment_root=args.experiment_root,
        permute_importance=bool(args.permute_importance),
        permute_repeats=int(args.permute_repeats),
        error_feature_names=args.error_features if args.error_features else None,
    )

    if args.out_dir is not None:
        cfg.experiment_root = args.out_dir

    if args.preset is not None:
        try:
            with open(args.preset, "r") as f:
                preset = json.load(f)
        except Exception as exc:
            raise SystemExit(f"Failed to load preset {args.preset}: {exc}")

        def _maybe_path_list(value) -> List[Path]:
            if isinstance(value, list):
                return [Path(x) for x in value]
            return [Path(value)]

        if "train_csv" in preset:
            cfg.train_csv = _maybe_path_list(preset["train_csv"])
        if "test_csv" in preset:
            cfg.test_csv = _maybe_path_list(preset["test_csv"])
        if cfg.input_cols is None and "input_cols" in preset:
            cfg.input_cols = [str(x) for x in preset["input_cols"]]
        if cfg.target_cols is None and "target_cols" in preset:
            cfg.target_cols = [str(x) for x in preset["target_cols"]]
        if cfg.input_cols_re is None and (
            "input_cols_re" in preset or "input_patterns" in preset
        ):
            src = preset.get("input_cols_re", preset.get("input_patterns"))
            if isinstance(src, list):
                cfg.input_cols_re = [str(x) for x in src]
        if cfg.target_cols_re is None and (
            "target_cols_re" in preset or "target_patterns" in preset
        ):
            src = preset.get("target_cols_re", preset.get("target_patterns"))
            if isinstance(src, list):
                cfg.target_cols_re = [str(x) for x in src]
        if "augment_flip" in preset:
            cfg.augment_flip = bool(preset["augment_flip"])
        if cfg.augment_profile is None and "augment_profile" in preset:
            cfg.augment_profile = str(preset["augment_profile"])
        if cfg.augment_config is None and "augment_config" in preset:
            try:
                cfg.augment_config = Path(preset["augment_config"])
            except Exception:
                pass
        for key in [
            "model_type",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "booster",
            "objective",
            "early_stopping_rounds",
            "n_jobs",
            "multioutput_n_jobs",
            "seed",
            "val_ratio",
        ]:
            if key in preset:
                setattr(cfg, key, preset[key])
        if "permute_importance" in preset:
            cfg.permute_importance = bool(preset["permute_importance"])
        if "permute_repeats" in preset:
            cfg.permute_repeats = int(preset["permute_repeats"])
        if cfg.error_feature_names is None and "error_feature_names" in preset:
            cfg.error_feature_names = [str(x) for x in preset["error_feature_names"]]

    return cfg


def _resolve_by_patterns(
    headers: List[str], patterns: Optional[List[str]]
) -> Optional[List[str]]:
    if not patterns:
        return None
    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error as exc:
            raise SystemExit(f"Invalid regex pattern '{pat}': {exc}")
    out: List[str] = []
    for h in headers:
        if any(rx.search(h) for rx in compiled):
            out.append(h)
    seen: set[str] = set()
    uniq = [c for c in out if not (c in seen or seen.add(c))]
    return uniq


def _load_csv(paths: Sequence[Path]) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    if len(paths) == 0:
        raise SystemExit("No CSV paths provided")
    frames = []
    originals: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        frames.append(df)
        originals.append(df.copy())
    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined, originals


def _apply_flip_augmentation(
    df: pd.DataFrame,
    *,
    augment_flip: bool,
    augment_profile: Optional[str],
    augment_config: Optional[Path],
) -> pd.DataFrame:
    if not augment_flip:
        return df
    flip_augmentor = load_augmentor(augment_config, augment_profile)
    try:
        if flip_augmentor is not None:
            flipped = flip_augmentor.apply_df(df)
            if flipped is not None and not flipped.empty:
                return pd.concat([df, flipped], axis=0, ignore_index=True)
    except Exception:
        pass
    return augment_with_flip(df)


def _build_base_estimator(cfg: CLIConfig):
    if cfg.model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise SystemExit(
                "LightGBM is not installed. Run `uv add lightgbm`."
            ) from exc
        return LGBMRegressor(
            objective=cfg.objective,
            boosting_type=cfg.booster,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth if cfg.max_depth > 0 else -1,
            num_leaves=cfg.num_leaves,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            min_child_samples=cfg.min_child_samples,
            n_jobs=cfg.n_jobs,
            random_state=cfg.seed,
            verbose=-1,
        )
    if cfg.model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise SystemExit("XGBoost is not installed. Run `uv add xgboost`.") from exc
        return XGBRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=None if cfg.max_depth <= 0 else cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            min_child_weight=cfg.min_child_samples,
            n_jobs=cfg.n_jobs,
            objective="reg:squarederror",
            random_state=cfg.seed,
        )
    if cfg.model_type == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise SystemExit(
                "CatBoost is not installed. Run `uv add catboost`."
            ) from exc
        depth = cfg.max_depth if cfg.max_depth > 0 else 6
        return CatBoostRegressor(
            iterations=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            depth=depth,
            subsample=cfg.subsample,
            random_seed=cfg.seed,
            l2_leaf_reg=cfg.reg_lambda,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
        )
    if cfg.model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        max_depth = None if cfg.max_depth <= 0 else cfg.max_depth
        max_features: Optional[float | str]
        if cfg.colsample_bytree <= 0:
            max_features = None
        elif cfg.colsample_bytree >= 1.0:
            max_features = 1.0
        else:
            max_features = cfg.colsample_bytree
        return RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=max(1, cfg.min_child_samples),
            max_features=max_features,
            n_jobs=cfg.n_jobs,
            random_state=cfg.seed,
        )
    raise SystemExit(f"Unsupported model type: {cfg.model_type}")


def _save_config(
    run_dir: Path,
    cfg: CLIConfig,
    input_cols: Sequence[str],
    target_cols: Sequence[str],
) -> None:
    cfg_dict = asdict(cfg)
    # Convert Paths to str for JSON serialization
    for key, value in list(cfg_dict.items()):
        if isinstance(value, Path):
            cfg_dict[key] = str(value)
        elif isinstance(value, list):
            cfg_dict[key] = [str(v) if isinstance(v, Path) else v for v in value]
    cfg_dict["resolved_input_cols"] = list(input_cols)
    cfg_dict["resolved_target_cols"] = list(target_cols)
    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg_dict, f, indent=2)


def _evaluate_and_save(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    input_cols: Sequence[str],
    target_cols: Sequence[str],
    split_name: str,
    run_dir: Path,
    *,
    error_feature_names: Optional[Sequence[str]] = None,
) -> None:
    y_pred = model.predict(X)
    y_true_df = pd.DataFrame(y_true, columns=target_cols)
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols)
    X_df = pd.DataFrame(X, columns=input_cols)

    report = EvaluationReport(model_name=f"rev03_tree_{split_name}")
    report.evaluate(y_true_df, y_pred_df)
    report.save_artifacts(
        str(run_dir),
        split_name=split_name,
        y_true_df=y_true_df,
        y_pred_df=y_pred_df,
        X_df=X_df,
        error_feature_names=list(error_feature_names) if error_feature_names else None,
    )

    if split_name == "test":
        error_df = (y_pred_df - y_true_df).abs()
        print("\nTop 10 absolute errors per target (test set):")
        for col in target_cols:
            top = error_df[col].nlargest(10)
            print(f"--- {col} ---")
            print(top.to_string())


def _aggregate_feature_importances(
    model, input_cols: Sequence[str], run_dir: Path
) -> None:
    mats: List[np.ndarray] = []
    if hasattr(model, "models"):
        for est in model.models:
            if hasattr(est, "feature_importances_"):
                mats.append(np.asarray(est.feature_importances_, dtype=float))
    elif hasattr(model, "estimators_"):
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                mats.append(np.asarray(est.feature_importances_, dtype=float))
    if not mats:
        return
    mat = np.vstack(mats)
    df = pd.DataFrame(
        {
            "feature": list(input_cols),
            "importance_mean": mat.mean(axis=0),
            "importance_std": mat.std(axis=0),
        }
    ).sort_values("importance_mean", ascending=False)
    df.to_csv(run_dir / "feature_importances_mean.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.bar(
            df["feature"], df["importance_mean"], yerr=df["importance_std"], alpha=0.85
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean importance")
        plt.title("Feature importances (mean across targets)")
        plt.tight_layout()
        plt.savefig(
            run_dir / "feature_importances_mean.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    except Exception:
        pass


def _compute_permutation_importance(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_repeats: int,
    random_state: int,
    run_dir: Path,
) -> None:
    rng = np.random.default_rng(random_state)

    def _rmse(y_a: np.ndarray, y_b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_a - y_b) ** 2)))

    baseline_pred = model.predict(X)
    base_score = np.mean(
        [_rmse(y_true[:, i], baseline_pred[:, i]) for i in range(y_true.shape[1])]
    )

    N, D = X.shape
    importances = np.zeros(D, dtype=float)
    stds = np.zeros(D, dtype=float)

    for j in range(D):
        scores: List[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            y_perm = model.predict(X_perm)
            score = np.mean(
                [_rmse(y_true[:, i], y_perm[:, i]) for i in range(y_true.shape[1])]
            )
            scores.append(score - base_score)
        importances[j] = float(np.mean(scores))
        stds[j] = float(np.std(scores))

    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": importances,
            "std": stds,
        }
    ).sort_values("importance", ascending=False)
    df.to_csv(run_dir / "permutation_importance_test.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.bar(df["feature"], df["importance"], yerr=df["std"], alpha=0.85)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("ΔRMSE")
        plt.title("Permutation feature importance (test)")
        plt.tight_layout()
        plt.savefig(
            run_dir / "permutation_importance_test.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    except Exception:
        pass


def main() -> None:
    cfg = _parse_args()
    set_all_seeds(cfg.seed)

    exp_root = (
        cfg.experiment_root
        if cfg.experiment_root is not None
        else _default_experiment_root()
    )
    run_dir = _start_run_dir(exp_root)
    _configure_matplotlib(run_dir)

    # Resolve columns
    train_paths = list(cfg.train_csv)
    train_headers = [list(pd.read_csv(p, nrows=0).columns) for p in train_paths]
    seen: set[str] = set()
    merged_train_headers: List[str] = []
    for cols in train_headers:
        for col in cols:
            if col not in seen:
                seen.add(col)
                merged_train_headers.append(col)

    test_headers = [list(pd.read_csv(p, nrows=0).columns) for p in cfg.test_csv]
    if len(test_headers) > 1:
        ref = set(test_headers[0])
        for idx, cols in enumerate(test_headers[1:], start=2):
            if set(cols) != ref:
                raise SystemExit(
                    f"Test CSV #{idx} columns differ from first file. Please align headers before training."
                )

    if cfg.input_cols is not None:
        input_cols = list(cfg.input_cols)
    else:
        input_cols = _resolve_by_patterns(
            merged_train_headers, cfg.input_cols_re
        )
    if cfg.target_cols is not None:
        target_cols = list(cfg.target_cols)
    else:
        target_cols = _resolve_by_patterns(
            merged_train_headers, cfg.target_cols_re
        ) 

    # Validate columns in every file
    problems: List[str] = []
    for path in train_paths:
        cols = set(pd.read_csv(path, nrows=0).columns)
        miss_in = [c for c in input_cols if c not in cols]
        miss_out = [c for c in target_cols if c not in cols]
        if miss_in or miss_out:
            msg = ", ".join(
                filter(
                    None,
                    [
                        f"missing input cols: {miss_in}" if miss_in else None,
                        f"missing target cols: {miss_out}" if miss_out else None,
                    ],
                )
            )
            problems.append(f"Train file {path}: {msg}")
    for idx, path in enumerate(cfg.test_csv, start=1):
        cols = set(pd.read_csv(path, nrows=0).columns)
        miss_in = [c for c in input_cols if c not in cols]
        miss_out = [c for c in target_cols if c not in cols]
        if miss_in or miss_out:
            msg = ", ".join(
                filter(
                    None,
                    [
                        f"missing input cols: {miss_in}" if miss_in else None,
                        f"missing target cols: {miss_out}" if miss_out else None,
                    ],
                )
            )
            problems.append(f"Test file {idx} ({path}): {msg}")
    if problems:
        raise SystemExit("Column validation failed: " + " | ".join(problems))

    print(f"Resolved input columns ({len(input_cols)}): {', '.join(input_cols)}")
    print(f"Resolved target columns ({len(target_cols)}): {', '.join(target_cols)}")
    print(f"Augment flip: {cfg.augment_flip}")
    print(f"Model type: {cfg.model_type}")

    train_df, _ = _load_csv(train_paths)
    train_df = _apply_flip_augmentation(
        train_df,
        augment_flip=cfg.augment_flip,
        augment_profile=cfg.augment_profile,
        augment_config=cfg.augment_config,
    )
    test_df, individual_test_dfs = _load_csv(cfg.test_csv)

    X_all = train_df[input_cols].to_numpy(dtype=np.float32)
    y_all = train_df[target_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(X_all).all() or not np.isfinite(y_all).all():
        raise SystemExit(
            "Training data contains NaN or infinite values. Please clean the dataset before training."
        )
    if cfg.model_type == "lightgbm" and cfg.objective == "gamma":
        if np.any(y_all <= 0):
            raise SystemExit(
                "LightGBM gamma objective requires strictly positive targets; "
                "found non-positive values in the training data."
            )
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
    )

    X_train = np.ascontiguousarray(X_train)
    y_train = np.ascontiguousarray(y_train)
    X_val = np.ascontiguousarray(X_val)
    y_val = np.ascontiguousarray(y_val)

    X_test = test_df[input_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_cols].to_numpy(dtype=np.float32)
    X_test = np.ascontiguousarray(X_test)
    y_test = np.ascontiguousarray(y_test)

    if cfg.model_type == "lightgbm":
        base_params = {
            "objective": cfg.objective,
            "boosting_type": cfg.booster,
            "n_estimators": cfg.n_estimators,
            "learning_rate": cfg.learning_rate,
            "max_depth": cfg.max_depth if cfg.max_depth > 0 else -1,
            "num_leaves": cfg.num_leaves,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "min_child_samples": cfg.min_child_samples,
            "n_jobs": cfg.n_jobs,
            "random_state": cfg.seed,
            "verbose": -1,
        }
        print("Training LightGBM with params:")
        for key, value in base_params.items():
            print(f"  {key}: {value}")
        print(f"  early_stopping_rounds: {cfg.early_stopping_rounds}")
        model = LightGBMMultiTargetRegressor(
            base_params,
            feature_names=input_cols,
            parallel_jobs=cfg.multioutput_n_jobs,
        )
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            Y_val=y_val,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
    else:
        model = MultiOutputRegressor(
            _build_base_estimator(cfg),
            n_jobs=cfg.multioutput_n_jobs,
        )
        print(
            f"Training {cfg.model_type} with default estimator params: {model.estimator}"
        )
        model.fit(X_train, y_train)

    _save_config(run_dir, cfg, input_cols, target_cols)
    joblib.dump(model, run_dir / "model.joblib")

    error_features = cfg.error_feature_names or input_cols[:3]
    _evaluate_and_save(
        model,
        X_train,
        y_train,
        input_cols,
        target_cols,
        "train",
        run_dir,
        error_feature_names=error_features,
    )
    _evaluate_and_save(
        model,
        X_val,
        y_val,
        input_cols,
        target_cols,
        "val",
        run_dir,
        error_feature_names=error_features,
    )
    _evaluate_and_save(
        model,
        X_test,
        y_test,
        input_cols,
        target_cols,
        "test",
        run_dir,
        error_feature_names=error_features,
    )

    for idx, df in enumerate(individual_test_dfs, start=1):
        X_ind = df[input_cols].to_numpy(dtype=np.float32)
        y_ind = df[target_cols].to_numpy(dtype=np.float32)
        _evaluate_and_save(
            model,
            X_ind,
            y_ind,
            input_cols,
            target_cols,
            f"test_file_{idx}",
            run_dir,
            error_feature_names=error_features,
        )

    _aggregate_feature_importances(model, input_cols, run_dir)

    if cfg.permute_importance:
        try:
            _compute_permutation_importance(
                model,
                X_test,
                y_test,
                input_cols,
                n_repeats=cfg.permute_repeats,
                random_state=cfg.seed,
                run_dir=run_dir,
            )
            print("Saved permutation importance results.")
        except Exception as exc:
            print(f"[WARN] Permutation importance failed: {exc}")

    print(f"Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
