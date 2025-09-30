"""CLI utility for serving predictions from SVGP experiment checkpoints.

The script mirrors the training CLI's preprocessing pipeline so that
downstream consumers can load a completed run directory and obtain
mean/standard-deviation predictions for arbitrary CSV inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import gpytorch
import numpy as np
import pandas as pd
import torch
from joblib import load

from models.gp.common import PCAPipeline
from models.gp.svgp_baseline import ComponentState, SingleTaskSVGP, _predict_components


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using an SVGP experiment directory")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to an svgp_baseline run directory")
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True, help="One or more CSV files to score")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for prediction outputs (defaults to run_dir/infer)")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for inference batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device for prediction (e.g. cpu, cuda:0)")
    parser.add_argument("--save-std", dest="save_std", action="store_true", help="Write per-target std CSV alongside predictions (default)")
    parser.add_argument("--no-save-std", dest="save_std", action="store_false", help="Skip writing std CSVs")
    parser.add_argument("--tau", type=float, default=None, help="Override tau scaling factor (multiplies std); defaults to tau.json when available")
    parser.add_argument("--no-tau", action="store_true", help="Disable tau scaling even if tau.json is present")
    parser.set_defaults(save_std=True)
    return parser.parse_args()


def _load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.json under {run_dir}")
    with config_path.open("r") as handle:
        return json.load(handle)


def _load_preprocessors(run_dir: Path) -> Tuple[object, object, Optional[PCAPipeline]]:
    checkpoints = run_dir / "checkpoints"
    scaler_x_path = checkpoints / "scaler_x.pkl"
    scaler_y_path = checkpoints / "scaler_y.pkl"
    if not scaler_x_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError("Expected scaler_x.pkl and scaler_y.pkl under checkpoints/")
    scaler_x = load(scaler_x_path)
    scaler_y = load(scaler_y_path)
    pca_path = checkpoints / "pca.pkl"
    pca_pipe: Optional[PCAPipeline]
    if pca_path.exists():
        pca_pipe = PCAPipeline.load(pca_path)
    else:
        pca_pipe = None
    return scaler_x, scaler_y, pca_pipe


_COMPONENT_PATTERN = re.compile(r"component_(\d+)\.pt$")


def _load_components(
    run_dir: Path,
    *,
    kernel: str,
    ard: bool,
    jitter: float,
) -> List[ComponentState]:
    def _sort_key(path: Path) -> Tuple[int, str]:
        match = _COMPONENT_PATTERN.match(path.name)
        if match:
            return int(match.group(1)), path.name
        return 10**9, path.name

    checkpoints = sorted((run_dir / "checkpoints").glob("component_*.pt"), key=_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No component_*.pt files found under {run_dir / 'checkpoints'}")

    components: List[ComponentState] = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu")
        model_state = payload.get("model_state")
        likelihood_state = payload.get("likelihood_state")
        if model_state is None or likelihood_state is None:
            raise RuntimeError(f"Checkpoint {path} missing model_state or likelihood_state")

        inducing = model_state.get("variational_strategy.inducing_points")
        if inducing is None:
            raise RuntimeError(f"Checkpoint {path} missing inducing point tensor")

        component_kernel = payload.get("kernel", kernel)
        component_ard = payload.get("ard", ard)

        inducing_tensor = inducing.detach().clone()
        model = SingleTaskSVGP(inducing_tensor, kernel=component_kernel, ard=component_ard, jitter=jitter)
        model.load_state_dict(model_state)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.load_state_dict(likelihood_state)
        components.append(ComponentState(model=model, likelihood=likelihood))

    return components


def _resolve_tau(run_dir: Path, *, tau_arg: Optional[float], disable_tau: bool) -> Optional[float]:
    if tau_arg is not None:
        return float(tau_arg)
    if disable_tau:
        return None
    tau_path = run_dir / "tau.json"
    if not tau_path.exists():
        return None
    with tau_path.open("r") as handle:
        payload = json.load(handle)
    value = payload.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid tau value stored in {tau_path}: {value}") from exc


def _prepare_inputs(df: pd.DataFrame, input_cols: Sequence[str]) -> np.ndarray:
    missing = [col for col in input_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe missing required columns: {missing}")
    data = df[input_cols].to_numpy(dtype=np.float32, copy=True)
    return data


def _decode_predictions(
    mean_latent: np.ndarray,
    var_latent: np.ndarray,
    *,
    scaler_y,
    pca: Optional[PCAPipeline],
    tau: Optional[float],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if pca is not None:
        mean_std = pca.inverse_transform(mean_latent)
        var_orig = pca.project_variance(var_latent, scale=scaler_y.scale_)
    else:
        mean_std = mean_latent
        scale_sq = np.square(scaler_y.scale_)
        if mean_latent.ndim == 2:
            var_orig = var_latent * scale_sq[None, :]
        else:
            var_orig = var_latent * scale_sq

    mean_orig = scaler_y.inverse_transform(mean_std)

    if var_orig is None:
        return mean_orig, None

    var_orig = np.maximum(var_orig, 0.0)
    std_orig = np.sqrt(var_orig)
    if tau is not None:
        std_orig = std_orig * float(tau)
    return mean_orig, std_orig


def _write_outputs(
    output_dir: Path,
    stem: str,
    target_cols: Sequence[str],
    mean: np.ndarray,
    std: Optional[np.ndarray],
    *,
    save_std: bool,
) -> Tuple[Path, Optional[Path]]:
    mean_df = pd.DataFrame(mean, columns=target_cols)
    mean_path = output_dir / f"{stem}_pred.csv"
    mean_df.to_csv(mean_path, index=False)

    std_path: Optional[Path] = None
    if save_std and std is not None:
        std_df = pd.DataFrame(std, columns=target_cols)
        std_path = output_dir / f"{stem}_pred_std.csv"
        std_df.to_csv(std_path, index=False)
    return mean_path, std_path


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")

    config = _load_config(run_dir)
    input_cols: List[str] = list(config.get("resolved_input_cols", []))
    target_cols: List[str] = list(config.get("resolved_target_cols", []))
    if not input_cols or not target_cols:
        raise RuntimeError("config.json must contain resolved_input_cols and resolved_target_cols")

    batch_size = args.batch_size or int(config.get("batch_size", 256))
    device = torch.device(args.device)

    kernel = str(config.get("kernel", "rbf"))
    ard = bool(config.get("ard", True))
    jitter = float(config.get("jitter", 1e-6))

    scaler_x, scaler_y, pca_pipe = _load_preprocessors(run_dir)
    components = _load_components(run_dir, kernel=kernel, ard=ard, jitter=jitter)
    tau_value = _resolve_tau(run_dir, tau_arg=args.tau, disable_tau=args.no_tau)

    output_dir = args.output_dir or (run_dir / "infer")
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in args.input_csv:
        df = pd.read_csv(csv_path)
        raw_inputs = _prepare_inputs(df, input_cols)
        inputs_std = scaler_x.transform(raw_inputs).astype(np.float32, copy=False)

        mean_latent, var_latent = _predict_components(components, inputs_std, batch_size=batch_size, device=device)
        mean_orig, std_orig = _decode_predictions(mean_latent, var_latent, scaler_y=scaler_y, pca=pca_pipe, tau=tau_value)

        stem = csv_path.stem
        mean_path, std_path = _write_outputs(
            output_dir,
            stem,
            target_cols,
            mean_orig,
            std_orig,
            save_std=args.save_std,
        )

        message = f"Wrote predictions to {mean_path}"
        if args.save_std and std_path is not None:
            message += f"; std to {std_path}"
        if tau_value is not None:
            message += f" (tau={tau_value:.4f})"
        print(message)


if __name__ == "__main__":
    main()
