"""Integration smoke tests for the SVGP training and inference CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"


def _write_dataset(path: Path, num_rows: int, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "F1": rng.normal(size=num_rows),
            "F2": rng.normal(size=num_rows),
            "N1_r": rng.normal(size=num_rows),
            "T_grad": rng.normal(size=num_rows),
            "R1": rng.normal(size=num_rows),
        }
    )
    frame.to_csv(path, index=False)


def _run_subprocess(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_svgp_training_and_infer_smoke(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_dataset(train_csv, num_rows=16, seed=0)
    _write_dataset(test_csv, num_rows=8, seed=1)

    experiment_root = tmp_path / "runs"
    experiment_root.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "-m",
        "models.gp.svgp_baseline",
        "--train-csv",
        str(train_csv),
        "--test-csv",
        str(test_csv),
        "--input-cols",
        "F1,F2,N1_r,T_grad",
        "--target-cols",
        "R1",
        "--experiment-root",
        str(experiment_root),
        "--val-ratio",
        "0.25",
        "--epochs",
        "2",
        "--batch-size",
        "8",
        "--inducing",
        "4",
        "--prewarm-epochs",
        "0",
        "--lr",
        "5e-2",
        "--no-augment-flip",
        "--print-freq",
        "0",
    ]

    _run_subprocess(train_cmd, cwd=REPO_ROOT)

    run_dirs = sorted(experiment_root.iterdir())
    assert len(run_dirs) == 1, "Expected exactly one SVGP run directory"
    run_dir = run_dirs[0]

    checkpoints_dir = run_dir / "checkpoints"
    assert checkpoints_dir.exists(), "SVGP run did not create checkpoints directory"
    assert any(checkpoints_dir.glob("component_*.pt")), "Missing component checkpoints"

    tau_path = run_dir / "tau.json"
    assert tau_path.exists(), "Expected tau calibration artifact"
    with tau_path.open("r") as handle:
        tau_payload = json.load(handle)
    tau_value = float(tau_payload["value"])

    infer_cmd = [
        sys.executable,
        "-m",
        "models.gp.infer",
        "--run-dir",
        str(run_dir),
        "--input-csv",
        str(test_csv),
    ]
    _run_subprocess(infer_cmd, cwd=REPO_ROOT)

    infer_dir = run_dir / "infer"
    pred_path = infer_dir / f"{test_csv.stem}_pred.csv"
    std_path = infer_dir / f"{test_csv.stem}_pred_std.csv"
    raw_std_path = infer_dir / f"{test_csv.stem}_pred_std_raw.csv"
    quantile_path = infer_dir / f"{test_csv.stem}_pred_quantiles.csv"
    assert pred_path.exists(), "Inference did not emit predictions"
    assert std_path.exists(), "Inference did not emit std predictions"
    assert raw_std_path.exists(), "Inference did not emit raw std predictions"
    assert quantile_path.exists(), "Inference did not emit quantile intervals"

    pred_df = pd.read_csv(pred_path)
    std_with_tau = pd.read_csv(std_path).to_numpy(dtype=np.float64)
    std_raw = pd.read_csv(raw_std_path).to_numpy(dtype=np.float64)
    quantile_df = pd.read_csv(quantile_path)

    quantile_tau_path = run_dir / "quantile_tau.json"
    assert quantile_tau_path.exists(), "Expected quantile_tau calibration artifact"
    with quantile_tau_path.open("r") as handle:
        quantile_payload = json.load(handle)
    raw_quantile_values = quantile_payload.get("values", {})
    quantile_tau = {float(key): float(value) for key, value in raw_quantile_values.items()}
    assert quantile_tau, "Quantile tau map should not be empty"

    mean_values = pred_df.to_numpy(dtype=np.float64)

    scaling_factor = tau_value
    per_group_tau_path = run_dir / "per_group_tau.json"
    if per_group_tau_path.exists():
        with per_group_tau_path.open("r") as handle:
            per_group_payload = json.load(handle)
        group_values = {
            str(key): float(value)
            for key, value in per_group_payload.get("values", {}).items()
            if isinstance(value, (int, float))
        }
        group_mapping = per_group_payload.get("groups", {})
        if isinstance(group_mapping, dict):
            for group_name, columns in group_mapping.items():
                if isinstance(columns, list) and "R1" in columns:
                    scaling_factor = group_values.get(group_name, scaling_factor)
                    break

    def _quantile_suffix(level: float) -> str:
        formatted = f"{level:.6f}".rstrip("0").rstrip(".")
        if not formatted:
            formatted = f"{level:.6f}"
        return f"q{formatted.replace('.', '_')}"

    def _normal_quantile(prob: float) -> float:
        distribution = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
        )
        tensor = torch.tensor(prob, dtype=torch.float64)
        return float(distribution.icdf(tensor).item())

    for level, tau_value in quantile_tau.items():
        suffix = _quantile_suffix(level)
        lower_col = f"R1_lower_{suffix}"
        upper_col = f"R1_upper_{suffix}"
        assert lower_col in quantile_df.columns
        assert upper_col in quantile_df.columns
        lower = quantile_df[lower_col].to_numpy(dtype=np.float64)
        upper = quantile_df[upper_col].to_numpy(dtype=np.float64)
        z = _normal_quantile(0.5 + level / 2.0)
        margin = z * std_raw[:, 0] * tau_value
        assert np.allclose(lower, mean_values[:, 0] - margin, rtol=1e-5, atol=1e-8)
        assert np.allclose(upper, mean_values[:, 0] + margin, rtol=1e-5, atol=1e-8)

    infer_cmd_no_tau = infer_cmd + ["--no-tau"]
    _run_subprocess(infer_cmd_no_tau, cwd=REPO_ROOT)

    std_without_tau = pd.read_csv(std_path).to_numpy(dtype=np.float64)
    std_raw_after = pd.read_csv(raw_std_path).to_numpy(dtype=np.float64)
    assert np.allclose(std_raw, std_raw_after, rtol=1e-6, atol=1e-9)
    assert np.allclose(std_without_tau, std_raw, rtol=1e-6, atol=1e-9)

    assert np.allclose(std_with_tau, std_without_tau * scaling_factor, rtol=1e-5, atol=1e-8)
