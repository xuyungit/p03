"""Integration smoke tests for the SVGP training and inference CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
    assert pred_path.exists(), "Inference did not emit predictions"
    assert std_path.exists(), "Inference did not emit std predictions"

    std_with_tau = pd.read_csv(std_path).to_numpy(dtype=np.float64)

    infer_cmd_no_tau = infer_cmd + ["--no-tau"]
    _run_subprocess(infer_cmd_no_tau, cwd=REPO_ROOT)

    std_without_tau = pd.read_csv(std_path).to_numpy(dtype=np.float64)

    assert np.allclose(std_with_tau, std_without_tau * tau_value, rtol=1e-5, atol=1e-8)
