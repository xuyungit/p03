"""
model_val.py

A dedicated script for model evaluation using a pre-trained checkpoint.
This script leverages `f_forward.py` to perform inference and generates
a comprehensive evaluation report, similar to the one produced at the
end of a training run.

This allows for repeated evaluations on different datasets without needing
to retrain the model.

Usage:
  uv run python models/model_val.py \
    --csv /path/to/your/test_data.csv \
    --checkpoint /path/to/your/model_checkpoint.pth \
    --out-dir /path/to/your/output_directory \
    --input-cols-re "^R(\\d+)$,^Ry_t\\d+$,^Rx_t\\d+$,^D_t\\d+$" \
    --target-cols-re "^F\\d+$,^N\\d+$"

  # Multiple CSV files (merged for evaluation):
  uv run python models/model_val.py \
    --csv /path/to/test1.csv /path/to/test2.csv /path/to/test3.csv \
    --checkpoint /path/to/your/model_checkpoint.pth \
    --out-dir /path/to/your/output_directory
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

# Ensure local package imports work when running as a script
import sys

from models.evaluation.report import EvaluationReport
import models.f_forward as f_forward
from models.training.predict import mc_dropout_predict


def _configure_matplotlib(run_dir: Path) -> None:
    """Set Matplotlib to a headless, English-safe config and writable cache."""
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


def _resolve_by_patterns(headers: List[str], patterns: Optional[List[str]]) -> Optional[List[str]]:
    """Resolves column names from a list of headers using regex patterns."""
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


def _analyze_and_print_errors(true_df: pd.DataFrame, pred_df: pd.DataFrame):
    """
    Analyzes prediction errors by comparing true and predicted values,
    printing the top 10 largest and smallest errors for each relevant channel.
    """
    # This function is adapted from analyze_errors.py
    # Ensure columns match
    if not all(true_df.columns == pred_df.columns):
        print("Error: Columns in true and prediction files do not match.", file=sys.stderr)
        return

    # Calculate real (signed) and absolute errors
    real_error_df = pred_df - true_df
    abs_error_df = real_error_df.abs()

    print("\n--- Analysis of Errors per Channel ---")
    for col in abs_error_df.columns:
        print(f"\n--- Channel: {col} ---")

        # --- LARGEST ERRORS ---
        top_10_indices = abs_error_df[col].nlargest(10).index
        largest_errors_df = pd.DataFrame({
            'Real Error': real_error_df.loc[top_10_indices, col],
            'True Value': true_df.loc[top_10_indices, col]
        })
        
        print("Top 10 LARGEST errors (by absolute value):")
        print(largest_errors_df.to_string(float_format='{:.4f}'.format))
        print() # Add a blank line for spacing

        # --- SMALLEST ERRORS ---
        bottom_10_indices = abs_error_df[col].nsmallest(10).index
        smallest_errors_df = pd.DataFrame({
            'Real Error': real_error_df.loc[bottom_10_indices, col],
            'True Value': true_df.loc[bottom_10_indices, col]
        })
        
        print("Top 10 SMALLEST errors (closest to zero):")
        print(smallest_errors_df.to_string(float_format='{:.4f}'.format))
        print("-" * (20 + len(col))) # Separator for next channel


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--csv", type=Path, required=True, nargs='+', help="Path(s) to the test CSV file(s). Multiple files will be merged for evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint file (e.g., best_full.pth).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to save evaluation artifacts.")
    parser.add_argument("--input-cols", type=str, default=None, help="Comma-separated input feature names.")
    parser.add_argument("--target-cols", type=str, default=None, help="Comma-separated target names.")
    parser.add_argument("--input-cols-re", type=str, default=None, help="Comma-separated regex for input columns.")
    parser.add_argument("--target-cols-re", type=str, default=None, help="Comma-separated regex for target columns.")
    parser.add_argument("--mc-samples", type=int, default=0, help="If >0, perform MC Dropout prediction with N samples.")
    parser.add_argument("--error-features", type=str, nargs='*', default=None, help="Feature names for error-vs-feature plots.")

    args = parser.parse_args()

    # --- Setup ---
    out_dir = Path.joinpath(args.out_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib(out_dir)
    print(f"Artifacts will be saved to: {out_dir}")

    # --- Load Data ---
    csv_files = args.csv
    dataframes = []
    
    for csv_file in csv_files:
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        df_temp = pd.read_csv(csv_file)
        dataframes.append(df_temp)
        print(f"Loaded data with {len(df_temp)} rows from {csv_file}")
    
    # Merge all dataframes
    df = pd.concat(dataframes, ignore_index=True)
    print(f"Merged data: total {len(df)} rows from {len(csv_files)} files")

    # --- Resolve Columns from Training Config ---
    config_path = args.checkpoint.parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config.json not found at expected path: {config_path}\n"
            f"Cannot determine the exact column order used for training."
        )

    print(f"Loading column configuration from: {config_path}")
    with config_path.open("r") as f:
        train_config = json.load(f)

    input_cols = train_config.get("columns", {}).get("input")
    target_cols = train_config.get("columns", {}).get("target")

    if not input_cols or not target_cols:
        raise ValueError(f"Could not read input/target columns from {config_path}")

    print("NOTE: Using column configuration from checkpoint's config.json. "
          f"Ignoring --input-cols-re and --target-cols-re arguments.")


    # Validate columns exist in the provided CSV
    missing_in = [c for c in input_cols if c not in df.columns]
    missing_tgt = [c for c in target_cols if c not in df.columns]
    if missing_in or missing_tgt:
        raise ValueError(f"Columns from config.json not found in CSV. Missing inputs: {missing_in}, Missing targets: {missing_tgt}")

    print(f"Using {len(input_cols)} input columns from config.json.")
    print(f"Using {len(target_cols)} target columns from config.json.")

    # --- Perform Inference ---
    f_forward.preload(
        input_cols=input_cols,
        target_cols=target_cols,
        checkpoint=args.checkpoint,
    )
    
    y_pred_df = f_forward.f(df)
    y_true_df = df[target_cols]
    X_df = df[input_cols]

    # --- Generate and Save Report ---
    model_name = args.checkpoint.stem.replace('_full', '').replace('_best', '')
    report = EvaluationReport(model_name=f"eval_{model_name}")
    report.evaluate(y_true_df, y_pred_df)

    # Detailed error analysis
    _analyze_and_print_errors(y_true_df, y_pred_df)
    
    # Determine which features to use for error plotting
    error_feature_names = args.error_features if args.error_features is not None else input_cols[:min(3, len(input_cols))]

    report.save_artifacts(
        str(out_dir),
        split_name="evaluation",
        y_true_df=y_true_df,
        y_pred_df=y_pred_df,
        X_df=X_df,
        error_feature_names=error_feature_names,
    )
    
    print("\nEvaluation metrics:")
    print(json.dumps(report.metrics, indent=2))

    # --- Optional: MC Dropout ---
    if args.mc_samples > 0:
        print(f"\nPerforming MC Dropout with {args.mc_samples} samples...")
        state = f_forward.get_state()
        model = state["model"]
        scaler_x = state["scaler_x"]
        scaler_y = state["scaler_y"]
        
        X_np = X_df.to_numpy(dtype=np.float32)
        X_s = scaler_x.transform(X_np)
        
        with torch.no_grad():
            X_t = torch.tensor(X_s, dtype=torch.float32, device=state["device"])
            mean_s, std_s = mc_dropout_predict(model, X_t, n_samples=args.mc_samples)
            
            mean_pred = scaler_y.inverse_transform(mean_s.cpu().numpy())
            # For StandardScaler: y = y_s * scale + mean => std_y = std_y_s * scale
            std_pred = std_s.cpu().numpy() * scaler_y.scale_.reshape(1, -1)

        pd.DataFrame(mean_pred, columns=target_cols).to_csv(out_dir / "pred_mc_mean.csv", index=False)
        pd.DataFrame(std_pred, columns=target_cols).to_csv(out_dir / "pred_mc_std.csv", index=False)
        print("Saved MC Dropout mean and std predictions.")

    # --- Save Config ---
    run_config = {
        "csv_paths": [str(csv_file) for csv_file in csv_files],
        "checkpoint_path": str(args.checkpoint),
        "out_dir": str(out_dir),
        "mc_samples": args.mc_samples,
        "columns": {
            "input": input_cols,
            "target": target_cols,
            "error_features": error_feature_names,
        }
    }
    with (out_dir / "eval_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nEvaluation complete. All artifacts saved in {out_dir}")


if __name__ == "__main__":
    main()
