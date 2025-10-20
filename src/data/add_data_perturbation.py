"""
add_data_perturbation.py

A utility script to apply various perturbations to a dataset, such as rounding
or adding noise. This is useful for testing model sensitivity and robustness.

The script reads an input CSV, applies the specified transformations to a
selection of columns, and saves the result to a new CSV file.

Usage (Rounding):
  uv run python add_data_perturbation.py \
    --input-csv /path/to/your/data.csv \
    --output-csv /path/to/your/data_rounded.csv \
    --round-digits 3 \
    --columns-re "^R(\\d+$" "^D_t\\d+$"

Usage (Proportional Noise):
  uv run python add_data_perturbation.py \
    --input-csv /path/to/your/data.csv \
    --output-csv /path/to/your/data_noisy.csv \
    --add-proportional-noise 0.05 \
    --columns-re "^R(\\d+$" \
    --noise-seed 42

Usage (Channel Bias):
  uv run python add_data_perturbation.py \
    --input-csv /path/to/your/data.csv \
    --output-csv /path/to/your/data_biased.csv \
    --columns-re "^channel_\\d+$" \
    --add-bias-range -0.5 0.5 \
    --bias-output-csv /path/to/bias.csv

Notes:
- By default, non-target columns are preserved byte-for-byte where possible
  (textual pass-through). This avoids unintended diffs on unrelated columns
  from CSV re-serialization (e.g., float formatting or BOM removal).
  You can disable this behavior with --no-preserve-untouched-text.

"""
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Optional


def _file_has_bom(path: Path) -> bool:
    try:
        with path.open('rb') as f:
            head = f.read(3)
        return head == b"\xef\xbb\xbf"
    except Exception:
        return False


def _format_float(val: float, digits: Optional[int], fmt_spec: Optional[str]) -> str:
    """Format a float value with either fixed digits or a format spec.

    - If `digits` is provided, fixed-point with that many decimals is used.
    - Else if `fmt_spec` is provided, Python format spec is used (e.g., '.6g', '.8f').
    - Else defaults to general format with reasonable precision via '.15g'.
    """
    if digits is not None:
        return f"{val:.{digits}f}"
    if fmt_spec:
        return format(val, fmt_spec)
    return format(val, ".15g")

def resolve_columns_by_regex(headers: List[str], patterns: List[str]) -> List[str]:
    """Resolves column names from a list of headers using regex patterns."""
    if not patterns:
        return []
    
    compiled_patterns = [re.compile(p) for p in patterns]
    
    matched_columns = []
    for col in headers:
        if any(p.search(col) for p in compiled_patterns):
            matched_columns.append(col)
            
    # Deduplicate while preserving order
    seen = set()
    unique_columns = [x for x in matched_columns if not (x in seen or seen.add(x))]
    return unique_columns

def main():
    parser = argparse.ArgumentParser(
        description="Apply perturbations like rounding or noise to a dataset."
    )
    parser.add_argument(
        "--input-csv", type=Path, required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output-csv", type=Path, required=True, help="Path to save the perturbed CSV file."
    )
    parser.add_argument(
        "--columns-re",
        type=str,
        nargs='+',
        required=True,
        help="One or more regex patterns to select columns for perturbation.",
    )
    
    # Perturbation options
    parser.add_argument(
        "--round-digits",
        type=int,
        help="Number of decimal places to round the selected columns to.",
    )
    parser.add_argument(
        "--add-proportional-noise",
        type=float,
        help="Add Gaussian noise proportional to each value. The value provided is the standard deviation of the noise relative to the value (e.g., 0.05 for 5%%)."
    )
    parser.add_argument(
        "--apply-angle-sensor-noise",
        action="store_true",
        help=(
            "Apply a Gaussian sensor noise model (σ=0.005° by default) and quantize to a 0.001° step. "
            "Intended for angle channels already expressed in radians."
        ),
    )
    parser.add_argument(
        "--sensor-sigma-deg",
        type=float,
        default=0.005,
        help="Override the sensor noise standard deviation in degrees. Only used with --apply-angle-sensor-noise.",
    )
    parser.add_argument(
        "--sensor-resolution-deg",
        type=float,
        default=0.001,
        help="Override the sensor quantization resolution in degrees. Only used with --apply-angle-sensor-noise.",
    )
    parser.add_argument(
        "--add-bias-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Add a fixed bias to each selected column. Each column receives a different uniformly sampled value within the provided inclusive range.",
    )
    parser.add_argument(
        "--bias-output-csv",
        type=Path,
        help="Optional path to write the per-column bias values when --add-bias-range is used. Defaults to bias.csv next to the output CSV.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        help="Optional random seed for noise generation to ensure reproducibility."
    )
    parser.add_argument(
        "--preserve-untouched-text",
        action="store_true",
        default=True,
        help=(
            "When set (default), non-selected columns are passed through as original text without re-formatting. "
            "Disabling this may reformat floats in unrelated columns due to CSV round-tripping."
        ),
    )
    parser.add_argument(
        "--no-preserve-untouched-text",
        dest="preserve_untouched_text",
        action="store_false",
        help="Disable text pass-through; entire CSV is re-serialized by pandas.",
    )
    parser.add_argument(
        "--float-format",
        type=str,
        default=None,
        help=(
            "Optional Python format spec (e.g., '.6g', '.8f') for formatting perturbed floats. "
            "Only applied to selected columns. Ignored when --round-digits is set (fixed decimals)."
        ),
    )

    args = parser.parse_args()

    sensor_sigma_rad = None
    sensor_resolution_rad = None
    if args.apply_angle_sensor_noise:
        if args.sensor_sigma_deg < 0:
            raise ValueError("--sensor-sigma-deg must be non-negative when applying angle sensor noise.")
        if args.sensor_resolution_deg <= 0:
            raise ValueError("--sensor-resolution-deg must be positive when applying angle sensor noise.")
        sensor_sigma_rad = float(np.deg2rad(args.sensor_sigma_deg))
        sensor_resolution_rad = float(np.deg2rad(args.sensor_resolution_deg))

    # --- Setup ---
    if args.noise_seed is not None:
        print(f"Using random seed for noise generation: {args.noise_seed}")
        np.random.seed(args.noise_seed)

    # --- Load Data ---
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    
    # Preserve BOM if present
    has_bom = _file_has_bom(args.input_csv)
    read_encoding = "utf-8-sig" if has_bom else None

    df = pd.read_csv(args.input_csv, encoding=read_encoding)
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    # --- Identify Columns ---
    columns_to_perturb = resolve_columns_by_regex(df.columns.tolist(), args.columns_re)
    
    if not columns_to_perturb:
        print("Warning: No columns matched the given regex patterns. Output will be identical to input.")
    else:
        print(f"Found {len(columns_to_perturb)} columns to perturb: {columns_to_perturb}")

    # --- Apply Perturbations ---
    modified = False

    if args.preserve_untouched_text:
        # Text-preserving path: load as text, only reformat selected columns.
        raw_df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False, encoding=read_encoding)

        if args.round_digits is not None:
            print(f"Applying rounding to {args.round_digits} decimal places (text-preserving mode)...")
            for col in columns_to_perturb:
                vals = pd.to_numeric(raw_df[col], errors='coerce')
                vals = np.round(vals, args.round_digits)
                # format back to strings
                formatted = [raw if np.isnan(v) else _format_float(float(v), args.round_digits, args.float_format) for raw, v in zip(raw_df[col].tolist(), vals.tolist())]
                raw_df[col] = formatted
            modified = True
            print("Rounding complete.")

        if args.add_proportional_noise is not None:
            print(f"Applying proportional noise with a standard deviation of {args.add_proportional_noise:.2%} (text-preserving mode)")
            for col in columns_to_perturb:
                base = pd.to_numeric(raw_df[col], errors='coerce')
                noise = base * args.add_proportional_noise * np.random.normal(0, 1, size=base.shape)
                new_vals = base + noise
                formatted = [raw if np.isnan(v) else _format_float(float(v), args.round_digits, args.float_format) for raw, v in zip(raw_df[col].tolist(), new_vals.tolist())]
                raw_df[col] = formatted
            modified = True
            print("Proportional noise added.")

        if args.add_bias_range is not None:
            bias_min, bias_max = args.add_bias_range
            if bias_min > bias_max:
                raise ValueError("--add-bias-range expects MIN to be less than or equal to MAX.")

            print(f"Applying per-column bias sampled uniformly from [{bias_min}, {bias_max}] (text-preserving mode)")
            bias_records = []
            for col in columns_to_perturb:
                bias_value = float(np.random.uniform(bias_min, bias_max))
                base = pd.to_numeric(raw_df[col], errors='coerce')
                new_vals = base + bias_value
                formatted = [raw if np.isnan(v) else _format_float(float(v), args.round_digits, args.float_format) for raw, v in zip(raw_df[col].tolist(), new_vals.tolist())]
                raw_df[col] = formatted
                bias_records.append({"column": col, "bias": bias_value})

            if bias_records:
                bias_output_path = args.bias_output_csv or args.output_csv.parent / "bias.csv"
                bias_output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(bias_records).to_csv(bias_output_path, index=False)
                print(f"Recorded channel biases in {bias_output_path}")
            else:
                print("No columns matched for bias application; no bias CSV created.")
            if bias_records:
                modified = True

        if args.apply_angle_sensor_noise:
            print(
                "Applying angle sensor noise with σ="
                f"{args.sensor_sigma_deg:.6f}° and resolution={args.sensor_resolution_deg:.6f}° "
                "(text-preserving mode)"
            )
            for col in columns_to_perturb:
                base = pd.to_numeric(raw_df[col], errors='coerce')
                noisy_vals = base + np.random.normal(0.0, sensor_sigma_rad, size=base.shape)
                quantized_vals = np.round(noisy_vals / sensor_resolution_rad) * sensor_resolution_rad
                formatted = [
                    raw if np.isnan(v) else _format_float(float(v), args.round_digits, args.float_format)
                    for raw, v in zip(raw_df[col].tolist(), quantized_vals.tolist())
                ]
                raw_df[col] = formatted
            modified = True
            print("Angle sensor noise applied.")

        # Save using same encoding signature as input (preserve BOM if present)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_encoding = "utf-8-sig" if has_bom else None
        raw_df.to_csv(args.output_csv, index=False, encoding=write_encoding)

    else:
        # Original pandas path: operate numerically, re-serialize entire CSV
        if args.round_digits is not None:
            print(f"Applying rounding to {args.round_digits} decimal places...")
            df[columns_to_perturb] = df[columns_to_perturb].round(args.round_digits)
            modified = True
            print("Rounding complete.")

        if args.add_proportional_noise is not None:
            print(f"Applying proportional noise with a standard deviation of {args.add_proportional_noise:.2%}")
            for col in columns_to_perturb:
                noise = df[col] * args.add_proportional_noise * np.random.normal(0, 1, size=df[col].shape)
                df[col] = df[col] + noise
            modified = True
            print("Proportional noise added.")

        if args.add_bias_range is not None:
            bias_min, bias_max = args.add_bias_range
            if bias_min > bias_max:
                raise ValueError("--add-bias-range expects MIN to be less than or equal to MAX.")

            print(f"Applying per-column bias sampled uniformly from [{bias_min}, {bias_max}]")
            bias_records = []
            for col in columns_to_perturb:
                bias_value = np.random.uniform(bias_min, bias_max)
                df[col] = df[col] + bias_value
                bias_records.append({"column": col, "bias": bias_value})

            if bias_records:
                bias_output_path = args.bias_output_csv or args.output_csv.parent / "bias.csv"
                bias_output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(bias_records).to_csv(bias_output_path, index=False)
                print(f"Recorded channel biases in {bias_output_path}")
            else:
                print("No columns matched for bias application; no bias CSV created.")
            if bias_records:
                modified = True

        if args.apply_angle_sensor_noise:
            print(
                "Applying angle sensor noise with σ="
                f"{args.sensor_sigma_deg:.6f}° and resolution={args.sensor_resolution_deg:.6f}°"
            )
            for col in columns_to_perturb:
                noise = np.random.normal(0.0, sensor_sigma_rad, size=df[col].shape)
                noisy_vals = df[col] + noise
                df[col] = np.round(noisy_vals / sensor_resolution_rad) * sensor_resolution_rad
            modified = True
            print("Angle sensor noise applied.")

        if not modified:
            print("No perturbation options were specified. Output file will be a copy of the input.")

        # Save using same encoding signature as input (preserve BOM if present)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_encoding = "utf-8-sig" if has_bom else None
        df.to_csv(args.output_csv, index=False, encoding=write_encoding)
    print(f"Successfully saved perturbed data to {args.output_csv}")


if __name__ == "__main__":
    main()
