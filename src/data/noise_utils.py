"""
Noise injection utilities for tabular datasets (post `pd.read_csv`).

Design goals:
- Apply realistic measurement noise to selected columns (e.g., reactions, deflections, rotations).
- Keep everything configurable, deterministic (seeded), and non-destructive (returns a copy).
- Offer simple presets and low-level building blocks.

Typical usage:
    import pandas as pd
    from data.noise_utils import add_noise, gaussian_snr, preset_bridge_measurements

    df = pd.read_csv("data/rfnd02_test.csv")
    cfg = preset_bridge_measurements(snr_db_reaction=30, sigma_deflection=1e-4)
    df_noisy = add_noise(df, cfg, seed=123)

Outputs can be saved by the caller (e.g., df_noisy.to_csv("data/noisy.csv", index=False)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------
# Core utilities
# ---------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _ensure_list(cols: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _select_columns(df: pd.DataFrame, patterns: Sequence[str]) -> List[str]:
    """Select columns by simple prefix patterns.

    Patterns are matched as:
      - exact name if pattern exactly equals a column name
      - prefix match if pattern ends with '*' (e.g., 'R*', 'D_t*', 'Ry_t*')

    Returns the deduplicated list in DataFrame column order.
    """
    cols: List[str] = []
    pat_list = _ensure_list(patterns)
    names = list(map(str, df.columns))
    name_set = set(names)
    for pat in pat_list:
        if pat.endswith("*"):
            pref = pat[:-1]
            for n in names:
                if n.startswith(pref) and n not in cols:
                    cols.append(n)
        else:
            if pat in name_set and pat not in cols:
                cols.append(pat)
    return cols


# ---------------------------
# Noise primitives
# ---------------------------

def gaussian_sigma(
    values: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Additive zero-mean Gaussian noise with absolute std `sigma`.

    Suitable for sensors with approximately constant absolute resolution/noise.
    """
    return values + rng.normal(loc=0.0, scale=sigma, size=values.shape)


def gaussian_relative(
    values: np.ndarray,
    rel_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian noise where std scales with |value|: std = rel_std * |x|.

    Useful when measurement error grows proportionally with magnitude.
    """
    scale = np.abs(values) * float(rel_std)
    return values + rng.normal(0.0, scale, size=values.shape)


def gaussian_snr(
    values: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise based on target SNR in dB.

    Uses amplitude definition: SNR_dB = 20*log10(rms(signal)/rms(noise)).
    """
    sig_rms = _rms(values)
    if sig_rms == 0.0:
        return values.copy()
    noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    return values + rng.normal(0.0, noise_rms, size=values.shape)


def laplace_sigma(values: np.ndarray, b: float, rng: np.random.Generator) -> np.ndarray:
    """Add Laplace noise with scale `b` (std = sqrt(2)*b). Robust alternative to Gaussian."""
    return values + rng.laplace(loc=0.0, scale=b, size=values.shape)


def uniform_range(values: np.ndarray, half_range: float, rng: np.random.Generator) -> np.ndarray:
    """Add uniform noise in [-half_range, +half_range]."""
    return values + rng.uniform(low=-half_range, high=half_range, size=values.shape)


def add_bias(values: np.ndarray, bias: float) -> np.ndarray:
    return values + float(bias)


def add_linear_drift(values: np.ndarray, drift_per_sample: float) -> np.ndarray:
    """Add a slow linear drift across the sample index (row-wise)."""
    n = values.shape[0]
    drift = np.linspace(0.0, drift_per_sample * max(n - 1, 0), num=n)
    return values + drift


def quantize(values: np.ndarray, step: float) -> np.ndarray:
    """Quantize by rounding to nearest multiple of `step`."""
    if step <= 0:
        return values
    return np.round(values / step) * step


def dropout(values: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly set entries to NaN with probability `p`. Caller may impute later."""
    if not (0.0 < p <= 1.0):
        return values
    mask = rng.random(values.shape) < p
    out = values.copy()
    out[mask] = np.nan
    return out


def spikes(values: np.ndarray, p: float, spike_std: float, rng: np.random.Generator) -> np.ndarray:
    """Inject occasional spikes: with prob `p`, add N(0, spike_std) impulse."""
    if p <= 0.0:
        return values
    mask = rng.random(values.shape) < p
    impulses = rng.normal(0.0, spike_std, size=values.shape)
    return values + impulses * mask


def pink_noise(values: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Approximate 1/f (pink) noise then scale to target SNR.

    Voss-McCartney algorithm for colored noise; adequate for quasi-static signals.
    """
    n = values.shape[0]
    if n <= 1:
        return values.copy()
    # Number of random sources ~ log2(n)
    n_sources = max(1, int(np.ceil(np.log2(n))))
    rows = rng.standard_normal((n_sources, n))
    out = np.zeros(n)
    # Cumulative sums at octaves
    step = 1
    for i in range(n_sources):
        # Repeat blocks of size `step` for averaging
        reps = int(np.ceil(n / step))
        row = np.repeat(np.add.reduceat(rows[i, : reps * step], np.arange(0, reps * step, step)) / step, step)[:n]
        out += row
        step *= 2
    out = out / max(1, n_sources)
    # Scale to target SNR
    sig_rms = _rms(values)
    if sig_rms == 0.0:
        return values.copy()
    noise_rms = _rms(out)
    if noise_rms == 0.0:
        return values.copy()
    scale = (sig_rms / (10.0 ** (snr_db / 20.0))) / noise_rms
    return values + out * scale


# ---------------------------
# High-level application to DataFrame
# ---------------------------

NoiseFunc = Callable[[np.ndarray, np.random.Generator], np.ndarray]


@dataclass
class ColumnNoise:
    columns: Sequence[str]
    fn: NoiseFunc


def make_gaussian_sigma(columns: Sequence[str], sigma: float) -> ColumnNoise:
    def _fn(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
        return gaussian_sigma(x, sigma=sigma, rng=r)

    return ColumnNoise(columns=columns, fn=_fn)


def make_gaussian_relative(columns: Sequence[str], rel_std: float) -> ColumnNoise:
    def _fn(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
        return gaussian_relative(x, rel_std=rel_std, rng=r)

    return ColumnNoise(columns=columns, fn=_fn)


def make_gaussian_snr(columns: Sequence[str], snr_db: float) -> ColumnNoise:
    def _fn(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
        return gaussian_snr(x, snr_db=snr_db, rng=r)

    return ColumnNoise(columns=columns, fn=_fn)


def make_laplace(columns: Sequence[str], b: float) -> ColumnNoise:
    def _fn(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
        return laplace_sigma(x, b=b, rng=r)

    return ColumnNoise(columns=columns, fn=_fn)


def make_uniform(columns: Sequence[str], half_range: float) -> ColumnNoise:
    def _fn(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
        return uniform_range(x, half_range=half_range, rng=r)

    return ColumnNoise(columns=columns, fn=_fn)


def make_bias(columns: Sequence[str], bias: float) -> ColumnNoise:
    def _fn(x: np.ndarray, _r: np.random.Generator) -> np.ndarray:
        return add_bias(x, bias=bias)

    return ColumnNoise(columns=columns, fn=_fn)


def make_drift(columns: Sequence[str], drift_per_sample: float) -> ColumnNoise:
    def _fn(x: np.ndarray, _r: np.random.Generator) -> np.ndarray:
        return add_linear_drift(x, drift_per_sample=drift_per_sample)

    return ColumnNoise(columns=columns, fn=_fn)


def make_quantize(columns: Sequence[str], step: float) -> ColumnNoise:
    def _fn(x: np.ndarray, _r: np.random.Generator) -> np.ndarray:
        return quantize(x, step=step)

    return ColumnNoise(columns=columns, fn=_fn)


def make_dropout(columns: Sequence[str], p: float) -> ColumnNoise:
    def _fn(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
        return dropout(x, p=p, rng=r)

    return ColumnNoise(columns=columns, fn=_fn)


def make_spikes(columns: Sequence[str], p: float, spike_std: float) -> ColumnNoise:
    def _fn(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
        return spikes(x, p=p, spike_std=spike_std, rng=r)

    return ColumnNoise(columns=columns, fn=_fn)


def make_pink(columns: Sequence[str], snr_db: float) -> ColumnNoise:
    def _fn(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
        return pink_noise(x, snr_db=snr_db, rng=r)

    return ColumnNoise(columns=columns, fn=_fn)


def add_noise(
    df: pd.DataFrame,
    schemes: Sequence[ColumnNoise],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Return a copy of `df` with configured noise applied to column subsets.

    Each ColumnNoise can target overlapping columns; operations are applied in order.
    """
    out = df.copy()
    rng = _rng(seed)
    for spec in schemes:
        cols = _ensure_list(spec.columns)
        for c in cols:
            if c not in out.columns:
                continue
            vals = out[c].to_numpy(dtype=float)
            vals = spec.fn(vals, rng)
            out[c] = vals
    return out


# ---------------------------
# Presets tailored for this repo's datasets
# ---------------------------

def infer_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Infer common measurement groups based on column name prefixes.

    Groups:
      - reaction: R1,R2,... or columns starting with 'R'
      - force: F1,F2,...
      - deflection: D_t*, D_st*, w_*, u_* (span samples), w_s*, u_s*
      - rotation: Ry_t*, phi_s*
      - nodes: N* (node indices/features) — left untouched by default
    """
    cols = list(map(str, df.columns))
    groups = {
        "reaction": [c for c in cols if c.startswith("R") and (len(c) == 2 or c[1].isdigit())],
        "force": [c for c in cols if c.startswith("F")],
        "deflection": [
            *[c for c in cols if c.startswith("D_t") or c.startswith("D_st")],
            *[c for c in cols if c.startswith("w_") or c.startswith("u_")],
            *[c for c in cols if c.startswith("w_s") or c.startswith("u_s")],
        ],
        "rotation": [c for c in cols if c.startswith("Ry_t") or c.startswith("phi_s")],
        "nodes": [c for c in cols if c.startswith("N")],
    }
    # Deduplicate while preserving order
    for k, lst in groups.items():
        seen = set()
        groups[k] = [x for x in lst if not (x in seen or seen.add(x))]
    return groups


def preset_bridge_measurements(
    df_like: Optional[pd.DataFrame] = None,
    *,
    # Reactions (load cells / derived forces)
    snr_db_reaction: Optional[float] = 35.0,
    sigma_reaction: Optional[float] = None,
    # Deflections (LVDT / displacement sensors)
    sigma_deflection: float = 1.0e-4,
    drift_deflection_per_sample: float = 0.0,
    # Rotations (inclinometers)
    sigma_rotation: float = 3.0e-5,
    drift_rotation_per_sample: float = 0.0,
    # Rare artifacts
    spike_prob: float = 0.0,
    spike_std_reaction: float = 300.0,
    spike_std_deflection: float = 5.0e-4,
    spike_std_rotation: float = 1.5e-4,
    # Quantization (optional)
    quant_step_deflection: float = 0.0,
    quant_step_rotation: float = 0.0,
    # Column patterns (optional overrides)
    reaction_cols: Optional[Sequence[str]] = None,
    deflection_cols: Optional[Sequence[str]] = None,
    rotation_cols: Optional[Sequence[str]] = None,
) -> List[ColumnNoise]:
    """Preset capturing plausible measurement noise for this project.

    - Reactions: primarily Gaussian noise; specify either SNR (recommended) or absolute sigma.
    - Deflections: small Gaussian noise (≈0.1 mm) + optional slow drift and quantization.
    - Rotations: small Gaussian noise (≈ tens of μrad) + optional drift.
    - Rare spikes: optional low-probability impulses in all groups for robustness testing.
    """
    dummy_df = df_like if isinstance(df_like, pd.DataFrame) else pd.DataFrame()
    groups = infer_groups(dummy_df) if not dummy_df.empty else {
        "reaction": reaction_cols or [],
        "deflection": deflection_cols or [],
        "rotation": rotation_cols or [],
    }
    if reaction_cols is not None:
        groups["reaction"] = list(reaction_cols)
    if deflection_cols is not None:
        groups["deflection"] = list(deflection_cols)
    if rotation_cols is not None:
        groups["rotation"] = list(rotation_cols)

    schemes: List[ColumnNoise] = []

    # Reactions
    if groups.get("reaction"):
        if sigma_reaction is not None:
            schemes.append(make_gaussian_sigma(groups["reaction"], sigma=sigma_reaction))
        elif snr_db_reaction is not None:
            schemes.append(make_gaussian_snr(groups["reaction"], snr_db=snr_db_reaction))
        if spike_prob > 0.0:
            schemes.append(make_spikes(groups["reaction"], p=spike_prob, spike_std=spike_std_reaction))

    # Deflections
    if groups.get("deflection"):
        schemes.append(make_gaussian_sigma(groups["deflection"], sigma=sigma_deflection))
        if drift_deflection_per_sample != 0.0:
            schemes.append(make_drift(groups["deflection"], drift_per_sample=drift_deflection_per_sample))
        if quant_step_deflection > 0.0:
            schemes.append(make_quantize(groups["deflection"], step=quant_step_deflection))
        if spike_prob > 0.0:
            schemes.append(make_spikes(groups["deflection"], p=spike_prob, spike_std=spike_std_deflection))

    # Rotations
    if groups.get("rotation"):
        schemes.append(make_gaussian_sigma(groups["rotation"], sigma=sigma_rotation))
        if drift_rotation_per_sample != 0.0:
            schemes.append(make_drift(groups["rotation"], drift_per_sample=drift_rotation_per_sample))
        if quant_step_rotation > 0.0:
            schemes.append(make_quantize(groups["rotation"], step=quant_step_rotation))
        if spike_prob > 0.0:
            schemes.append(make_spikes(groups["rotation"], p=spike_prob, spike_std=spike_std_rotation))

    return schemes


# ---------------------------
# Optional CLI helper
# ---------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Add measurement noise to a CSV file.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snr-db-reaction", type=float, default=35.0, help="Target SNR for reaction columns (dB).")
    p.add_argument("--sigma-reaction", type=float, default=None, help="Absolute sigma for reaction columns (overrides SNR if set).")
    p.add_argument("--sigma-deflection", type=float, default=1.0e-4)
    p.add_argument("--sigma-rotation", type=float, default=3.0e-5)
    p.add_argument("--drift-defl", type=float, default=0.0, help="Drift per sample for deflections (units/sample).")
    p.add_argument("--drift-rot", type=float, default=0.0, help="Drift per sample for rotations (rad/sample).")
    p.add_argument("--spike-prob", type=float, default=0.0)
    p.add_argument("--quant-defl", type=float, default=0.0, help="Quantization step for deflections (units).")
    p.add_argument("--quant-rot", type=float, default=0.0, help="Quantization step for rotations (rad).")
    p.add_argument(
        "--reaction-cols",
        nargs="*",
        default=None,
        help="Column names or prefixes ending with * (e.g., R*). If omitted, inferred.")
    p.add_argument(
        "--deflection-cols",
        nargs="*",
        default=None,
        help="Column names or prefixes (e.g., D_t*, D_st*, w_*, u_*). If omitted, inferred.")
    p.add_argument(
        "--rotation-cols",
        nargs="*",
        default=None,
        help="Column names or prefixes (e.g., Ry_t*, phi_s*). If omitted, inferred.")

    args = p.parse_args(argv)

    df = pd.read_csv(args.input)

    # Resolve patterns to concrete columns when provided
    def _resolve(cols: Optional[Sequence[str]]) -> Optional[List[str]]:
        if cols is None:
            return None
        return _select_columns(df, cols)

    schemes = preset_bridge_measurements(
        df_like=df,
        snr_db_reaction=args.snr_db_reaction if args.sigma_reaction is None else None,
        sigma_reaction=args.sigma_reaction,
        sigma_deflection=args.sigma_deflection,
        sigma_rotation=args.sigma_rotation,
        drift_deflection_per_sample=args.drift_defl,
        drift_rotation_per_sample=args.drift_rot,
        spike_prob=args.spike_prob,
        quant_step_deflection=args.quant_defl,
        quant_step_rotation=args.quant_rot,
        reaction_cols=_resolve(args.reaction_cols),
        deflection_cols=_resolve(args.deflection_cols),
        rotation_cols=_resolve(args.rotation_cols),
    )

    out = add_noise(df, schemes, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[OK] Wrote noisy CSV: {args.output} (rows={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

