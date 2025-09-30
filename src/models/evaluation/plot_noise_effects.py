#!/usr/bin/env python3
"""
Noise robustness visualization utilities and CLI.

Supports multiple noisy prediction sets (e.g., different SNR/sigma levels) and
any number of targets (e.g., 7 outputs). Produces professional plots:
 - Clean vs Noisy scatter with identity line (hue = noise level)
 - Bland–Altman (mean vs difference) plots per target (hue = noise level)
 - Residual distribution comparison (clean vs noisy) via violin/box
 - Robustness curves: metric vs noise level (RMSE/MAE/R2)
 - Sensitivity ranking: median |Δŷ| per target (optionally normalized)

Usage example:
  uv run python models/evaluation/plot_noise_effects.py \
    --ytrue data/y_true.csv \
    --yhat-clean data/yhat_clean.csv \
    --noisy SNR35=data/yhat_snr35.csv \
    --noisy SNR30=data/yhat_snr30.csv \
    --outdir experiments/noise_plots
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import Metrics


# -------------------------
# Data prep
# -------------------------

def _align_frames(ref: pd.DataFrame, other: pd.DataFrame, name: str) -> pd.DataFrame:
    """Reindex and reorder columns of `other` to match `ref`.
    Raises if columns are missing.
    """
    missing = [c for c in ref.columns if c not in other.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    out = other.loc[ref.index, ref.columns]
    return out


def melt_long(
    y_true: pd.DataFrame,
    yhat_clean: pd.DataFrame,
    yhat_noisy_by_level: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return a long-form DataFrame for plotting across targets and levels.

    Columns: ['idx','target','y_true','yhat','yhat_clean','delta','level']
    """
    yhat_clean = _align_frames(y_true, yhat_clean, "yhat_clean")
    parts = []
    for level, df in yhat_noisy_by_level.items():
        yh = _align_frames(y_true, df, f"yhat_noisy[{level}]")
        tmp = pd.DataFrame({
            "idx": y_true.index,
            "level": level,
        })
        for col in y_true.columns:
            part = pd.DataFrame({
                "idx": y_true.index,
                "target": col,
                "y_true": y_true[col].to_numpy(),
                "yhat": yh[col].to_numpy(),
                "yhat_clean": yhat_clean[col].to_numpy(),
            })
            parts.append(part.assign(level=level))
    long = pd.concat(parts, axis=0, ignore_index=True)
    long["delta"] = long["yhat"] - long["yhat_clean"]
    return long


# -------------------------
# Plotting helpers
# -------------------------

def _grid(n: int) -> Tuple[int, int]:
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    return n_rows, n_cols


def plot_scatter_identity(long: pd.DataFrame, out: Path, title: str = "Clean vs Noisy (ŷ)") -> None:
    targets = long["target"].unique().tolist()
    n_rows, n_cols = _grid(len(targets))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    for i, tgt in enumerate(targets):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        dd = long[long["target"] == tgt]
        sns.scatterplot(data=dd, x="yhat_clean", y="yhat", hue="level", alpha=0.5, s=16, ax=ax)
        # identity line
        all_vals = np.concatenate([dd["yhat_clean"].to_numpy(), dd["yhat"].to_numpy()])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
        ax.set_title(str(tgt))
        ax.set_xlabel("ŷ (clean)")
        ax.set_ylabel("ŷ (noisy)")
        ax.grid(True, alpha=0.3)
    # tidy legends
    handles, labels = axes[0][0].get_legend_handles_labels()
    for row in axes:
        for ax in row:
            if ax is not axes[0][0]:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.96, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_bland_altman(long: pd.DataFrame, out: Path, title: str = "Bland–Altman: mean(ŷ) vs Δŷ") -> None:
    long = long.copy()
    long["mean_hat"] = 0.5 * (long["yhat"] + long["yhat_clean"])
    targets = long["target"].unique().tolist()
    n_rows, n_cols = _grid(len(targets))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    for i, tgt in enumerate(targets):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        dd = long[long["target"] == tgt]
        sns.scatterplot(data=dd, x="mean_hat", y="delta", hue="level", alpha=0.5, s=16, ax=ax)
        # mean and ±1.96 std per level
        for lvl, g in dd.groupby("level"):
            mu = np.nanmean(g["delta"]) if len(g) else 0.0
            sd = np.nanstd(g["delta"]) if len(g) else 0.0
            ax.axhline(mu, color="gray", ls=":", lw=1)
            ax.axhline(mu + 1.96*sd, color="gray", ls="--", lw=0.8)
            ax.axhline(mu - 1.96*sd, color="gray", ls="--", lw=0.8)
        ax.set_title(str(tgt))
        ax.set_xlabel("mean(ŷ_clean, ŷ_noisy)")
        ax.set_ylabel("Δŷ = ŷ_noisy − ŷ_clean")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for row in axes:
        for ax in row:
            if ax is not axes[0][0]:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.96, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_residual_violin(
    y_true: pd.DataFrame,
    yhat_clean: pd.DataFrame,
    yhat_noisy_by_level: Dict[str, pd.DataFrame],
    out: Path,
    title: str = "Residual distributions (clean vs noisy)",
) -> None:
    frames = []
    # clean
    res_clean = y_true - _align_frames(y_true, yhat_clean, "yhat_clean")
    frames.append(res_clean.assign(level="CLEAN").melt(id_vars=["level"], var_name="target", value_name="residual"))
    # noisy
    for lvl, yh in yhat_noisy_by_level.items():
        res = y_true - _align_frames(y_true, yh, f"yhat_noisy[{lvl}]")
        frames.append(res.assign(level=str(lvl)).melt(id_vars=["level"], var_name="target", value_name="residual"))
    dfv = pd.concat(frames, ignore_index=True)

    targets = y_true.columns.tolist()
    n_rows, n_cols = _grid(len(targets))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    for i, tgt in enumerate(targets):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        dd = dfv[dfv["target"] == tgt]
        sns.violinplot(data=dd, x="level", y="residual", inner="box", cut=0, ax=ax)
        ax.axhline(0.0, color="k", ls="--", lw=1)
        ax.set_title(str(tgt))
        ax.set_xlabel("")
        ax.set_ylabel("y − ŷ")
        ax.grid(True, alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def compute_metric_table(
    y_true: pd.DataFrame,
    yhat_clean: pd.DataFrame,
    yhat_noisy_by_level: Dict[str, pd.DataFrame],
    metric: str = "rmse",
) -> pd.DataFrame:
    """Return wide table (targets x levels) with selected metric.
    metric in {"rmse","mae","r2"}
    """
    yhat_clean = _align_frames(y_true, yhat_clean, "yhat_clean")
    cols = y_true.columns
    rows = []
    for lvl, yh in yhat_noisy_by_level.items():
        yh = _align_frames(y_true, yh, f"yhat_noisy[{lvl}]")
        if metric == "rmse":
            md = Metrics.root_mean_squared_error(y_true[cols], yh[cols])
        elif metric == "mae":
            md = Metrics.mean_absolute_error(y_true[cols], yh[cols])
        elif metric == "r2":
            md = Metrics.r2_score(y_true[cols], yh[cols])
        elif metric == "mape":
            md = Metrics.mean_absolute_percentage_error(y_true[cols], yh[cols])
        elif metric == "mae_percent_mean_abs":
            md = Metrics.mean_absolute_error_percent(y_true[cols], yh[cols], denom="mean_abs")
        elif metric == "mae_percent_rms":
            md = Metrics.mean_absolute_error_percent(y_true[cols], yh[cols], denom="rms")
        elif metric == "mae_percent_p95":
            md = Metrics.mean_absolute_error_percent(y_true[cols], yh[cols], denom="p95")
        else:
            raise ValueError("Unsupported metric")
        # filter only target columns (exclude 'overall')
        row = {"level": lvl}
        row.update({c: md[c] for c in cols})
        rows.append(row)
    dfm = pd.DataFrame(rows).set_index("level")
    return dfm


def plot_robustness_curves(dfm: pd.DataFrame, out: Path, title: str = "Robustness curve") -> None:
    # Assume index = levels, columns = targets
    # Sort levels if they look numeric
    try:
        dfm = dfm.copy()
        dfm.index = pd.Index([float(x) if str(x).replace(".", "", 1).isdigit() else x for x in dfm.index], name="level")
        dfm = dfm.sort_index()
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in dfm.columns:
        ax.plot(dfm.index, dfm[col], marker="o", label=str(col))
    ax.set_xlabel("Noise level (label)")
    ax.set_ylabel(dfm.columns.name or "metric")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Target", ncol=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_bar(long: pd.DataFrame, out: Path, normalize: bool = False, title: str = "Noise sensitivity") -> None:
    # Compute per-target sensitivity: median |Δŷ| aggregated across levels
    agg = long.copy()
    agg["abs_delta"] = agg["delta"].abs()
    sens = agg.groupby("target")["abs_delta"].median().sort_values(ascending=False)
    if normalize:
        # normalize by median |ŷ_clean| to compare across scales
        scale = long.groupby("target")["yhat_clean"].apply(lambda s: np.median(np.abs(s)))
        sens = sens / (scale + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=sens.index, y=sens.values, ax=ax, color="#4C78A8")
    ax.set_ylabel("median |Δŷ|")
    ax.set_xlabel("target")
    ax.set_title(title + (" (normalized)" if normalize else ""))
    ax.grid(True, axis="y", alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_baseline_metric(
    y_true: pd.DataFrame,
    yhat_clean: pd.DataFrame,
    metric: str = "rmse",
) -> pd.Series:
    cols = y_true.columns
    yh = _align_frames(y_true, yhat_clean, "yhat_clean")
    if metric == "rmse":
        md = Metrics.root_mean_squared_error(y_true[cols], yh[cols])
    elif metric == "mae":
        md = Metrics.mean_absolute_error(y_true[cols], yh[cols])
    elif metric == "r2":
        md = Metrics.r2_score(y_true[cols], yh[cols])
    else:
        raise ValueError("Unsupported metric")
    return pd.Series({c: md[c] for c in cols}, name="baseline")


def plot_metric_heatmap(
    y_true: pd.DataFrame,
    yhat_clean: pd.DataFrame,
    yhat_noisy_by_level: Dict[str, pd.DataFrame],
    out: Path,
    metric: str = "rmse",
    include_clean: bool = True,
    title: Optional[str] = None,
) -> None:
    """Plot a target-by-level heatmap for the selected metric.

    - Rows: targets
    - Columns: noise levels (optionally with a leading CLEAN column)
    """
    # Metric by level (levels x targets)
    dfm = compute_metric_table(y_true, yhat_clean, yhat_noisy_by_level, metric=metric)
    # Optionally prepend baseline (clean predictions vs y_true)
    if include_clean:
        base = compute_baseline_metric(y_true, yhat_clean, metric=metric)
        dfm = pd.concat([pd.DataFrame([base.to_dict()], index=["CLEAN"]), dfm], axis=0)
    # Reindex levels to keep textual order as-is
    # Build matrix in shape (targets x levels)
    targets = list(y_true.columns)
    mat = dfm[targets].T  # rows=targets, cols=levels

    # Choose colormap: for RMSE/MAE lower is better, use rocket_r; for R2 higher is better, use viridis
    if metric.lower() in {"rmse", "mae", "mape", "mae_percent_mean_abs", "mae_percent_rms", "mae_percent_p95"}:
        cmap = "rocket_r"
    else:
        cmap = "viridis"

    fig_w = max(6.0, 1.2 * mat.shape[1])
    fig_h = max(4.0, 0.6 * mat.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(mat, annot=False, cmap=cmap, cbar=True, ax=ax)
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Target")
    ax.set_title(title or f"{metric.upper()} by noise level")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_degradation_table(
    dfm_noisy: pd.DataFrame,
    baseline: pd.Series,
    relative: bool = False,
) -> pd.DataFrame:
    # Align by columns
    baseline = baseline.reindex(dfm_noisy.columns)
    if relative:
        # percentage change: (noisy - baseline)/baseline * 100
        return (dfm_noisy - baseline) / (baseline.replace(0, np.nan)) * 100.0
    else:
        return dfm_noisy - baseline


def plot_heatmap(
    df: pd.DataFrame,
    out: Path,
    title: str = "Noise Sensitivity Heatmap",
    cmap: str = "RdYlGn_r",
    fmt: str = ".3g",
    cbar_label: Optional[str] = None,
) -> None:
    # Try sort index numerically if possible (e.g., SNR labels) while preserving string labels
    try:
        # Extract trailing number for labels like "SNR35"
        def _num(x: str):
            import re
            m = re.search(r"([+-]?\d+(?:\.\d+)?)", str(x))
            return float(m.group(1)) if m else np.nan
        order = sorted(df.index, key=_num)
        df = df.loc[order]
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(1.2 * len(df.columns) + 2, 0.5 * len(df.index) + 2.5))
    hm = sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, linecolor="#EEE", cbar_kws={"label": cbar_label or ""}, ax=ax)
    ax.set_xlabel("Target")
    ax.set_ylabel("Noise level")
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# CLI
# -------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot noise robustness with multiple noisy predictions.")
    p.add_argument("--ytrue", type=Path, required=True)
    p.add_argument("--yhat-clean", dest="yhat_clean", type=Path, required=True)
    p.add_argument(
        "--noisy",
        action="append",
        default=[],
        help="Pairs label=path for noisy predictions (e.g., SNR35=path.csv). Can repeat.",
    )
    p.add_argument("--targets", nargs="*", default=None, help="Optional list of target columns to plot.")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--metric", choices=["rmse", "mae", "r2"], default="rmse")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    df_true = pd.read_csv(args.ytrue)
    df_clean = pd.read_csv(args.yhat_clean)
    levels: Dict[str, pd.DataFrame] = {}
    for item in args.noisy:
        if "=" not in item:
            raise SystemExit(f"Invalid --noisy '{item}'. Use label=path.csv")
        label, path = item.split("=", 1)
        levels[label] = pd.read_csv(path)

    # Optionally subset targets
    if args.targets:
        keep = [c for c in args.targets if c in df_true.columns]
        if not keep:
            raise SystemExit("No requested targets found in y_true")
        df_true = df_true[keep]
        df_clean = df_clean[keep]
        for k in list(levels.keys()):
            levels[k] = levels[k][keep]

    # Prepare
    long = melt_long(df_true, df_clean, levels)
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Plots
    plot_scatter_identity(long, args.outdir / "scatter_clean_vs_noisy.png")
    plot_bland_altman(long, args.outdir / "bland_altman.png")
    plot_residual_violin(df_true, df_clean, levels, args.outdir / "residual_violin.png")
    dfm = compute_metric_table(df_true, df_clean, levels, metric=args.metric)
    dfm.columns.name = args.metric
    plot_robustness_curves(dfm, args.outdir / f"robustness_{args.metric}.png", title=f"Robustness ({args.metric})")
    plot_sensitivity_bar(long, args.outdir / "sensitivity_bar_abs.png", normalize=False)
    plot_sensitivity_bar(long, args.outdir / "sensitivity_bar_norm.png", normalize=True)
    print(f"[OK] Plots saved to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
