import numpy as np
import pandas as pd

from models.gp.uncertainty import (
    build_uncertainty_summary,
    calibrate_global_tau,
    calibrate_quantile_tau_map,
)


def _make_synthetic_frame(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    num_samples = 4000
    num_targets = 3
    columns = [f"t{idx}" for idx in range(num_targets)]

    y_pred = rng.normal(loc=0.0, scale=1.0, size=(num_samples, num_targets))
    residual = rng.normal(loc=0.0, scale=1.5, size=(num_samples, num_targets))
    y_true = y_pred + residual
    y_std = np.ones_like(y_pred)

    true_df = pd.DataFrame(y_true, columns=columns)
    pred_df = pd.DataFrame(y_pred, columns=columns)
    std_df = pd.DataFrame(y_std, columns=columns)
    return true_df, pred_df, std_df


def test_multi_quantile_calibration_matches_global_scalar():
    true_df, pred_df, std_df = _make_synthetic_frame(seed=123)
    coverage_levels = [0.5, 0.9, 0.95]

    global_tau = calibrate_global_tau(true_df, pred_df, std_df, coverage_levels)
    assert global_tau is not None
    assert np.isfinite(global_tau)

    quantile_tau = calibrate_quantile_tau_map(true_df, pred_df, std_df, coverage_levels)
    assert "0.9" not in quantile_tau  # sanity: keys should be floats
    assert len(quantile_tau) == len([lvl for lvl in coverage_levels if 0.0 < lvl < 1.0])

    representative_levels = [0.5, 0.9, 0.95]
    for level in representative_levels:
        tau_value = quantile_tau.get(level)
        assert tau_value is not None
        assert np.isfinite(tau_value)
        # With synthetic data (sigma_true ~= 1.5), all calibrated taus should be close
        # to the underlying ratio.
        assert np.isclose(tau_value, 1.5, atol=0.25)

    # Multi-quantile calibration should agree with the scalar solution within tolerance.
    for tau_value in quantile_tau.values():
        assert np.isclose(tau_value, global_tau, atol=0.25)

    summary = build_uncertainty_summary(
        true_df,
        pred_df,
        std_df,
        coverage_levels,
        quantile_tau=quantile_tau,
    )
    assert summary is not None
    assert "quantile_tau_scaled" in summary
    quantile_section = summary["quantile_tau_scaled"]
    assert "values" in quantile_section
    assert "coverage" in quantile_section

    for level in representative_levels:
        level_key = f"{level:g}"
        coverage = quantile_section["coverage"][level_key]["overall"]
        assert abs(coverage - level) < 0.05
