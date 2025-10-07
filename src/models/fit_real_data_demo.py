#!/usr/bin/env python3
"""Demonstration of parameter fitting using real bridge data.

This script reads data from dt_24hours_data_1.csv and demonstrates:
1. Loading real bridge reaction measurements
2. Fitting structural parameters (settlements, ei_factors, kv_factors)
3. Fitting thermal state (temperature gradients)
4. Comparing fitted results with true values
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from models.parameter_fitter import (
    BridgeParameterFitter,
    FitConfig,
    FitResult,
    ReactionMeasurement,
)


def load_sample(csv_path: Path, sample_idx: int = 0) -> Dict:
    """Load a single sample from the CSV file."""
    df = pd.read_csv(csv_path)
    if sample_idx >= len(df):
        raise ValueError(f"Sample index {sample_idx} out of range (max: {len(df) - 1})")
    
    row = df.iloc[sample_idx]
    
    # Extract true values
    true_data = {
        "settlements": (
            float(row["settlement_a_mm"]),
            float(row["settlement_b_mm"]),
            float(row["settlement_c_mm"]),
            float(row["settlement_d_mm"]),
        ),
        "ei_factors": (
            float(row["ei_factor_s1"]),
            float(row["ei_factor_s2"]),
            float(row["ei_factor_s3"]),
        ),
        "kv_factors": (
            float(row["kv_factor_a"]),
            float(row["kv_factor_b"]),
            float(row["kv_factor_c"]),
            float(row["kv_factor_d"]),
        ),
        "dT_spans": (
            float(row["dT_s1_C"]),
            float(row["dT_s2_C"]),
            float(row["dT_s3_C"]),
        ),
        "reactions": np.array([
            float(row["R_a_kN"]),
            float(row["R_b_kN"]),
            float(row["R_c_kN"]),
            float(row["R_d_kN"]),
        ]),
        "uniform_load": float(row["uniform_load_N_per_mm"]),
        "sample_id": int(row["sample_id"]),
    }
    
    return true_data


def print_comparison(name: str, true_vals: tuple | np.ndarray, fitted_vals: tuple | np.ndarray) -> None:
    """Print a comparison between true and fitted values."""
    true_arr = np.array(true_vals)
    fitted_arr = np.array(fitted_vals)
    errors = fitted_arr - true_arr
    rel_errors = errors / (np.abs(true_arr) + 1e-10) * 100
    
    print(f"\n{name}:")
    print(f"  True:    {true_arr}")
    print(f"  Fitted:  {fitted_arr}")
    print(f"  Error:   {errors}")
    print(f"  RelErr: {rel_errors} %")


def main():
    parser = argparse.ArgumentParser(description="Fit parameters from real bridge data")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/augmented/dt_24hours_data_1.csv"),
        help="Path to CSV data file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to use (default: 0)",
    )
    parser.add_argument(
        "--fit-settlements",
        action="store_true",
        help="Fit settlement parameters",
    )
    parser.add_argument(
        "--fit-ei",
        action="store_true",
        help="Fit EI factor parameters",
    )
    parser.add_argument(
        "--fit-kv",
        action="store_true",
        help="Fit support stiffness factor parameters",
    )
    parser.add_argument(
        "--fit-temp",
        action="store_true",
        help="Fit temperature gradient parameters",
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Fit all parameters simultaneously",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum iterations for optimization (default: 200)",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading sample {args.sample} from {args.data}")
    true_data = load_sample(args.data, args.sample)
    
    print("\n" + "=" * 60)
    print(f"Sample ID: {true_data['sample_id']}")
    print("=" * 60)
    
    # Print true values
    print("\n--- TRUE VALUES ---")
    print(f"Settlements (mm): {true_data['settlements']}")
    print(f"EI factors:       {true_data['ei_factors']}")
    print(f"KV factors:       {true_data['kv_factors']}")
    print(f"Temperature (°C): {true_data['dT_spans']}")
    print(f"Reactions (kN):   {true_data['reactions']}")
    print(f"Uniform load:     {true_data['uniform_load']} N/mm")
    
    # Create measurement
    measurement = ReactionMeasurement(
        reactions_kN=true_data["reactions"],
        timestamp=0.0,
    )
    
    # Create fitter
    fitter = BridgeParameterFitter(
        uniform_load=true_data["uniform_load"],
        ne_per_span=64,
    )
    
    # Determine what to fit
    if args.fit_all:
        fit_settlements = fit_ei = fit_kv = fit_temp = True
    else:
        fit_settlements = args.fit_settlements
        fit_ei = args.fit_ei
        fit_kv = args.fit_kv
        fit_temp = args.fit_temp
    
    # If nothing specified, fit temperature only (simplest case)
    if not any([fit_settlements, fit_ei, fit_kv, fit_temp]):
        print("\nNo fit options specified, fitting temperature gradients only.")
        fit_temp = True
    
    # Create fit config
    config = FitConfig(
        fit_settlements=fit_settlements,
        fit_ei_factors=fit_ei,
        fit_kv_factors=fit_kv,
        fit_temperature=fit_temp,
        maxiter=args.maxiter,
        ftol=1e-8,
        gtol=1e-8,
    )
    
    # Set initial guess (use true values with some perturbation)
    np.random.seed(42)
    if not fit_settlements:
        fitter.set_settlements(true_data["settlements"])
    else:
        # Perturb initial guess
        perturbed = tuple(s + np.random.randn() * 0.5 for s in true_data["settlements"])
        fitter.set_settlements(perturbed)
    
    if not fit_ei:
        fitter.set_ei_factors(true_data["ei_factors"])
    else:
        perturbed = tuple(max(0.1, e + np.random.randn() * 0.1) for e in true_data["ei_factors"])
        fitter.set_ei_factors(perturbed)
    
    if not fit_kv:
        fitter.set_kv_factors(true_data["kv_factors"])
    else:
        perturbed = tuple(max(0.1, k + np.random.randn() * 0.1) for k in true_data["kv_factors"])
        fitter.set_kv_factors(perturbed)
    
    print("\n" + "=" * 60)
    print("FITTING CONFIGURATION")
    print("=" * 60)
    print(f"Fit settlements:  {config.fit_settlements}")
    print(f"Fit EI factors:   {config.fit_ei_factors}")
    print(f"Fit KV factors:   {config.fit_kv_factors}")
    print(f"Fit temperature:  {config.fit_temperature}")
    print(f"Max iterations:   {config.maxiter}")
    
    # Perform fit
    print("\n" + "=" * 60)
    print("RUNNING OPTIMIZATION...")
    print("=" * 60)
    
    result = fitter.fit_single_measurement(measurement, config)
    
    # Print results
    print("\n" + "=" * 60)
    print("FIT RESULTS")
    print("=" * 60)
    print(f"Success:     {result.success}")
    print(f"Message:     {result.message}")
    print(f"Iterations:  {result.nit}")
    print(f"Func evals:  {result.nfev}")
    print(f"Final cost:  {result.cost:.6e}")
    print(f"Optimality:  {result.optimality:.6e}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON")
    print("=" * 60)
    
    if config.fit_settlements or args.fit_all:
        print_comparison("Settlements (mm)", true_data["settlements"], result.fitted_params.settlements)
    
    if config.fit_ei_factors or args.fit_all:
        print_comparison("EI factors", true_data["ei_factors"], result.fitted_params.ei_factors)
    
    if config.fit_kv_factors or args.fit_all:
        print_comparison("KV factors", true_data["kv_factors"], result.fitted_params.kv_factors)
    
    if config.fit_temperature or args.fit_all:
        print_comparison("Temperature (°C)", true_data["dT_spans"], result.fitted_thermal.dT_spans)
    
    # Compare reactions
    print_comparison("Reactions (kN)", true_data["reactions"], result.fitted_reactions)
    
    # Print summary statistics
    reaction_errors = result.fitted_reactions - true_data["reactions"]
    print(f"\nReaction fit statistics:")
    print(f"  RMSE:     {np.sqrt(np.mean(reaction_errors**2)):.4f} kN")
    print(f"  Max err:  {np.max(np.abs(reaction_errors)):.4f} kN")
    print(f"  Mean err: {np.mean(reaction_errors):.4f} kN")
    

if __name__ == "__main__":
    main()
