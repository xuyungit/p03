"""Parameter fitting module for bridge structural parameters.

This module provides tools to fit structural parameters (settlements, EI factors,
KV factors) and thermal states (temperature gradients) from measured reaction forces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from models.bridge_forward_model import (
    StructuralParams,
    ThermalState,
    evaluate_forward,
)


@dataclass
class ReactionMeasurement:
    """A single measurement of bridge reactions."""
    
    reactions_kN: np.ndarray
    timestamp: float = 0.0
    weights: np.ndarray | None = None  # Optional per-reaction weights


@dataclass
class ParameterBounds:
    """Bounds for parameters during optimization."""
    
    settlement_range: Tuple[float, float] = (-20.0, 20.0)  # mm
    ei_factor_range: Tuple[float, float] = (0.1, 2.0)
    kv_factor_range: Tuple[float, float] = (0.1, 3.0)
    dT_range: Tuple[float, float] = (-20.0, 20.0)  # °C


@dataclass
class FitConfig:
    """Configuration for parameter fitting."""
    
    fit_settlements: bool = False
    fit_ei_factors: bool = False
    fit_kv_factors: bool = False
    fit_temperature: bool = True
    
    bounds: ParameterBounds = field(default_factory=ParameterBounds)
    
    maxiter: int = 100
    ftol: float = 1e-6
    gtol: float = 1e-6
    verbose: int = 1


@dataclass
class FittedParameters:
    """Fitted structural parameters."""
    
    settlements: Tuple[float, float, float, float]
    ei_factors: Tuple[float, float, float]
    kv_factors: Tuple[float, float, float, float]


@dataclass
class FittedThermal:
    """Fitted thermal state."""
    
    dT_spans: Tuple[float, float, float]


@dataclass
class FitResult:
    """Results from parameter fitting."""
    
    success: bool
    message: str
    fitted_params: FittedParameters
    fitted_thermal: FittedThermal
    fitted_reactions: np.ndarray
    cost: float
    optimality: float
    nfev: int
    nit: int


class BridgeParameterFitter:
    """Fits bridge parameters from reaction measurements using forward model."""
    
    def __init__(
        self,
        uniform_load: float,
        ne_per_span: int = 64,
    ):
        """Initialize fitter.
        
        Args:
            uniform_load: Uniform distributed load in N/mm
            ne_per_span: Number of elements per span for forward model
        """
        self.uniform_load = uniform_load
        self.ne_per_span = ne_per_span
        
        # Current parameter values (will be updated during fitting)
        self._settlements = (0.0, 0.0, 0.0, 0.0)
        self._ei_factors = (1.0, 1.0, 1.0)
        self._kv_factors = (1.0, 1.0, 1.0, 1.0)
    
    def set_settlements(self, values: Sequence[float]) -> None:
        """Set current settlement values (mm)."""
        self._settlements = tuple(values)
    
    def set_ei_factors(self, values: Sequence[float]) -> None:
        """Set current EI factor values."""
        self._ei_factors = tuple(values)
    
    def set_kv_factors(self, values: Sequence[float]) -> None:
        """Set current vertical stiffness factor values."""
        self._kv_factors = tuple(values)
    
    def _build_x0_and_bounds(
        self,
        config: FitConfig,
        dT_initial: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Build initial guess and bounds for optimization."""
        x0_list = []
        lower_list = []
        upper_list = []
        
        bounds = config.bounds
        
        if config.fit_settlements:
            x0_list.extend(self._settlements)
            lower_list.extend([bounds.settlement_range[0]] * 4)
            upper_list.extend([bounds.settlement_range[1]] * 4)
        
        if config.fit_ei_factors:
            x0_list.extend(self._ei_factors)
            lower_list.extend([bounds.ei_factor_range[0]] * 3)
            upper_list.extend([bounds.ei_factor_range[1]] * 3)
        
        if config.fit_kv_factors:
            x0_list.extend(self._kv_factors)
            lower_list.extend([bounds.kv_factor_range[0]] * 4)
            upper_list.extend([bounds.kv_factor_range[1]] * 4)
        
        if config.fit_temperature:
            x0_list.extend(dT_initial)
            lower_list.extend([bounds.dT_range[0]] * 3)
            upper_list.extend([bounds.dT_range[1]] * 3)
        
        x0 = np.array(x0_list)
        lower = np.array(lower_list)
        upper = np.array(upper_list)
        
        return x0, (lower, upper)
    
    def _unpack_params(
        self,
        x: np.ndarray,
        config: FitConfig,
    ) -> Tuple[
        Tuple[float, ...],
        Tuple[float, ...],
        Tuple[float, ...],
        Tuple[float, ...],
    ]:
        """Unpack optimization vector into parameter tuples."""
        idx = 0
        
        if config.fit_settlements:
            settlements = tuple(x[idx:idx+4])
            idx += 4
        else:
            settlements = self._settlements
        
        if config.fit_ei_factors:
            ei_factors = tuple(x[idx:idx+3])
            idx += 3
        else:
            ei_factors = self._ei_factors
        
        if config.fit_kv_factors:
            kv_factors = tuple(x[idx:idx+4])
            idx += 4
        else:
            kv_factors = self._kv_factors
        
        if config.fit_temperature:
            dT_spans = tuple(x[idx:idx+3])
            idx += 3
        else:
            dT_spans = (0.0, 0.0, 0.0)
        
        return settlements, ei_factors, kv_factors, dT_spans
    
    def _residual(
        self,
        x: np.ndarray,
        config: FitConfig,
        measured_reactions: np.ndarray,
        weights: np.ndarray | None,
    ) -> np.ndarray:
        """Compute residual between measured and predicted reactions."""
        settlements, ei_factors, kv_factors, dT_spans = self._unpack_params(x, config)
        
        # Create structural parameters
        struct_params = StructuralParams(
            settlements=settlements,
            kv_factors=kv_factors,
            ei_factors=ei_factors,
            uniform_load=self.uniform_load,
            ne_per_span=self.ne_per_span,
        )
        
        # Create thermal state
        thermal_state = ThermalState(dT_spans=dT_spans)
        
        # Run forward model
        response = evaluate_forward(struct_params, thermal_state)
        
        # Compute residuals
        predicted_reactions = response.reactions_kN
        residuals = predicted_reactions - measured_reactions
        
        # Apply weights if provided
        if weights is not None:
            residuals = residuals * weights
        
        return residuals
    
    def fit_single_measurement(
        self,
        measurement: ReactionMeasurement,
        config: FitConfig,
        dT_initial: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> FitResult:
        """Fit parameters from a single reaction measurement.
        
        Args:
            measurement: Reaction measurement data
            config: Fitting configuration
            dT_initial: Initial guess for temperature gradients
        
        Returns:
            FitResult with fitted parameters and diagnostics
        """
        # Build optimization problem
        x0, bounds = self._build_x0_and_bounds(config, dT_initial)
        
        # Run optimization
        result = least_squares(
            self._residual,
            x0,
            bounds=bounds,
            args=(config, measurement.reactions_kN, measurement.weights),
            ftol=config.ftol,
            gtol=config.gtol,
            max_nfev=config.maxiter * 100,
            verbose=config.verbose,
        )
        
        # Unpack solution
        settlements, ei_factors, kv_factors, dT_spans = self._unpack_params(
            result.x, config
        )
        
        # Compute final reactions using forward model
        struct_params = StructuralParams(
            settlements=settlements,  # type: ignore[arg-type]
            kv_factors=kv_factors,  # type: ignore[arg-type]
            ei_factors=ei_factors,  # type: ignore[arg-type]
            uniform_load=self.uniform_load,
            ne_per_span=self.ne_per_span,
        )
        thermal_state = ThermalState(dT_spans=dT_spans)  # type: ignore[arg-type]
        final_response = evaluate_forward(struct_params, thermal_state)
        
        return FitResult(
            success=result.success,
            message=result.message,
            fitted_params=FittedParameters(
                settlements=settlements,  # type: ignore[arg-type]
                ei_factors=ei_factors,  # type: ignore[arg-type]
                kv_factors=kv_factors,  # type: ignore[arg-type]
            ),
            fitted_thermal=FittedThermal(dT_spans=dT_spans),  # type: ignore[arg-type]
            fitted_reactions=final_response.reactions_kN,
            cost=result.cost,
            optimality=result.optimality,
            nfev=result.nfev,
            nit=result.njev if hasattr(result, 'njev') else 0,
        )


__all__ = [
    "ReactionMeasurement",
    "ParameterBounds",
    "FitConfig",
    "FittedParameters",
    "FittedThermal",
    "FitResult",
    "BridgeParameterFitter",
]
