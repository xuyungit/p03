"""Temperature parameterizations for bridge fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import OptimizationConfig


class TemperatureBasis(Protocol):
    """Protocol for temperature parameterizations."""

    n_cases: int
    temp_segments: int

    def num_params(self) -> int:
        """Number of scalar parameters required by this basis."""

    def initial(self, opt_config: OptimizationConfig) -> np.ndarray:
        """Initial parameter vector."""

    def bounds(self, opt_config: OptimizationConfig) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds for parameters."""

    def expand(self, params: np.ndarray) -> np.ndarray:
        """Expand parameter vector to a (n_cases, temp_segments) temperature matrix."""

    def describe(self) -> str:
        """Human readable description for logging."""


@dataclass
class RawTemperatureBasis:
    """Baseline parameterization: independent temperature per case and segment."""

    n_cases: int
    temp_segments: int

    def num_params(self) -> int:
        return self.n_cases * self.temp_segments

    def initial(self, opt_config: OptimizationConfig) -> np.ndarray:
        init = np.full(self.num_params(), opt_config.temp_gradient_initial, dtype=float)
        return init

    def bounds(self, opt_config: OptimizationConfig) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.num_params(), opt_config.temp_gradient_lower, dtype=float)
        upper = np.full(self.num_params(), opt_config.temp_gradient_upper, dtype=float)
        return lower, upper

    def expand(self, params: np.ndarray) -> np.ndarray:
        return params.reshape(self.n_cases, self.temp_segments)

    def describe(self) -> str:
        return "逐工况温度 (无降维)"


@dataclass
class FourierTemperatureBasis:
    """Fourier series parameterization for smooth, low-dimensional temperature fields."""

    n_cases: int
    temp_segments: int
    harmonics: int = 3
    include_bias: bool = True
    time_values: np.ndarray | None = None
    fundamental_period: float | None = None
    coeff_lower: float | None = None
    coeff_upper: float | None = None
    coeff_initial: np.ndarray | float | None = None

    def __post_init__(self) -> None:
        if self.harmonics < 0:
            raise ValueError("harmonics must be non-negative")

        if self.time_values is None:
            self._time = np.arange(self.n_cases, dtype=float)
        else:
            self._time = np.asarray(self.time_values, dtype=float)
            if self._time.shape != (self.n_cases,):
                raise ValueError("time_values must have shape (n_cases,)")

        period = self.fundamental_period or float(self.n_cases)
        if period <= 0.0:
            raise ValueError("fundamental_period must be positive")

        omega = 2.0 * np.pi / period

        columns: list[np.ndarray] = []
        if self.include_bias:
            columns.append(np.ones_like(self._time))

        for h in range(1, self.harmonics + 1):
            angle = h * omega * self._time
            columns.append(np.cos(angle))
            columns.append(np.sin(angle))

        if not columns:
            columns.append(np.ones_like(self._time))
            self.include_bias = True

        self._design_matrix = np.column_stack(columns).astype(float)
        self._n_coeff = self._design_matrix.shape[1]

    def num_params(self) -> int:
        return self.temp_segments * self._n_coeff

    def initial(self, opt_config: OptimizationConfig) -> np.ndarray:
        if self.coeff_initial is None:
            base = np.zeros(self._n_coeff, dtype=float)
            if self.include_bias:
                base[0] = opt_config.temp_gradient_initial
        else:
            base = np.asarray(self.coeff_initial, dtype=float)
            if base.shape not in {(self._n_coeff,), ()}:
                raise ValueError(f"coeff_initial must have shape ({self._n_coeff},) or scalar")
            base = np.broadcast_to(base, (self._n_coeff,)).copy()

        init = np.tile(base, (self.temp_segments, 1))
        return init.ravel()

    def bounds(self, opt_config: OptimizationConfig) -> tuple[np.ndarray, np.ndarray]:
        lower_value = self.coeff_lower if self.coeff_lower is not None else opt_config.temp_gradient_lower
        upper_value = self.coeff_upper if self.coeff_upper is not None else opt_config.temp_gradient_upper
        lower = np.full((self.temp_segments, self._n_coeff), lower_value, dtype=float)
        upper = np.full((self.temp_segments, self._n_coeff), upper_value, dtype=float)
        return lower.ravel(), upper.ravel()

    def expand(self, params: np.ndarray) -> np.ndarray:
        coeffs = params.reshape(self.temp_segments, self._n_coeff)
        dT_matrix = self._design_matrix @ coeffs.T
        return dT_matrix.astype(float)

    def describe(self) -> str:
        bias_text = "含常数项" if self.include_bias else "无常数项"
        return f"傅里叶基 (k={self.harmonics}, {bias_text}, 每段 {self._n_coeff} 参数)"

