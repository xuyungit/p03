"""Reusable bridge forward model built on shared mechanics utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from bridge import mechanics as _mech


# Re-export core constants and data structures for backwards compatibility.
E = _mech.E
Iy = _mech.Iy
EI_BASE = _mech.EI_BASE
Q_UNIFORM = _mech.Q_UNIFORM
KV0 = _mech.KV0
ALPHA = _mech.ALPHA
H_MM = _mech.H_MM

StructuralParams = _mech.StructuralParams
ThermalState = _mech.ThermalState
BridgeResponse = _mech.BridgeResponse
MeshData = _mech.MeshData
StructuralSystem = _mech.StructuralSystem


def get_span_lengths_mm() -> tuple[float, float, float]:
    return _mech.get_span_lengths_mm()


def set_span_lengths_mm(lengths_mm: Sequence[float]) -> None:
    _mech.set_span_lengths_mm(lengths_mm)


def span_breakpoints_mm() -> tuple[float, float, float, float]:
    return _mech.span_breakpoints_mm()


def total_length_mm() -> float:
    return _mech.total_length_mm()


def build_mesh(ne_per_span: int) -> MeshData:
    return _mech.build_mesh(ne_per_span)


def assemble_structural_system(struct_params: StructuralParams) -> StructuralSystem:
    return _mech.assemble_structural_system(struct_params)


def thermal_load_vector(system: StructuralSystem, thermal_state: ThermalState) -> np.ndarray:
    return _mech.thermal_load_vector(system, thermal_state)


def solve_linear(system: StructuralSystem, rhs: np.ndarray) -> np.ndarray:
    return _mech.solve_linear(system, rhs)


def solve_with_system(system: StructuralSystem, thermal_state: ThermalState) -> tuple[np.ndarray, np.ndarray]:
    return _mech.solve_with_system(system, thermal_state)


def solve_case(struct_params: StructuralParams, thermal_state: ThermalState) -> tuple[np.ndarray, np.ndarray]:
    return _mech.solve_case(struct_params, thermal_state)


def sample_19pt_deflection_rotation(ne_per_span: int, U: np.ndarray) -> dict[str, float]:
    return _mech.sample_19pt_deflection_rotation(ne_per_span, U)


def rotation_sampling_matrix(ne_per_span: int, rotation_keys: Sequence[str]) -> np.ndarray:
    return _mech.rotation_sampling_matrix(ne_per_span, rotation_keys)


def evaluate_forward(struct_params: StructuralParams, thermal_state: ThermalState) -> BridgeResponse:
    return _mech.evaluate_forward(struct_params, thermal_state)


def evaluate_forward_with_system(system: StructuralSystem, thermal_state: ThermalState) -> BridgeResponse:
    return _mech.evaluate_forward_with_system(system, thermal_state)


__all__ = [
    "StructuralParams",
    "ThermalState",
    "BridgeResponse",
    "MeshData",
    "StructuralSystem",
    "assemble_structural_system",
    "solve_with_system",
    "evaluate_forward_with_system",
    "evaluate_forward",
    "solve_case",
    "sample_19pt_deflection_rotation",
    "rotation_sampling_matrix",
    "TOTAL_LEN",
    "Q_UNIFORM",
    "set_span_lengths_mm",
    "get_span_lengths_mm",
    "span_breakpoints_mm",
    "total_length_mm",
    "solve_linear",
    "thermal_load_vector",
    "build_mesh",
]


def __getattr__(name: str):
    if name == "TOTAL_LEN":
        return _mech.total_length_mm()
    raise AttributeError(name)
