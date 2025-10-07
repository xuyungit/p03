"""Reusable bridge forward model for reaction and rotation evaluation.

This module extracts the core finite-element routine from
``data/bridge_reaction_sampling.py`` and fixes the support stiffness factors to
unity. It exposes a minimal API that maps structural parameters and thermal
states to reactions and sensor rotations, which can serve as the forward model
\(\mathcal{M}\) in the bias-estimation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve

# ----------------- 常量与几何参数 -----------------
E = 206_000.0                  # N/mm^2（钢）
Iy = 3.154279e11               # mm^4
EI_BASE = E * Iy               # N·mm^2

LMM = 40_000.0                 # 每跨长度 mm
SPAN_LEN = [LMM, LMM, LMM]
TOTAL_LEN = float(sum(SPAN_LEN))

# 均布恒载（向下为负，单位 N/mm）
Q_UNIFORM = -(12.0 + 20.0 + 29.64)

# 支座基准刚度（N/mm）
KV0 = 5_000.0 * 1_000.0

# 温差到曲率：κ = α·ΔT/h
ALPHA = 1.2e-5                 # 1/°C
H_MM = 2_000.0                 # mm


# ----------------- 数据结构 -----------------
@dataclass
class StructuralParams:
    """Parameters held constant within a measurement window."""

    settlements: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ei_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    kv_factors: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    uniform_load: float = Q_UNIFORM
    ne_per_span: int = 64


@dataclass
class ThermalState:
    """Per-sample temperature gradient parameters."""

    dT_spans: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BridgeResponse:
    """Outputs from a forward evaluation."""

    reactions_kN: np.ndarray
    sensor_map: Dict[str, float]
    displacements: np.ndarray


@dataclass
class MeshData:
    """Precomputed mesh description for a given discretisation."""

    x_coords: np.ndarray
    elements: Tuple[Tuple[int, int, float, int], ...]


@dataclass
class StructuralSystem:
    """Linear system assembled for fixed structural parameters."""

    struct_params: StructuralParams
    mesh: MeshData
    lu: np.ndarray
    piv: np.ndarray
    load_base: np.ndarray
    kv_values: np.ndarray
    settlements: np.ndarray
    support_nodes: Tuple[int, int, int, int]
    span_ei: Tuple[float, float, float]


# ----------------- 梁单元 FE 工具 -----------------
def k_beam(ei: float, L: float) -> np.ndarray:
    fac = ei / (L ** 3)
    L2 = L * L
    return fac * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2],
        ],
        dtype=float,
    )


def f_udl(q: float, L: float) -> np.ndarray:
    return np.array([q * L / 2.0, q * L * L / 12.0, q * L / 2.0, -q * L * L / 12.0], dtype=float)


def f_thermal(ei: float, kappa: float) -> np.ndarray:
    # 固定端弯矩等效：EIκ * [0, -1, 0, +1]
    return ei * kappa * np.array([0.0, -1.0, 0.0, +1.0], dtype=float)


@lru_cache(maxsize=None)
def build_mesh(ne_per_span: int) -> MeshData:
    elements: List[Tuple[int, int, float, int]] = []
    x = [0.0]
    for si, L in enumerate(SPAN_LEN):
        dx = L / ne_per_span
        for _ in range(ne_per_span):
            n1 = len(x) - 1
            n2 = n1 + 1
            elements.append((n1, n2, dx, si))
            x.append(x[-1] + dx)
    mesh = MeshData(x_coords=np.array(x, dtype=float), elements=tuple(elements))
    return mesh


# ----------------- 采样矩阵工具 -----------------
def _sensor_points_labels() -> Tuple[Tuple[float, ...], Tuple[str, ...]]:
    pts: List[float] = []
    labs: List[str] = []
    # A
    pts.append(0.0); labs.append("A")
    # S1
    for k in range(1, 6):
        pts.append(k / 6.0 * LMM)
        labs.append(f"S1-{k}/6L")
    # B
    pts.append(1.0 * LMM); labs.append("B")
    # S2
    for k in range(1, 6):
        pts.append(LMM + k / 6.0 * LMM)
        labs.append(f"S2-{k}/6L")
    # C
    pts.append(2.0 * LMM); labs.append("C")
    # S3
    for k in range(1, 6):
        pts.append(2 * LMM + k / 6.0 * LMM)
        labs.append(f"S3-{k}/6L")
    # D
    pts.append(3.0 * LMM); labs.append("D")
    return tuple(pts), tuple(labs)


def _find_element(x_coords: np.ndarray, x: float) -> Tuple[int, float, float]:
    if x >= x_coords[-1]:
        i = len(x_coords) - 2
    else:
        j = int(np.searchsorted(x_coords, x, side="right"))
        i = max(0, j - 1)
    xi = x_coords[i]
    L = x_coords[i + 1] - x_coords[i]
    return i, L, x - xi


@lru_cache(maxsize=None)
def _sampling_matrices(ne_per_span: int) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    mesh = build_mesh(ne_per_span)
    x_coords = mesh.x_coords
    ndof = 2 * len(x_coords)
    pts, labs = _sensor_points_labels()

    def_mat = np.zeros((len(labs), ndof), dtype=float)
    rot_mat = np.zeros((len(labs), ndof), dtype=float)

    for row, (x, lab) in enumerate(zip(pts, labs)):
        i, L, xl = _find_element(x_coords, x)
        dof = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
        xi = xl / L
        N1 = 1.0 - 3.0 * xi * xi + 2.0 * xi * xi * xi
        N2 = L * (xi - 2.0 * xi * xi + xi * xi * xi)
        N3 = 3.0 * xi * xi - 2.0 * xi * xi * xi
        N4 = L * (-xi * xi + xi * xi * xi)
        dN1 = (-6.0 * xi + 6.0 * xi * xi) / L
        dN2 = 1.0 - 4.0 * xi + 3.0 * xi * xi
        dN3 = (6.0 * xi - 6.0 * xi * xi) / L
        dN4 = -2.0 * xi + 3.0 * xi * xi

        def_mat[row, dof] = [N1, N2, N3, N4]
        rot_mat[row, dof] = [dN1, dN2, dN3, dN4]

    rotation_keys = tuple(f"theta_{lab.replace('/', '_')}_rad" for lab in labs)
    return rotation_keys, def_mat, rot_mat


# ----------------- 结构系统缓存 -----------------
def _support_nodes(mesh: MeshData) -> Tuple[int, int, int, int]:
    support_x = [0.0, LMM, 2 * LMM, 3 * LMM]
    nodes: List[int] = []
    for xm in support_x:
        idx = int(np.argmin(np.abs(mesh.x_coords - xm)))
        nodes.append(idx)
    return tuple(nodes)


def assemble_structural_system(struct_params: StructuralParams) -> StructuralSystem:
    """Precompute stiffness factorisation and base loads for given parameters."""

    mesh = build_mesh(struct_params.ne_per_span)
    ndof = 2 * len(mesh.x_coords)
    K = np.zeros((ndof, ndof), dtype=float)
    F_base = np.zeros(ndof, dtype=float)

    span_ei = tuple(EI_BASE * float(fac) for fac in struct_params.ei_factors)

    for n1, n2, L, si in mesh.elements:
        ei = span_ei[si]
        ke = k_beam(ei, L)
        fe = f_udl(struct_params.uniform_load, L)
        dof = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        K[np.ix_(dof, dof)] += ke
        F_base[dof] += fe

    support_nodes = _support_nodes(mesh)
    settlements = np.asarray(struct_params.settlements, dtype=float)
    kv_scale = np.asarray(struct_params.kv_factors, dtype=float)
    if kv_scale.shape != (4,):
        raise ValueError("kv_factors must contain four scale values for supports A-D.")
    Kv = KV0 * kv_scale
    for i, idx in enumerate(support_nodes):
        K[2 * idx, 2 * idx] += Kv[i]
        F_base[2 * idx] += Kv[i] * settlements[i]

    lu, piv = lu_factor(K)

    return StructuralSystem(
        struct_params=struct_params,
        mesh=mesh,
        lu=lu,
        piv=piv,
        load_base=F_base,
        kv_values=Kv,
        settlements=settlements,
        support_nodes=support_nodes,
        span_ei=span_ei,
    )


def thermal_load_vector(system: StructuralSystem, thermal_state: ThermalState) -> np.ndarray:
    loads = np.zeros_like(system.load_base)
    kappa_span = tuple((ALPHA / H_MM) * np.array(thermal_state.dT_spans))
    for n1, n2, L, si in system.mesh.elements:
        ei = system.span_ei[si]
        fth = f_thermal(ei, kappa_span[si])
        dof = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        loads[dof] += fth
    return loads


def solve_with_system(system: StructuralSystem, thermal_state: ThermalState) -> Tuple[np.ndarray, np.ndarray]:
    th = thermal_load_vector(system, thermal_state)
    F = system.load_base - th
    U = lu_solve((system.lu, system.piv), F)

    reactions = []
    for i, idx in enumerate(system.support_nodes):
        ui = float(U[2 * idx])
        Ri = -system.kv_values[i] * (ui - system.settlements[i]) / 1_000.0  # kN
        reactions.append(Ri)

    return U, np.array(reactions, dtype=float)


# ----------------- 前向求解 -----------------
def solve_case(struct_params: StructuralParams, thermal_state: ThermalState) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the bridge response for the given structural parameters and thermal state."""

    system = assemble_structural_system(struct_params)
    return solve_with_system(system, thermal_state)


def sample_19pt_deflection_rotation(ne_per_span: int, U: np.ndarray) -> Dict[str, float]:
    """Match the 19 labelled sensor points and return deflection/rotation values."""

    _, def_mat, rot_mat = _sampling_matrices(ne_per_span)
    displacements = np.asarray(U, dtype=float).reshape(-1)
    deflections = def_mat @ displacements
    rotations = rot_mat @ displacements

    _, labs = _sensor_points_labels()
    out: Dict[str, float] = {}
    for idx, lab in enumerate(labs):
        safe_lab = lab.replace("/", "_")
        out[f"v_{safe_lab}_mm"] = float(deflections[idx])
        out[f"theta_{safe_lab}_rad"] = float(rotations[idx])
    return out


def rotation_sampling_matrix(ne_per_span: int, rotation_keys: Sequence[str]) -> np.ndarray:
    """Return a dense matrix mapping nodal DOFs to requested rotation channels."""

    base_keys, _, rot_mat = _sampling_matrices(ne_per_span)
    key_to_idx = {key: i for i, key in enumerate(base_keys)}
    indices = [key_to_idx[k] for k in rotation_keys]
    return rot_mat[indices]


def evaluate_forward(struct_params: StructuralParams, thermal_state: ThermalState) -> BridgeResponse:
    """Convenience wrapper returning reactions and 19-point sensor outputs."""

    U, reactions = solve_case(struct_params, thermal_state)
    sensor_map = sample_19pt_deflection_rotation(struct_params.ne_per_span, U)
    return BridgeResponse(reactions_kN=reactions, sensor_map=sensor_map, displacements=U)


def evaluate_forward_with_system(system: StructuralSystem, thermal_state: ThermalState) -> BridgeResponse:
    """Evaluate the forward model using a pre-assembled structural system."""

    U, reactions = solve_with_system(system, thermal_state)
    sensor_map = sample_19pt_deflection_rotation(system.struct_params.ne_per_span, U)
    return BridgeResponse(reactions_kN=reactions, sensor_map=sensor_map, displacements=U)


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
]
