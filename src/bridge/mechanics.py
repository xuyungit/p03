"""Shared bridge mechanics utilities for sampling and forward solves.

This module centralises the finite-element assembly, geometry utilities,
and sampling helpers so that both the data-generation scripts and the
learning-time forward models stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:  # SciPy is optional; fall back to NumPy solves when unavailable.
    from scipy.linalg import lu_factor, lu_solve  # type: ignore
except Exception:  # pragma: no cover - SciPy missing in lightweight envs
    lu_factor = None  # type: ignore
    lu_solve = None  # type: ignore


# ----------------- 常量与几何参数（可按需参数化） -----------------
E = 206_000.0                  # N/mm^2（钢）
Iy = 3.154279e11               # mm^4
EI_BASE = E * Iy               # N·mm^2

DEFAULT_SPAN_MM = 40_000.0
_SPAN_LENGTHS_MM: Tuple[float, float, float] = (
    DEFAULT_SPAN_MM,
    DEFAULT_SPAN_MM,
    DEFAULT_SPAN_MM,
)
_TOTAL_LENGTH_MM: float = float(sum(_SPAN_LENGTHS_MM))

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
    settlements: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ei_factors: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    kv_factors: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    uniform_load: float = Q_UNIFORM
    ne_per_span: int = 64


@dataclass
class ThermalState:
    dT_spans: Tuple[float, ...] = (0.0, 0.0, 0.0)


@dataclass
class BridgeResponse:
    reactions_kN: np.ndarray
    sensor_map: Dict[str, float]
    displacements: np.ndarray


@dataclass(frozen=True)
class MeshData:
    x_coords: np.ndarray
    elements: Tuple[Tuple[int, int, float, int], ...]


@dataclass
class StructuralSystem:
    struct_params: StructuralParams
    mesh: MeshData
    stiffness: np.ndarray
    load_base: np.ndarray
    kv_values: np.ndarray
    settlements: np.ndarray
    support_nodes: Tuple[int, int, int, int]
    span_ei: Tuple[float, float, float]
    lu: np.ndarray | None
    piv: np.ndarray | None


# ----------------- 几何工具 -----------------
def get_span_lengths_mm() -> Tuple[float, float, float]:
    return _SPAN_LENGTHS_MM


def set_span_lengths_mm(lengths_mm: Sequence[float]) -> None:
    if len(lengths_mm) != 3:
        raise ValueError("span_lengths must contain exactly three values for A-B, B-C, C-D spans")
    lengths = tuple(float(L) for L in lengths_mm)
    if any(L <= 0.0 for L in lengths):
        raise ValueError("span_lengths must be positive")
    global _SPAN_LENGTHS_MM, _TOTAL_LENGTH_MM
    _SPAN_LENGTHS_MM = lengths
    _TOTAL_LENGTH_MM = float(sum(lengths))
    _mesh_cache.clear()
    _sampling_cache.clear()


def span_breakpoints_mm() -> Tuple[float, float, float, float]:
    pos = [0.0]
    for L in _SPAN_LENGTHS_MM:
        pos.append(pos[-1] + L)
    return tuple(pos)  # A, B, C, D


def total_length_mm() -> float:
    return _TOTAL_LENGTH_MM


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
    return ei * kappa * np.array([0.0, -1.0, 0.0, +1.0], dtype=float)


_mesh_cache: Dict[Tuple[int, Tuple[float, float, float]], MeshData] = {}


def build_mesh(ne_per_span: int, span_lengths: Sequence[float] | None = None) -> MeshData:
    spans = tuple(float(L) for L in (span_lengths or _SPAN_LENGTHS_MM))
    key = (int(ne_per_span), spans)
    if key in _mesh_cache:
        return _mesh_cache[key]

    elements: List[Tuple[int, int, float, int]] = []
    x = [0.0]
    for si, L in enumerate(spans):
        dx = L / ne_per_span
        for _ in range(ne_per_span):
            n1 = len(x) - 1
            n2 = n1 + 1
            elements.append((n1, n2, dx, si))
            x.append(x[-1] + dx)

    mesh = MeshData(x_coords=np.array(x, dtype=float), elements=tuple(elements))
    _mesh_cache[key] = mesh
    return mesh


# ----------------- 采样矩阵工具 -----------------
def _sensor_points_labels(span_lengths: Sequence[float]) -> Tuple[Tuple[float, ...], Tuple[str, ...]]:
    pts: List[float] = []
    labs: List[str] = []
    support_labels = ["A", "B", "C", "D"]
    support_positions = span_breakpoints_mm()
    pts.append(support_positions[0]); labs.append(support_labels[0])
    for si, span_len in enumerate(span_lengths):
        start = support_positions[si]
        for k in range(1, 6):
            pts.append(start + k / 6.0 * span_len)
            labs.append(f"S{si + 1}-{k}/6L")
        pts.append(support_positions[si + 1])
        labs.append(support_labels[si + 1])
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


_sampling_cache: Dict[Tuple[int, Tuple[float, float, float]], Tuple[Tuple[str, ...], np.ndarray, np.ndarray]] = {}


def sampling_matrices(ne_per_span: int, span_lengths: Sequence[float] | None = None) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    spans = tuple(float(L) for L in (span_lengths or _SPAN_LENGTHS_MM))
    key = (int(ne_per_span), spans)
    cached = _sampling_cache.get(key)
    if cached is not None:
        return cached

    mesh = build_mesh(ne_per_span, spans)
    x_coords = mesh.x_coords
    ndof = 2 * len(x_coords)
    pts, labs = _sensor_points_labels(spans)

    def_mat = np.zeros((len(labs), ndof), dtype=float)
    rot_mat = np.zeros((len(labs), ndof), dtype=float)

    for row, (x, _lab) in enumerate(zip(pts, labs)):
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

    result = (labs, def_mat, rot_mat)
    _sampling_cache[key] = result
    return result


# ----------------- 结构系统装配 -----------------
def _support_nodes(mesh: MeshData) -> Tuple[int, int, int, int]:
    support_positions = span_breakpoints_mm()
    nodes: List[int] = []
    for xm in support_positions:
        idx = int(np.argmin(np.abs(mesh.x_coords - xm)))
        nodes.append(idx)
    return tuple(nodes)


def assemble_structural_system(struct_params: StructuralParams) -> StructuralSystem:
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

    if lu_factor is not None:
        lu, piv = lu_factor(K)
    else:  # pragma: no cover - SciPy missing
        lu = None
        piv = None

    return StructuralSystem(
        struct_params=struct_params,
        mesh=mesh,
        stiffness=K,
        load_base=F_base,
        kv_values=Kv,
        settlements=settlements,
        support_nodes=support_nodes,
        span_ei=span_ei,
        lu=lu,
        piv=piv,
    )


def thermal_load_vector(system: StructuralSystem, thermal_state: ThermalState) -> np.ndarray:
    loads = np.zeros_like(system.load_base)

    dT = tuple(thermal_state.dT_spans)
    nvals = len(dT)
    scale = (ALPHA / H_MM)

    if nvals == 3:
        kappa_span = tuple(scale * np.array(dT))
        for n1, n2, _L, si in system.mesh.elements:
            ei = system.span_ei[si]
            fth = f_thermal(ei, kappa_span[si])
            dof = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            loads[dof] += fth
        return loads

    if nvals == 2:
        x_coords = system.mesh.x_coords
        half = total_length_mm() / 2.0
        kappa_left = scale * dT[0]
        kappa_right = scale * dT[1]
        for n1, n2, _L, si in system.mesh.elements:
            x_mid = 0.5 * (x_coords[n1] + x_coords[n2])
            kappa = kappa_left if x_mid < half else kappa_right
            ei = system.span_ei[si]
            fth = f_thermal(ei, kappa)
            dof = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            loads[dof] += fth
        return loads

    if nvals == 1:
        kappa = scale * dT[0]
        for n1, n2, _L, si in system.mesh.elements:
            ei = system.span_ei[si]
            fth = f_thermal(ei, kappa)
            dof = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            loads[dof] += fth
        return loads

    raise ValueError(
        f"Unsupported number of temperature values: {nvals}. Provide 3 (per span) or 2 (half/half)."
    )


def solve_linear(system: StructuralSystem, rhs: np.ndarray) -> np.ndarray:
    if system.lu is not None and lu_solve is not None:
        return lu_solve((system.lu, system.piv), rhs)
    return np.linalg.solve(system.stiffness, rhs)


def solve_with_system(system: StructuralSystem, thermal_state: ThermalState) -> Tuple[np.ndarray, np.ndarray]:
    th = thermal_load_vector(system, thermal_state)
    F = system.load_base - th
    U = solve_linear(system, F)

    reactions = []
    for i, idx in enumerate(system.support_nodes):
        ui = float(U[2 * idx])
        Ri = -system.kv_values[i] * (ui - system.settlements[i]) / 1_000.0  # kN
        reactions.append(Ri)

    return U, np.array(reactions, dtype=float)


def solve_case(struct_params: StructuralParams, thermal_state: ThermalState) -> Tuple[np.ndarray, np.ndarray]:
    system = assemble_structural_system(struct_params)
    return solve_with_system(system, thermal_state)


def sample_19pt_deflection_rotation(ne_per_span: int, U: np.ndarray) -> Dict[str, float]:
    labs, def_mat, rot_mat = sampling_matrices(ne_per_span)
    displacements = np.asarray(U, dtype=float).reshape(-1)
    deflections = def_mat @ displacements
    rotations = rot_mat @ displacements

    out: Dict[str, float] = {}
    for idx, lab in enumerate(labs):
        safe_lab = lab.replace("/", "_")
        out[f"v_{safe_lab}_mm"] = float(deflections[idx])
        out[f"theta_{safe_lab}_rad"] = float(rotations[idx])
    return out


def rotation_sampling_matrix(ne_per_span: int, rotation_keys: Sequence[str]) -> np.ndarray:
    labs, _, rot_mat = sampling_matrices(ne_per_span)
    key_to_idx = {
        f"theta_{lab.replace('/', '_')}_rad": i
        for i, lab in enumerate(labs)
    }
    try:
        indices = [key_to_idx[k] for k in rotation_keys]
    except KeyError as exc:  # pragma: no cover - defensive path for mis-specified keys
        missing = str(exc.args[0])
        raise KeyError(f"Rotation key not found in sampling matrix: {missing}") from exc
    return rot_mat[indices]


def evaluate_forward(struct_params: StructuralParams, thermal_state: ThermalState) -> BridgeResponse:
    U, reactions = solve_case(struct_params, thermal_state)
    sensor_map = sample_19pt_deflection_rotation(struct_params.ne_per_span, U)
    return BridgeResponse(reactions_kN=reactions, sensor_map=sensor_map, displacements=U)


def evaluate_forward_with_system(system: StructuralSystem, thermal_state: ThermalState) -> BridgeResponse:
    U, reactions = solve_with_system(system, thermal_state)
    sensor_map = sample_19pt_deflection_rotation(system.struct_params.ne_per_span, U)
    return BridgeResponse(reactions_kN=reactions, sensor_map=sensor_map, displacements=U)


__all__ = [
    "E",
    "Iy",
    "EI_BASE",
    "Q_UNIFORM",
    "KV0",
    "ALPHA",
    "H_MM",
    "StructuralParams",
    "ThermalState",
    "BridgeResponse",
    "MeshData",
    "StructuralSystem",
    "get_span_lengths_mm",
    "set_span_lengths_mm",
    "span_breakpoints_mm",
    "total_length_mm",
    "build_mesh",
    "sampling_matrices",
    "assemble_structural_system",
    "thermal_load_vector",
    "solve_linear",
    "solve_with_system",
    "solve_case",
    "sample_19pt_deflection_rotation",
    "rotation_sampling_matrix",
    "evaluate_forward",
    "evaluate_forward_with_system",
]
