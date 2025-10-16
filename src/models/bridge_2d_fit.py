#!/usr/bin/env python3
"""Further optimized multi-case fitting with vectorized operations and flexible measurements.

Additional optimizations over fit_multi_case_optimized.py:
1. Batch thermal load computation - vectorize temperature load vectors
2. Batch solve - solve all cases simultaneously using BLAS Level 3
3. Vectorized regularization computation
4. Pre-allocate arrays to reduce memory allocations
5. Support for multiple measurement types (reactions, displacements, rotations, span rotations)

Expected performance improvement: 1.5-2x over fit_multi_case_optimized.py

Usage:
    # Basic usage (默认固定 settlement_a = 0.0 作为参考点，仅使用反力)
    uv run python src/models/fit_multi_case_v2.py \
        --data data/augmented/dt_24hours_data_4.csv \
        --max-samples 100 \
        --maxiter 500
    
    # Multiple files (will be concatenated with temporal smoothness across boundaries)
    uv run python src/models/fit_multi_case_v2.py \
        --data data/augmented/dt_24hours_data_new.csv \
        --data data/augmented/dt_24hours_data_new2.csv \
        --maxiter 500
    
    # Fixed KV factors (not fitted, use given values)
    uv run python src/models/fit_multi_case_v2.py \
        --data data/augmented/dt_24hours_data_4.csv \
        --fixed-kv 1.0 1.0 1.0 1.0 \
        --maxiter 500
    
    # 不固定第一个沉降（拟合全部4个沉降）
    uv run python src/models/fit_multi_case_v2.py \
        --data data/augmented/dt_24hours_data_4.csv \
        --no-fix-first-settlement \
        --maxiter 500
    
    # 使用位移和转角测量值（权重可调）
    uv run python src/models/fit_multi_case_v2.py \
        --data data/augmented/dt_24hours_data_new2.csv \
        --use-displacements \
        --use-rotations \
        --displacement-weight 10.0 \
        --rotation-weight 1000.0 \
        --maxiter 500
    
    # 使用跨中转角测量值增强鲁棒性
    uv run python src/models/fit_multi_case_v2.py \
        --data data/augmented/dt_24hours_data_new2_round4.csv \
        --use-rotations \
        --use-span-rotations \
        --rotation-weight 100.0 \
        --span-rotation-weight 50.0 \
        --maxiter 500
    
    # 仅使用位移（不使用反力和转角）
    uv run python src/models/fit_multi_case_v2.py \
        --data data/augmented/dt_24hours_data_new2.csv \
        --use-displacements \
        --maxiter 500
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from models.bridge_forward_model import (
    StructuralParams,
    StructuralSystem,
    ThermalState,
    assemble_structural_system,
    set_span_lengths_mm,
    solve_linear,
    ALPHA,
    H_MM,
)


# ============================================================================
# Constants and Configuration Classes
# ============================================================================

class ColumnNames:
    """数据列名常量"""
    # 测量值列名
    REACTIONS = ['R_a_kN', 'R_b_kN', 'R_c_kN', 'R_d_kN']
    DISPLACEMENTS = ['v_A_mm', 'v_B_mm', 'v_C_mm', 'v_D_mm']
    ROTATIONS = ['theta_A_rad', 'theta_B_rad', 'theta_C_rad', 'theta_D_rad']
    SPAN_ROTATIONS = ['theta_S1-5_6L_rad', 'theta_S2-4_6L_rad', 'theta_S3-5_6L_rad']
    
    # 温度列名
    TEMPS_3_SPAN = ['dT_s1_C', 'dT_s2_C', 'dT_s3_C']
    TEMPS_2_SEG = ['dT_left_C', 'dT_right_C']
    
    # 结构参数列名
    SPAN_LENGTHS = ['span_length_s1_mm', 'span_length_s2_mm', 'span_length_s3_mm']
    SETTLEMENTS = ['settlement_a_mm', 'settlement_b_mm', 'settlement_c_mm', 'settlement_d_mm']
    EI_FACTORS = ['ei_factor_s1', 'ei_factor_s2', 'ei_factor_s3']
    KV_FACTORS = ['kv_factor_a', 'kv_factor_b', 'kv_factor_c', 'kv_factor_d']
    
    # 其他
    UNIFORM_LOAD = 'uniform_load_N_per_mm'
    SAMPLE_ID = 'sample_id'


@dataclass
class OptimizationConfig:
    """优化配置参数"""
    # 参数边界
    settlement_lower: float = -20.0
    settlement_upper: float = 20.0
    ei_factor_lower: float = 0.3
    ei_factor_upper: float = 1.5
    kv_factor_lower: float = 0.1
    kv_factor_upper: float = 3.0
    temp_gradient_lower: float = -10.0
    temp_gradient_upper: float = 20.0
    temp_gradient_initial: float = 10.0
    
    # 正则化阈值
    temp_spatial_diff_thresh: float = 3.0
    temp_temporal_diff_thresh: float = 1.0
    
    # 优化器设置
    ftol: float = 1e-8
    gtol: float = 1e-8
    xtol: float = 1e-12
    
    # 数值稳定性
    min_scale_threshold: float = 1e-10


@dataclass
class MeasurementConfig:
    """Configuration for which measurements to use in fitting.
    
    Attributes:
        use_reactions: Use support reactions (R_a, R_b, R_c, R_d)
        use_displacements: Use support vertical displacements (v_A, v_B, v_C, v_D)
        use_rotations: Use support rotations (theta_A, theta_B, theta_C, theta_D)
        use_span_rotations: Use span rotations (theta_S1-5_6L, theta_S2-4_6L, theta_S3-5_6L)
        displacement_weight: Weight for displacement residuals relative to reactions
        rotation_weight: Weight for rotation residuals relative to reactions
        span_rotation_weight: Weight for span rotation residuals relative to reactions
        auto_normalize: Automatically normalize residuals by measurement std (recommended)
    """
    use_reactions: bool = True
    use_displacements: bool = False
    use_rotations: bool = False
    use_span_rotations: bool = False
    displacement_weight: float = 1.0
    rotation_weight: float = 1.0
    span_rotation_weight: float = 1.0
    auto_normalize: bool = True  # 自动归一化（推荐）
    
    def __post_init__(self):
        if not any([self.use_reactions, self.use_displacements, self.use_rotations, self.use_span_rotations]):
            raise ValueError("At least one measurement type must be enabled")
    
    def count_measurements_per_case(self) -> int:
        """Count number of measurements per case."""
        count = 0
        if self.use_reactions:
            count += 4  # 4 support reactions
        if self.use_displacements:
            count += 4  # 4 support displacements
        if self.use_rotations:
            count += 4  # 4 support rotations
        if self.use_span_rotations:
            count += 3  # 3 span rotations (S1-5/6L, S2-4/6L, S3-5/6L)
        return count
    
    def describe(self) -> str:
        """Return human-readable description."""
        parts = []
        if self.use_reactions:
            parts.append("反力(4)")
        if self.use_displacements:
            parts.append(f"位移(4, 权重={self.displacement_weight})")
        if self.use_rotations:
            parts.append(f"转角(4, 权重={self.rotation_weight})")
        if self.use_span_rotations:
            parts.append(f"跨中转角(3, 权重={self.span_rotation_weight})")
        
        if self.auto_normalize:
            parts.append("自动归一化")
        
        return " + ".join(parts)


def load_multi_case_data(
    csv_paths: List[Path], 
    max_samples: int | None = None,
    measurement_config: Optional[MeasurementConfig] = None,
) -> Tuple[pd.DataFrame, dict, MeasurementConfig]:
    """Load and concatenate multi-case data from multiple files.
    
    Args:
        csv_paths: List of CSV file paths to load and concatenate
        max_samples: Optional limit on total samples after concatenation
        measurement_config: Optional measurement configuration. If None, auto-detect from data.
    
    Returns:
        Combined DataFrame, constant structural parameters, and measurement configuration
    """
    dfs = []
    
    print(f"\n加载 {len(csv_paths)} 个数据文件:")
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        print(f"  文件 {i+1}: {csv_path.name} ({len(df)} 条数据)")
        dfs.append(df)
    
    # Concatenate all dataframes (once)
    df = pd.concat(dfs, ignore_index=True)
    
    # Reset sample_id to be continuous across all files
    df['sample_id'] = range(len(df))
    
    print(f"\n合并后数据集:")
    print(f"  总样本数: {len(df)}")
    
    if max_samples is not None and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()
        df['sample_id'] = range(len(df))
        print(f"  限制到前 {max_samples} 条数据")
    
    # Auto-detect available measurements if config not provided
    if measurement_config is None:
        has_reactions = all(col in df.columns for col in ColumnNames.REACTIONS)
        has_displacements = all(col in df.columns for col in ColumnNames.DISPLACEMENTS)
        has_rotations = all(col in df.columns for col in ColumnNames.ROTATIONS)
        has_span_rotations = all(col in df.columns for col in ColumnNames.SPAN_ROTATIONS)
        
        print(f"\n可用测量数据:")
        print(f"  反力 (R_a~R_d): {'✓' if has_reactions else '✗'}")
        print(f"  位移 (v_A~v_D): {'✓' if has_displacements else '✗'}")
        print(f"  转角 (theta_A~theta_D): {'✓' if has_rotations else '✗'}")
        print(f"  跨中转角 (S1-5/6L, S2-4/6L, S3-5/6L): {'✓' if has_span_rotations else '✗'}")
        
        # Default: use only reactions
        measurement_config = MeasurementConfig(
            use_reactions=has_reactions,
            use_displacements=False,
            use_rotations=False,
            use_span_rotations=False,
        )
        print(f"\n默认使用: 仅反力")
    
    has_span_cols = all(col in df.columns for col in ColumnNames.SPAN_LENGTHS)
    if has_span_cols:
        spans_mm = tuple(float(df[col].iloc[0]) for col in ColumnNames.SPAN_LENGTHS)
        for i, col in enumerate(ColumnNames.SPAN_LENGTHS):
            if not np.isclose(df[col], spans_mm[i]).all():
                raise ValueError("span_length_x_mm 列应为常数")
        set_span_lengths_mm(spans_mm)
        print(
            "  侦测到跨度配置: "
            f"S1={spans_mm[0]/1000:.2f} m, S2={spans_mm[1]/1000:.2f} m, S3={spans_mm[2]/1000:.2f} m"
        )
    else:
        print("  未检测到跨度列，使用默认 40:40:40 m")

    # Verify structural parameters are constant
    struct_cols = (
        ColumnNames.SETTLEMENTS +
        ColumnNames.EI_FACTORS +
        ColumnNames.KV_FACTORS
    )
    if has_span_cols:
        struct_cols.extend(ColumnNames.SPAN_LENGTHS)
    
    const_params = {}
    for col in struct_cols:
        unique_vals = df[col].unique()
        if len(unique_vals) != 1:
            print(f"  警告: {col} 不是常数，有 {len(unique_vals)} 个不同值")
        const_params[col] = float(df[col].iloc[0])
    
    print(f"  结构参数验证: {'✓ 全部为常数' if all(len(df[c].unique()) == 1 for c in struct_cols) else '✗ 存在变化'}")
    
    return df, const_params, measurement_config


def f_thermal_element(ei: float, kappa: float) -> np.ndarray:
    """Thermal equivalent load for a single element."""
    return np.array([0.0, ei * kappa, 0.0, -ei * kappa])


def batch_thermal_load_vectors(
    system: StructuralSystem,
    dT_matrix: np.ndarray,  # shape: (n_cases, 2 or 3)
) -> np.ndarray:
    """Vectorized computation of thermal load vectors for all cases.

    Supports both 3-span (per-span) and 2-segment (50/50) temperature inputs.

    Args:
        system: Structural system
        dT_matrix: Temperature gradients, shape (n_cases, 2 or 3)

    Returns:
        Load vectors, shape (n_cases, ndof)
    """
    n_cases, nT = dT_matrix.shape
    ndof = len(system.load_base)

    # Pre-allocate result
    loads_all = np.zeros((n_cases, ndof), dtype=float)

    # Convert temperatures to curvatures
    kappa_matrix = (ALPHA / H_MM) * dT_matrix

    # Iterate over elements (assembly is inherently sparse)
    x_coords = system.mesh.x_coords
    half = x_coords[-1] / 2.0
    for n1, n2, L, si in system.mesh.elements:
        ei = system.span_ei[si]
        if nT == 3:
            kappa_cases = kappa_matrix[:, si]
        elif nT == 2:
            x_mid = 0.5 * (x_coords[n1] + x_coords[n2])
            idx = 0 if x_mid < half else 1
            kappa_cases = kappa_matrix[:, idx]
        else:
            # Treat as uniform across bridge if a single column is supplied
            kappa_cases = kappa_matrix[:, 0]

        dof_indices = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        loads_all[:, dof_indices[1]] -= ei * kappa_cases  # node 1 moment
        loads_all[:, dof_indices[3]] += ei * kappa_cases  # node 2 moment

    return loads_all


def batch_solve_with_system(
    system: StructuralSystem,
    dT_matrix: np.ndarray,  # shape: (n_cases, 2 or 3)
    rot_mat_span_cache: Optional[np.ndarray] = None,  # Pre-computed rotation matrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch solve for all cases simultaneously.
    
    Args:
        system: Structural system
        dT_matrix: Temperature gradients, shape (n_cases, 2 or 3)
        rot_mat_span_cache: Optional pre-computed rotation sampling matrix for performance
    
    Returns:
        U_all: Displacements, shape (n_cases, ndof)
        reactions_all: Reactions, shape (n_cases, 4)
        displacements_all: Support vertical displacements, shape (n_cases, 4)
        rotations_all: Support rotations, shape (n_cases, 4)
        span_rotations_all: Span rotations at 3 locations, shape (n_cases, 3)
                           [theta_S1-5_6L, theta_S2-4_6L, theta_S3-5_6L]
    """
    n_cases = dT_matrix.shape[0]
    
    # Batch compute thermal loads: (n_cases, ndof)
    thermal_loads = batch_thermal_load_vectors(system, dT_matrix)
    
    # Compute right-hand sides: (n_cases, ndof)
    F_all = system.load_base[np.newaxis, :] - thermal_loads
    
    # Batch solve using shared linear solver (SciPy if available, otherwise NumPy)
    U_all = solve_linear(system, F_all.T).T  # shape: (n_cases, ndof)
    
    # Extract reactions, displacements, and rotations for all cases
    reactions_all = np.zeros((n_cases, 4), dtype=float)
    displacements_all = np.zeros((n_cases, 4), dtype=float)
    rotations_all = np.zeros((n_cases, 4), dtype=float)
    
    for i, idx in enumerate(system.support_nodes):
        # Vertical displacement at support
        ui_all = U_all[:, 2 * idx]
        displacements_all[:, i] = ui_all
        
        # Rotation at support
        theta_all = U_all[:, 2 * idx + 1]
        rotations_all[:, i] = theta_all
        
        # Reaction force
        reactions_all[:, i] = -system.kv_values[i] * (ui_all - system.settlements[i]) / 1_000.0
    
    # Extract span rotations using pre-computed rotation sampling matrix (if provided)
    # Apply rotation matrix to all displacement vectors: (3, ndof) @ (ndof, n_cases) -> (3, n_cases)
    if rot_mat_span_cache is not None:
        span_rotations_all = (rot_mat_span_cache @ U_all.T).T  # shape: (n_cases, 3)
    else:
        # Fallback: compute on-the-fly (should not happen in optimized path)
        from models.bridge_forward_model import rotation_sampling_matrix
        span_rotation_keys = [
            'theta_S1-5_6L_rad',
            'theta_S2-4_6L_rad',
            'theta_S3-5_6L_rad',
        ]
        rot_mat_span = rotation_sampling_matrix(system.struct_params.ne_per_span, span_rotation_keys)
        span_rotations_all = (rot_mat_span @ U_all.T).T  # shape: (n_cases, 3)
    
    return U_all, reactions_all, displacements_all, rotations_all, span_rotations_all
    
    return U_all, reactions_all, displacements_all, rotations_all, span_rotations_all


class VectorizedMultiCaseFitter:
    """Highly optimized fitter with vectorized operations."""
    
    def __init__(
        self,
        reactions_matrix: np.ndarray,  # shape: (n_cases, 4)
        uniform_load: float,
        ne_per_span: int = 64,
        temp_spatial_weight: float = 1.0,
        temp_temporal_weight: float = 1.0,
        fixed_kv_factors: tuple[float, float, float, float] | None = None,
        fix_first_settlement: bool = True,
        temp_segments: int = 3,
        measurement_config: Optional[MeasurementConfig] = None,
        displacements_matrix: Optional[np.ndarray] = None,  # shape: (n_cases, 4)
        rotations_matrix: Optional[np.ndarray] = None,  # shape: (n_cases, 4)
        span_rotations_matrix: Optional[np.ndarray] = None,  # shape: (n_cases, 3)
        opt_config: Optional[OptimizationConfig] = None,
    ):
        """
        Args:
            reactions_matrix: Measured reactions, shape (n_cases, 4)
            uniform_load: Uniform load in N/mm
            ne_per_span: Number of elements per span
            temp_spatial_weight: Weight for spatial temperature smoothness
            temp_temporal_weight: Weight for temporal temperature smoothness
            fixed_kv_factors: If provided, use these fixed KV values instead of fitting
            fix_first_settlement: If True, fix settlement_a to 0.0 as reference (default: True)
            temp_segments: Number of temperature segments (2 or 3)
            measurement_config: Configuration for which measurements to use
            displacements_matrix: Measured displacements, shape (n_cases, 4) if using
            rotations_matrix: Measured rotations, shape (n_cases, 4) if using
            span_rotations_matrix: Measured span rotations, shape (n_cases, 3) if using
                                  [theta_S1-5_6L, theta_S2-4_6L, theta_S3-5_6L]
            opt_config: Optimization configuration (uses defaults if None)
        """
        self.reactions_matrix = reactions_matrix
        self.n_cases = reactions_matrix.shape[0]
        self.uniform_load = uniform_load
        self.ne_per_span = ne_per_span
        self.temp_spatial_weight = temp_spatial_weight
        self.temp_temporal_weight = temp_temporal_weight
        self.fixed_kv_factors = fixed_kv_factors
        self.fix_first_settlement = fix_first_settlement
        if temp_segments not in (2, 3):
            raise ValueError("temp_segments must be 2 or 3")
        self.temp_segments = temp_segments
        
        # Optimization configuration
        self.opt_config = opt_config if opt_config is not None else OptimizationConfig()
        
        # Measurement configuration
        if measurement_config is None:
            measurement_config = MeasurementConfig(use_reactions=True)
        self.measurement_config = measurement_config
        
        # Store measurement matrices
        self.displacements_matrix = displacements_matrix
        self.rotations_matrix = rotations_matrix
        self.span_rotations_matrix = span_rotations_matrix
        
        # Validate measurement configuration
        if measurement_config.use_displacements and displacements_matrix is None:
            raise ValueError("displacements_matrix required when use_displacements=True")
        if measurement_config.use_rotations and rotations_matrix is None:
            raise ValueError("rotations_matrix required when use_rotations=True")
        if measurement_config.use_span_rotations and span_rotations_matrix is None:
            raise ValueError("span_rotations_matrix required when use_span_rotations=True")
        
        # Compute normalization scales (standard deviations) for auto-normalization
        self.reaction_scale = np.std(reactions_matrix) if measurement_config.auto_normalize else 1.0
        self.displacement_scale = (
            np.std(displacements_matrix) if (measurement_config.auto_normalize and displacements_matrix is not None) 
            else 1.0
        )
        self.rotation_scale = (
            np.std(rotations_matrix) if (measurement_config.auto_normalize and rotations_matrix is not None) 
            else 1.0
        )
        self.span_rotation_scale = (
            np.std(span_rotations_matrix) if (measurement_config.auto_normalize and span_rotations_matrix is not None) 
            else 1.0
        )
        
        # Avoid division by zero
        if self.reaction_scale < self.opt_config.min_scale_threshold:
            self.reaction_scale = 1.0
        if self.displacement_scale < self.opt_config.min_scale_threshold:
            self.displacement_scale = 1.0
        if self.rotation_scale < self.opt_config.min_scale_threshold:
            self.rotation_scale = 1.0
        if self.span_rotation_scale < self.opt_config.min_scale_threshold:
            self.span_rotation_scale = 1.0
        
        # Cache
        self._cached_system: StructuralSystem | None = None
        self._cached_struct_params: tuple | None = None
        
        # Pre-compute rotation sampling matrix for span rotations (cached for performance)
        # This matrix is constant for a given mesh and will be attached to each system
        if measurement_config.use_span_rotations:
            from models.bridge_forward_model import rotation_sampling_matrix
            span_rotation_keys = [
                'theta_S1-5_6L_rad',
                'theta_S2-4_6L_rad',
                'theta_S3-5_6L_rad',
            ]
            self._rot_mat_span_cache = rotation_sampling_matrix(ne_per_span, span_rotation_keys)
        else:
            self._rot_mat_span_cache = None
        
        # Pre-allocate arrays for physics residuals
        n_measurements = measurement_config.count_measurements_per_case()
        self._physics_residuals = np.zeros(self.n_cases * n_measurements, dtype=float)
        
        # Pre-allocate arrays for regularization (sizes depend on temp_segments)
        self._spatial_residuals = np.zeros(self.n_cases * max(0, self.temp_segments - 1), dtype=float)
        self._temporal_residuals = np.zeros((self.n_cases - 1) * self.temp_segments, dtype=float)
        
        # Statistics
        self._n_residual_calls = 0
        self._n_system_assemblies = 0
        
        kv_status = f"固定为 {fixed_kv_factors}" if fixed_kv_factors else "参与拟合"
        settlement_status = "Settlement_A 固定为 0.0（参考点）" if fix_first_settlement else "全部拟合"
        
        print(f"\n多工况拟合设置 (向量化版本 v2):")
        print(f"  工况数量: {self.n_cases}")
        print(f"  每工况约束: {n_measurements} ({measurement_config.describe()})")
        print(f"  总约束数: {self.n_cases * n_measurements}")
        print(f"  KV参数: {kv_status}")
        print(f"  沉降参数: {settlement_status}")
        print(f"  优化特性: 批量求解 + 向量化温度载荷 (nT={self.temp_segments})")
        
        if measurement_config.auto_normalize:
            print(f"\n归一化标度 (基于测量数据的标准差):")
            if measurement_config.use_reactions:
                print(f"  反力:   {self.reaction_scale:.6e}")
            if measurement_config.use_displacements:
                print(f"  位移:   {self.displacement_scale:.6e}")
            if measurement_config.use_rotations:
                print(f"  转角:   {self.rotation_scale:.6e}")
    
    def _build_x0_and_bounds(
        self,
        fit_struct: bool = True,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Build initial guess and bounds."""
        x0_list = []
        lower_list = []
        upper_list = []
        
        if fit_struct:
            # Settlements (4 or 3 if first is fixed)
            if self.fix_first_settlement:
                # Only optimize settlements B, C, D (A is fixed at 0.0)
                x0_list.extend([0.0, 0.0, 0.0])
                lower_list.extend([self.opt_config.settlement_lower] * 3)
                upper_list.extend([self.opt_config.settlement_upper] * 3)
            else:
                # Optimize all 4 settlements
                x0_list.extend([0.0, 0.0, 0.0, 0.0])
                lower_list.extend([self.opt_config.settlement_lower] * 4)
                upper_list.extend([self.opt_config.settlement_upper] * 4)
            
            # EI factors (3)
            x0_list.extend([1.0, 1.0, 1.0])
            lower_list.extend([self.opt_config.ei_factor_lower] * 3)
            upper_list.extend([self.opt_config.ei_factor_upper] * 3)
            
            # Kv factors (4) - only if not fixed
            if self.fixed_kv_factors is None:
                x0_list.extend([1.0, 1.0, 1.0, 1.0])
                lower_list.extend([self.opt_config.kv_factor_lower] * 4)
                upper_list.extend([self.opt_config.kv_factor_upper] * 4)
        
        # Temperature gradients (nT per case)
        nT = self.temp_segments
        for _ in range(self.n_cases):
            x0_list.extend([self.opt_config.temp_gradient_initial] * nT)
            lower_list.extend([self.opt_config.temp_gradient_lower] * nT)
            upper_list.extend([self.opt_config.temp_gradient_upper] * nT)
        
        x0 = np.array(x0_list)
        lower = np.array(lower_list)
        upper = np.array(upper_list)
        
        # Count parameters
        n_settlement = (3 if self.fix_first_settlement else 4) if fit_struct else 0
        n_ei = 3 if fit_struct else 0
        n_kv = 0 if (not fit_struct or self.fixed_kv_factors is not None) else 4
        n_struct = n_settlement + n_ei + n_kv
        n_temp = self.n_cases * self.temp_segments
        
        print(f"\n参数空间:")
        print(f"  结构参数: {n_struct} (沉降: {n_settlement}, EI: {n_ei}, KV: {n_kv})")
        print(f"  温度参数: {n_temp} ({self.n_cases} × {self.temp_segments})")
        print(f"  总参数数: {len(x0)}")
        
        return x0, (lower, upper)
    
    def _unpack_params(
        self,
        x: np.ndarray,
        fit_struct: bool = True,
    ) -> Tuple[StructuralParams, np.ndarray]:
        """Unpack parameters into StructuralParams and temperature matrix.
        
        Returns:
            struct_params: StructuralParams object
            dT_matrix: Temperature matrix, shape (n_cases, nT)
        """
        idx = 0
        
        if fit_struct:
            # Settlements: either (B, C, D) with A=0.0, or (A, B, C, D)
            if self.fix_first_settlement:
                settlements = (0.0, x[idx], x[idx+1], x[idx+2])
                idx += 3
            else:
                settlements = tuple(x[idx:idx+4])
                idx += 4
            
            ei_factors = tuple(x[idx:idx+3])
            idx += 3
            
            # Use fixed KV if provided, otherwise extract from optimization variables
            if self.fixed_kv_factors is not None:
                kv_factors = self.fixed_kv_factors
            else:
                kv_factors = tuple(x[idx:idx+4])
                idx += 4
        else:
            settlements = (0.0, 0.0, 0.0, 0.0)
            ei_factors = (1.0, 1.0, 1.0)
            kv_factors = self.fixed_kv_factors if self.fixed_kv_factors is not None else (1.0, 1.0, 1.0, 1.0)
        
        struct_params = StructuralParams(
            settlements=settlements,  # type: ignore[arg-type]
            ei_factors=ei_factors,  # type: ignore[arg-type]
            kv_factors=kv_factors,  # type: ignore[arg-type]
            uniform_load=self.uniform_load,
            ne_per_span=self.ne_per_span,
        )
        
        # Extract temperature matrix: (n_cases, nT)
        dT_matrix = x[idx:].reshape(self.n_cases, self.temp_segments)
        
        return struct_params, dT_matrix
    
    def _get_or_assemble_system(self, struct_params: StructuralParams) -> StructuralSystem:
        """Get cached system or assemble new one."""
        param_key = (
            struct_params.settlements,
            struct_params.ei_factors,
            struct_params.kv_factors,
        )
        
        if self._cached_system is not None and self._cached_struct_params == param_key:
            return self._cached_system
        
        self._n_system_assemblies += 1
        system = assemble_structural_system(struct_params)
        
        self._cached_system = system
        self._cached_struct_params = param_key
        
        return system
    
    def _residual(
        self,
        x: np.ndarray,
        fit_struct: bool = True,
    ) -> np.ndarray:
        """Vectorized residual computation with auto-normalization."""
        self._n_residual_calls += 1
        
        struct_params, dT_matrix = self._unpack_params(x, fit_struct)
        system = self._get_or_assemble_system(struct_params)
        
        # 1. Physics residuals - compute all responses
        _, reactions_all, displacements_all, rotations_all, span_rotations_all = batch_solve_with_system(
            system, dT_matrix, self._rot_mat_span_cache
        )
        
        # Build physics residuals based on measurement configuration
        # Apply normalization by std and user-specified weights
        residuals_list = []
        
        if self.measurement_config.use_reactions:
            reaction_residuals = (reactions_all - self.reactions_matrix).ravel()
            # Normalize by std (if auto_normalize) then apply user weight
            reaction_residuals = reaction_residuals / self.reaction_scale
            residuals_list.append(reaction_residuals)
        
        if self.measurement_config.use_displacements:
            displacement_residuals = (displacements_all - self.displacements_matrix).ravel()
            # Normalize by std (if auto_normalize) then apply user weight
            displacement_residuals = (displacement_residuals / self.displacement_scale) * self.measurement_config.displacement_weight
            residuals_list.append(displacement_residuals)
        
        if self.measurement_config.use_rotations:
            rotation_residuals = (rotations_all - self.rotations_matrix).ravel()
            # Normalize by std (if auto_normalize) then apply user weight
            rotation_residuals = (rotation_residuals / self.rotation_scale) * self.measurement_config.rotation_weight
            residuals_list.append(rotation_residuals)
        
        if self.measurement_config.use_span_rotations:
            span_rotation_residuals = (span_rotations_all - self.span_rotations_matrix).ravel()
            # Normalize by std (if auto_normalize) then apply user weight
            span_rotation_residuals = (span_rotation_residuals / self.span_rotation_scale) * self.measurement_config.span_rotation_weight
            residuals_list.append(span_rotation_residuals)
        
        physics_residuals = np.concatenate(residuals_list)

        # 2. Spatial regularization
        if self.temp_spatial_weight > 0:
            if self.temp_segments == 3:
                diff_01 = dT_matrix[:, 1] - dT_matrix[:, 0]
                diff_12 = dT_matrix[:, 2] - dT_matrix[:, 1]
                spatial_penalty_01 = np.maximum(0.0, np.abs(diff_01) - self.opt_config.temp_spatial_diff_thresh) * self.temp_spatial_weight
                spatial_penalty_12 = np.maximum(0.0, np.abs(diff_12) - self.opt_config.temp_spatial_diff_thresh) * self.temp_spatial_weight
                self._spatial_residuals[0::2] = spatial_penalty_01
                self._spatial_residuals[1::2] = spatial_penalty_12
            else:
                # Only one adjacent pair per case in 2-seg mode
                diff = dT_matrix[:, 1] - dT_matrix[:, 0]
                spatial_penalty = np.maximum(0.0, np.abs(diff) - self.opt_config.temp_spatial_diff_thresh) * self.temp_spatial_weight
                self._spatial_residuals[:] = spatial_penalty
        else:
            self._spatial_residuals[:] = 0.0

        # 3. Temporal regularization
        if self.temp_temporal_weight > 0:
            dT_diff = dT_matrix[1:, :] - dT_matrix[:-1, :]
            temporal_penalty = np.maximum(0.0, np.abs(dT_diff) - self.opt_config.temp_temporal_diff_thresh) * self.temp_temporal_weight
            self._temporal_residuals[:] = temporal_penalty.ravel()
        else:
            self._temporal_residuals[:] = 0.0
        
        # Combine all residuals
        return np.concatenate([
            physics_residuals,
            self._spatial_residuals,
            self._temporal_residuals,
        ])
    
    def fit(
        self,
        fit_struct: bool = True,
        maxiter: int = 200,
        verbose: int = 2,
    ) -> dict:
        """Perform joint fitting."""
        print(f"\n{'='*60}")
        print("开始优化 (向量化版本 v2)...")
        print(f"{'='*60}")
        
        self._n_residual_calls = 0
        self._n_system_assemblies = 0
        
        x0, bounds = self._build_x0_and_bounds(fit_struct)
        
        result = least_squares(
            self._residual,
            x0,
            bounds=bounds,
            args=(fit_struct,),
            ftol=self.opt_config.ftol,
            gtol=self.opt_config.gtol,
            xtol=self.opt_config.xtol,
            max_nfev=maxiter * 1,
            verbose=verbose,
        )
        
        struct_params, dT_matrix = self._unpack_params(result.x, fit_struct)
        
        # Convert dT_matrix back to ThermalState list for compatibility
        thermal_states = [ThermalState(dT_spans=tuple(row)) for row in dT_matrix]  # type: ignore[arg-type]
        
        # Compute final residuals by measurement type
        _, reactions_all, displacements_all, rotations_all, span_rotations_all = batch_solve_with_system(
            self._get_or_assemble_system(struct_params), dT_matrix, self._rot_mat_span_cache
        )
        
        reaction_residuals = reactions_all - self.reactions_matrix if self.measurement_config.use_reactions else None
        displacement_residuals = displacements_all - self.displacements_matrix if self.measurement_config.use_displacements else None
        rotation_residuals = rotations_all - self.rotations_matrix if self.measurement_config.use_rotations else None
        span_rotation_residuals = span_rotations_all - self.span_rotations_matrix if self.measurement_config.use_span_rotations else None
        
        print(f"\n性能统计:")
        print(f"  残差函数调用次数: {self._n_residual_calls}")
        print(f"  系统组装次数: {self._n_system_assemblies}")
        print(f"  缓存命中率: {100 * (1 - self._n_system_assemblies / self._n_residual_calls):.1f}%")
        print(f"  优化特性: 批量求解 + 向量化")
        
        return {
            'success': result.success,
            'message': result.message,
            'struct_params': struct_params,
            'thermal_states': thermal_states,
            'reaction_residuals': reaction_residuals,
            'displacement_residuals': displacement_residuals,
            'rotation_residuals': rotation_residuals,
            'span_rotation_residuals': span_rotation_residuals,
            'reactions_computed': reactions_all,
            'displacements_computed': displacements_all,
            'rotations_computed': rotations_all,
            'span_rotations_computed': span_rotations_all,
            'cost': result.cost,
            'optimality': result.optimality,
            'nfev': result.nfev,
            'x': result.x,
            'n_residual_calls': self._n_residual_calls,
            'n_system_assemblies': self._n_system_assemblies,
        }


def print_fitting_results(
    result: dict,
    const_params: dict,
    true_temps: np.ndarray,
    reactions_matrix: np.ndarray,
    displacements_matrix: Optional[np.ndarray],
    rotations_matrix: Optional[np.ndarray],
    span_rotations_matrix: Optional[np.ndarray],
    fitter: VectorizedMultiCaseFitter,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Print fitting results and perform forward verification.
    
    Args:
        result: Fitting result dictionary from fitter.fit()
        const_params: Dictionary of constant structural parameters from data
        true_temps: True temperature matrix, shape (n_cases, 2 or 3)
        reactions_matrix: Observed reactions matrix
        displacements_matrix: Observed displacements matrix (optional)
        rotations_matrix: Observed rotations matrix (optional)
        span_rotations_matrix: Observed span rotations matrix (optional)
        fitter: VectorizedMultiCaseFitter instance
    
    Returns:
        Tuple of (recomputed_reactions, recomputed_displacements, recomputed_rotations,
                  recomputed_span_rotations, reaction_forward_res)
    """
    # Print basic results
    print(f"\n{'='*60}")
    print("拟合结果")
    print(f"{'='*60}")
    print(f"成功: {result['success']}")
    print(f"消息: {result['message']}")
    print(f"函数评估次数: {result['nfev']}")
    print(f"最终cost: {result['cost']:.6e}")
    print(f"最优性: {result['optimality']:.6e}")
    
    # Structural parameters
    sp = result['struct_params']
    print(f"\n拟合的结构参数:")
    print(f"  Settlements (mm): {sp.settlements}")
    print(f"  EI factors:       {sp.ei_factors}")
    print(f"  Kv factors:       {sp.kv_factors}")
    
    print(f"\n真实结构参数:")
    print(f"  Settlements (mm): ({const_params['settlement_a_mm']:.4f}, "
          f"{const_params['settlement_b_mm']:.4f}, "
          f"{const_params['settlement_c_mm']:.4f}, "
          f"{const_params['settlement_d_mm']:.4f})")
    print(f"  EI factors:       ({const_params['ei_factor_s1']:.4f}, "
          f"{const_params['ei_factor_s2']:.4f}, "
          f"{const_params['ei_factor_s3']:.4f})")
    print(f"  Kv factors:       ({const_params['kv_factor_a']:.4f}, "
          f"{const_params['kv_factor_b']:.4f}, "
          f"{const_params['kv_factor_c']:.4f}, "
          f"{const_params['kv_factor_d']:.4f})")
    
    # Compare parameters
    true_settlements = np.array([
        const_params['settlement_a_mm'],
        const_params['settlement_b_mm'],
        const_params['settlement_c_mm'],
        const_params['settlement_d_mm'],
    ])
    fitted_settlements = np.array(sp.settlements)
    
    true_ei = np.array([
        const_params['ei_factor_s1'],
        const_params['ei_factor_s2'],
        const_params['ei_factor_s3'],
    ])
    fitted_ei = np.array(sp.ei_factors)
    
    true_kv = np.array([
        const_params['kv_factor_a'],
        const_params['kv_factor_b'],
        const_params['kv_factor_c'],
        const_params['kv_factor_d'],
    ])
    fitted_kv = np.array(sp.kv_factors)
    
    # Temperature RMSE calculation
    fitted_temps = np.array([ts.dT_spans for ts in result['thermal_states']])
    temp_rmse = np.sqrt(np.mean((fitted_temps - true_temps)**2))
    
    print(f"\n结构参数误差:")
    print(f"  Settlements RMSE: {np.sqrt(np.mean((fitted_settlements - true_settlements)**2)):.4f} mm")
    print(f"  EI factors RMSE:  {np.sqrt(np.mean((fitted_ei - true_ei)**2)):.6f}")
    print(f"  Kv factors RMSE:  {np.sqrt(np.mean((fitted_kv - true_kv)**2)):.6f}")
    print(f"  Temperature RMSE: {temp_rmse:.4f} °C")
    
    # Residuals by measurement type
    print(f"\n残差统计 ({len(result['thermal_states'])} 工况):")
    
    if result['reaction_residuals'] is not None:
        reaction_res = result['reaction_residuals']
        print(f"\n  反力残差:")
        print(f"    RMSE:     {np.sqrt(np.mean(reaction_res**2)):.6e} kN")
        print(f"    最大残差: {np.max(np.abs(reaction_res)):.6e} kN")
        print(f"    平均残差: {np.mean(np.abs(reaction_res)):.6e} kN")
    
    if result['displacement_residuals'] is not None:
        disp_res = result['displacement_residuals']
        print(f"\n  位移残差:")
        print(f"    RMSE:     {np.sqrt(np.mean(disp_res**2)):.6e} mm")
        print(f"    最大残差: {np.max(np.abs(disp_res)):.6e} mm")
        print(f"    平均残差: {np.mean(np.abs(disp_res)):.6e} mm")
    
    if result['rotation_residuals'] is not None:
        rot_res = result['rotation_residuals']
        print(f"\n  转角残差:")
        print(f"    RMSE:     {np.sqrt(np.mean(rot_res**2)):.6e} rad")
        print(f"    最大残差: {np.max(np.abs(rot_res)):.6e} rad")
        print(f"    平均残差: {np.mean(np.abs(rot_res)):.6e} rad")
    
    if result['span_rotation_residuals'] is not None:
        span_rot_res = result['span_rotation_residuals']
        print(f"\n  跨中转角残差:")
        print(f"    RMSE:     {np.sqrt(np.mean(span_rot_res**2)):.6e} rad")
        print(f"    最大残差: {np.max(np.abs(span_rot_res)):.6e} rad")
        print(f"    平均残差: {np.mean(np.abs(span_rot_res)):.6e} rad")
    
    # Temperature examples
    print(f"\n温度梯度示例 (前5个工况):")
    for i in range(min(5, len(result['thermal_states']))):
        fitted_dT = tuple(result['thermal_states'][i].dT_spans)
        true_dT = tuple(true_temps[i])
        print(f"  Case {i}:")
        if len(fitted_dT) == 3 and len(true_dT) == 3:
            print(f"    真实: [{true_dT[0]:6.2f}, {true_dT[1]:6.2f}, {true_dT[2]:6.2f}]°C")
            print(f"    拟合: [{fitted_dT[0]:6.2f}, {fitted_dT[1]:6.2f}, {fitted_dT[2]:6.2f}]°C")
        elif len(fitted_dT) == 2 and len(true_dT) == 2:
            print(f"    真实: [L={true_dT[0]:6.2f}, R={true_dT[1]:6.2f}]°C")
            print(f"    拟合: [L={fitted_dT[0]:6.2f}, R={fitted_dT[1]:6.2f}]°C")
        else:
            print(f"    真实: {true_temps[i]}")
            print(f"    拟合: {fitted_dT}")
    
    # Forward verification: recompute all responses using fitted parameters
    print(f"\n{'='*60}")
    print("正模型验算 - 使用拟合参数重新计算所有响应")
    print(f"{'='*60}")
    
    # Assemble system with fitted structural parameters
    fitted_system = assemble_structural_system(result['struct_params'])
    
    # Extract fitted temperature matrix and recompute all responses
    fitted_dT_matrix = np.array([ts.dT_spans for ts in result['thermal_states']])
    _, recomputed_reactions, recomputed_displacements, recomputed_rotations, recomputed_span_rotations = batch_solve_with_system(
        fitted_system, fitted_dT_matrix, fitter._rot_mat_span_cache
    )
    
    # Compare with observed measurements
    print(f"\n正模型验算残差:")
    
    reaction_forward_res = recomputed_reactions - reactions_matrix
    print(f"  反力:")
    print(f"    RMSE:     {np.sqrt(np.mean(reaction_forward_res**2)):.6e} kN")
    print(f"    最大残差: {np.max(np.abs(reaction_forward_res)):.6e} kN")
    
    if displacements_matrix is not None:
        disp_forward_res = recomputed_displacements - displacements_matrix
        print(f"  位移:")
        print(f"    RMSE:     {np.sqrt(np.mean(disp_forward_res**2)):.6e} mm")
        print(f"    最大残差: {np.max(np.abs(disp_forward_res)):.6e} mm")
    
    if rotations_matrix is not None:
        rot_forward_res = recomputed_rotations - rotations_matrix
        print(f"  转角:")
        print(f"    RMSE:     {np.sqrt(np.mean(rot_forward_res**2)):.6e} rad")
        print(f"    最大残差: {np.max(np.abs(rot_forward_res)):.6e} rad")
    
    if span_rotations_matrix is not None:
        span_rot_forward_res = recomputed_span_rotations - span_rotations_matrix
        print(f"  跨中转角:")
        print(f"    RMSE:     {np.sqrt(np.mean(span_rot_forward_res**2)):.6e} rad")
        print(f"    最大残差: {np.max(np.abs(span_rot_forward_res)):.6e} rad")
    
    # Compare optimizer residuals vs forward residuals for reactions
    if result['reaction_residuals'] is not None:
        optimizer_rmse = np.sqrt(np.mean(result['reaction_residuals']**2))
        forward_rmse = np.sqrt(np.mean(reaction_forward_res**2))
        
        print(f"\n反力残差对比:")
        print(f"  优化器残差 RMSE: {optimizer_rmse:.6e} kN")
        print(f"  正模型残差 RMSE: {forward_rmse:.6e} kN")
        print(f"  差异: {abs(optimizer_rmse - forward_rmse):.6e} kN")
        
        if abs(optimizer_rmse - forward_rmse) > 1e-6:
            print(f"  ⚠️  警告: 优化器残差与正模型残差存在显著差异，可能存在数值问题")
        else:
            print(f"  ✓ 优化器残差与正模型残差一致，验算通过")
    
    # Show detailed comparison for first few cases
    print(f"\n前5个工况的反力详细对比:")
    print(f"  {'Case':<6} {'Support':<8} {'Observed':>12} {'Recomputed':>12} {'Residual':>12}")
    print(f"  {'-'*60}")
    support_names = ['A', 'B', 'C', 'D']
    for i in range(min(5, len(reactions_matrix))):
        for j, name in enumerate(support_names):
            obs = reactions_matrix[i, j]
            rec = recomputed_reactions[i, j]
            res = reaction_forward_res[i, j]
            print(f"  {i:<6} {name:<8} {obs:>12.6f} {rec:>12.6f} {res:>12.6e}")
    
    return recomputed_reactions, recomputed_displacements, recomputed_rotations, recomputed_span_rotations, reaction_forward_res


def main():
    parser = argparse.ArgumentParser(
        description="向量化优化版多工况拟合 (v2) - 支持多文件输入和多种测量类型"
    )
    parser.add_argument('--data', type=Path, action='append', required=True, 
                        help='CSV数据文件 (可多次指定以连接多个文件)')
    parser.add_argument('--max-samples', type=int, help='限制样本数 (拼接后)')
    parser.add_argument('--no-fit-struct', action='store_true', help='不拟合结构参数')
    parser.add_argument('--maxiter', type=int, default=200, help='最大迭代次数')
    parser.add_argument('--temp-spatial-weight', type=float, default=1.0)
    parser.add_argument('--temp-temporal-weight', type=float, default=1.0)
    parser.add_argument('--fixed-kv', type=float, nargs=4, metavar=('KVA', 'KVB', 'KVC', 'KVD'),
                        help='固定KV参数 (4个值: kv_a kv_b kv_c kv_d)，不参与拟合')
    parser.add_argument('--no-fix-first-settlement', action='store_true',
                        help='不固定第一个沉降为0（默认固定settlement_a=0作为参考点）')
    parser.add_argument('--output', type=Path, help='输出CSV文件')
    parser.add_argument('--temp-segments', type=int, default=3, choices=[2, 3],
                        help='温度分段数：3=每跨；2=按全长50/50')
    
    # Measurement configuration
    parser.add_argument('--use-displacements', action='store_true',
                        help='使用支座位移测量值 (v_A, v_B, v_C, v_D)')
    parser.add_argument('--use-rotations', action='store_true',
                        help='使用支座转角测量值 (theta_A, theta_B, theta_C, theta_D)')
    parser.add_argument('--use-span-rotations', action='store_true',
                        help='使用跨中转角测量值 (theta_S1-5_6L, theta_S2-4_6L, theta_S3-5_6L)')
    parser.add_argument('--displacement-weight', type=float, default=1.0,
                        help='位移残差权重 (相对于反力，默认1.0)')
    parser.add_argument('--rotation-weight', type=float, default=1.0,
                        help='转角残差权重 (相对于反力，默认1.0)')
    parser.add_argument('--span-rotation-weight', type=float, default=1.0,
                        help='跨中转角残差权重 (相对于反力，默认1.0)')
    parser.add_argument('--no-auto-normalize', action='store_true',
                        help='禁用自动归一化（默认启用，基于测量数据标准差归一化）')
    
    args = parser.parse_args()
    
    # Load and concatenate data from multiple files
    print(f"{'='*60}")
    print("数据加载")
    print(f"{'='*60}")
    
    # Create measurement configuration from command line args
    measurement_config = MeasurementConfig(
        use_reactions=True,  # Always use reactions
        use_displacements=args.use_displacements,
        use_rotations=args.use_rotations,
        use_span_rotations=args.use_span_rotations,
        displacement_weight=args.displacement_weight,
        rotation_weight=args.rotation_weight,
        span_rotation_weight=args.span_rotation_weight,
        auto_normalize=not args.no_auto_normalize,  # Default: enabled
    )
    
    df, const_params, detected_config = load_multi_case_data(
        args.data, args.max_samples, measurement_config
    )
    
    # Extract measurement matrices based on what's available and requested
    reactions_matrix = df[ColumnNames.REACTIONS].to_numpy()
    
    displacements_matrix = None
    if measurement_config.use_displacements:
        if all(col in df.columns for col in ColumnNames.DISPLACEMENTS):
            displacements_matrix = df[ColumnNames.DISPLACEMENTS].to_numpy()
            print(f"\n✓ 使用支座位移测量值 (权重={measurement_config.displacement_weight})")
        else:
            print(f"\n✗ 警告: 请求使用位移但数据中不包含，将跳过位移约束")
            measurement_config.use_displacements = False
    
    rotations_matrix = None
    if measurement_config.use_rotations:
        if all(col in df.columns for col in ColumnNames.ROTATIONS):
            rotations_matrix = df[ColumnNames.ROTATIONS].to_numpy()
            print(f"\n✓ 使用支座转角测量值 (权重={measurement_config.rotation_weight})")
        else:
            print(f"\n✗ 警告: 请求使用转角但数据中不包含，将跳过转角约束")
            measurement_config.use_rotations = False
    
    span_rotations_matrix = None
    if measurement_config.use_span_rotations:
        if all(col in df.columns for col in ColumnNames.SPAN_ROTATIONS):
            span_rotations_matrix = df[ColumnNames.SPAN_ROTATIONS].to_numpy()
            print(f"\n✓ 使用跨中转角测量值 (权重={measurement_config.span_rotation_weight})")
        else:
            print(f"\n✗ 警告: 请求使用跨中转角但数据中不包含，将跳过跨中转角约束")
            measurement_config.use_span_rotations = False
    
    # Temperature columns detection
    requested_segments = int(args.temp_segments)
    if requested_segments == 3:
        if all(c in df.columns for c in ColumnNames.TEMPS_3_SPAN):
            true_temps = df[ColumnNames.TEMPS_3_SPAN].to_numpy()
            actual_segments = 3
        elif all(c in df.columns for c in ColumnNames.TEMPS_2_SEG):
            print('提示: 数据文件仅含2段温度列，自动切换为2段模式。')
            true_temps = df[ColumnNames.TEMPS_2_SEG].to_numpy()
            actual_segments = 2
        else:
            raise SystemExit('缺少温度列：需要 dT_s1_C/dT_s2_C/dT_s3_C 或 dT_left_C/dT_right_C')
    else:
        if all(c in df.columns for c in ColumnNames.TEMPS_2_SEG):
            true_temps = df[ColumnNames.TEMPS_2_SEG].to_numpy()
            actual_segments = 2
        elif all(c in df.columns for c in ColumnNames.TEMPS_3_SPAN):
            print('提示: 数据文件仅含3段温度列，自动切换为3段模式。')
            true_temps = df[ColumnNames.TEMPS_3_SPAN].to_numpy()
            actual_segments = 3
        else:
            raise SystemExit('缺少温度列：需要 dT_left_C/dT_right_C 或 dT_s1_C/dT_s2_C/dT_s3_C')
    uniform_load = float(df[ColumnNames.UNIFORM_LOAD].iloc[0])
    
    # Parse fixed KV if provided
    fixed_kv_factors = None
    if args.fixed_kv is not None:
        fixed_kv_factors = tuple(args.fixed_kv)
        print(f"\n使用固定KV参数: {fixed_kv_factors}")
    
    # Determine if first settlement should be fixed
    fix_first_settlement = not args.no_fix_first_settlement
    
    # Create fitter
    fitter = VectorizedMultiCaseFitter(
        reactions_matrix=reactions_matrix,
        uniform_load=uniform_load,
        temp_spatial_weight=args.temp_spatial_weight,
        temp_temporal_weight=args.temp_temporal_weight,
        fixed_kv_factors=fixed_kv_factors,
        fix_first_settlement=fix_first_settlement,
        temp_segments=actual_segments,
        measurement_config=measurement_config,
        displacements_matrix=displacements_matrix,
        rotations_matrix=rotations_matrix,
        span_rotations_matrix=span_rotations_matrix,
    )
    
    # Fit
    result = fitter.fit(
        fit_struct=not args.no_fit_struct,
        maxiter=args.maxiter,
        verbose=2,
    )
    
    # Print results and perform forward verification
    recomputed_reactions, recomputed_displacements, recomputed_rotations, recomputed_span_rotations, reaction_forward_res = print_fitting_results(
        result=result,
        const_params=const_params,
        true_temps=true_temps,
        reactions_matrix=reactions_matrix,
        displacements_matrix=displacements_matrix,
        rotations_matrix=rotations_matrix,
        span_rotations_matrix=span_rotations_matrix,
        fitter=fitter,
    )
    
    # Extract structural parameters for output
    sp = result['struct_params']
    
    # Save results
    if args.output:
        output_df = df.copy()
        
        # Fitted structural parameters
        output_df['settlement_a_fitted'] = sp.settlements[0]
        output_df['settlement_b_fitted'] = sp.settlements[1]
        output_df['settlement_c_fitted'] = sp.settlements[2]
        output_df['settlement_d_fitted'] = sp.settlements[3]
        output_df['ei_factor_s1_fitted'] = sp.ei_factors[0]
        output_df['ei_factor_s2_fitted'] = sp.ei_factors[1]
        output_df['ei_factor_s3_fitted'] = sp.ei_factors[2]
        output_df['kv_factor_a_fitted'] = sp.kv_factors[0]
        output_df['kv_factor_b_fitted'] = sp.kv_factors[1]
        output_df['kv_factor_c_fitted'] = sp.kv_factors[2]
        output_df['kv_factor_d_fitted'] = sp.kv_factors[3]
        
        # Fitted temperatures
        for i, ts in enumerate(result['thermal_states']):
            if len(ts.dT_spans) == 3:
                output_df.loc[i, 'dT_s1_fitted'] = ts.dT_spans[0]
                output_df.loc[i, 'dT_s2_fitted'] = ts.dT_spans[1]
                output_df.loc[i, 'dT_s3_fitted'] = ts.dT_spans[2]
            else:
                output_df.loc[i, 'dT_left_fitted'] = ts.dT_spans[0]
                output_df.loc[i, 'dT_right_fitted'] = ts.dT_spans[1]
        
        # Recomputed responses
        support_names = ['a', 'b', 'c', 'd']
        for i, name in enumerate(support_names):
            output_df[f'R_{name}_recomputed_kN'] = recomputed_reactions[:, i]
            output_df[f'v_{name.upper()}_recomputed_mm'] = recomputed_displacements[:, i]
            output_df[f'theta_{name.upper()}_recomputed_rad'] = recomputed_rotations[:, i]
        
        # Recomputed span rotations
        span_names = ['S1-5_6L', 'S2-4_6L', 'S3-5_6L']
        for i, name in enumerate(span_names):
            output_df[f'theta_{name}_recomputed_rad'] = recomputed_span_rotations[:, i]
        
        # Optimizer residuals
        if result['reaction_residuals'] is not None:
            for i in range(4):
                output_df[f'residual_optimizer_R{i+1}'] = result['reaction_residuals'][:, i]
        
        if result['displacement_residuals'] is not None:
            for i in range(4):
                output_df[f'residual_optimizer_v{i+1}'] = result['displacement_residuals'][:, i]
        
        if result['rotation_residuals'] is not None:
            for i in range(4):
                output_df[f'residual_optimizer_theta{i+1}'] = result['rotation_residuals'][:, i]
        
        if result['span_rotation_residuals'] is not None:
            for i in range(3):
                output_df[f'residual_optimizer_span_theta{i+1}'] = result['span_rotation_residuals'][:, i]
        
        # Forward model residuals
        for i in range(4):
            output_df[f'residual_forward_R{i+1}'] = reaction_forward_res[:, i]
        
        if displacements_matrix is not None:
            disp_forward_res = recomputed_displacements - displacements_matrix
            for i in range(4):
                output_df[f'residual_forward_v{i+1}'] = disp_forward_res[:, i]
        
        if rotations_matrix is not None:
            rot_forward_res = recomputed_rotations - rotations_matrix
            for i in range(4):
                output_df[f'residual_forward_theta{i+1}'] = rot_forward_res[:, i]
        
        if span_rotations_matrix is not None:
            span_rot_forward_res = recomputed_span_rotations - span_rotations_matrix
            for i in range(3):
                output_df[f'residual_forward_span_theta{i+1}'] = span_rot_forward_res[:, i]
        
        output_df.to_csv(args.output, index=False)
        print(f"\n结果已保存到: {args.output}")
        print(f"\n输出文件包含:")
        print(f"  - 拟合的结构参数 (11个)")
        print(f"  - 拟合的温度参数 (每工况 {actual_segments} 个)")
        print(f"  - 正模型重算的响应 (反力、位移、转角、跨中转角)")
        print(f"  - 优化器残差 ({measurement_config.describe()})")
        print(f"  - 正模型残差 (用于验证)")


if __name__ == '__main__':
    main()
