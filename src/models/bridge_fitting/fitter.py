"""Core fitting engine with vectorized operations."""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from models.bridge_forward_model import (
    StructuralParams,
    StructuralSystem,
    ThermalState,
    assemble_structural_system,
    solve_linear,
    ALPHA,
    H_MM,
)

from .config import MeasurementConfig, OptimizationConfig


def batch_thermal_load_vectors(
    system: StructuralSystem,
    dT_matrix: np.ndarray,
) -> np.ndarray:
    """Vectorized computation of thermal load vectors for all cases."""
    n_cases, nT = dT_matrix.shape
    ndof = len(system.load_base)
    loads_all = np.zeros((n_cases, ndof), dtype=float)
    kappa_matrix = (ALPHA / H_MM) * dT_matrix
    
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
            kappa_cases = kappa_matrix[:, 0]
        
        dof_indices = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        loads_all[:, dof_indices[1]] -= ei * kappa_cases
        loads_all[:, dof_indices[3]] += ei * kappa_cases
    
    return loads_all


def batch_solve_with_system(
    system: StructuralSystem,
    dT_matrix: np.ndarray,
    rot_mat_span_cache: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch solve for all cases simultaneously."""
    n_cases = dT_matrix.shape[0]
    thermal_loads = batch_thermal_load_vectors(system, dT_matrix)
    F_all = system.load_base[np.newaxis, :] - thermal_loads
    U_all = solve_linear(system, F_all.T).T
    
    reactions_all = np.zeros((n_cases, 4), dtype=float)
    displacements_all = np.zeros((n_cases, 4), dtype=float)
    rotations_all = np.zeros((n_cases, 4), dtype=float)
    
    for i, idx in enumerate(system.support_nodes):
        ui_all = U_all[:, 2 * idx]
        displacements_all[:, i] = ui_all
        theta_all = U_all[:, 2 * idx + 1]
        rotations_all[:, i] = theta_all
        reactions_all[:, i] = -system.kv_values[i] * (ui_all - system.settlements[i]) / 1_000.0
    
    if rot_mat_span_cache is not None:
        span_rotations_all = (rot_mat_span_cache @ U_all.T).T
    else:
        from models.bridge_forward_model import rotation_sampling_matrix
        span_rotation_keys = ['theta_S1-5_6L_rad', 'theta_S2-4_6L_rad', 'theta_S3-5_6L_rad']
        rot_mat_span = rotation_sampling_matrix(system.struct_params.ne_per_span, span_rotation_keys)
        span_rotations_all = (rot_mat_span @ U_all.T).T
    
    return U_all, reactions_all, displacements_all, rotations_all, span_rotations_all


class VectorizedMultiCaseFitter:
    """Highly optimized fitter with vectorized operations."""
    
    def __init__(
        self,
        reactions_matrix: np.ndarray,
        uniform_load: float,
        ne_per_span: int = 64,
        temp_spatial_weight: float = 1.0,
        temp_temporal_weight: float = 1.0,
        fixed_kv_factors: tuple[float, float, float, float] | None = None,
        fix_first_settlement: bool = True,
        temp_segments: int = 3,
        measurement_config: Optional[MeasurementConfig] = None,
        displacements_matrix: Optional[np.ndarray] = None,
        rotations_matrix: Optional[np.ndarray] = None,
        span_rotations_matrix: Optional[np.ndarray] = None,
        opt_config: Optional[OptimizationConfig] = None,
    ):
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
        
        self.opt_config = opt_config if opt_config is not None else OptimizationConfig()
        
        if measurement_config is None:
            measurement_config = MeasurementConfig(use_reactions=True)
        self.measurement_config = measurement_config
        
        self.displacements_matrix = displacements_matrix
        self.rotations_matrix = rotations_matrix
        self.span_rotations_matrix = span_rotations_matrix
        
        if measurement_config.use_displacements and displacements_matrix is None:
            raise ValueError("displacements_matrix required when use_displacements=True")
        if measurement_config.use_rotations and rotations_matrix is None:
            raise ValueError("rotations_matrix required when use_rotations=True")
        if measurement_config.use_span_rotations and span_rotations_matrix is None:
            raise ValueError("span_rotations_matrix required when use_span_rotations=True")
        
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
        
        if self.reaction_scale < self.opt_config.min_scale_threshold:
            self.reaction_scale = 1.0
        if self.displacement_scale < self.opt_config.min_scale_threshold:
            self.displacement_scale = 1.0
        if self.rotation_scale < self.opt_config.min_scale_threshold:
            self.rotation_scale = 1.0
        if self.span_rotation_scale < self.opt_config.min_scale_threshold:
            self.span_rotation_scale = 1.0
        
        self._cached_system: StructuralSystem | None = None
        self._cached_struct_params: tuple | None = None
        
        if measurement_config.use_span_rotations:
            from models.bridge_forward_model import rotation_sampling_matrix
            span_rotation_keys = ['theta_S1-5_6L_rad', 'theta_S2-4_6L_rad', 'theta_S3-5_6L_rad']
            self._rot_mat_span_cache = rotation_sampling_matrix(ne_per_span, span_rotation_keys)
        else:
            self._rot_mat_span_cache = None
        
        n_measurements = measurement_config.count_measurements_per_case()
        self._physics_residuals = np.zeros(self.n_cases * n_measurements, dtype=float)
        self._spatial_residuals = np.zeros(self.n_cases * max(0, self.temp_segments - 1), dtype=float)
        self._temporal_residuals = np.zeros((self.n_cases - 1) * self.temp_segments, dtype=float)
        
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
            if self.fix_first_settlement:
                x0_list.extend([0.0, 0.0, 0.0])
                lower_list.extend([self.opt_config.settlement_lower] * 3)
                upper_list.extend([self.opt_config.settlement_upper] * 3)
            else:
                x0_list.extend([0.0, 0.0, 0.0, 0.0])
                lower_list.extend([self.opt_config.settlement_lower] * 4)
                upper_list.extend([self.opt_config.settlement_upper] * 4)
            
            x0_list.extend([1.0, 1.0, 1.0])
            lower_list.extend([self.opt_config.ei_factor_lower] * 3)
            upper_list.extend([self.opt_config.ei_factor_upper] * 3)
            
            if self.fixed_kv_factors is None:
                x0_list.extend([1.0, 1.0, 1.0, 1.0])
                lower_list.extend([self.opt_config.kv_factor_lower] * 4)
                upper_list.extend([self.opt_config.kv_factor_upper] * 4)
        
        nT = self.temp_segments
        for _ in range(self.n_cases):
            x0_list.extend([self.opt_config.temp_gradient_initial] * nT)
            lower_list.extend([self.opt_config.temp_gradient_lower] * nT)
            upper_list.extend([self.opt_config.temp_gradient_upper] * nT)
        
        x0 = np.array(x0_list)
        lower = np.array(lower_list)
        upper = np.array(upper_list)
        
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
        """Unpack parameters into StructuralParams and temperature matrix."""
        idx = 0
        
        if fit_struct:
            if self.fix_first_settlement:
                settlements = (0.0, x[idx], x[idx+1], x[idx+2])
                idx += 3
            else:
                settlements = tuple(x[idx:idx+4])
                idx += 4
            
            ei_factors = tuple(x[idx:idx+3])
            idx += 3
            
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
        
        _, reactions_all, displacements_all, rotations_all, span_rotations_all = batch_solve_with_system(
            system, dT_matrix, self._rot_mat_span_cache
        )
        
        residuals_list = []
        
        if self.measurement_config.use_reactions:
            reaction_residuals = (reactions_all - self.reactions_matrix).ravel()
            reaction_residuals = reaction_residuals / self.reaction_scale
            residuals_list.append(reaction_residuals)
        
        if self.measurement_config.use_displacements:
            displacement_residuals = (displacements_all - self.displacements_matrix).ravel()
            displacement_residuals = (displacement_residuals / self.displacement_scale) * self.measurement_config.displacement_weight
            residuals_list.append(displacement_residuals)
        
        if self.measurement_config.use_rotations:
            rotation_residuals = (rotations_all - self.rotations_matrix).ravel()
            rotation_residuals = (rotation_residuals / self.rotation_scale) * self.measurement_config.rotation_weight
            residuals_list.append(rotation_residuals)
        
        if self.measurement_config.use_span_rotations:
            span_rotation_residuals = (span_rotations_all - self.span_rotations_matrix).ravel()
            span_rotation_residuals = (span_rotation_residuals / self.span_rotation_scale) * self.measurement_config.span_rotation_weight
            residuals_list.append(span_rotation_residuals)
        
        physics_residuals = np.concatenate(residuals_list)

        if self.temp_spatial_weight > 0:
            if self.temp_segments == 3:
                diff_01 = dT_matrix[:, 1] - dT_matrix[:, 0]
                diff_12 = dT_matrix[:, 2] - dT_matrix[:, 1]
                spatial_penalty_01 = np.maximum(0.0, np.abs(diff_01) - self.opt_config.temp_spatial_diff_thresh) * self.temp_spatial_weight
                spatial_penalty_12 = np.maximum(0.0, np.abs(diff_12) - self.opt_config.temp_spatial_diff_thresh) * self.temp_spatial_weight
                self._spatial_residuals[0::2] = spatial_penalty_01
                self._spatial_residuals[1::2] = spatial_penalty_12
            else:
                diff = dT_matrix[:, 1] - dT_matrix[:, 0]
                spatial_penalty = np.maximum(0.0, np.abs(diff) - self.opt_config.temp_spatial_diff_thresh) * self.temp_spatial_weight
                self._spatial_residuals[:] = spatial_penalty
        else:
            self._spatial_residuals[:] = 0.0

        if self.temp_temporal_weight > 0:
            dT_diff = dT_matrix[1:, :] - dT_matrix[:-1, :]
            temporal_penalty = np.maximum(0.0, np.abs(dT_diff) - self.opt_config.temp_temporal_diff_thresh) * self.temp_temporal_weight
            self._temporal_residuals[:] = temporal_penalty.ravel()
        else:
            self._temporal_residuals[:] = 0.0
        
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
        thermal_states = [ThermalState(dT_spans=tuple(row)) for row in dT_matrix]  # type: ignore[arg-type]
        
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
