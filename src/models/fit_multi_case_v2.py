#!/usr/bin/env python3
"""Further optimized multi-case fitting with vectorized operations.

Additional optimizations over fit_multi_case_optimized.py:
1. Batch thermal load computation - vectorize temperature load vectors
2. Batch solve - solve all cases simultaneously using BLAS Level 3
3. Vectorized regularization computation
4. Pre-allocate arrays to reduce memory allocations

Expected performance improvement: 1.5-2x over fit_multi_case_optimized.py

Usage:
    # Basic usage (默认固定 settlement_a = 0.0 作为参考点)
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
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.linalg import lu_solve
from scipy.optimize import least_squares

from models.bridge_forward_model import (
    StructuralParams,
    StructuralSystem,
    ThermalState,
    assemble_structural_system,
    ALPHA,
    H_MM,
)


def load_multi_case_data(csv_paths: List[Path], max_samples: int | None = None) -> Tuple[pd.DataFrame, dict]:
    """Load and concatenate multi-case data from multiple files.
    
    Args:
        csv_paths: List of CSV file paths to load and concatenate
        max_samples: Optional limit on total samples after concatenation
    
    Returns:
        Combined DataFrame with continuous sample_id, and constant structural parameters
    """
    dfs = []
    
    print(f"\n加载 {len(csv_paths)} 个数据文件:")
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        print(f"  文件 {i+1}: {csv_path.name} ({len(df)} 条数据)")
        dfs.append(df)
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Reset sample_id to be continuous across all files
    df['sample_id'] = range(len(df))
    
    print(f"\n合并后数据集:")
    print(f"  总样本数: {len(df)}")
    
    if max_samples is not None and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()
        df['sample_id'] = range(len(df))
        print(f"  限制到前 {max_samples} 条数据")
    
    # Verify structural parameters are constant
    struct_cols = [
        'settlement_a_mm', 'settlement_b_mm', 'settlement_c_mm', 'settlement_d_mm',
        'ei_factor_s1', 'ei_factor_s2', 'ei_factor_s3',
        'kv_factor_a', 'kv_factor_b', 'kv_factor_c', 'kv_factor_d',
    ]
    
    const_params = {}
    for col in struct_cols:
        unique_vals = df[col].unique()
        if len(unique_vals) != 1:
            print(f"  警告: {col} 不是常数，有 {len(unique_vals)} 个不同值")
        const_params[col] = float(df[col].iloc[0])
    
    print(f"  结构参数验证: {'✓ 全部为常数' if all(len(df[c].unique()) == 1 for c in struct_cols) else '✗ 存在变化'}")
    
    return df, const_params


def f_thermal_element(ei: float, kappa: float) -> np.ndarray:
    """Thermal equivalent load for a single element."""
    return np.array([0.0, ei * kappa, 0.0, -ei * kappa])


def batch_thermal_load_vectors(
    system: StructuralSystem,
    dT_matrix: np.ndarray,  # shape: (n_cases, 3) - temperatures for all cases
) -> np.ndarray:
    """Vectorized computation of thermal load vectors for all cases.
    
    Args:
        system: Structural system
        dT_matrix: Temperature gradients, shape (n_cases, 3)
    
    Returns:
        Load vectors, shape (n_cases, ndof)
    """
    n_cases = dT_matrix.shape[0]
    ndof = len(system.load_base)
    
    # Pre-allocate result
    loads_all = np.zeros((n_cases, ndof), dtype=float)
    
    # Convert temperatures to curvatures: (n_cases, 3)
    kappa_matrix = (ALPHA / H_MM) * dT_matrix
    
    # Iterate over elements (this part is hard to vectorize due to assembly)
    for n1, n2, L, si in system.mesh.elements:
        ei = system.span_ei[si]
        # Thermal loads for all cases at once: (n_cases,)
        kappa_cases = kappa_matrix[:, si]
        
        # Compute thermal loads for this element across all cases
        # f_thermal = ei * kappa * [0, -1, 0, +1]
        # 注意：这里是固定端弯矩等效载荷
        dof_indices = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        
        # Vectorized assembly - 注意符号！
        loads_all[:, dof_indices[1]] -= ei * kappa_cases  # moment at node 1: -EI*kappa
        loads_all[:, dof_indices[3]] += ei * kappa_cases  # moment at node 2: +EI*kappa
    
    return loads_all


def batch_solve_with_system(
    system: StructuralSystem,
    dT_matrix: np.ndarray,  # shape: (n_cases, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch solve for all cases simultaneously.
    
    Args:
        system: Structural system
        dT_matrix: Temperature gradients, shape (n_cases, 3)
    
    Returns:
        U_all: Displacements, shape (n_cases, ndof)
        reactions_all: Reactions, shape (n_cases, 4)
    """
    n_cases = dT_matrix.shape[0]
    
    # Batch compute thermal loads: (n_cases, ndof)
    thermal_loads = batch_thermal_load_vectors(system, dT_matrix)
    
    # Compute right-hand sides: (n_cases, ndof)
    F_all = system.load_base[np.newaxis, :] - thermal_loads
    
    # Batch solve using BLAS Level 3: solve K @ U = F for multiple RHS
    # lu_solve accepts (ndof, n_cases) format
    U_all = lu_solve((system.lu, system.piv), F_all.T).T  # shape: (n_cases, ndof)
    
    # Extract reactions for all cases
    reactions_all = np.zeros((n_cases, 4), dtype=float)
    for i, idx in enumerate(system.support_nodes):
        ui_all = U_all[:, 2 * idx]  # vertical displacements at support
        reactions_all[:, i] = -system.kv_values[i] * (ui_all - system.settlements[i]) / 1_000.0
    
    return U_all, reactions_all


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
        """
        self.reactions_matrix = reactions_matrix
        self.n_cases = reactions_matrix.shape[0]
        self.uniform_load = uniform_load
        self.ne_per_span = ne_per_span
        self.temp_spatial_weight = temp_spatial_weight
        self.temp_temporal_weight = temp_temporal_weight
        self.fixed_kv_factors = fixed_kv_factors
        self.fix_first_settlement = fix_first_settlement
        
        # Cache
        self._cached_system: StructuralSystem | None = None
        self._cached_struct_params: tuple | None = None
        
        # Pre-allocate arrays for regularization
        self._spatial_residuals = np.zeros(self.n_cases * 2, dtype=float)
        self._temporal_residuals = np.zeros((self.n_cases - 1) * 3, dtype=float)
        
        # Statistics
        self._n_residual_calls = 0
        self._n_system_assemblies = 0
        
        kv_status = f"固定为 {fixed_kv_factors}" if fixed_kv_factors else "参与拟合"
        settlement_status = "Settlement_A 固定为 0.0（参考点）" if fix_first_settlement else "全部拟合"
        print(f"\n多工况拟合设置 (向量化版本 v2):")
        print(f"  工况数量: {self.n_cases}")
        print(f"  每工况约束: 4 (反力)")
        print(f"  总约束数: {self.n_cases * 4}")
        print(f"  KV参数: {kv_status}")
        print(f"  沉降参数: {settlement_status}")
        print(f"  优化特性: 批量求解 + 向量化温度载荷")
    
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
                lower_list.extend([-20.0] * 3)
                upper_list.extend([20.0] * 3)
            else:
                # Optimize all 4 settlements
                x0_list.extend([0.0, 0.0, 0.0, 0.0])
                lower_list.extend([-20.0] * 4)
                upper_list.extend([20.0] * 4)
            
            # EI factors (3)
            x0_list.extend([1.0, 1.0, 1.0])
            lower_list.extend([0.1] * 3)
            upper_list.extend([2.0] * 3)
            
            # Kv factors (4) - only if not fixed
            if self.fixed_kv_factors is None:
                x0_list.extend([1.0, 1.0, 1.0, 1.0])
                lower_list.extend([0.1] * 4)
                upper_list.extend([3.0] * 4)
        
        # Temperature gradients (3 per case)
        for _ in range(self.n_cases):
            x0_list.extend([10.0, 10.0, 10.0])
            lower_list.extend([-10.0] * 3)
            upper_list.extend([20.0] * 3)
        
        x0 = np.array(x0_list)
        lower = np.array(lower_list)
        upper = np.array(upper_list)
        
        # Count parameters
        n_settlement = (3 if self.fix_first_settlement else 4) if fit_struct else 0
        n_ei = 3 if fit_struct else 0
        n_kv = 0 if (not fit_struct or self.fixed_kv_factors is not None) else 4
        n_struct = n_settlement + n_ei + n_kv
        n_temp = self.n_cases * 3
        
        print(f"\n参数空间:")
        print(f"  结构参数: {n_struct} (沉降: {n_settlement}, EI: {n_ei}, KV: {n_kv})")
        print(f"  温度参数: {n_temp} ({self.n_cases} × 3)")
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
            dT_matrix: Temperature matrix, shape (n_cases, 3)
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
        
        # Extract temperature matrix: (n_cases, 3)
        dT_matrix = x[idx:].reshape(self.n_cases, 3)
        
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
        """Vectorized residual computation."""
        self._n_residual_calls += 1
        
        struct_params, dT_matrix = self._unpack_params(x, fit_struct)
        system = self._get_or_assemble_system(struct_params)
        
        # 1. Batch solve for all cases at once
        _, reactions_all = batch_solve_with_system(system, dT_matrix)
        
        # Physics residuals: (n_cases, 4) -> (n_cases * 4,)
        physics_residuals = (reactions_all - self.reactions_matrix).ravel()
        
        # 2. Vectorized spatial regularization
        if self.temp_spatial_weight > 0:
            # Compute differences between adjacent spans within each case
            # dT_matrix: (n_cases, 3), we want diff[:, 1] - diff[:, 0] and diff[:, 2] - diff[:, 1]
            diff_01 = dT_matrix[:, 1] - dT_matrix[:, 0]  # (n_cases,)
            diff_12 = dT_matrix[:, 2] - dT_matrix[:, 1]  # (n_cases,)
            
            # Soft L1 penalty: max(0, |diff| - 5.0)
            spatial_penalty_01 = np.maximum(0.0, np.abs(diff_01) - 5.0) * self.temp_spatial_weight
            spatial_penalty_12 = np.maximum(0.0, np.abs(diff_12) - 5.0) * self.temp_spatial_weight
            
            # Interleave: [case0_01, case0_12, case1_01, case1_12, ...]
            self._spatial_residuals[0::2] = spatial_penalty_01
            self._spatial_residuals[1::2] = spatial_penalty_12
        else:
            self._spatial_residuals[:] = 0.0
        
        # 3. Vectorized temporal regularization
        if self.temp_temporal_weight > 0:
            # Compute differences between consecutive cases
            # NOTE: This works across file boundaries when multiple files are concatenated,
            # ensuring smooth temperature transitions throughout the entire time series
            # dT_matrix: (n_cases, 3), we want dT[i+1] - dT[i] for each span
            dT_diff = dT_matrix[1:, :] - dT_matrix[:-1, :]  # (n_cases-1, 3)
            
            # Soft L1 penalty: max(0, |diff| - 3.0)
            temporal_penalty = np.maximum(0.0, np.abs(dT_diff) - 3.0) * self.temp_temporal_weight
            
            # Flatten: (n_cases-1, 3) -> ((n_cases-1) * 3,)
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
            ftol=1e-8,
            gtol=1e-8,
            xtol=1e-12,
            max_nfev=maxiter * 100,
            verbose=verbose,
        )
        
        struct_params, dT_matrix = self._unpack_params(result.x, fit_struct)
        
        # Convert dT_matrix back to ThermalState list for compatibility
        thermal_states = [ThermalState(dT_spans=tuple(row)) for row in dT_matrix]  # type: ignore[arg-type]
        
        # Compute final residuals
        all_residuals = self._residual(result.x, fit_struct)
        n_physics = self.n_cases * 4
        physics_residuals = all_residuals[:n_physics].reshape(self.n_cases, 4)
        
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
            'residuals': physics_residuals,
            'cost': result.cost,
            'optimality': result.optimality,
            'nfev': result.nfev,
            'x': result.x,
            'n_residual_calls': self._n_residual_calls,
            'n_system_assemblies': self._n_system_assemblies,
        }


def main():
    parser = argparse.ArgumentParser(
        description="向量化优化版多工况拟合 (v2) - 支持多文件输入"
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
    
    args = parser.parse_args()
    
    # Load and concatenate data from multiple files
    print(f"{'='*60}")
    print("数据加载")
    print(f"{'='*60}")
    df, const_params = load_multi_case_data(args.data, args.max_samples)
    
    reactions_matrix = df[['R_a_kN', 'R_b_kN', 'R_c_kN', 'R_d_kN']].to_numpy()
    true_temps = df[['dT_s1_C', 'dT_s2_C', 'dT_s3_C']].to_numpy()
    uniform_load = float(df['uniform_load_N_per_mm'].iloc[0])
    
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
    )
    
    # Fit
    result = fitter.fit(
        fit_struct=not args.no_fit_struct,
        maxiter=args.maxiter,
        verbose=2,
    )
    
    # Print results
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
    
    print(f"\n结构参数误差:")
    print(f"  Settlements RMSE: {np.sqrt(np.mean((fitted_settlements - true_settlements)**2)):.4f} mm")
    print(f"  EI factors RMSE:  {np.sqrt(np.mean((fitted_ei - true_ei)**2)):.6f}")
    print(f"  Kv factors RMSE:  {np.sqrt(np.mean((fitted_kv - true_kv)**2)):.6f}")
    
    # Residuals
    residuals = result['residuals']
    print(f"\n反力残差统计 ({len(residuals)} 工况):")
    print(f"  全局 RMSE: {np.sqrt(np.mean(residuals**2)):.6e} kN")
    print(f"  最大残差:  {np.max(np.abs(residuals)):.6e} kN")
    print(f"  平均残差:  {np.mean(np.abs(residuals)):.6e} kN")
    
    # Temperature examples
    print(f"\n温度梯度示例 (前5个工况):")
    for i in range(min(5, len(result['thermal_states']))):
        fitted_dT = result['thermal_states'][i].dT_spans
        true_dT = true_temps[i]
        print(f"  Case {i}:")
        print(f"    真实: [{true_dT[0]:6.2f}, {true_dT[1]:6.2f}, {true_dT[2]:6.2f}]°C")
        print(f"    拟合: [{fitted_dT[0]:6.2f}, {fitted_dT[1]:6.2f}, {fitted_dT[2]:6.2f}]°C")
    
    # Forward verification: recompute reactions using fitted parameters
    print(f"\n{'='*60}")
    print("正模型验算 - 使用拟合参数重新计算反力")
    print(f"{'='*60}")
    
    # Assemble system with fitted structural parameters
    fitted_system = assemble_structural_system(result['struct_params'])
    
    # Extract fitted temperature matrix
    fitted_dT_matrix = np.array([ts.dT_spans for ts in result['thermal_states']])
    
    # Recompute reactions using forward model
    _, recomputed_reactions = batch_solve_with_system(fitted_system, fitted_dT_matrix)
    
    # Compare with observed reactions
    forward_residuals = recomputed_reactions - reactions_matrix
    
    print(f"\n正模型验算残差统计:")
    print(f"  RMSE: {np.sqrt(np.mean(forward_residuals**2)):.6e} kN")
    print(f"  最大残差: {np.max(np.abs(forward_residuals)):.6e} kN")
    print(f"  平均残差: {np.mean(np.abs(forward_residuals)):.6e} kN")
    
    # Compare optimizer residuals vs forward residuals
    optimizer_rmse = np.sqrt(np.mean(residuals**2))
    forward_rmse = np.sqrt(np.mean(forward_residuals**2))
    
    print(f"\n残差对比:")
    print(f"  优化器残差 RMSE: {optimizer_rmse:.6e} kN")
    print(f"  正模型残差 RMSE: {forward_rmse:.6e} kN")
    print(f"  差异: {abs(optimizer_rmse - forward_rmse):.6e} kN")
    
    if abs(optimizer_rmse - forward_rmse) > 1e-6:
        print(f"  ⚠️  警告: 优化器残差与正模型残差存在显著差异，可能存在数值问题")
    else:
        print(f"  ✓ 优化器残差与正模型残差一致，验算通过")
    
    # Show detailed comparison for first few cases
    print(f"\n前5个工况的详细对比:")
    print(f"  {'Case':<6} {'Support':<8} {'Observed':>12} {'Recomputed':>12} {'Residual':>12}")
    print(f"  {'-'*60}")
    support_names = ['A', 'B', 'C', 'D']
    for i in range(min(5, len(reactions_matrix))):
        for j, name in enumerate(support_names):
            obs = reactions_matrix[i, j]
            rec = recomputed_reactions[i, j]
            res = forward_residuals[i, j]
            print(f"  {i:<6} {name:<8} {obs:>12.6f} {rec:>12.6f} {res:>12.6e}")
    
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
            output_df.loc[i, 'dT_s1_fitted'] = ts.dT_spans[0]
            output_df.loc[i, 'dT_s2_fitted'] = ts.dT_spans[1]
            output_df.loc[i, 'dT_s3_fitted'] = ts.dT_spans[2]
        
        # Optimizer residuals (from optimization process)
        for i in range(4):
            output_df[f'residual_optimizer_R{i+1}'] = residuals[:, i]
        
        # Forward model verification: recomputed reactions
        support_names = ['a', 'b', 'c', 'd']
        for i, name in enumerate(support_names):
            output_df[f'R_{name}_recomputed_kN'] = recomputed_reactions[:, i]
        
        # Forward model residuals (observed - recomputed)
        for i in range(4):
            output_df[f'residual_forward_R{i+1}'] = forward_residuals[:, i]
        
        output_df.to_csv(args.output, index=False)
        print(f"\n结果已保存到: {args.output}")
        print(f"\n输出文件包含:")
        print(f"  - 拟合的结构参数 (11个)")
        print(f"  - 拟合的温度参数 (每工况3个)")
        print(f"  - 优化器残差 (4个反力残差)")
        print(f"  - 正模型重算的反力 (4个)")
        print(f"  - 正模型残差 (4个，用于验证多解)")


if __name__ == '__main__':
    main()
