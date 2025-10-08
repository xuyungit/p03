#!/usr/bin/env python3
"""Multi-case parameter fitting for time-series bridge data.

Scenario: Structural parameters (settlements, EI, Kv) remain constant,
while temperature gradients vary over time. We have multiple measurements
at different times with different temperatures.

This script jointly fits:
- Fixed structural parameters (shared across all cases)
- Variable temperature gradients (one per measurement)

Usage:
    uv run python src/models/fit_multi_case.py \
        --data data/augmented/dt_24hours_data_4.csv \
        --max-samples 100 \
        --maxiter 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from models.bridge_forward_model import (
    StructuralParams,
    ThermalState,
    evaluate_forward,
)


def load_multi_case_data(csv_path: Path, max_samples: int | None = None) -> Tuple[pd.DataFrame, dict]:
    """Load multi-case data and verify structural parameters are constant."""
    df = pd.read_csv(csv_path)
    
    if max_samples is not None and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()
        print(f"限制到前 {max_samples} 条数据")
    
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
            print(f"警告: {col} 不是常数，有 {len(unique_vals)} 个不同值")
        const_params[col] = float(df[col].iloc[0])
    
    print(f"\n数据集信息:")
    print(f"  总样本数: {len(df)}")
    print(f"  结构参数验证: {'✓ 全部为常数' if all(len(df[c].unique()) == 1 for c in struct_cols) else '✗ 存在变化'}")
    
    return df, const_params


class MultiCaseFitter:
    """Joint fitting of constant structural parameters and varying temperatures with regularization."""
    
    def __init__(
        self,
        reactions_matrix: np.ndarray,  # shape: (n_cases, 4)
        uniform_load: float,
        ne_per_span: int = 64,
        temp_spatial_weight: float = 1.0,  # Weight for spatial smoothness
        temp_temporal_weight: float = 1.0,  # Weight for temporal smoothness
    ):
        """
        Args:
            reactions_matrix: Measured reactions, shape (n_cases, 4)
            uniform_load: Uniform load in N/mm
            ne_per_span: Number of elements per span
            temp_spatial_weight: Weight for spatial temperature smoothness penalty
            temp_temporal_weight: Weight for temporal temperature smoothness penalty
        """
        self.reactions_matrix = reactions_matrix
        self.n_cases = reactions_matrix.shape[0]
        self.uniform_load = uniform_load
        self.ne_per_span = ne_per_span
        self.temp_spatial_weight = temp_spatial_weight
        self.temp_temporal_weight = temp_temporal_weight
        
        print(f"\n多工况拟合设置:")
        print(f"  工况数量: {self.n_cases}")
        print(f"  每工况约束: 4 (反力)")
        print(f"  总约束数: {self.n_cases * 4}")
        print(f"  温度空间正则化权重: {temp_spatial_weight}")
        print(f"  温度时间正则化权重: {temp_temporal_weight}")
    
    def _build_x0_and_bounds(
        self,
        fit_struct: bool = True,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Build initial guess and bounds with tighter temperature constraints."""
        x0_list = []
        lower_list = []
        upper_list = []
        
        # Structural parameters (constant across all cases)
        if fit_struct:
            # Settlements (4)
            x0_list.extend([0.0, 0.0, 0.0, 0.0])
            lower_list.extend([-20.0] * 4)
            upper_list.extend([20.0] * 4)
            
            # EI factors (3)
            x0_list.extend([1.0, 1.0, 1.0])
            lower_list.extend([0.1] * 3)
            upper_list.extend([2.0] * 3)
            
            # Kv factors (4)
            x0_list.extend([1.0, 1.0, 1.0, 1.0])
            lower_list.extend([0.1] * 4)
            upper_list.extend([3.0] * 4)
        
        # Temperature gradients (3 per case, varying)
        # Tighter bounds: [-10, 20]°C
        for _ in range(self.n_cases):
            x0_list.extend([10.0, 10.0, 10.0])  # Start near middle of expected range
            lower_list.extend([-10.0] * 3)
            upper_list.extend([20.0] * 3)
        
        x0 = np.array(x0_list)
        lower = np.array(lower_list)
        upper = np.array(upper_list)
        
        n_struct_params = 11 if fit_struct else 0
        n_temp_params = self.n_cases * 3
        
        print(f"\n参数空间:")
        print(f"  结构参数 (固定): {n_struct_params if fit_struct else 0}")
        print(f"  温度参数 (变化): {n_temp_params} ({self.n_cases} 工况 × 3)")
        print(f"  温度范围: [-10, 20]°C")
        print(f"  总参数数: {len(x0)}")
        print(f"  自由度: {self.n_cases * 4} 约束 - {len(x0)} 参数 = {self.n_cases * 4 - len(x0)}")
        
        return x0, (lower, upper)
    
    def _unpack_params(
        self,
        x: np.ndarray,
        fit_struct: bool = True,
    ) -> Tuple[StructuralParams, list[ThermalState]]:
        """Unpack optimization vector."""
        idx = 0
        
        if fit_struct:
            settlements = tuple(x[idx:idx+4])
            idx += 4
            ei_factors = tuple(x[idx:idx+3])
            idx += 3
            kv_factors = tuple(x[idx:idx+4])
            idx += 4
        else:
            # Use default values
            settlements = (0.0, 0.0, 0.0, 0.0)
            ei_factors = (1.0, 1.0, 1.0)
            kv_factors = (1.0, 1.0, 1.0, 1.0)
        
        struct_params = StructuralParams(
            settlements=settlements,  # type: ignore[arg-type]
            ei_factors=ei_factors,  # type: ignore[arg-type]
            kv_factors=kv_factors,  # type: ignore[arg-type]
            uniform_load=self.uniform_load,
            ne_per_span=self.ne_per_span,
        )
        
        # Temperature states (one per case)
        thermal_states = []
        for _ in range(self.n_cases):
            dT_spans = tuple(x[idx:idx+3])
            thermal_states.append(ThermalState(dT_spans=dT_spans))  # type: ignore[arg-type]
            idx += 3
        
        return struct_params, thermal_states
    
    def _residual(
        self,
        x: np.ndarray,
        fit_struct: bool = True,
    ) -> np.ndarray:
        """Compute stacked residuals for all cases with temperature regularization."""
        struct_params, thermal_states = self._unpack_params(x, fit_struct)
        
        # 1. Physics residuals (reaction mismatch)
        physics_residuals = []
        for i in range(self.n_cases):
            # Forward model
            response = evaluate_forward(struct_params, thermal_states[i])
            
            # Residual for this case
            residual = response.reactions_kN - self.reactions_matrix[i]
            physics_residuals.append(residual)
        
        physics_residuals = np.concatenate(physics_residuals)
        
        # 2. Temperature spatial smoothness regularization
        # Always return fixed-size array for scipy.optimize.least_squares
        # Use soft penalty instead of conditional penalty
        spatial_residuals = []
        if self.temp_spatial_weight > 0:
            for i in range(self.n_cases):
                dT = np.array(thermal_states[i].dT_spans)
                for j in range(2):  # 2 adjacent pairs per case
                    diff = dT[j+1] - dT[j]
                    # Soft L1 penalty: only penalize beyond threshold
                    penalty = max(0.0, abs(diff) - 5.0) * self.temp_spatial_weight
                    spatial_residuals.append(penalty)
        
        # 3. Temperature temporal smoothness regularization
        temporal_residuals = []
        if self.temp_temporal_weight > 0:
            for i in range(self.n_cases - 1):  # n_cases - 1 pairs
                dT_curr = np.array(thermal_states[i].dT_spans)
                dT_next = np.array(thermal_states[i+1].dT_spans)
                for j in range(3):  # 3 spans
                    diff = dT_next[j] - dT_curr[j]
                    # Soft L1 penalty
                    penalty = max(0.0, abs(diff) - 3.0) * self.temp_temporal_weight
                    temporal_residuals.append(penalty)
        
        # Combine all residuals with fixed size
        all_residuals = [physics_residuals]
        if spatial_residuals:
            all_residuals.append(np.array(spatial_residuals))
        if temporal_residuals:
            all_residuals.append(np.array(temporal_residuals))
        
        return np.concatenate(all_residuals)
    
    def fit(
        self,
        fit_struct: bool = True,
        maxiter: int = 200,
        verbose: int = 2,
    ) -> dict:
        """Perform joint fitting."""
        print(f"\n{'='*60}")
        print("开始优化...")
        print(f"{'='*60}")
        
        x0, bounds = self._build_x0_and_bounds(fit_struct)
        
        result = least_squares(
            self._residual,
            x0,
            bounds=bounds,
            args=(fit_struct,),
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=maxiter * 100,
            verbose=verbose,
        )
        
        struct_params, thermal_states = self._unpack_params(result.x, fit_struct)
        
        # Compute final residuals
        all_residuals_combined = self._residual(result.x, fit_struct)
        n_physics = self.n_cases * 4
        physics_residuals = all_residuals_combined[:n_physics].reshape(self.n_cases, 4)
        n_regularization = len(all_residuals_combined) - n_physics
        
        return {
            'success': result.success,
            'message': result.message,
            'struct_params': struct_params,
            'thermal_states': thermal_states,
            'residuals': physics_residuals,
            'n_regularization_terms': n_regularization,
            'cost': result.cost,
            'optimality': result.optimality,
            'nfev': result.nfev,
            'x': result.x,
        }


def main():
    parser = argparse.ArgumentParser(
        description="多工况联合拟合：固定结构参数 + 变化温度梯度 (带温度正则化)"
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='CSV数据文件路径',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='限制使用的样本数（用于快速测试）',
    )
    parser.add_argument(
        '--no-fit-struct',
        action='store_true',
        help='不拟合结构参数（仅拟合温度）',
    )
    parser.add_argument(
        '--maxiter',
        type=int,
        default=200,
        help='最大迭代次数',
    )
    parser.add_argument(
        '--temp-spatial-weight',
        type=float,
        default=1.0,
        help='温度空间平滑性权重（相邻梁之间）',
    )
    parser.add_argument(
        '--temp-temporal-weight',
        type=float,
        default=1.0,
        help='温度时间平滑性权重（相邻时刻之间）',
    )
    parser.add_argument(
        '--use-true-temps',
        action='store_true',
        help='使用真实温度（不拟合温度，仅拟合结构参数）',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='输出结果CSV文件',
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"加载数据: {args.data}")
    df, const_params = load_multi_case_data(args.data, args.max_samples)
    
    # Extract reactions
    reactions_matrix = df[['R_a_kN', 'R_b_kN', 'R_c_kN', 'R_d_kN']].to_numpy()
    true_temps = df[['dT_s1_C', 'dT_s2_C', 'dT_s3_C']].to_numpy()
    uniform_load = float(df['uniform_load_N_per_mm'].iloc[0])
    
    # If using true temperatures, only fit structural parameters
    if args.use_true_temps:
        print("\n使用真实温度 - 仅拟合结构参数")
        # Create a special fitter that uses fixed temperatures
        class FixedTempFitter(MultiCaseFitter):
            def __init__(self, *args, true_temps, **kwargs):
                super().__init__(*args, **kwargs)
                self.true_temps = true_temps
            
            def _build_x0_and_bounds(self, fit_struct=True):
                # Only structural parameters
                x0_list = []
                lower_list = []
                upper_list = []
                
                # Settlements (4)
                x0_list.extend([0.0, 0.0, 0.0, 0.0])
                lower_list.extend([-20.0] * 4)
                upper_list.extend([20.0] * 4)
                
                # EI factors (3)
                x0_list.extend([1.0, 1.0, 1.0])
                lower_list.extend([0.1] * 3)
                upper_list.extend([2.0] * 3)
                
                # Kv factors (4)
                x0_list.extend([1.0, 1.0, 1.0, 1.0])
                lower_list.extend([0.1] * 4)
                upper_list.extend([3.0] * 4)
                
                x0 = np.array(x0_list)
                lower = np.array(lower_list)
                upper = np.array(upper_list)
                
                print(f"\n参数空间:")
                print(f"  结构参数: 11")
                print(f"  温度参数: 0 (使用真实值)")
                print(f"  总参数数: {len(x0)}")
                
                return x0, (lower, upper)
            
            def _unpack_params(self, x, fit_struct=True):
                # Only structural parameters
                settlements = tuple(x[0:4])
                ei_factors = tuple(x[4:7])
                kv_factors = tuple(x[7:11])
                
                struct_params = StructuralParams(
                    settlements=settlements,
                    ei_factors=ei_factors,
                    kv_factors=kv_factors,
                    uniform_load=self.uniform_load,
                    ne_per_span=self.ne_per_span,
                )
                
                # Use true temperatures
                thermal_states = []
                for i in range(self.n_cases):
                    dT_spans = tuple(self.true_temps[i])
                    thermal_states.append(ThermalState(dT_spans=dT_spans))
                
                return struct_params, thermal_states
            
            def _residual(self, x, fit_struct=True):
                # No temperature regularization needed
                struct_params, thermal_states = self._unpack_params(x, fit_struct)
                
                physics_residuals = []
                for i in range(self.n_cases):
                    response = evaluate_forward(struct_params, thermal_states[i])
                    residual = response.reactions_kN - self.reactions_matrix[i]
                    physics_residuals.append(residual)
                
                return np.concatenate(physics_residuals)
        
        fitter = FixedTempFitter(
            reactions_matrix=reactions_matrix,
            uniform_load=uniform_load,
            true_temps=true_temps,
        )
    else:
        # Create fitter
        fitter = MultiCaseFitter(
            reactions_matrix=reactions_matrix,
            uniform_load=uniform_load,
            temp_spatial_weight=args.temp_spatial_weight,
            temp_temporal_weight=args.temp_temporal_weight,
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
    print(f"正则化项数: {result['n_regularization_terms']}")
    
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
    
    # Compare structural parameters
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
    
    # Residual statistics
    residuals = result['residuals']
    print(f"\n反力残差统计 ({len(residuals)} 工况):")
    print(f"  全局 RMSE: {np.sqrt(np.mean(residuals**2)):.6e} kN")
    print(f"  最大残差:  {np.max(np.abs(residuals)):.6e} kN")
    print(f"  平均残差:  {np.mean(np.abs(residuals)):.6e} kN")
    
    # Temperature smoothness statistics
    print(f"\n温度梯度平滑性统计:")
    
    # Spatial smoothness (within same case)
    spatial_diffs = []
    for ts in result['thermal_states']:
        dT = np.array(ts.dT_spans)
        for j in range(2):
            spatial_diffs.append(abs(dT[j+1] - dT[j]))
    
    spatial_diffs = np.array(spatial_diffs)
    print(f"  空间差异 (同一工况相邻梁):")
    print(f"    平均: {np.mean(spatial_diffs):.3f}°C")
    print(f"    最大: {np.max(spatial_diffs):.3f}°C")
    print(f"    超过5°C的比例: {100 * np.mean(spatial_diffs > 5.0):.1f}%")
    
    # Temporal smoothness (between consecutive cases)
    temporal_diffs = []
    for i in range(len(result['thermal_states']) - 1):
        dT_curr = np.array(result['thermal_states'][i].dT_spans)
        dT_next = np.array(result['thermal_states'][i+1].dT_spans)
        for j in range(3):
            temporal_diffs.append(abs(dT_next[j] - dT_curr[j]))
    
    temporal_diffs = np.array(temporal_diffs)
    print(f"  时间差异 (相邻工况同一梁):")
    print(f"    平均: {np.mean(temporal_diffs):.3f}°C")
    print(f"    最大: {np.max(temporal_diffs):.3f}°C")
    print(f"    超过3°C的比例: {100 * np.mean(temporal_diffs > 3.0):.1f}%")
    
    # Temperature statistics (first few cases)
    print(f"\n温度梯度示例 (前5个工况):")
    for i in range(min(5, len(result['thermal_states']))):
        fitted_dT = result['thermal_states'][i].dT_spans
        true_dT = true_temps[i]
        print(f"  Case {i}:")
        print(f"    真实: [{true_dT[0]:6.2f}, {true_dT[1]:6.2f}, {true_dT[2]:6.2f}]°C")
        print(f"    拟合: [{fitted_dT[0]:6.2f}, {fitted_dT[1]:6.2f}, {fitted_dT[2]:6.2f}]°C")
    
    # Save results
    if args.output:
        output_df = df.copy()
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
        
        for i, ts in enumerate(result['thermal_states']):
            output_df.loc[i, 'dT_s1_fitted'] = ts.dT_spans[0]
            output_df.loc[i, 'dT_s2_fitted'] = ts.dT_spans[1]
            output_df.loc[i, 'dT_s3_fitted'] = ts.dT_spans[2]
        
        for i in range(4):
            output_df[f'residual_R{i+1}'] = residuals[:, i]
        
        output_df.to_csv(args.output, index=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
