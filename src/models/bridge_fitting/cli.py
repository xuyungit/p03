"""Command-line interface for bridge fitting."""

import argparse
from pathlib import Path

import numpy as np

from .config import ColumnNames, MeasurementConfig
from .data_loader import load_multi_case_data
from .fitter import VectorizedMultiCaseFitter
from .measurements import setup_measurement_matrices
from .residuals import add_residual_columns
from .results import print_fitting_results


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
    
    print(f"{'='*60}")
    print("数据加载")
    print(f"{'='*60}")
    
    measurement_config = MeasurementConfig(
        use_reactions=True,
        use_displacements=args.use_displacements,
        use_rotations=args.use_rotations,
        use_span_rotations=args.use_span_rotations,
        displacement_weight=args.displacement_weight,
        rotation_weight=args.rotation_weight,
        span_rotation_weight=args.span_rotation_weight,
        auto_normalize=not args.no_auto_normalize,
    )
    
    df, const_params, detected_config = load_multi_case_data(
        args.data, args.max_samples, measurement_config
    )
    
    reactions_matrix, displacements_matrix, rotations_matrix, span_rotations_matrix, measurement_config = setup_measurement_matrices(
        df, measurement_config
    )
    
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
    
    fixed_kv_factors = None
    if args.fixed_kv is not None:
        fixed_kv_factors = tuple(args.fixed_kv)
        print(f"\n使用固定KV参数: {fixed_kv_factors}")
    
    fix_first_settlement = not args.no_fix_first_settlement
    
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
    
    result = fitter.fit(
        fit_struct=not args.no_fit_struct,
        maxiter=args.maxiter,
        verbose=2,
    )
    
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
    
    if args.output:
        output_df = df.copy()
        
        sp = result['struct_params']
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
            if len(ts.dT_spans) == 3:
                output_df.loc[i, 'dT_s1_fitted'] = ts.dT_spans[0]
                output_df.loc[i, 'dT_s2_fitted'] = ts.dT_spans[1]
                output_df.loc[i, 'dT_s3_fitted'] = ts.dT_spans[2]
            else:
                output_df.loc[i, 'dT_left_fitted'] = ts.dT_spans[0]
                output_df.loc[i, 'dT_right_fitted'] = ts.dT_spans[1]
        
        support_names = ['a', 'b', 'c', 'd']
        for i, name in enumerate(support_names):
            output_df[f'R_{name}_recomputed_kN'] = recomputed_reactions[:, i]
            output_df[f'v_{name.upper()}_recomputed_mm'] = recomputed_displacements[:, i]
            output_df[f'theta_{name.upper()}_recomputed_rad'] = recomputed_rotations[:, i]
        
        span_names = ['S1-5_6L', 'S2-4_6L', 'S3-5_6L']
        for i, name in enumerate(span_names):
            output_df[f'theta_{name}_recomputed_rad'] = recomputed_span_rotations[:, i]
        
        add_residual_columns(output_df, result['reaction_residuals'], 'residual_optimizer_R', 4)
        add_residual_columns(output_df, result['displacement_residuals'], 'residual_optimizer_v', 4)
        add_residual_columns(output_df, result['rotation_residuals'], 'residual_optimizer_theta', 4)
        add_residual_columns(output_df, result['span_rotation_residuals'], 'residual_optimizer_span_theta', 3)
        
        add_residual_columns(output_df, reaction_forward_res, 'residual_forward_R', 4)
        
        if displacements_matrix is not None:
            disp_forward_res = recomputed_displacements - displacements_matrix
            add_residual_columns(output_df, disp_forward_res, 'residual_forward_v', 4)
        
        if rotations_matrix is not None:
            rot_forward_res = recomputed_rotations - rotations_matrix
            add_residual_columns(output_df, rot_forward_res, 'residual_forward_theta', 4)
        
        if span_rotations_matrix is not None:
            span_rot_forward_res = recomputed_span_rotations - span_rotations_matrix
            add_residual_columns(output_df, span_rot_forward_res, 'residual_forward_span_theta', 3)
        
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
