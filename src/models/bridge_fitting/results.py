"""Results printing and forward verification."""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from models.bridge_forward_model import assemble_structural_system

from .residuals import print_residual_stats

if TYPE_CHECKING:
    from .fitter import VectorizedMultiCaseFitter


def extract_true_parameters(const_params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从常数参数字典提取真实参数数组。"""
    true_settlements = np.array([
        const_params['settlement_a_mm'],
        const_params['settlement_b_mm'],
        const_params['settlement_c_mm'],
        const_params['settlement_d_mm'],
    ])
    
    true_ei = np.array([
        const_params['ei_factor_s1'],
        const_params['ei_factor_s2'],
        const_params['ei_factor_s3'],
    ])
    
    true_kv = np.array([
        const_params['kv_factor_a'],
        const_params['kv_factor_b'],
        const_params['kv_factor_c'],
        const_params['kv_factor_d'],
    ])
    
    return true_settlements, true_ei, true_kv


def print_fitting_results(
    result: dict,
    const_params: dict,
    true_temps: np.ndarray,
    reactions_matrix: np.ndarray,
    displacements_matrix: Optional[np.ndarray],
    rotations_matrix: Optional[np.ndarray],
    span_rotations_matrix: Optional[np.ndarray],
    fitter: 'VectorizedMultiCaseFitter',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Print fitting results and perform forward verification."""
    from .fitter import batch_solve_with_system
    
    print(f"\n{'='*60}")
    print("拟合结果")
    print(f"{'='*60}")
    print(f"成功: {result['success']}")
    print(f"消息: {result['message']}")
    print(f"函数评估次数: {result['nfev']}")
    print(f"最终cost: {result['cost']:.6e}")
    print(f"最优性: {result['optimality']:.6e}")
    
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

    rotation_biases = result.get('rotation_biases')
    rotation_bias_labels = result.get('rotation_bias_labels') or []
    if rotation_biases is not None and len(rotation_bias_labels) == len(rotation_biases):
        print(f"\n拟合的转角偏差 (rad):")
        for label, bias in zip(rotation_bias_labels, rotation_biases):
            print(f"  {label}: {bias:+.6e}")
    
    true_settlements, true_ei, true_kv = extract_true_parameters(const_params)
    fitted_settlements = np.array(sp.settlements)
    fitted_ei = np.array(sp.ei_factors)
    fitted_kv = np.array(sp.kv_factors)
    
    fitted_temps = np.array([ts.dT_spans for ts in result['thermal_states']])
    temp_rmse = np.sqrt(np.mean((fitted_temps - true_temps)**2))
    
    print(f"\n结构参数误差:")
    print(f"  Settlements RMSE: {np.sqrt(np.mean((fitted_settlements - true_settlements)**2)):.4f} mm")
    print(f"  EI factors RMSE:  {np.sqrt(np.mean((fitted_ei - true_ei)**2)):.6f}")
    print(f"  Kv factors RMSE:  {np.sqrt(np.mean((fitted_kv - true_kv)**2)):.6f}")
    print(f"  Temperature RMSE: {temp_rmse:.4f} °C")
    
    print(f"\n残差统计 ({len(result['thermal_states'])} 工况):")
    print_residual_stats(result['reaction_residuals'], "反力", "kN")
    print_residual_stats(result['displacement_residuals'], "位移", "mm")
    print_residual_stats(result['rotation_residuals'], "转角", "rad")
    print_residual_stats(result['span_rotation_residuals'], "跨中转角", "rad")
    
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
    
    print(f"\n{'='*60}")
    print("正模型验算 - 使用拟合参数重新计算所有响应")
    print(f"{'='*60}")
    
    fitted_system = assemble_structural_system(result['struct_params'])
    fitted_dT_matrix = np.array([ts.dT_spans for ts in result['thermal_states']])
    _, recomputed_reactions, recomputed_displacements, recomputed_rotations, recomputed_span_rotations = batch_solve_with_system(
        fitted_system, fitted_dT_matrix, fitter._rot_mat_span_cache
    )
    
    print(f"\n正模型验算残差:")
    
    reaction_forward_res = recomputed_reactions - reactions_matrix
    print(f"  反力:")
    print(f"    RMSE:     {np.sqrt(np.mean(reaction_forward_res**2)):.6e} kN")
    print(f"    最大残差: {np.max(np.abs(reaction_forward_res)):.6e} kN")
    
    if displacements_matrix is not None:
        disp_forward_res = recomputed_displacements - displacements_matrix
        print_residual_stats(disp_forward_res, "位移", "mm")
    
    if rotations_matrix is not None:
        rot_forward_res = recomputed_rotations - rotations_matrix
        if rotation_biases is not None and fitter._rotation_bias_support_count:
            bias_support = rotation_biases[:fitter._rotation_bias_support_count]
            rot_forward_res = recomputed_rotations + bias_support[np.newaxis, :] - rotations_matrix
        print_residual_stats(rot_forward_res, "转角", "rad")
    
    if span_rotations_matrix is not None:
        span_rot_forward_res = recomputed_span_rotations - span_rotations_matrix
        if rotation_biases is not None and fitter._rotation_bias_span_count:
            bias_span = rotation_biases[fitter._rotation_bias_support_count:]
            span_rot_forward_res = recomputed_span_rotations + bias_span[np.newaxis, :] - span_rotations_matrix
        print_residual_stats(span_rot_forward_res, "跨中转角", "rad")
    
    if result['reaction_residuals'] is not None:
        optimizer_rmse = np.sqrt(np.mean(result['reaction_residuals']**2))
        forward_rmse = np.sqrt(np.mean(reaction_forward_res**2))
        
        print(f"\n反力残差对比:")
        print(f"  优化器残差 RMSE: {optimizer_rmse:.6e} kN")
        print(f"  正模型残差 RMSE: {forward_rmse:.6e} kN")
        print(f"  差异: {abs(optimizer_rmse - forward_rmse):.6e} kN")
        
        if abs(optimizer_rmse - forward_rmse) > 1e-6:
            print(f"  ⚠️  警告: 优化器残差与正模型残差存在显著差异")
        else:
            print(f"  ✓ 优化器残差与正模型残差一致，验算通过")
    
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
