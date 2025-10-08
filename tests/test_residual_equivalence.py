#!/usr/bin/env python3
"""比较 optimized 和 v2 版本的残差函数输出"""

import numpy as np
import pandas as pd
from pathlib import Path
from models.fit_multi_case_optimized import OptimizedMultiCaseFitter
from models.fit_multi_case_v2 import VectorizedMultiCaseFitter

def load_test_data():
    """加载测试数据"""
    csv_path = Path('data/augmented/dt_24hours_data_4.csv')
    df = pd.read_csv(csv_path)
    df = df.head(10)  # 只用10条数据测试
    
    reactions = df[['R_a_kN', 'R_b_kN', 'R_c_kN', 'R_d_kN']].to_numpy()
    uniform_load = float(df['uniform_load_N_per_mm'].iloc[0])
    
    return reactions, uniform_load


def test_residual_equivalence():
    """测试残差函数的等价性"""
    
    print("="*70)
    print("残差函数等价性测试")
    print("="*70)
    
    reactions, uniform_load = load_test_data()
    n_cases = len(reactions)
    
    print(f"\n测试配置:")
    print(f"  工况数: {n_cases}")
    print(f"  均布荷载: {uniform_load} N/mm")
    
    # Create fitters
    fitter_opt = OptimizedMultiCaseFitter(
        reactions_matrix=reactions,
        uniform_load=uniform_load,
        temp_spatial_weight=1.0,
        temp_temporal_weight=1.0,
    )
    
    fitter_v2 = VectorizedMultiCaseFitter(
        reactions_matrix=reactions,
        uniform_load=uniform_load,
        temp_spatial_weight=1.0,
        temp_temporal_weight=1.0,
    )
    
    # Build identical x0
    x0_opt, bounds_opt = fitter_opt._build_x0_and_bounds(fit_struct=True)
    x0_v2, bounds_v2 = fitter_v2._build_x0_and_bounds(fit_struct=True)
    
    assert np.allclose(x0_opt, x0_v2), "Initial guess should be identical"
    assert np.allclose(bounds_opt[0], bounds_v2[0]), "Lower bounds should be identical"
    assert np.allclose(bounds_opt[1], bounds_v2[1]), "Upper bounds should be identical"
    
    print(f"\n✓ 初始猜测和边界一致")
    print(f"  参数数量: {len(x0_opt)}")
    
    # Test 1: Residual at x0
    print(f"\n测试 1: 初始点 x0 的残差")
    res_opt = fitter_opt._residual(x0_opt, fit_struct=True)
    res_v2 = fitter_v2._residual(x0_v2, fit_struct=True)
    
    print(f"  Optimized 残差长度: {len(res_opt)}")
    print(f"  V2 残差长度:        {len(res_v2)}")
    
    if len(res_opt) != len(res_v2):
        print(f"  ✗ 残差向量长度不同!")
        print(f"    差异: {len(res_opt) - len(res_v2)}")
        return False
    
    diff = res_opt - res_v2
    print(f"  最大绝对差异: {np.max(np.abs(diff)):.2e}")
    print(f"  平均绝对差异: {np.mean(np.abs(diff)):.2e}")
    print(f"  RMS 差异:     {np.sqrt(np.mean(diff**2)):.2e}")
    
    # Analyze by section
    n_physics = n_cases * 4
    n_spatial = n_cases * 2
    n_temporal = (n_cases - 1) * 3
    
    print(f"\n分段分析:")
    print(f"  物理残差 (0:{n_physics}):")
    diff_physics = res_opt[:n_physics] - res_v2[:n_physics]
    print(f"    最大差异: {np.max(np.abs(diff_physics)):.2e}")
    
    print(f"  空间正则化 ({n_physics}:{n_physics+n_spatial}):")
    diff_spatial = res_opt[n_physics:n_physics+n_spatial] - res_v2[n_physics:n_physics+n_spatial]
    print(f"    最大差异: {np.max(np.abs(diff_spatial)):.2e}")
    
    print(f"  时间正则化 ({n_physics+n_spatial}:):")
    diff_temporal = res_opt[n_physics+n_spatial:] - res_v2[n_physics+n_spatial:]
    print(f"    最大差异: {np.max(np.abs(diff_temporal)):.2e}")
    
    # Test 2: Perturbed parameters
    print(f"\n测试 2: 扰动参数的残差")
    x_perturbed = x0_opt.copy()
    x_perturbed[:4] += np.array([1.0, -0.5, 0.8, -1.2])  # Perturb settlements
    x_perturbed[4:7] += np.array([0.1, -0.05, 0.08])      # Perturb EI
    x_perturbed[11:14] += np.array([2.0, -1.5, 3.0])      # Perturb first temp case
    
    res_opt_p = fitter_opt._residual(x_perturbed, fit_struct=True)
    res_v2_p = fitter_v2._residual(x_perturbed, fit_struct=True)
    
    diff_p = res_opt_p - res_v2_p
    print(f"  最大绝对差异: {np.max(np.abs(diff_p)):.2e}")
    print(f"  平均绝对差异: {np.mean(np.abs(diff_p)):.2e}")
    
    # Final verdict
    print(f"\n结论:")
    tol = 1e-10
    if np.max(np.abs(diff)) < tol and np.max(np.abs(diff_p)) < tol:
        print(f"  ✓✓✓ 残差函数完全等价（误差 < {tol}）")
        return True
    elif np.max(np.abs(diff)) < 1e-6 and np.max(np.abs(diff_p)) < 1e-6:
        print(f"  ✓✓ 残差函数数值等价（误差 < 1e-6）")
        return True
    else:
        print(f"  ✗ 残差函数存在差异")
        print(f"\n  差异详情（初始点）:")
        print(f"    物理残差:   {np.max(np.abs(diff_physics)):.2e}")
        print(f"    空间正则化: {np.max(np.abs(diff_spatial)):.2e}")
        print(f"    时间正则化: {np.max(np.abs(diff_temporal)):.2e}")
        return False


if __name__ == '__main__':
    success = test_residual_equivalence()
    exit(0 if success else 1)
