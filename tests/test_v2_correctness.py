#!/usr/bin/env python3
"""诊断 V2 版本的数值准确性"""

import numpy as np
from models.bridge_forward_model import (
    StructuralParams,
    ThermalState,
    assemble_structural_system,
    solve_with_system,
    thermal_load_vector,
)
from models.fit_multi_case_v2 import batch_thermal_load_vectors

def test_batch_thermal_load():
    """测试批量温度载荷计算的正确性"""
    
    print("="*70)
    print("诊断测试：批量温度载荷计算")
    print("="*70)
    
    # 创建测试参数
    struct_params = StructuralParams(
        settlements=(0.0, 1.0, 2.0, 3.0),
        ei_factors=(0.9, 1.2, 1.1),
        kv_factors=(1.0, 1.0, 1.0, 1.0),
    )
    
    # 组装系统
    system = assemble_structural_system(struct_params)
    
    # 测试温度
    n_cases = 5
    dT_matrix = np.array([
        [10.0, 8.0, 12.0],
        [5.0, 15.0, 10.0],
        [8.0, 8.0, 8.0],
        [12.0, 6.0, 14.0],
        [7.0, 9.0, 11.0],
    ])
    
    print(f"\n测试配置:")
    print(f"  工况数: {n_cases}")
    print(f"  温度矩阵:\n{dT_matrix}")
    
    # 方法1：循环计算（原版方法）
    print(f"\n方法1：循环计算（参考）")
    loads_loop = []
    for i in range(n_cases):
        thermal_state = ThermalState(dT_spans=tuple(dT_matrix[i]))
        load_vec = thermal_load_vector(system, thermal_state)
        loads_loop.append(load_vec)
    loads_loop = np.array(loads_loop)
    
    print(f"  结果形状: {loads_loop.shape}")
    print(f"  前3个载荷向量的前10个元素:")
    for i in range(min(3, n_cases)):
        print(f"    Case {i}: {loads_loop[i, :10]}")
    
    # 方法2：批量计算（V2方法）
    print(f"\n方法2：批量计算（V2）")
    loads_batch = batch_thermal_load_vectors(system, dT_matrix)
    
    print(f"  结果形状: {loads_batch.shape}")
    print(f"  前3个载荷向量的前10个元素:")
    for i in range(min(3, n_cases)):
        print(f"    Case {i}: {loads_batch[i, :10]}")
    
    # 比较
    print(f"\n差异分析:")
    diff = loads_loop - loads_batch
    print(f"  最大绝对差异: {np.max(np.abs(diff)):.2e}")
    print(f"  平均绝对差异: {np.mean(np.abs(diff)):.2e}")
    print(f"  相对差异 (RMS): {np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(loads_loop**2)) * 100:.6f}%")
    
    # 详细检查非零元素
    nonzero_mask = np.abs(loads_loop) > 1e-10
    if np.any(nonzero_mask):
        rel_diff = np.abs(diff[nonzero_mask] / loads_loop[nonzero_mask])
        print(f"  最大相对差异（非零元素）: {np.max(rel_diff):.2e}")
    
    # 判断
    print(f"\n结论:")
    if np.max(np.abs(diff)) < 1e-10:
        print("  ✓✓✓ 完全一致（机器精度内）")
        return True
    elif np.max(np.abs(diff)) < 1e-8:
        print("  ✓✓ 非常接近（数值误差可接受）")
        return True
    else:
        print("  ✗ 存在显著差异 - V2 实现可能有问题")
        print(f"\n  最大差异位置:")
        max_diff_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"    工况 {max_diff_idx[0]}, DOF {max_diff_idx[1]}")
        print(f"    循环方法: {loads_loop[max_diff_idx]:.6e}")
        print(f"    批量方法: {loads_batch[max_diff_idx]:.6e}")
        print(f"    差异:     {diff[max_diff_idx]:.6e}")
        return False


def test_batch_solve():
    """测试批量求解的正确性"""
    
    print("\n" + "="*70)
    print("诊断测试：批量求解")
    print("="*70)
    
    from scipy.linalg import lu_solve
    
    # 创建测试系统
    struct_params = StructuralParams()
    system = assemble_structural_system(struct_params)
    
    n_cases = 3
    dT_matrix = np.array([
        [10.0, 8.0, 12.0],
        [5.0, 15.0, 10.0],
        [8.0, 8.0, 8.0],
    ])
    
    print(f"\n测试配置:")
    print(f"  工况数: {n_cases}")
    
    # 计算温度载荷
    from models.fit_multi_case_v2 import batch_thermal_load_vectors
    thermal_loads = batch_thermal_load_vectors(system, dT_matrix)
    
    # 方法1：循环求解
    print(f"\n方法1：循环求解")
    U_loop = []
    reactions_loop = []
    for i in range(n_cases):
        thermal_state = ThermalState(dT_spans=tuple(dT_matrix[i]))
        U, reactions = solve_with_system(system, thermal_state)
        U_loop.append(U)
        reactions_loop.append(reactions)
    U_loop = np.array(U_loop)
    reactions_loop = np.array(reactions_loop)
    
    print(f"  位移形状: {U_loop.shape}")
    print(f"  反力:\n{reactions_loop}")
    
    # 方法2：批量求解
    print(f"\n方法2：批量求解")
    F_all = system.load_base[np.newaxis, :] - thermal_loads
    U_batch = lu_solve((system.lu, system.piv), F_all.T).T
    
    # 计算反力
    reactions_batch = np.zeros((n_cases, 4))
    for i, idx in enumerate(system.support_nodes):
        ui_all = U_batch[:, 2 * idx]
        reactions_batch[:, i] = -system.kv_values[i] * (ui_all - system.settlements[i]) / 1_000.0
    
    print(f"  位移形状: {U_batch.shape}")
    print(f"  反力:\n{reactions_batch}")
    
    # 比较
    print(f"\n差异分析:")
    diff_U = U_loop - U_batch
    diff_R = reactions_loop - reactions_batch
    
    print(f"  位移最大差异: {np.max(np.abs(diff_U)):.2e}")
    print(f"  反力最大差异: {np.max(np.abs(diff_R)):.2e} kN")
    
    print(f"\n结论:")
    if np.max(np.abs(diff_U)) < 1e-10 and np.max(np.abs(diff_R)) < 1e-10:
        print("  ✓✓✓ 完全一致")
        return True
    else:
        print("  ⚠ 存在差异")
        return False


if __name__ == '__main__':
    test1_pass = test_batch_thermal_load()
    test2_pass = test_batch_solve()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    if test1_pass and test2_pass:
        print("✓ 所有测试通过 - V2 实现正确")
    else:
        print("✗ 存在问题 - 需要修复")
