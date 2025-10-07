"""测试参数拟合功能

验证 parameter_fitter 模块能否正确反演结构参数。
"""

from __future__ import annotations

import numpy as np
import pytest

from models.bridge_forward_model import (
    StructuralParams,
    ThermalState,
    evaluate_forward,
)
from models.parameter_fitter import (
    FittingConfig,
    ParameterFitter,
    fit_parameters_from_reactions,
)


def test_fit_identity_case():
    """测试：如果测量值就是用初值生成的，应该能收敛回初值"""
    
    # 定义真实参数
    true_params = StructuralParams(
        settlements=(0.0, 0.0, 0.0, 0.0),
        ei_factors=(1.0, 1.0, 1.0),
        kv_factors=(1.0, 1.0, 1.0, 1.0),
    )
    true_thermal = ThermalState(dT_spans=(0.0, 0.0, 0.0))
    
    # 生成"测量"反力
    response = evaluate_forward(true_params, true_thermal)
    measured_reactions = response.reactions_kN
    
    # 拟合参数
    config = FittingConfig(verbose=0)
    result = fit_parameters_from_reactions(
        measured_reactions=measured_reactions,
        initial_params=true_params,
        initial_thermal=true_thermal,
        config=config,
    )
    
    # 验证
    assert result.success, f"优化未收敛: {result.optimize_result.message}"
    assert result.rmse < 1e-6, f"RMSE过大: {result.rmse}"
    
    # 验证反力误差
    np.testing.assert_allclose(
        result.reactions_fitted,
        measured_reactions,
        rtol=1e-5,
        atol=1e-6,
    )


def test_fit_with_settlements():
    """测试：拟合有沉降的情况"""
    
    # 真实参数（有沉降）
    true_params = StructuralParams(
        settlements=(5.0, -3.0, 2.0, -1.0),  # mm
        ei_factors=(1.0, 1.0, 1.0),
        kv_factors=(1.0, 1.0, 1.0, 1.0),
    )
    true_thermal = ThermalState(dT_spans=(0.0, 0.0, 0.0))
    
    # 生成测量反力
    response = evaluate_forward(true_params, true_thermal)
    measured_reactions = response.reactions_kN
    
    # 从零初值拟合
    initial_params = StructuralParams()
    initial_thermal = ThermalState()
    
    config = FittingConfig(
        fit_settlements=True,
        fit_ei_factors=False,  # 固定EI
        fit_kv_factors=False,  # 固定Kv
        fit_temperature=False,  # 固定温度
        verbose=0,
    )
    
    result = fit_parameters_from_reactions(
        measured_reactions=measured_reactions,
        initial_params=initial_params,
        initial_thermal=initial_thermal,
        config=config,
    )
    
    # 验证
    assert result.success
    assert result.rmse < 1e-3  # 应该能拟合得很好
    
    # 验证沉降值（允许一定误差）
    np.testing.assert_allclose(
        result.settlements,
        true_params.settlements,
        rtol=0.01,
        atol=0.1,  # ±0.1 mm
    )


def test_fit_with_temperature():
    """测试：拟合有温度梯度的情况"""
    
    # 真实参数（有温度）
    true_params = StructuralParams()
    true_thermal = ThermalState(dT_spans=(10.0, -5.0, 8.0))  # °C
    
    # 生成测量反力
    response = evaluate_forward(true_params, true_thermal)
    measured_reactions = response.reactions_kN
    
    # 从零初值拟合
    config = FittingConfig(
        fit_settlements=False,
        fit_ei_factors=False,
        fit_kv_factors=False,
        fit_temperature=True,  # 只拟合温度
        verbose=0,
    )
    
    result = fit_parameters_from_reactions(
        measured_reactions=measured_reactions,
        initial_params=StructuralParams(),
        initial_thermal=ThermalState(),
        config=config,
    )
    
    # 验证
    assert result.success
    assert result.rmse < 1e-3
    
    # 验证温度值
    np.testing.assert_allclose(
        result.dT_spans,
        true_thermal.dT_spans,
        rtol=0.01,
        atol=0.1,  # ±0.1 °C
    )


def test_fit_all_parameters():
    """测试：同时拟合所有参数（欠定问题，但应该能找到一组解）"""
    
    # 真实参数
    true_params = StructuralParams(
        settlements=(2.0, -1.0, 1.5, -0.5),
        ei_factors=(1.1, 0.9, 1.05),
        kv_factors=(1.2, 0.8, 1.1, 0.95),
    )
    true_thermal = ThermalState(dT_spans=(5.0, -3.0, 4.0))
    
    # 生成测量反力
    response = evaluate_forward(true_params, true_thermal)
    measured_reactions = response.reactions_kN
    
    # 拟合所有参数
    config = FittingConfig(
        fit_settlements=True,
        fit_ei_factors=True,
        fit_kv_factors=True,
        fit_temperature=True,
        verbose=0,
    )
    
    # 使用真实值附近的初值
    initial_params = StructuralParams(
        settlements=(0.0, 0.0, 0.0, 0.0),
        ei_factors=(1.0, 1.0, 1.0),
        kv_factors=(1.0, 1.0, 1.0, 1.0),
    )
    initial_thermal = ThermalState(dT_spans=(0.0, 0.0, 0.0))
    
    result = fit_parameters_from_reactions(
        measured_reactions=measured_reactions,
        initial_params=initial_params,
        initial_thermal=initial_thermal,
        config=config,
    )
    
    # 验证：至少应该能重现反力（即使参数可能不唯一）
    assert result.success
    assert result.rmse < 0.1  # 反力拟合误差应该很小
    
    np.testing.assert_allclose(
        result.reactions_fitted,
        measured_reactions,
        rtol=1e-3,
        atol=0.01,  # ±0.01 kN
    )


def test_batch_fitting():
    """测试：批量拟合多组数据"""
    
    # 生成多组测量数据
    n_samples = 5
    true_settlements = [
        (i * 0.5, -i * 0.3, i * 0.2, -i * 0.1)
        for i in range(n_samples)
    ]
    
    measured_reactions_list = []
    for settlements in true_settlements:
        params = StructuralParams(settlements=settlements)
        thermal = ThermalState()
        response = evaluate_forward(params, thermal)
        measured_reactions_list.append(response.reactions_kN)
    
    measured_reactions_batch = np.array(measured_reactions_list)
    
    # 批量拟合
    config = FittingConfig(
        fit_settlements=True,
        fit_ei_factors=False,
        fit_kv_factors=False,
        fit_temperature=False,
        verbose=0,
    )
    
    fitter = ParameterFitter(
        config=config,
        initial_params=StructuralParams(),
        initial_thermal=ThermalState(),
    )
    
    results = fitter.fit_batch(measured_reactions_batch)
    
    # 验证
    assert len(results) == n_samples
    
    for i, result in enumerate(results):
        assert result.success, f"Sample {i} failed"
        assert result.rmse < 1e-3
        
        # 验证沉降值
        np.testing.assert_allclose(
            result.settlements,
            true_settlements[i],
            rtol=0.01,
            atol=0.1,
        )


def test_parameter_bounds():
    """测试：参数边界约束"""
    
    # 生成测量反力
    true_params = StructuralParams(settlements=(10.0, -10.0, 5.0, -5.0))
    response = evaluate_forward(true_params, ThermalState())
    measured_reactions = response.reactions_kN
    
    # 设置严格的边界
    from models.parameter_fitter import ParameterBounds
    
    bounds = ParameterBounds(
        settlement_lower=-8.0,  # 比真实值更严格
        settlement_upper=8.0,
    )
    
    config = FittingConfig(
        fit_settlements=True,
        fit_ei_factors=False,
        fit_kv_factors=False,
        fit_temperature=False,
        bounds=bounds,
        verbose=0,
    )
    
    result = fit_parameters_from_reactions(
        measured_reactions=measured_reactions,
        config=config,
    )
    
    # 验证：所有参数应该在边界内
    for s in result.settlements:
        assert bounds.settlement_lower <= s <= bounds.settlement_upper


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
