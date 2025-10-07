"""参数拟合示例脚本

演示如何使用 parameter_fitter 模块根据测量的反力来拟合结构参数。

用法:
    # 从单次测量拟合参数
    uv run python src/models/fit_params_demo.py --reactions 1250.0 1350.0 1400.0 1300.0
    
    # 从CSV文件批量拟合
    uv run python src/models/fit_params_demo.py --csv data/measured_reactions.csv --output fitted_params.csv
    
    # 只拟合部分参数（固定其他参数）
    uv run python src/models/fit_params_demo.py --reactions 1250.0 1350.0 1400.0 1300.0 --no-fit-kv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.bridge_forward_model import StructuralParams, ThermalState
from models.parameter_fitter import (
    FittingConfig,
    ParameterBounds,
    ParameterFitter,
    fit_parameters_from_reactions,
)


def print_fitting_result(result, label: str = "") -> None:
    """打印拟合结果"""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("  拟合结果")
        print(f"{'='*60}")
    
    print("\n拟合的参数:")
    print(f"  Settlements (mm): {result.settlements}")
    print(f"  EI factors:       {result.ei_factors}")
    print(f"  Kv factors:       {result.kv_factors}")
    print(f"  dT spans (°C):    {result.dT_spans}")
    
    print(f"\n拟合后的反力 (kN): {result.reactions_fitted}")
    
    print(f"\n残差统计:")
    print(f"  RMSE:            {result.rmse:.6f} kN")
    print(f"  最大绝对误差:     {result.max_abs_error:.6f} kN")
    print(f"  残差范数:         {result.residual_norm:.6f}")
    
    print(f"\n优化状态:")
    print(f"  成功:            {result.success}")
    print(f"  迭代次数:         {result.optimize_result.nfev}")
    print(f"  优化消息:         {result.optimize_result.message}")


def fit_single_measurement(
    reactions: list[float],
    initial_params: Optional[StructuralParams] = None,
    initial_thermal: Optional[ThermalState] = None,
    fit_settlements: bool = True,
    fit_ei: bool = True,
    fit_kv: bool = True,
    fit_temperature: bool = True,
    verbose: int = 2,
) -> None:
    """拟合单次测量数据"""
    
    print("输入的测量反力 (kN):", reactions)
    
    # 设置拟合配置
    config = FittingConfig(
        fit_settlements=fit_settlements,
        fit_ei_factors=fit_ei,
        fit_kv_factors=fit_kv,
        fit_temperature=fit_temperature,
        verbose=verbose,
    )
    
    # 执行拟合
    result = fit_parameters_from_reactions(
        measured_reactions=reactions,
        initial_params=initial_params,
        initial_thermal=initial_thermal,
        config=config,
    )
    
    # 打印结果
    print_fitting_result(result)


def fit_from_csv(
    csv_path: Path,
    output_path: Optional[Path] = None,
    reaction_cols: Optional[list[str]] = None,
    fit_settlements: bool = True,
    fit_ei: bool = True,
    fit_kv: bool = True,
    fit_temperature: bool = True,
    verbose: int = 1,
) -> None:
    """从CSV文件批量拟合参数"""
    
    # 读取测量数据
    df = pd.read_csv(csv_path)
    print(f"读取 {len(df)} 条测量记录")
    
    # 确定反力列
    if reaction_cols is None:
        # 自动检测形如 R1, R2, R3, R4 的列
        reaction_cols = [col for col in df.columns if col in ['R1', 'R2', 'R3', 'R4']]
        if len(reaction_cols) != 4:
            raise ValueError(
                f"未找到4个反力列。请使用 --reaction-cols 指定列名。"
                f"可用列: {df.columns.tolist()}"
            )
    
    print(f"使用反力列: {reaction_cols}")
    
    # 提取反力数据
    reactions_data = df[reaction_cols].to_numpy(dtype=float)
    
    # 设置拟合配置
    config = FittingConfig(
        fit_settlements=fit_settlements,
        fit_ei_factors=fit_ei,
        fit_kv_factors=fit_kv,
        fit_temperature=fit_temperature,
        verbose=verbose,
    )
    
    # 创建拟合器
    fitter = ParameterFitter(
        config=config,
        initial_params=StructuralParams(),
        initial_thermal=ThermalState(),
    )
    
    # 批量拟合
    print("\n开始批量拟合...")
    results = fitter.fit_batch(reactions_data)
    
    # 统计成功率
    n_success = sum(1 for r in results if r.success)
    print(f"\n拟合完成: {n_success}/{len(results)} 次成功")
    
    # 构建结果DataFrame
    result_data = {
        'settlement_A': [r.settlements[0] for r in results],
        'settlement_B': [r.settlements[1] for r in results],
        'settlement_C': [r.settlements[2] for r in results],
        'settlement_D': [r.settlements[3] for r in results],
        'ei_factor_1': [r.ei_factors[0] for r in results],
        'ei_factor_2': [r.ei_factors[1] for r in results],
        'ei_factor_3': [r.ei_factors[2] for r in results],
        'kv_factor_A': [r.kv_factors[0] for r in results],
        'kv_factor_B': [r.kv_factors[1] for r in results],
        'kv_factor_C': [r.kv_factors[2] for r in results],
        'kv_factor_D': [r.kv_factors[3] for r in results],
        'dT_span1': [r.dT_spans[0] for r in results],
        'dT_span2': [r.dT_spans[1] for r in results],
        'dT_span3': [r.dT_spans[2] for r in results],
        'R1_fitted': [r.reactions_fitted[0] for r in results],
        'R2_fitted': [r.reactions_fitted[1] for r in results],
        'R3_fitted': [r.reactions_fitted[2] for r in results],
        'R4_fitted': [r.reactions_fitted[3] for r in results],
        'rmse': [r.rmse for r in results],
        'max_abs_error': [r.max_abs_error for r in results],
        'success': [r.success for r in results],
    }
    
    result_df = pd.DataFrame(result_data)
    
    # 添加原始测量值
    for i, col in enumerate(reaction_cols):
        result_df[f'{col}_measured'] = reactions_data[:, i]
    
    # 保存结果
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_fitted.csv"
    
    result_df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 打印统计信息
    print(f"\n拟合质量统计:")
    print(f"  平均 RMSE:        {result_df['rmse'].mean():.6f} kN")
    print(f"  平均最大误差:      {result_df['max_abs_error'].mean():.6f} kN")
    print(f"  成功率:           {n_success}/{len(results)} ({100*n_success/len(results):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="根据测量反力拟合结构参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--reactions',
        nargs=4,
        type=float,
        metavar=('R1', 'R2', 'R3', 'R4'),
        help='单次测量的4个反力值 (kN)',
    )
    input_group.add_argument(
        '--csv',
        type=Path,
        help='包含多次测量的CSV文件',
    )
    
    # 输出选项
    parser.add_argument(
        '--output',
        type=Path,
        help='输出CSV文件路径（仅用于 --csv 模式）',
    )
    
    parser.add_argument(
        '--reaction-cols',
        nargs=4,
        metavar=('COL1', 'COL2', 'COL3', 'COL4'),
        help='CSV中反力列的名称（默认自动检测 R1, R2, R3, R4）',
    )
    
    # 拟合选项
    parser.add_argument(
        '--no-fit-settlements',
        action='store_true',
        help='不拟合支座沉降（使用初值固定）',
    )
    parser.add_argument(
        '--no-fit-ei',
        action='store_true',
        help='不拟合EI因子（使用初值固定）',
    )
    parser.add_argument(
        '--no-fit-kv',
        action='store_true',
        help='不拟合支座刚度因子（使用初值固定）',
    )
    parser.add_argument(
        '--no-fit-temperature',
        action='store_true',
        help='不拟合温度梯度（使用初值固定）',
    )
    
    # 其他选项
    parser.add_argument(
        '--verbose',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='详细程度（0=静默, 1=进度, 2=详细）',
    )
    
    args = parser.parse_args()
    
    # 执行拟合
    if args.reactions:
        # 单次测量模式
        fit_single_measurement(
            reactions=args.reactions,
            fit_settlements=not args.no_fit_settlements,
            fit_ei=not args.no_fit_ei,
            fit_kv=not args.no_fit_kv,
            fit_temperature=not args.no_fit_temperature,
            verbose=args.verbose,
        )
    else:
        # CSV批量模式
        if not args.csv.exists():
            print(f"错误: 文件不存在: {args.csv}", file=sys.stderr)
            sys.exit(1)
        
        fit_from_csv(
            csv_path=args.csv,
            output_path=args.output,
            reaction_cols=args.reaction_cols,
            fit_settlements=not args.no_fit_settlements,
            fit_ei=not args.no_fit_ei,
            fit_kv=not args.no_fit_kv,
            fit_temperature=not args.no_fit_temperature,
            verbose=args.verbose,
        )


if __name__ == '__main__':
    main()
