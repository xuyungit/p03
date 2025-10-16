"""Residual computation and statistics."""

from typing import Optional

import numpy as np
import pandas as pd


def print_residual_stats(residuals: Optional[np.ndarray], name: str, unit: str) -> None:
    """打印残差统计信息。
    
    Args:
        residuals: 残差数组
        name: 残差类型名称
        unit: 单位
    """
    if residuals is None:
        return
    
    print(f"\n  {name}残差:")
    print(f"    RMSE:     {np.sqrt(np.mean(residuals**2)):.6e} {unit}")
    print(f"    最大残差: {np.max(np.abs(residuals)):.6e} {unit}")
    print(f"    平均残差: {np.mean(np.abs(residuals)):.6e} {unit}")


def add_residual_columns(
    df: pd.DataFrame,
    residuals: Optional[np.ndarray],
    column_prefix: str,
    n_cols: int
) -> None:
    """添加残差列到DataFrame。
    
    Args:
        df: 输出DataFrame
        residuals: 残差数组 (n_cases, n_cols)
        column_prefix: 列名前缀
        n_cols: 列数
    """
    if residuals is not None:
        for i in range(n_cols):
            df[f'{column_prefix}{i+1}'] = residuals[:, i]
