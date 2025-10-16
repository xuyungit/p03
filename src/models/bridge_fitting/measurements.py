"""Measurement data extraction and processing."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ColumnNames, MeasurementConfig


def extract_measurement_matrix(
    df: pd.DataFrame,
    col_names: List[str],
    measurement_type: str,
    enabled: bool,
    weight: float
) -> Tuple[Optional[np.ndarray], bool]:
    """提取测量矩阵的通用函数。
    
    Args:
        df: 数据DataFrame
        col_names: 列名列表
        measurement_type: 测量类型名称
        enabled: 是否启用该测量类型
        weight: 测量权重
    
    Returns:
        (matrix, enabled): 测量矩阵和更新后的启用状态
    """
    if not enabled:
        return None, False
    
    if all(col in df.columns for col in col_names):
        matrix = df[col_names].to_numpy()
        print(f"\n✓ 使用{measurement_type} (权重={weight})")
        return matrix, True
    else:
        print(f"\n✗ 警告: 请求使用{measurement_type}但数据中不包含，将跳过{measurement_type}约束")
        return None, False


def setup_measurement_matrices(
    df: pd.DataFrame,
    measurement_config: MeasurementConfig
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], MeasurementConfig]:
    """设置所有测量矩阵。
    
    Args:
        df: 数据DataFrame
        measurement_config: 测量配置
    
    Returns:
        (reactions_matrix, displacements_matrix, rotations_matrix, 
         span_rotations_matrix, updated_config)
    """
    reactions_matrix = df[ColumnNames.REACTIONS].to_numpy()
    
    displacements_matrix, use_displacements = extract_measurement_matrix(
        df, ColumnNames.DISPLACEMENTS, "支座位移测量值",
        measurement_config.use_displacements, measurement_config.displacement_weight
    )
    
    rotations_matrix, use_rotations = extract_measurement_matrix(
        df, ColumnNames.ROTATIONS, "支座转角测量值",
        measurement_config.use_rotations, measurement_config.rotation_weight
    )
    
    span_rotations_matrix, use_span_rotations = extract_measurement_matrix(
        df, ColumnNames.SPAN_ROTATIONS, "跨中转角测量值",
        measurement_config.use_span_rotations, measurement_config.span_rotation_weight
    )
    
    updated_config = MeasurementConfig(
        use_reactions=True,
        use_displacements=use_displacements,
        use_rotations=use_rotations,
        use_span_rotations=use_span_rotations,
        displacement_weight=measurement_config.displacement_weight,
        rotation_weight=measurement_config.rotation_weight,
        span_rotation_weight=measurement_config.span_rotation_weight,
        auto_normalize=measurement_config.auto_normalize,
    )
    
    return reactions_matrix, displacements_matrix, rotations_matrix, span_rotations_matrix, updated_config
