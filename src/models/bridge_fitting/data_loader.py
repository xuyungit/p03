"""Data loading utilities."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from models.bridge_forward_model import set_span_lengths_mm

from .config import ColumnNames, MeasurementConfig


def load_multi_case_data(
    csv_paths: List[Path], 
    max_samples: int | None = None,
    measurement_config: Optional[MeasurementConfig] = None,
) -> Tuple[pd.DataFrame, dict, MeasurementConfig]:
    """Load and concatenate multi-case data from multiple files."""
    dfs = []
    
    print(f"\n加载 {len(csv_paths)} 个数据文件:")
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        print(f"  文件 {i+1}: {csv_path.name} ({len(df)} 条数据)")
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df['sample_id'] = range(len(df))
    
    print(f"\n合并后数据集:")
    print(f"  总样本数: {len(df)}")
    
    if max_samples is not None and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()
        df['sample_id'] = range(len(df))
        print(f"  限制到前 {max_samples} 条数据")
    
    if measurement_config is None:
        has_reactions = all(col in df.columns for col in ColumnNames.REACTIONS)
        has_displacements = all(col in df.columns for col in ColumnNames.DISPLACEMENTS)
        has_rotations = all(col in df.columns for col in ColumnNames.ROTATIONS)
        has_span_rotations = all(col in df.columns for col in ColumnNames.SPAN_ROTATIONS)
        
        print(f"\n可用测量数据:")
        print(f"  反力 (R_a~R_d): {'✓' if has_reactions else '✗'}")
        print(f"  位移 (v_A~v_D): {'✓' if has_displacements else '✗'}")
        print(f"  转角 (theta_A~theta_D): {'✓' if has_rotations else '✗'}")
        print(f"  跨中转角 (S1-5/6L, S2-4/6L, S3-5/6L): {'✓' if has_span_rotations else '✗'}")
        
        measurement_config = MeasurementConfig(
            use_reactions=has_reactions,
            use_displacements=False,
            use_rotations=False,
            use_span_rotations=False,
        )
        print(f"\n默认使用: 仅反力")
    
    has_span_cols = all(col in df.columns for col in ColumnNames.SPAN_LENGTHS)
    if has_span_cols:
        spans_mm = tuple(float(df[col].iloc[0]) for col in ColumnNames.SPAN_LENGTHS)
        for i, col in enumerate(ColumnNames.SPAN_LENGTHS):
            if not np.isclose(df[col], spans_mm[i]).all():
                raise ValueError("span_length_x_mm 列应为常数")
        set_span_lengths_mm(spans_mm)
        print(
            "  侦测到跨度配置: "
            f"S1={spans_mm[0]/1000:.2f} m, S2={spans_mm[1]/1000:.2f} m, S3={spans_mm[2]/1000:.2f} m"
        )
    else:
        print("  未检测到跨度列，使用默认 40:40:40 m")

    struct_cols = (
        ColumnNames.SETTLEMENTS +
        ColumnNames.EI_FACTORS +
        ColumnNames.KV_FACTORS
    )
    if has_span_cols:
        struct_cols.extend(ColumnNames.SPAN_LENGTHS)
    
    const_params = {}
    for col in struct_cols:
        unique_vals = df[col].unique()
        if len(unique_vals) != 1:
            print(f"  警告: {col} 不是常数，有 {len(unique_vals)} 个不同值")
        const_params[col] = float(df[col].iloc[0])
    
    print(f"  结构参数验证: {'✓ 全部为常数' if all(len(df[c].unique()) == 1 for c in struct_cols) else '✗ 存在变化'}")
    
    return df, const_params, measurement_config
