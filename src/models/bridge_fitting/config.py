"""Configuration classes for bridge fitting."""

from dataclasses import dataclass


class ColumnNames:
    """数据列名常量"""
    REACTIONS = ['R_a_kN', 'R_b_kN', 'R_c_kN', 'R_d_kN']
    DISPLACEMENTS = ['v_A_mm', 'v_B_mm', 'v_C_mm', 'v_D_mm']
    ROTATIONS = ['theta_A_rad', 'theta_B_rad', 'theta_C_rad', 'theta_D_rad']
    SPAN_ROTATIONS = ['theta_S1-5_6L_rad', 'theta_S2-4_6L_rad', 'theta_S3-5_6L_rad']
    
    TEMPS_3_SPAN = ['dT_s1_C', 'dT_s2_C', 'dT_s3_C']
    TEMPS_2_SEG = ['dT_left_C', 'dT_right_C']
    
    SPAN_LENGTHS = ['span_length_s1_mm', 'span_length_s2_mm', 'span_length_s3_mm']
    SETTLEMENTS = ['settlement_a_mm', 'settlement_b_mm', 'settlement_c_mm', 'settlement_d_mm']
    EI_FACTORS = ['ei_factor_s1', 'ei_factor_s2', 'ei_factor_s3']
    KV_FACTORS = ['kv_factor_a', 'kv_factor_b', 'kv_factor_c', 'kv_factor_d']
    
    UNIFORM_LOAD = 'uniform_load_N_per_mm'
    SAMPLE_ID = 'sample_id'


@dataclass
class OptimizationConfig:
    """优化配置参数"""
    settlement_lower: float = -20.0
    settlement_upper: float = 20.0
    ei_factor_lower: float = 0.3
    ei_factor_upper: float = 1.5
    kv_factor_lower: float = 0.1
    kv_factor_upper: float = 3.0
    temp_gradient_lower: float = -10.0
    temp_gradient_upper: float = 20.0
    temp_gradient_initial: float = 10.0
    
    temp_spatial_diff_thresh: float = 3.0
    temp_temporal_diff_thresh: float = 1.0
    
    ftol: float = 1e-8
    gtol: float = 1e-8
    xtol: float = 1e-12
    
    min_scale_threshold: float = 1e-10


@dataclass
class MeasurementConfig:
    """Configuration for which measurements to use in fitting."""
    use_reactions: bool = True
    use_displacements: bool = False
    use_rotations: bool = False
    use_span_rotations: bool = False
    displacement_weight: float = 1.0
    rotation_weight: float = 1.0
    span_rotation_weight: float = 1.0
    auto_normalize: bool = True
    
    def __post_init__(self):
        if not any([self.use_reactions, self.use_displacements, self.use_rotations, self.use_span_rotations]):
            raise ValueError("At least one measurement type must be enabled")
    
    def count_measurements_per_case(self) -> int:
        """Count number of measurements per case."""
        count = 0
        if self.use_reactions:
            count += 4
        if self.use_displacements:
            count += 4
        if self.use_rotations:
            count += 4
        if self.use_span_rotations:
            count += 3
        return count
    
    def describe(self) -> str:
        """Return human-readable description."""
        parts = []
        if self.use_reactions:
            parts.append("反力(4)")
        if self.use_displacements:
            parts.append(f"位移(4, 权重={self.displacement_weight})")
        if self.use_rotations:
            parts.append(f"转角(4, 权重={self.rotation_weight})")
        if self.use_span_rotations:
            parts.append(f"跨中转角(3, 权重={self.span_rotation_weight})")
        
        if self.auto_normalize:
            parts.append("自动归一化")
        
        return " + ".join(parts)
