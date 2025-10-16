"""Bridge fitting package - Modular implementation for bridge parameter fitting."""

from .cli import main
from .config import ColumnNames, MeasurementConfig, OptimizationConfig
from .data_loader import load_multi_case_data
from .fitter import VectorizedMultiCaseFitter, batch_solve_with_system
from .measurements import setup_measurement_matrices
from .residuals import add_residual_columns, print_residual_stats
from .results import extract_true_parameters, print_fitting_results

__all__ = [
    'main',
    'ColumnNames',
    'MeasurementConfig',
    'OptimizationConfig',
    'load_multi_case_data',
    'VectorizedMultiCaseFitter',
    'batch_solve_with_system',
    'setup_measurement_matrices',
    'add_residual_columns',
    'print_residual_stats',
    'extract_true_parameters',
    'print_fitting_results',
]
