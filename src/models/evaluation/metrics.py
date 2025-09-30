"""
Evaluation metrics for model performance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Metrics:
    """
    Class for computing evaluation metrics
    """
    
    @staticmethod
    def mean_absolute_error(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        """
        Compute MAE for each target
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with MAE for each target
        """
        mae_dict = {}
        
        for col in y_true.columns:
            if col in y_pred.columns:
                mae = mean_absolute_error(y_true[col], y_pred[col])
                mae_dict[col] = mae
            else:
                logger.warning(f"Column '{col}' not found in predictions")
                
        # Overall MAE
        mae_dict['overall'] = np.mean(list(mae_dict.values()))
        
        return mae_dict
        
    @staticmethod
    def mean_squared_error(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        """
        Compute MSE for each target
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with MSE for each target
        """
        mse_dict = {}
        
        for col in y_true.columns:
            if col in y_pred.columns:
                mse = mean_squared_error(y_true[col], y_pred[col])
                mse_dict[col] = mse
            else:
                logger.warning(f"Column '{col}' not found in predictions")
                
        # Overall MSE
        mse_dict['overall'] = np.mean(list(mse_dict.values()))
        
        return mse_dict
        
    @staticmethod
    def root_mean_squared_error(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        """
        Compute RMSE for each target
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with RMSE for each target
        """
        mse_dict = Metrics.mean_squared_error(y_true, y_pred)
        rmse_dict = {k: np.sqrt(v) for k, v in mse_dict.items()}
        
        return rmse_dict
        
    @staticmethod
    def r2_score(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        """
        Compute R² score for each target
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with R² for each target
        """
        r2_dict = {}
        
        for col in y_true.columns:
            if col in y_pred.columns:
                r2 = r2_score(y_true[col], y_pred[col])
                r2_dict[col] = r2
            else:
                logger.warning(f"Column '{col}' not found in predictions")
                
        # Overall R² (average)
        r2_dict['overall'] = np.mean(list(r2_dict.values()))
        
        return r2_dict
        
    @staticmethod
    def mean_absolute_percentage_error(y_true: pd.DataFrame, y_pred: pd.DataFrame, 
                                     epsilon: float = 1e-10) -> Dict[str, float]:
        """
        Compute MAPE for each target
        
        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            Dictionary with MAPE for each target
        """
        mape_dict = {}
        
        for col in y_true.columns:
            if col in y_pred.columns:
                # Avoid division by (near-)zero by using a dynamic threshold
                # Based on data scale: 0.5% of the 95th percentile magnitude, with a hard lower bound `epsilon`.
                col_vals = y_true[col].to_numpy()
                # Use percentiles for robustness against outliers
                col_scale = np.nanpercentile(np.abs(col_vals), 95)
                dyn_eps = max(epsilon, 0.005 * col_scale)
                mask = np.abs(y_true[col]) > dyn_eps
                if mask.sum() > 0:
                    mape = np.mean(np.abs((y_true[col][mask] - y_pred[col][mask]) / y_true[col][mask])) * 100
                    mape_dict[col] = mape
                else:
                    logger.warning(f"Column '{col}' has all zero or near-zero values (threshold={dyn_eps:.4g})")
                    mape_dict[col] = np.nan
            else:
                logger.warning(f"Column '{col}' not found in predictions")
                
        # Overall MAPE
        valid_mapes = [v for v in mape_dict.values() if not np.isnan(v)]
        mape_dict['overall'] = np.mean(valid_mapes) if valid_mapes else np.nan
        
        return mape_dict

    @staticmethod
    def mean_absolute_error_percent(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        denom: str = "mean_abs",
        epsilon: float = 1e-10,
    ) -> Dict[str, float]:
        """
        Compute MAE expressed as a percentage of a column-level typical true magnitude.

        This is more stable than MAPE when y_true has near-zeros.

        denom options:
          - "mean_abs": 100 * MAE / mean(|y_true|)
          - "rms":      100 * MAE / rms(y_true)
          - "p95":      100 * MAE / (0.95-quantile magnitude), i.e., q95(|y_true|)
        A dynamic floor `epsilon` avoids division by near-zero denominators.
        """
        def _rms(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

        out: Dict[str, float] = {}
        for col in y_true.columns:
            if col not in y_pred.columns:
                logging.warning(f"Column '{col}' not found in predictions")
                continue
            t = y_true[col].to_numpy()
            p = y_pred[col].to_numpy()
            mae = float(mean_absolute_error(t, p))
            mag = np.abs(t)
            if denom == "mean_abs":
                d = float(np.mean(mag))
            elif denom == "rms":
                d = _rms(t)
            elif denom == "p95":
                d = float(np.nanpercentile(mag, 95))
            else:
                raise ValueError("Unsupported denom; use 'mean_abs', 'rms', or 'p95'")
            d = max(d, epsilon)
            out[col] = 100.0 * mae / d

        # Overall (average across columns)
        out['overall'] = float(np.mean(list(out.values()))) if out else np.nan
        return out
        
    @staticmethod
    def compute_all_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all metrics
        """
        # Ensure column names match
        y_pred.columns = y_true.columns
        
        metrics = {
            'mae': Metrics.mean_absolute_error(y_true, y_pred),
            'mse': Metrics.mean_squared_error(y_true, y_pred),
            'rmse': Metrics.root_mean_squared_error(y_true, y_pred),
            'r2': Metrics.r2_score(y_true, y_pred),
            'mape': Metrics.mean_absolute_percentage_error(y_true, y_pred)
        }
        
        return metrics
        
    @staticmethod
    def check_physical_constraints(y_pred: pd.DataFrame, 
                                 stiffness_cols: List[str],
                                 height_cols: List[str]) -> Dict[str, Any]:
        """
        Check if predictions satisfy physical constraints
        
        Args:
            y_pred: Predicted values
            stiffness_cols: Column names for stiffness values (should be positive)
            height_cols: Column names for height values
            
        Returns:
            Dictionary with constraint violation information
        """
        violations = {}
        
        # Check stiffness positivity
        for col in stiffness_cols:
            if col in y_pred.columns:
                negative_mask = y_pred[col] < 0
                if negative_mask.any():
                    violations[f'{col}_negative'] = {
                        'count': negative_mask.sum(),
                        'percentage': (negative_mask.sum() / len(y_pred)) * 100,
                        'min_value': y_pred[col].min()
                    }
                    
        # Summary
        violations['total_violations'] = sum(v['count'] for v in violations.values() if isinstance(v, dict))
        violations['has_violations'] = violations['total_violations'] > 0
        
        return violations

    @staticmethod
    def detailed_error_analysis(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Perform detailed error analysis for each parameter
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with detailed error analysis for each parameter
        """
        error_analysis = {}
        
        for col in y_true.columns:
            if col not in y_pred.columns:
                logger.warning(f"Column '{col}' not found in predictions")
                continue
                
            true_vals = y_true[col].values
            pred_vals = y_pred[col].values
            
            # Absolute errors
            abs_errors = np.abs(true_vals - pred_vals)
            
            # Relative errors (percentage)
            # Avoid division by zero
            epsilon = 1e-10
            mask = np.abs(true_vals) > epsilon
            rel_errors = np.zeros_like(true_vals)
            if mask.sum() > 0:
                rel_errors[mask] = ((true_vals[mask] - pred_vals[mask]) / true_vals[mask]) * 100
                rel_errors[~mask] = np.nan
            
            # Error statistics
            analysis = {
                # Basic statistics
                'mean_absolute_error': np.mean(abs_errors),
                'std_absolute_error': np.std(abs_errors),
                'max_absolute_error': np.max(abs_errors),
                'min_absolute_error': np.min(abs_errors),
                
                # Relative error statistics (excluding NaN values)
                'mean_relative_error': np.nanmean(rel_errors),
                'std_relative_error': np.nanstd(rel_errors),
                'max_relative_error': np.nanmax(rel_errors),
                'min_relative_error': np.nanmin(rel_errors),
                
                # Error ranges
                'error_range_abs': f"{np.min(abs_errors):.4f} ~ {np.max(abs_errors):.4f}",
                'error_range_rel': f"{np.nanmin(rel_errors):.2f}% ~ {np.nanmax(rel_errors):.2f}%",
                
                # Percentile analysis
                'abs_error_percentiles': {
                    '25%': np.percentile(abs_errors, 25),
                    '50%': np.percentile(abs_errors, 50),
                    '75%': np.percentile(abs_errors, 75),
                    '90%': np.percentile(abs_errors, 90),
                    '95%': np.percentile(abs_errors, 95),
                    '99%': np.percentile(abs_errors, 99)
                },
                'rel_error_percentiles': {
                    '25%': np.nanpercentile(rel_errors, 25),
                    '50%': np.nanpercentile(rel_errors, 50),
                    '75%': np.nanpercentile(rel_errors, 75),
                    '90%': np.nanpercentile(rel_errors, 90),
                    '95%': np.nanpercentile(rel_errors, 95),
                    '99%': np.nanpercentile(rel_errors, 99)
                },
                
                # Error distribution analysis
                'within_5_percent': np.nansum(np.abs(rel_errors) <= 5) / len(rel_errors) * 100,
                'within_10_percent': np.nansum(np.abs(rel_errors) <= 10) / len(rel_errors) * 100,
                'within_20_percent': np.nansum(np.abs(rel_errors) <= 20) / len(rel_errors) * 100,
                
                # Bias analysis
                'mean_bias': np.mean(true_vals - pred_vals),
                'bias_percentage': np.nanmean(rel_errors),
                
                # Sample statistics for context
                'true_mean': np.mean(true_vals),
                'true_std': np.std(true_vals),
                'pred_mean': np.mean(pred_vals),
                'pred_std': np.std(pred_vals)
            }
            
            error_analysis[col] = analysis
            
        return error_analysis
