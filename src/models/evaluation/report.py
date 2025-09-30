"""
Evaluation report generation
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json

from .metrics import Metrics

logger = logging.getLogger(__name__)


class EvaluationReport:
    """
    Class for generating evaluation reports
    """
    
    def __init__(self, model_name: str):
        """
        Initialize evaluation report
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.timestamp = datetime.now().isoformat()
        self.metrics = None
        self.predictions = None
        self.true_values = None
        
    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Evaluate model performance
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        # Store data
        self.true_values = y_true
        self.predictions = y_pred
        
        # Compute metrics
        self.metrics = Metrics.compute_all_metrics(y_true, y_pred)
        
        # Compute detailed error analysis
        self.error_analysis = Metrics.detailed_error_analysis(y_true, y_pred)
            
        logger.info(f"Evaluation completed for model '{self.model_name}'")
        
    def generate_text_report(self) -> str:
        """
        Generate text report
        
        Returns:
            Text report as string
        """
        if self.metrics is None:
            raise RuntimeError("Must run evaluate() before generating report")
            
        report_lines = [
            f"=" * 80,
            f"Model Evaluation Report: {self.model_name}",
            f"Timestamp: {self.timestamp}",
            f"=" * 80,
            "",
            "Overall Performance Metrics:",
            f"  - R² Score: {self.metrics['r2']['overall']:.4f}",
            f"  - RMSE: {self.metrics['rmse']['overall']:.4f}",
            f"  - MAE: {self.metrics['mae']['overall']:.4f}",
            f"  - MAPE: {self.metrics['mape']['overall']:.2f}%",
            "",
            "Per-Target Performance:",
            "-" * 40
        ]
        
        # Per-target metrics
        for target in self.true_values.columns:
            report_lines.extend([
                f"\n{target}:",
                f"  R²: {self.metrics['r2'][target]:.4f}",
                f"  RMSE: {self.metrics['rmse'][target]:.4f}",
                f"  MAE: {self.metrics['mae'][target]:.4f}",
                f"  MAPE: {self.metrics['mape'][target]:.2f}%"
            ])
                    
        report_lines.extend(["", "=" * 80])
        
        return "\n".join(report_lines)
        
    def generate_detailed_error_report(self) -> str:
        """
        Generate detailed error analysis report
        
        Returns:
            Detailed error report as string
        """
        if self.error_analysis is None:
            raise RuntimeError("Must run evaluate() before generating detailed error report")
            
        report_lines = [
            f"=" * 80,
            f"Detailed Error Analysis Report: {self.model_name}",
            f"Timestamp: {self.timestamp}",
            f"=" * 80,
            ""
        ]
        
        # Group parameters by type
        reaction_cols = [col for col in self.true_values.columns if col.startswith('R')]
        displacement_cols = [col for col in self.true_values.columns if col.startswith('D')]
        
        # Summary statistics
        report_lines.extend([
            "SUMMARY STATISTICS:",
            "-" * 40
        ])
        
        # Overall error statistics
        all_rel_errors = []
        for col in self.true_values.columns:
            if col in self.error_analysis:
                analysis = self.error_analysis[col]
                all_rel_errors.append(analysis['mean_relative_error'])
                
        if all_rel_errors:
            report_lines.extend([
                f"Overall Mean Relative Error: {np.mean(all_rel_errors):.2f}%",
                f"Overall Error Range: {np.min(all_rel_errors):.2f}% ~ {np.max(all_rel_errors):.2f}%",
                ""
            ])
        
        # Reaction forces analysis
        if reaction_cols:
            report_lines.extend([
                "REACTION FORCES (R1, R2, R3, R4) ANALYSIS:",
                "-" * 50
            ])
            
            for col in reaction_cols:
                if col in self.error_analysis:
                    analysis = self.error_analysis[col]
                    report_lines.extend([
                        f"\n{col}:",
                        f"  平均绝对误差: {analysis['mean_absolute_error']:.4f}",
                        f"  平均相对误差: {analysis['mean_relative_error']:.2f}%",
                        f"  误差范围: {analysis['error_range_rel']}",
                        f"  误差分布:",
                        f"    ±5% 以内: {analysis['within_5_percent']:.1f}%",
                        f"    ±10% 以内: {analysis['within_10_percent']:.1f}%",
                        f"    ±20% 以内: {analysis['within_20_percent']:.1f}%",
                        f"  偏差: {analysis['mean_bias']:.4f} ({analysis['bias_percentage']:.2f}%)",
                        f"  真实值范围: {analysis['true_mean']:.2f} ± {analysis['true_std']:.2f}",
                        f"  预测值范围: {analysis['pred_mean']:.2f} ± {analysis['pred_std']:.2f}"
                    ])
        
        # Displacement analysis
        if displacement_cols:
            report_lines.extend([
                "",
                "DISPLACEMENTS (D1, D2, D3) ANALYSIS:",
                "-" * 50
            ])
            
            for col in displacement_cols:
                if col in self.error_analysis:
                    analysis = self.error_analysis[col]
                    report_lines.extend([
                        f"\n{col}:",
                        f"  平均绝对误差: {analysis['mean_absolute_error']:.4f}",
                        f"  平均相对误差: {analysis['mean_relative_error']:.2f}%",
                        f"  误差范围: {analysis['error_range_rel']}",
                        f"  误差分布:",
                        f"    ±5% 以内: {analysis['within_5_percent']:.1f}%",
                        f"    ±10% 以内: {analysis['within_10_percent']:.1f}%",
                        f"    ±20% 以内: {analysis['within_20_percent']:.1f}%",
                        f"  偏差: {analysis['mean_bias']:.4f} ({analysis['bias_percentage']:.2f}%)",
                        f"  真实值范围: {analysis['true_mean']:.2f} ± {analysis['true_std']:.2f}",
                        f"  预测值范围: {analysis['pred_mean']:.2f} ± {analysis['pred_std']:.2f}"
                    ])
        
        # Percentile analysis
        report_lines.extend([
            "",
            "PERCENTILE ANALYSIS:",
            "-" * 40
        ])
        
        for col in self.true_values.columns:
            if col in self.error_analysis:
                analysis = self.error_analysis[col]
                report_lines.extend([
                    f"\n{col} 相对误差百分位数:",
                    f"  25%: {analysis['rel_error_percentiles']['25%']:.2f}%",
                    f"  50%: {analysis['rel_error_percentiles']['50%']:.2f}%",
                    f"  75%: {analysis['rel_error_percentiles']['75%']:.2f}%",
                    f"  90%: {analysis['rel_error_percentiles']['90%']:.2f}%",
                    f"  95%: {analysis['rel_error_percentiles']['95%']:.2f}%",
                    f"  99%: {analysis['rel_error_percentiles']['99%']:.2f}%"
                ])
        
        report_lines.extend(["", "=" * 80])
        
        return "\n".join(report_lines)
        
    def plot_predictions(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot actual vs predicted values
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if self.predictions is None:
            raise RuntimeError("Must run evaluate() before plotting")
            
        # Ensure English-friendly matplotlib configuration
        try:
            matplotlib.rcParams.update({
                'font.family': 'DejaVu Sans',
                'axes.unicode_minus': False,
            })
        except Exception:
            pass

        n_targets = len(self.true_values.columns)
        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols

        # Scale figure height with number of rows to prevent cramped axes
        row_height = 3.0  # inches per row
        dynamic_figsize = (figsize[0], max(figsize[1], row_height * n_rows))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=dynamic_figsize, constrained_layout=True
        )
        axes = np.atleast_1d(axes).flatten()
        
        for idx, col in enumerate(self.true_values.columns):
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(self.true_values[col], self.predictions[col], alpha=0.5)
            
            # Perfect prediction line
            min_val = min(self.true_values[col].min(), self.predictions[col].min())
            max_val = max(self.true_values[col].max(), self.predictions[col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            # Labels and title
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{col}\nR² = {self.metrics["r2"][col]:.3f}')
            ax.grid(True, alpha=0.3)
            
        # Hide extra subplots
        for idx in range(n_targets, len(axes)):
            axes[idx].set_visible(False)
            
        fig.suptitle(f'Predictions vs True Values - {self.model_name}', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to {save_path}")
            
        plt.close()
        
    def plot_residuals(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot residual distributions
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if self.predictions is None:
            raise RuntimeError("Must run evaluate() before plotting")
            
        # Ensure English-friendly matplotlib configuration
        try:
            matplotlib.rcParams.update({
                'font.family': 'DejaVu Sans',
                'axes.unicode_minus': False,
            })
        except Exception:
            pass

        n_targets = len(self.true_values.columns)
        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols

        # Scale figure height with number of rows to prevent cramped axes
        row_height = 3.0  # inches per row
        dynamic_figsize = (figsize[0], max(figsize[1], row_height * n_rows))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=dynamic_figsize, constrained_layout=True
        )
        axes = np.atleast_1d(axes).flatten()
        
        for idx, col in enumerate(self.true_values.columns):
            ax = axes[idx]
            
            # Calculate residuals
            residuals = self.true_values[col] - self.predictions[col]
            
            # Histogram
            ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            
            # Labels and title
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{col}\nMean: {residuals.mean():.3f}, Std: {residuals.std():.3f}')
            ax.grid(True, alpha=0.3)
            
        # Hide extra subplots
        for idx in range(n_targets, len(axes)):
            axes[idx].set_visible(False)
            
        fig.suptitle(f'Residual Distributions - {self.model_name}', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
            
        plt.close()
        
    def save_report(self, directory: str):
        """
        Save complete report to directory
        
        Args:
            directory: Directory to save report files
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save text report
        text_report = self.generate_text_report()
        with open(os.path.join(directory, 'report.txt'), 'w') as f:
            f.write(text_report)
            
        # Save metrics as JSON
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_metrics = convert_to_serializable(self.metrics)
        with open(os.path.join(directory, 'metrics.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        # Save plots
        self.plot_predictions(os.path.join(directory, 'predictions.png'))
        self.plot_residuals(os.path.join(directory, 'residuals.png'))
        
        logger.info(f"Report saved to {directory}")

    def save_artifacts(
        self,
        directory: str,
        *,
        split_name: str,
        y_true_df: Optional[pd.DataFrame] = None,
        y_pred_df: Optional[pd.DataFrame] = None,
        X_df: Optional[pd.DataFrame] = None,
        error_feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Save metrics, plots, and optional CSVs with split-specific prefixes.

        Files:
          - {split}_report.txt
          - {split}_metrics.json
          - {split}_predictions.png
          - {split}_residuals.png
          - {split}_true.csv (optional)
          - {split}_pred.csv (optional)
        """
        import os
        os.makedirs(directory, exist_ok=True)

        # Save text report
        text_report = self.generate_text_report()
        with open(os.path.join(directory, f"{split_name}_report.txt"), "w") as f:
            f.write(text_report)

        # Save metrics as JSON (serializable)
        def convert_to_serializable(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_metrics = convert_to_serializable(self.metrics)
        with open(os.path.join(directory, f"{split_name}_metrics.json"), "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        # Save plots
        self.plot_predictions(os.path.join(directory, f"{split_name}_predictions.png"))
        self.plot_residuals(os.path.join(directory, f"{split_name}_residuals.png"))

        # Optional CSVs
        if y_true_df is not None:
            y_true_df.to_csv(os.path.join(directory, f"{split_name}_true.csv"), index=False)
        if y_pred_df is not None:
            y_pred_df.to_csv(os.path.join(directory, f"{split_name}_pred.csv"), index=False)
        # Optional error-feature plots
        if X_df is not None and error_feature_names:
            try:
                self.plot_error_vs_features(
                    X_df,
                    error_feature_names,
                    os.path.join(directory, f"{split_name}_error_vs_features.png"),
                )
            except Exception as e:
                logger.warning(f"Failed to generate error-vs-feature plot: {e}")
        
    @staticmethod
    def compare_models(reports: List['EvaluationReport']) -> pd.DataFrame:
        """
        Compare multiple model reports
        
        Args:
            reports: List of evaluation reports
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for report in reports:
            if report.metrics is None:
                logger.warning(f"Model '{report.model_name}' has no metrics")
                continue
                
            row = {
                'model': report.model_name,
                'overall_r2': report.metrics['r2']['overall'],
                'overall_rmse': report.metrics['rmse']['overall'],
                'overall_mae': report.metrics['mae']['overall'],
                'overall_mape': report.metrics['mape']['overall']
            }
            
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('overall_r2', ascending=False)
        
        return comparison_df

    def plot_error_vs_features(
        self,
        X_df: pd.DataFrame,
        feature_names: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> None:
        """
        Plot scatter of overall absolute error per sample vs selected input features.

        Overall absolute error is computed as mean absolute error across all targets
        for each sample (in original units).
        """
        if self.predictions is None or self.true_values is None:
            raise RuntimeError("Must run evaluate() before plotting error vs features")

        # Ensure English-friendly matplotlib configuration
        try:
            matplotlib.rcParams.update({
                'font.family': 'DejaVu Sans',
                'axes.unicode_minus': False,
            })
        except Exception:
            pass

        # Compute per-sample mean absolute error over targets
        abs_err = (self.true_values.to_numpy() - self.predictions.to_numpy())
        abs_err = np.abs(abs_err).mean(axis=1)

        # Determine layout
        n_feats = len(feature_names)
        n_cols = min(3, n_feats)
        n_rows = (n_feats + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize[0], max(figsize[1], 4 * n_rows)),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).flatten()

        for i, feat in enumerate(feature_names):
            ax = axes[i]
            if feat not in X_df.columns:
                ax.set_visible(False)
                continue
            ax.scatter(X_df[feat].to_numpy(), abs_err, alpha=0.5)
            ax.set_xlabel(feat)
            ax.set_ylabel('Mean Absolute Error (overall)')
            ax.set_title(f'Error vs {feat}')
            ax.grid(True, alpha=0.3)

        for j in range(n_feats, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'Error vs Features - {self.model_name}', fontsize=16)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error-vs-feature plot saved to {save_path}")
        plt.close()
