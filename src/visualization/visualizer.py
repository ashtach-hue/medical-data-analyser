"""Enhanced visualization with trend plotting."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class HealthVisualizer:
    """Create visualizations for health data including trends."""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize HealthVisualizer.
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.fig = None
        self.ax = None
        sns.set_palette("husl")

    def plot_trends(self, data: Dict[str, Any], figsize: tuple = (14, 8)) -> None:
        """Plot health trends over time.
        
        Args:
            data: Dictionary with metrics data
            figsize: Figure size
        """
        if not data or not isinstance(data, dict):
            logger.warning("Invalid data for trend plotting")
            return
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Plot trend data
        if 'summary_stats' in data:
            stats_df = pd.DataFrame(data['summary_stats'])
            for col in stats_df.columns:
                self.ax.plot(stats_df.index, stats_df[col], marker='o', label=col, linewidth=2)
        
        self.ax.set_title('Health Metrics Trends Over Time', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('Measurement Index', fontsize=12)
        self.ax.set_ylabel('Value', fontsize=12)
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        logger.info("Trend plot created")

    def plot_trend_heatmap(self, df: pd.DataFrame, patient_id_col: str = 'patient_id', 
                           date_col: str = 'date', figsize: tuple = (14, 8)) -> None:
        """Plot heatmap of health metrics trends.
        
        Args:
            df: Patient data
            patient_id_col: Column name for patient ID
            date_col: Column name for date
            figsize: Figure size
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != patient_id_col]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for heatmap")
            return
        
        # Normalize data
        df_numeric = df[numeric_cols].copy()
        df_normalized = (df_numeric - df_numeric.mean()) / df_numeric.std()
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        sns.heatmap(df_normalized.iloc[:30].T, cmap='RdYlGn', center=0, 
                    ax=self.ax, cbar_kws={'label': 'Normalized Value'})
        
        self.ax.set_title('Health Metrics Heatmap (Normalized)', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('Record Index', fontsize=12)
        self.ax.set_ylabel('Health Metric', fontsize=12)
        
        plt.tight_layout()
        logger.info("Heatmap created")

    def plot_patient_comparison(self, df: pd.DataFrame, metric_col: str, 
                               patient_ids: List[str], date_col: str = 'date',
                               patient_id_col: str = 'patient_id',
                               figsize: tuple = (14, 6)) -> None:
        """Compare a metric across multiple patients.
        
        Args:
            df: Patient data
            metric_col: Column to compare
            patient_ids: List of patient IDs to compare
            date_col: Column name for date
            patient_id_col: Column name for patient ID
            figsize: Figure size
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        for patient_id in patient_ids:
            patient_data = df[df[patient_id_col] == patient_id].copy()
            if date_col in patient_data.columns:
                patient_data[date_col] = pd.to_datetime(patient_data[date_col])
                patient_data = patient_data.sort_values(date_col)
                self.ax.plot(patient_data[date_col], patient_data[metric_col], 
                           marker='o', label=f'Patient {patient_id}', linewidth=2)
            else:
                self.ax.plot(patient_data[metric_col], marker='o', 
                           label=f'Patient {patient_id}', linewidth=2)
        
        self.ax.set_title(f'{metric_col} Comparison Across Patients', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('Date' if date_col in df.columns else 'Measurement Index', fontsize=12)
        self.ax.set_ylabel(metric_col, fontsize=12)
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        logger.info("Patient comparison plot created")

    def plot_metrics(self, data: Dict[str, Any], figsize: tuple = (12, 6)) -> None:
        """Plot health metrics.
        
        Args:
            data: Dictionary with metrics data
            figsize: Figure size
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        if 'summary_stats' in data:
            stats_df = pd.DataFrame(data['summary_stats'])
            stats_df.plot(ax=self.ax)
            self.ax.set_title('Health Metrics Summary', fontsize=14, fontweight='bold')
            self.ax.set_ylabel('Value', fontsize=12)
        
        plt.tight_layout()
        logger.info("Metrics plot created")

    def plot_correlation_matrix(self, df: pd.DataFrame, figsize: tuple = (10, 8)) -> None:
        """Plot correlation heatmap.
        
        Args:
            df: Input DataFrame
            figsize: Figure size
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=self.ax, cmap='coolwarm', center=0)
        self.ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        logger.info("Correlation plot created")

    def save_figure(self, filepath: str, dpi: int = 300) -> None:
        """Save current figure.
        
        Args:
            filepath: Output file path
            dpi: Resolution
        """
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {filepath}")

    def show(self) -> None:
        """Display the current figure."""
        if self.fig is None:
            logger.warning("No figure to display")
            return
        
        plt.show()

    def close(self) -> None:
        """Close the current figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
