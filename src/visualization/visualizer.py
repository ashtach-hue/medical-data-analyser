"""Health data visualization utilities."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HealthVisualizer:
    """Create visualizations for health data."""

    def __init__(self, style: str = 'seaborn'):
        """Initialize HealthVisualizer.
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        self.fig = None
        self.ax = None

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
            self.ax.set_title('Health Metrics Summary')
            self.ax.set_ylabel('Value')
            
        plt.tight_layout()
        logger.info("Plot created")

    def plot_correlation_matrix(self, df: pd.DataFrame, figsize: tuple = (10, 8)) -> None:
        """Plot correlation heatmap.
        
        Args:
            df: Input DataFrame
            figsize: Figure size
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=self.ax)
        self.ax.set_title('Feature Correlation Matrix')
        
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
