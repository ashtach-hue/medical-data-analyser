"""Health metrics analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class HealthAnalysis:
    """Perform statistical analysis on health data."""

    def __init__(self):
        """Initialize HealthAnalysis."""
        self.results = None

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze health metrics in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'summary_stats': df.describe().to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'shape': df.shape,
            'columns': df.columns.tolist()
        }
        
        self.results = results
        logger.info("Analysis completed")
        return results

    def get_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Correlation matrix
        """
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr()

    def identify_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Identify outliers using z-score.
        
        Args:
            df: Input DataFrame
            column: Column to analyze
            threshold: Z-score threshold
            
        Returns:
            DataFrame with outliers marked
        """
        from scipy import stats
        
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = df[np.abs(stats.zscore(df[column].dropna())) > threshold]
        
        return outliers
