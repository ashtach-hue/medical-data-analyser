"""Health data statistical analysis"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class HealthAnalysis:
    """Statistical analysis for health data"""
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
        """Compute correlation matrix
        
        Args:
            df: Input DataFrame
            numeric_cols: Columns to include
            
        Returns:
            Correlation matrix
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return df[numeric_cols].corr()
    
    @staticmethod
    def statistical_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with statistical summaries
        """
        return {
            'describe': df.describe().to_dict(),
            'skewness': df.skew().to_dict(),
            'kurtosis': df.kurtosis().to_dict()
        }
    
    @staticmethod
    def compare_groups(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
        """Compare values across groups using t-test
        
        Args:
            df: Input DataFrame
            group_col: Column with group labels
            value_col: Column with values to compare
            
        Returns:
            Dictionary with comparison results
        """
        groups = df[group_col].unique()
        values = [df[df[group_col] == g][value_col].dropna().values for g in groups]
        
        t_stat, p_value = stats.ttest_ind(*values)
        
        return {
            'groups': groups.tolist(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
