"""Data cleaning utilities"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Utilities for data cleaning and validation"""
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from a column
        
        Args:
            df: Input DataFrame
            column: Column name
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[column] >= lower) & (df_clean[column] <= upper)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_clean[column].dropna()))
            df_clean = df_clean[z_scores < threshold]
        
        logger.info(f"Outliers removed from {column}")
        return df_clean
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df
