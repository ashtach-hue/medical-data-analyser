"""Data validation module"""

import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and integrity"""
    
    @staticmethod
    def validate_schema(df: pd.DataFrame, schema: Dict[str, str]) -> bool:
        """Validate DataFrame against expected schema
        
        Args:
            df: Input DataFrame
            schema: Dictionary of column names and expected types
            
        Returns:
            True if schema is valid
        """
        for col, dtype in schema.items():
            if col not in df.columns:
                logger.error(f"Missing column: {col}")
                return False
            if str(df[col].dtype) != dtype:
                logger.warning(f"Column {col} has type {df[col].dtype}, expected {dtype}")
        return True
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality metrics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
