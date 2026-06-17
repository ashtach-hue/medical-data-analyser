"""Main data processing module for ETL pipelines"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles ETL processes for medical data"""
    
    def __init__(self, data_dir: Union[str, Path] = 'data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directories ready at {self.data_dir}")
    
    def load_data(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from various file formats
        
        Args:
            filepath: Path to data file
            **kwargs: Additional arguments for read function
            
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        logger.info(f"Loading data from {filepath}")
        
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath, **kwargs)
        elif filepath.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(filepath, **kwargs)
        elif filepath.suffix == '.json':
            return pd.read_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def save_data(self, df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
        """Save DataFrame to file
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            **kwargs: Additional arguments for save function
        """
        filepath = Path(filepath)
        logger.info(f"Saving data to {filepath}")
        
        if filepath.suffix == '.csv':
            df.to_csv(filepath, index=False, **kwargs)
        elif filepath.suffix in ['.xlsx', '.xls']:
            df.to_excel(filepath, index=False, **kwargs)
        elif filepath.suffix == '.json':
            df.to_json(filepath, **kwargs)
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean',
                             numerical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values in dataset
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', 'forward_fill', 'drop'
            numerical_cols: List of numerical columns
            
        Returns:
            DataFrame with missing values handled
        """
        df_copy = df.copy()
        logger.info(f"Handling missing values using {strategy} strategy")
        
        if strategy == 'drop':
            return df_copy.dropna()
        elif strategy == 'forward_fill':
            return df_copy.fillna(method='ffill')
        elif strategy in ['mean', 'median']:
            if numerical_cols is None:
                numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if col in df_copy.columns:
                    if strategy == 'mean':
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                    else:
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
        
        logger.info(f"Missing values handled. Remaining NaN count: {df_copy.isna().sum().sum()}")
        return df_copy
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_shape = df.shape
        df_clean = df.drop_duplicates(subset=subset)
        removed = initial_shape[0] - df_clean.shape[0]
        logger.info(f"Removed {removed} duplicate rows")
        return df_clean
    
    def normalize_data(self, df: pd.DataFrame, numerical_cols: Optional[List[str]] = None,
                      method: str = 'minmax') -> pd.DataFrame:
        """Normalize numerical data
        
        Args:
            df: Input DataFrame
            numerical_cols: Columns to normalize
            method: 'minmax' or 'zscore'
            
        Returns:
            DataFrame with normalized values
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        df_copy = df.copy()
        
        if numerical_cols is None:
            numerical_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
        
        logger.info(f"Data normalized using {method} method")
        return df_copy
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data summary
        """
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isna().sum().to_dict(),
            'numeric_summary': df.describe().to_dict()
        }
