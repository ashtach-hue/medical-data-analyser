"""Data processing and ETL pipeline."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle data loading, cleaning, and preprocessing."""

    def __init__(self, random_state: int = 42):
        """Initialize DataProcessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def load_and_clean(self, filepath: str) -> pd.DataFrame:
        """Load and clean patient data.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data from {filepath} with shape {df.shape}")
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            df = df.dropna()
            
            logger.info(f"Cleaned data with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split as sklearn_split
        
        train_df, test_df = sklearn_split(
            df, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"Split data: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df
