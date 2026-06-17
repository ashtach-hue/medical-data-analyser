"""Tests for data processor module"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'bp': [120, 130, 125, 140, 135],
            'glucose': [100, 110, 105, 120, 115]
        })
    
    def test_processor_initialization(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor()
        assert processor.raw_dir.exists()
        assert processor.processed_dir.exists()
    
    def test_handle_missing_values(self, sample_df):
        """Test missing value handling"""
        processor = DataProcessor()
        df_with_nan = sample_df.copy()
        df_with_nan.loc[0, 'age'] = np.nan
        
        df_clean = processor.handle_missing_values(df_with_nan, strategy='mean')
        assert df_clean['age'].isna().sum() == 0
    
    def test_remove_duplicates(self, sample_df):
        """Test duplicate removal"""
        processor = DataProcessor()
        df_dup = pd.concat([sample_df, sample_df.iloc[0:1]], ignore_index=True)
        
        df_clean = processor.remove_duplicates(df_dup)
        assert len(df_clean) < len(df_dup)
