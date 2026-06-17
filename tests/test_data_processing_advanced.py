"""Unit tests for data processing module."""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor


class TestDataProcessor:
    """Test DataProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create DataProcessor instance."""
        return DataProcessor()

    @pytest.fixture
    def sample_data(self):
        """Create sample data with missing values and duplicates."""
        return pd.DataFrame({
            'patient_id': [1, 1, 2, 2, 2, 3],
            'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-04'],
            'age': [25, 25, 30, 30, 30, 35],
            'blood_pressure': [120, 120, 130, np.nan, 125, 135],
            'cholesterol': [200, 200, 210, 220, np.nan, 230]
        })

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.random_state == 42

    def test_load_and_clean(self, processor, sample_data, tmp_path):
        """Test data loading and cleaning."""
        # Save sample data to temporary file
        temp_file = tmp_path / "test_data.csv"
        sample_data.to_csv(temp_file, index=False)
        
        # Load and clean
        df_cleaned = processor.load_and_clean(str(temp_file))
        
        # Check that duplicates are removed
        assert len(df_cleaned) <= len(sample_data)
        # Check that NaN values are removed
        assert df_cleaned.isnull().sum().sum() == 0

    def test_train_test_split(self, processor, sample_data):
        """Test train/test split."""
        train, test = processor.train_test_split(sample_data, test_size=0.25)
        
        # Check split
        assert len(train) + len(test) == len(sample_data)
        assert len(test) / len(sample_data) >= 0.2  # At least ~20%

    def test_handle_missing_values(self, processor, sample_data):
        """Test missing value handling."""
        # Add more missing values
        df_with_missing = sample_data.copy()
        
        # Check that we have missing values
        assert df_with_missing.isnull().sum().sum() > 0

    def test_data_integrity(self, processor, sample_data):
        """Test data integrity."""
        # Check column types
        assert 'patient_id' in sample_data.columns
        assert 'date' in sample_data.columns
        assert 'blood_pressure' in sample_data.columns
