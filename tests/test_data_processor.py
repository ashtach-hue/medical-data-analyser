"""Tests for data processing module."""

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
        """Create sample data."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40],
            'blood_pressure': [120, 130, 125, 135],
            'cholesterol': [200, 210, 205, 220]
        })

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.random_state == 42

    def test_train_test_split(self, processor, sample_data):
        """Test train/test split."""
        train, test = processor.train_test_split(sample_data, test_size=0.25)
        assert len(train) + len(test) == len(sample_data)
