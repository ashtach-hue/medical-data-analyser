"""Tests for disease predictor module."""

import pytest
import pandas as pd
import numpy as np
from src.models import DiseasePredictor


class TestDiseasePredictor:
    """Test DiseasePredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create DiseasePredictor instance."""
        return DiseasePredictor(model_type='ensemble')

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = pd.DataFrame({
            'age': np.random.randint(20, 80, 100),
            'blood_pressure': np.random.randint(100, 180, 100),
            'cholesterol': np.random.randint(150, 300, 100)
        })
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_predictor_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.model_type == 'ensemble'
        assert predictor.confidence_threshold == 0.7

    def test_fit_predict(self, predictor, sample_data):
        """Test model training and prediction."""
        X, y = sample_data
        predictor.fit(X, y)
        
        assert predictor.model is not None
        assert predictor.feature_names is not None
        
        results = predictor.predict(X[:5])
        assert 'predictions' in results
        assert 'probabilities' in results
        assert 'confidence' in results
