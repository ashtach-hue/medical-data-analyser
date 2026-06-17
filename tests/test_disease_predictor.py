"""Tests for disease predictor model"""

import pytest
import pandas as pd
import numpy as np
from src.models import DiseasePredictor


class TestDiseasePredictor:
    """Test cases for DiseasePredictor"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y
    
    def test_predictor_initialization(self):
        """Test DiseasePredictor initialization"""
        predictor = DiseasePredictor(model_type='random_forest')
        assert predictor.model is not None
    
    def test_model_training(self, sample_data):
        """Test model training"""
        X, y = sample_data
        predictor = DiseasePredictor(model_type='random_forest')
        predictor.train(X, y)
        
        predictions = predictor.predict(X)
        assert predictions.shape[0] == len(X)
