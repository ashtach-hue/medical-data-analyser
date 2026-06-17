"""Integration tests for the Medical Data Analyser."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing import DataProcessor
from src.trends import TrendAnalyzer
from src.analysis import HealthAnalysis
from src.visualization import HealthVisualizer
from src.models import DiseasePredictor


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def sample_patient_data(self):
        """Create sample patient dataset."""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        data = []
        
        for patient_id in [1, 2, 3]:
            for i, date in enumerate(dates):
                data.append({
                    'patient_id': patient_id,
                    'date': date,
                    'age': 40 + patient_id * 5,
                    'blood_pressure': 120 + i + patient_id * 2,
                    'cholesterol': 200 + i * 2,
                    'heart_rate': 70 + i,
                    'glucose': 100 + i
                })
        
        return pd.DataFrame(data)

    def test_full_workflow(self, sample_patient_data):
        """Test complete workflow from data processing to visualization."""
        
        # Step 1: Data Processing
        processor = DataProcessor()
        df = sample_patient_data.copy()
        
        # Step 2: Train/Test Split
        train_df, test_df = processor.train_test_split(df, test_size=0.2)
        assert len(train_df) > 0
        assert len(test_df) > 0
        
        # Step 3: Trend Analysis
        analyzer = TrendAnalyzer()
        trends = analyzer.analyze_patient_trends(df)
        assert len(trends) == 3  # 3 patients
        
        # Step 4: Statistical Analysis
        analysis = HealthAnalysis()
        summary = analysis.get_trend_summary(df)
        assert summary['total_patients'] == 3
        assert summary['total_records'] == 60
        
        # Step 5: Get correlations
        correlations = analysis.get_correlations(df)
        assert correlations is not None
        assert len(correlations) > 0

    def test_trend_forecast_workflow(self, sample_patient_data):
        """Test trend forecasting workflow."""
        analyzer = TrendAnalyzer()
        
        # Forecast for first patient
        forecast = analyzer.forecast_trend(
            sample_patient_data,
            'blood_pressure',
            patient_id=1,
            periods=5
        )
        
        assert 'forecast_values' in forecast
        assert len(forecast['forecast_values']) == 5
        assert 'trend_direction' in forecast

    def test_patient_comparison_workflow(self, sample_patient_data):
        """Test multi-patient comparison workflow."""
        analyzer = TrendAnalyzer()
        
        # Compare blood pressure trends
        comparison = analyzer.compare_patient_trends(
            sample_patient_data,
            'blood_pressure',
            [1, 2, 3]
        )
        
        assert len(comparison) == 3
        for patient_id in [1, 2, 3]:
            assert patient_id in comparison
            assert 'slope' in comparison[patient_id]
            assert 'mean' in comparison[patient_id]

    def test_model_prediction_workflow(self, sample_patient_data):
        """Test ML model prediction workflow."""
        # Prepare data for ML
        X = sample_patient_data[['age', 'blood_pressure', 'cholesterol', 'heart_rate', 'glucose']]
        y = (sample_patient_data['blood_pressure'] > 130).astype(int)
        
        # Train model
        predictor = DiseasePredictor()
        predictor.fit(X, y)
        
        # Make predictions
        predictions = predictor.predict(X.head(5))
        
        assert 'predictions' in predictions
        assert 'probabilities' in predictions
        assert 'confidence' in predictions
        assert len(predictions['predictions']) == 5

    def test_change_detection_workflow(self, sample_patient_data):
        """Test trend change detection workflow."""
        analyzer = TrendAnalyzer()
        
        # Detect changes
        changes = analyzer.detect_trend_changes(
            sample_patient_data,
            'blood_pressure',
            window_size=3
        )
        
        assert isinstance(changes, dict)
        for patient_id in [1, 2, 3]:
            assert patient_id in changes
            assert isinstance(changes[patient_id], list)
