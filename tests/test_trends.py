"""Tests for trend analysis module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.trends import TrendAnalyzer


class TestTrendAnalyzer:
    """Test TrendAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create TrendAnalyzer instance."""
        return TrendAnalyzer()

    @pytest.fixture
    def sample_patient_data(self):
        """Create sample patient time-series data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = {
            'patient_id': [1] * 10,
            'date': dates,
            'blood_pressure': np.linspace(120, 130, 10),
            'cholesterol': np.linspace(200, 210, 10),
            'heart_rate': np.linspace(70, 75, 10)
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def multi_patient_data(self):
        """Create multi-patient time-series data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = []
        
        for patient_id in [1, 2, 3]:
            for i, date in enumerate(dates):
                data.append({
                    'patient_id': patient_id,
                    'date': date,
                    'blood_pressure': 120 + i + patient_id * 2,
                    'cholesterol': 200 + i * 2,
                    'heart_rate': 70 + i
                })
        
        return pd.DataFrame(data)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.date_column == 'date'

    def test_analyze_patient_trends(self, analyzer, sample_patient_data):
        """Test patient trend analysis."""
        trends = analyzer.analyze_patient_trends(sample_patient_data)
        
        assert 1 in trends
        assert 'blood_pressure' in trends[1]
        assert 'trend_direction' in trends[1]['blood_pressure']

    def test_calculate_growth_rates(self, analyzer, sample_patient_data):
        """Test growth rate calculation."""
        growth_rates = analyzer.calculate_growth_rates(sample_patient_data, 'blood_pressure')
        
        assert 1 in growth_rates
        assert isinstance(growth_rates[1], (int, float))

    def test_detect_trend_changes(self, analyzer, multi_patient_data):
        """Test trend change detection."""
        changes = analyzer.detect_trend_changes(multi_patient_data, 'blood_pressure')
        
        assert isinstance(changes, dict)
        for patient_id in [1, 2, 3]:
            assert patient_id in changes
            assert isinstance(changes[patient_id], list)

    def test_get_trend_summary(self, analyzer, multi_patient_data):
        """Test trend summary calculation."""
        summary = analyzer.get_trend_summary(multi_patient_data)
        
        assert summary['total_patients'] == 3
        assert summary['total_records'] == 30
        assert 'date_range' in summary
        assert 'metrics' in summary

    def test_forecast_trend(self, analyzer, sample_patient_data):
        """Test trend forecasting."""
        forecast = analyzer.forecast_trend(
            sample_patient_data, 
            'blood_pressure', 
            patient_id=1, 
            periods=5
        )
        
        assert 'forecast_values' in forecast
        assert len(forecast['forecast_values']) == 5

    def test_compare_patient_trends(self, analyzer, multi_patient_data):
        """Test patient trend comparison."""
        comparison = analyzer.compare_patient_trends(
            multi_patient_data,
            'blood_pressure',
            [1, 2, 3]
        )
        
        assert len(comparison) == 3
        for patient_id in [1, 2, 3]:
            assert patient_id in comparison
            assert 'slope' in comparison[patient_id]
            assert 'trend' in comparison[patient_id]
