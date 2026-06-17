"""Patient trend analysis and time-series analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyze trends in patient health metrics over time."""

    def __init__(self, date_column: str = 'date'):
        """Initialize TrendAnalyzer.
        
        Args:
            date_column: Name of the date column in data
        """
        self.date_column = date_column
        self.scaler = StandardScaler()

    def analyze_patient_trends(self, df: pd.DataFrame, patient_id_col: str = 'patient_id') -> Dict[str, Any]:
        """Analyze trends for all patients in the dataset.
        
        Args:
            df: DataFrame with patient data
            patient_id_col: Column name for patient ID
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df = df.sort_values([patient_id_col, self.date_column])
            
            trends = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for patient_id in df[patient_id_col].unique():
                patient_data = df[df[patient_id_col] == patient_id].copy()
                trends[patient_id] = self._analyze_single_patient(patient_data, numeric_cols)
            
            logger.info(f"Analyzed trends for {len(trends)} patients")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            raise

    def _analyze_single_patient(self, patient_df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze trends for a single patient.
        
        Args:
            patient_df: Patient data
            numeric_cols: List of numeric columns to analyze
            
        Returns:
            Trend analysis for patient
        """
        trends = {}
        
        for col in numeric_cols:
            if col in patient_df.columns and len(patient_df) > 1:
                values = patient_df[col].dropna().values
                
                if len(values) > 1:
                    trend = self._calculate_trend(values)
                    trends[col] = {
                        'trend_direction': trend['direction'],
                        'trend_strength': trend['strength'],
                        'slope': trend['slope'],
                        'r_squared': trend['r_squared'],
                        'latest_value': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        return trends

    def _calculate_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Calculate trend for a series of values.
        
        Args:
            values: Array of values
            
        Returns:
            Trend statistics
        """
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        strength = abs(r_value)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'direction': direction,
            'strength': strength
        }

    def calculate_growth_rates(self, df: pd.DataFrame, metric_col: str, 
                              period: str = 'D', patient_id_col: str = 'patient_id') -> Dict[str, float]:
        """Calculate growth rates for a health metric.
        
        Args:
            df: Patient data
            metric_col: Column to analyze
            period: Period for growth calculation ('D', 'W', 'M')
            patient_id_col: Column name for patient ID
            
        Returns:
            Dictionary with growth rates by patient
        """
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values([patient_id_col, self.date_column])
        
        growth_rates = {}
        
        for patient_id in df[patient_id_col].unique():
            patient_data = df[df[patient_id_col] == patient_id].copy()
            
            if len(patient_data) > 1:
                patient_data.set_index(self.date_column, inplace=True)
                resampled = patient_data[metric_col].resample(period).last()
                
                if len(resampled) > 1:
                    growth_rate = ((resampled.iloc[-1] - resampled.iloc[0]) / resampled.iloc[0]) * 100
                    growth_rates[patient_id] = growth_rate
        
        return growth_rates

    def detect_trend_changes(self, df: pd.DataFrame, metric_col: str, 
                            window_size: int = 3, patient_id_col: str = 'patient_id') -> Dict[str, List[Dict]]:
        """Detect significant changes in trends.
        
        Args:
            df: Patient data
            metric_col: Column to analyze
            window_size: Window size for change detection
            patient_id_col: Column name for patient ID
            
        Returns:
            Dictionary with detected changes by patient
        """
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values([patient_id_col, self.date_column])
        
        changes = {}
        
        for patient_id in df[patient_id_col].unique():
            patient_data = df[df[patient_id_col] == patient_id].copy()
            values = patient_data[metric_col].dropna().values
            
            if len(values) > window_size:
                patient_changes = []
                
                for i in range(window_size, len(values) - window_size):
                    prev_trend = np.polyfit(range(window_size), values[i-window_size:i], 1)[0]
                    next_trend = np.polyfit(range(window_size), values[i:i+window_size], 1)[0]
                    
                    change_magnitude = abs(next_trend - prev_trend)
                    
                    if change_magnitude > np.std(values) * 0.5:
                        patient_changes.append({
                            'index': i,
                            'date': patient_data.iloc[i][self.date_column] if self.date_column in patient_data.columns else i,
                            'change_magnitude': float(change_magnitude),
                            'previous_trend': float(prev_trend),
                            'current_trend': float(next_trend)
                        })
                
                changes[patient_id] = patient_changes
        
        return changes

    def get_trend_summary(self, df: pd.DataFrame, patient_id_col: str = 'patient_id') -> Dict[str, Any]:
        """Get statistical summary of trends.
        
        Args:
            df: Patient data
            patient_id_col: Column name for patient ID
            
        Returns:
            Summary statistics
        """
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary = {
            'total_patients': df[patient_id_col].nunique(),
            'total_records': len(df),
            'date_range': {
                'start': df[self.date_column].min().strftime('%Y-%m-%d'),
                'end': df[self.date_column].max().strftime('%Y-%m-%d')
            },
            'metrics': {}
        }
        
        for col in numeric_cols:
            if col not in [patient_id_col, self.date_column]:
                summary['metrics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
        
        return summary

    def forecast_trend(self, df: pd.DataFrame, metric_col: str, 
                      patient_id: Any, periods: int = 5) -> Dict[str, Any]:
        """Forecast future trend using linear regression.
        
        Args:
            df: Patient data
            metric_col: Column to forecast
            patient_id: Patient ID to forecast
            periods: Number of periods to forecast
            
        Returns:
            Forecast data with confidence intervals
        """
        try:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            patient_data = df[df['patient_id'] == patient_id].copy()
            patient_data = patient_data.sort_values(self.date_column)
            
            values = patient_data[metric_col].dropna().values
            x = np.arange(len(values))
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Forecast
            forecast_x = np.arange(len(values), len(values) + periods)
            forecast_y = slope * forecast_x + intercept
            
            # Confidence interval
            residuals = values - (slope * x + intercept)
            std_residuals = np.std(residuals)
            
            return {
                'forecast_values': forecast_y.tolist(),
                'confidence_interval': (std_residuals * 1.96),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'r_squared': r_value ** 2
            }
            
        except Exception as e:
            logger.error(f"Error forecasting trend: {str(e)}")
            return {}

    def compare_patient_trends(self, df: pd.DataFrame, metric_col: str, 
                              patient_ids: List[Any], patient_id_col: str = 'patient_id') -> Dict[str, Any]:
        """Compare trends between multiple patients.
        
        Args:
            df: Patient data
            metric_col: Column to compare
            patient_ids: List of patient IDs to compare
            patient_id_col: Column name for patient ID
            
        Returns:
            Comparison of trends
        """
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        comparison = {}
        
        for patient_id in patient_ids:
            patient_data = df[df[patient_id_col] == patient_id].copy()
            values = patient_data[metric_col].dropna().values
            
            if len(values) > 1:
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                comparison[patient_id] = {
                    'slope': float(slope),
                    'mean': float(np.mean(values)),
                    'trend': 'increasing' if slope > 0 else 'decreasing',
                    'r_squared': float(r_value ** 2),
                    'latest_value': float(values[-1])
                }
        
        return comparison
