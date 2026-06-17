"""Disease prediction models"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """Predicts disease risk using ensemble models"""
    
    def __init__(self, model_type: str = 'ensemble', model_path: Union[str, Path] = None):
        """Initialize disease predictor
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'ensemble'
            model_path: Path to load trained model
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the model based on type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'ensemble':
            # Ensemble of both models
            self.model = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == 'ensemble':
            for model_name, model in self.model.items():
                logger.info(f"Training {model_name}...")
                model.fit(X_scaled, y_train)
        else:
            logger.info(f"Training {self.model_type}...")
            self.model.fit(X_scaled, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make disease risk predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'ensemble':
            # Average predictions from both models
            rf_pred = self.model['rf'].predict_proba(X_scaled)[:, 1]
            gb_pred = self.model['gb'].predict_proba(X_scaled)[:, 1]
            return (rf_pred + gb_pred) / 2
        else:
            return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predictions with confidence scores
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary with predictions and confidence
        """
        proba = self.predict(X)
        predictions = (proba >= 0.5).astype(int)
        
        return {
            'predictions': predictions,
            'probabilities': proba,
            'confidence': np.abs(proba - 0.5) * 2  # 0-1 confidence score
        }
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save trained model
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_names},
                   filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """Load trained model
        
        Args:
            filepath: Path to load model from
        """
        filepath = Path(filepath)
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['features']
        logger.info(f"Model loaded from {filepath}")
