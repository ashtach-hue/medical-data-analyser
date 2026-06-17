"""Disease prediction models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """Predict disease risk using machine learning models."""

    def __init__(self, model_type: str = 'ensemble', confidence_threshold: float = 0.7):
        """Initialize DiseasePredictor.
        
        Args:
            model_type: Type of model ('ensemble', 'random_forest', 'xgboost')
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the disease prediction model.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        from sklearn.ensemble import RandomForestClassifier
        
        self.feature_names = X.columns.tolist()
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        logger.info(f"Model trained with {len(self.feature_names)} features")

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make disease risk predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': np.max(probabilities, axis=1)
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained or does not support feature importance.")
        
        importance_dict = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
