"""Model training utilities"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Utilities for model training and validation"""
    
    @staticmethod
    def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                              random_state: int = 42) -> Tuple:
        """Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    @staticmethod
    def cross_validate(model: Any, X: pd.DataFrame, y: pd.Series,
                      cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation
        
        Args:
            model: Sklearn model
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        logger.info(f"Cross-validation scores: {scores}")
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
