"""Model evaluation utilities"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates model performance"""
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['roc_curve'] = roc_curve(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
        """Get feature importance from tree-based models
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
