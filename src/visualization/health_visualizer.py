"""Health data visualization utilities"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class HealthVisualizer:
    """Visualization utilities for health data"""
    
    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str, bins: int = 30,
                         figsize: tuple = (10, 6)) -> None:
        """Plot distribution of a column
        
        Args:
            df: Input DataFrame
            column: Column to plot
            bins: Number of bins
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(corr_matrix: pd.DataFrame, figsize: tuple = (12, 10)) -> None:
        """Plot correlation heatmap
        
        Args:
            corr_matrix: Correlation matrix
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 15,
                               figsize: tuple = (10, 6)) -> None:
        """Plot feature importance
        
        Args:
            feature_importance_df: DataFrame with feature and importance columns
            top_n: Number of top features to show
            figsize: Figure size
        """
        top_features = feature_importance_df.head(top_n)
        plt.figure(figsize=figsize)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
                      figsize: tuple = (8, 6)) -> None:
        """Plot ROC curve
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc: AUC score
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()
