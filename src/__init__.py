"""Medical Data Analyser Package"""

__version__ = '0.1.0'
__author__ = 'ashtach-hue'

from src.data_processing import DataProcessor
from src.models import DiseasePredictor

__all__ = ['DataProcessor', 'DiseasePredictor']
