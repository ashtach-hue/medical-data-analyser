"""Configuration management"""

from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models' / 'trained'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = LOGS_DIR / os.getenv('LOG_FILE', 'app.log')

# Model configuration
MODEL_TYPE = os.getenv('MODEL_TYPE', 'ensemble')
MODEL_PATH = os.getenv('MODEL_PATH', str(MODELS_DIR / 'disease_predictor.pkl'))

# API configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
API_DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./medical_db.db')
