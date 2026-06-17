"""Helper utility functions"""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(log_level: int = logging.INFO,
                     log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging for the application
    
    Args:
        log_level: Logging level
        log_file: Optional file path for log output
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('medical_data_analyser')
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
