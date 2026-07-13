"""
Logging setup utilities.
"""
import os
import logging
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Sets up and returns a logger with custom formatting.
    
    Args:
        name: Name of the logger.
        log_file: Optional file path to write log messages.
        
    Returns:
        A logging.Logger object.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
