"""
Professional logging setup for AlphaRL-Quant.
Provides structured logging with proper formatting and log levels.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Sets up both console and file handlers with consistent formatting.
    Prevents duplicate handlers if called multiple times.
    
    Args:
        log_file: Path to log file. If None, only console logging.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string. Uses default if None.
    
    Returns:
        Configured root logger
    
"""
    if format_string is None:
        format_string = (
            '[%(asctime)s] %(levelname)-8s '
            '[%(name)s.%(funcName)s:%(lineno)d] - %(message)s'
        )
    
    formatter = logging.Formatter(
        fmt=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Uses the standard Python logging hierarchy, so loggers inherit
    configuration from the root logger set up by setup_logging().
    
    Args:
        name: Logger name (typically __name__ from calling module)
    
    Returns:
        Logger instance
    
"""
    return logging.getLogger(name)


# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

default_log_file = Path(f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log")
setup_logging(log_file=default_log_file, level=logging.INFO)
