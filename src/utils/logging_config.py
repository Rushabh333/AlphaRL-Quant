"""
AlphaRL-Quant Structured Logging Configuration
Provides JSON-formatted logging for production monitoring and analysis.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import traceback


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Makes logs machine-readable for aggregation tools (ELK, Splunk, etc.)
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add stack info if present
        if record.stack_info:
            log_data["stack_info"] = record.stack_info
        
        # Add extra fields (custom context)
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in [
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'message', 'pathname', 'process', 'processName',
                    'relativeCreated', 'thread', 'threadName', 'exc_info',
                    'exc_text', 'stack_info'
                ]:
                    try:
                        # Only add JSON-serializable values
                        json.dumps({key: value})
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        return json.dumps(log_data)


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual information to all log messages.
    Useful for tracking requests, users, or pipelines across multiple log entries.
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        super().__init__(logger, context)
    
    def process(self, msg, kwargs):
        """Add context to log record."""
        # Merge context into extra
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        return msg, kwargs


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
    console_output: bool = True
) -> None:
    """
    Configure application-wide logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON formatting (recommended for production)
        console_output: Also output to console/stdout
    
    Example:
        >>> setup_logging(log_level="INFO", log_file="logs/app.log")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started", extra={"version": "1.0"})
    """
    
    # Determine formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        # Human-readable format for local development
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Configure handlers
    handlers = []
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set third-party library log levels (reduce noise)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "log_file": log_file,
            "json_format": json_format
        }
    )


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (typically __name__)
        context: Optional context dict to add to all log messages
    
    Returns:
        Logger instance (or ContextLogger if context provided)
    
    Example:
        >>> logger = get_logger(__name__, context={"user_id": 123})
        >>> logger.info("Processing data")  # Will include user_id in log
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextLogger(logger, context)
    
    return logger


# Convenience functions for common logging patterns
def log_function_call(func):
    """
    Decorator to automatically log function entry and exit.
    
    Example:
        >>> @log_function_call
        >>> def process_data(data):
        >>>     return data * 2
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log entry
        logger.debug(
            f"Entering {func.__name__}",
            extra={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
        )
        
        try:
            result = func(*args, **kwargs)
            
            # Log successful exit
            logger.debug(
                f"Exiting {func.__name__}",
                extra={"function": func.__name__, "success": True}
            )
            
            return result
            
        except Exception as e:
            # Log exception
            logger.error(
                f"Error in {func.__name__}",
                extra={
                    "function": func.__name__,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            raise
    
    return wrapper


def log_performance(operation: str):
    """
    Context manager for logging operation performance.
    
    Example:
        >>> with log_performance("data_processing"):
        >>>     process_large_dataset()
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timer():
        logger = logging.getLogger(__name__)
        start = time.time()
        
        logger.info(f"Starting {operation}")
        
        try:
            yield
        finally:
            elapsed = time.time() - start
            logger.info(
                f"Completed {operation}",
                extra={
                    "operation": operation,
                    "duration_seconds": round(elapsed, 3)
                }
            )
    
    return timer()


# Example usage and testing
if __name__ == "__main__":
    # Setup logging with JSON format
    setup_logging(
        log_level="DEBUG",
        log_file="logs/test.log",
        json_format=True
    )
    
    # Get logger
    logger = get_logger(__name__)
    
    # Basic logging
    logger.debug("Debug message")
    logger.info("Info message", extra={"user": "trader", "action": "login"})
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Exception logging
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Caught division by zero")
    
    # Context logger
    ctx_logger = get_logger(__name__, context={"pipeline_id": "abc123"})
    ctx_logger.info("Processing data")  # Will include pipeline_id
    
    # Performance logging
    with log_performance("test_operation"):
        import time
        time.sleep(0.1)
    
    print("\n✅ Logging test complete. Check logs/test.log")
