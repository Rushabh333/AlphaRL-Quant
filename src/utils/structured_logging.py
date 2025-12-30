"""
Structured logging formatter for production environments.
Outputs logs in JSON format for easy parsing by log aggregators.
"""
import json
import logging
import traceback
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    Format logs as JSON for easy parsing by log aggregation tools.
    
    Compatible with ELK stack, Datadog, CloudWatch, etc.
    """
    
    def format(self, record):
    """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields if present
        if hasattr(record, 'ticker'):
            log_data['ticker'] = record.ticker
        if hasattr(record, 'stage'):
            log_data['stage'] = record.stage
        if hasattr(record, 'duration'):
            log_data['duration_seconds'] = record.duration
        
        return json.dumps(log_data)


def setup_structured_logging(log_file: str = "logs/pipeline.json"):
    """Set up structured JSON logging."""
    from pathlib import Path
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    json_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    json_handler.setFormatter(StructuredFormatter())
    json_handler.setLevel(logging.DEBUG)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(json_handler)
    
    logging.info("Structured JSON logging enabled", extra={'log_file': log_file})
