import os
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import json
import uuid
import warnings

class ContextFilter(logging.Filter):
    """Add transaction ID to log records"""
    def filter(self, record):
        if not hasattr(record, 'transaction_id'):
            record.transaction_id = 'N/A'
        return True

class JsonFormatter(logging.Formatter):
    """Format log records as JSON strings"""
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName,
            'transaction_id': record.transaction_id
        }
        
        # Include exception info if available
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def setup_logger():
    """Create and configure logger with enhanced capabilities"""
    logger = logging.getLogger("CryptoAnalyzer")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Add context filter for transaction IDs
    logger.addFilter(ContextFilter())
    
    # Standard console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Detailed file handler
    fh = RotatingFileHandler(
        'crypto_analysis.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    
    # JSON file handler for machine-readable logs
    json_fh = RotatingFileHandler(
        'crypto_analysis.json', 
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    json_fh.setLevel(logging.DEBUG)
    
    # Enhanced text formatter
    text_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - [%(transaction_id)s] - %(message)s'
    )
    
    # JSON formatter
    json_formatter = JsonFormatter()
    
    # Set formatters
    ch.setFormatter(text_formatter)
    fh.setFormatter(text_formatter)
    json_fh.setFormatter(json_formatter)
    
    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(json_fh)
    
    # Capture warnings as logs
    logging.captureWarnings(True)
    warnings.filterwarnings("always", category=UserWarning)
    
    # Log initialization success
    logger.info("Logger initialized successfully", extra={'transaction_id': 'SYSTEM_INIT'})
    
    return logger

class NumpyEncoder(json.JSONEncoder):
    # Existing implementation remains unchanged
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_to_native(obj):
    # Existing implementation remains unchanged
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    else:
        return obj