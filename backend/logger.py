"""
Simple in-memory logger for Railway deployment
Stores logs in memory and provides API access
"""
import logging
from datetime import datetime
from typing import List, Dict
from collections import deque

# Global log buffer (thread-safe deque)
LOG_BUFFER = deque(maxlen=500)  # Keep last 500 logs

class MemoryHandler(logging.Handler):
    """Custom handler that stores logs in memory"""

    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': self.format(record),
                'module': record.module,
                'funcName': record.funcName,
            }
            LOG_BUFFER.append(log_entry)
        except Exception:
            self.handleError(record)

def setup_logging():
    """Setup logging with memory handler"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler (for Railway logs view)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Memory handler (for our API)
    memory_handler = MemoryHandler()
    memory_handler.setLevel(logging.DEBUG)
    memory_formatter = logging.Formatter('%(message)s')
    memory_handler.setFormatter(memory_formatter)
    logger.addHandler(memory_handler)

    return logger

def get_logs(limit: int = 100, level: str = None) -> List[Dict]:
    """Get logs from buffer"""
    logs = list(LOG_BUFFER)

    # Filter by level if specified
    if level:
        logs = [log for log in logs if log['level'] == level.upper()]

    # Return last N logs
    return logs[-limit:]

def clear_logs():
    """Clear log buffer"""
    LOG_BUFFER.clear()

def add_custom_log(level: str, message: str):
    """Manually add a log entry"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level.upper(),
        'message': message,
        'module': 'custom',
        'funcName': 'manual',
    }
    LOG_BUFFER.append(log_entry)
