"""
Logger configuration for PhotoPrint.
Provides centralized logging setup for all modules.
"""

import logging
import os
from pathlib import Path

# Константи для логування
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT_CONSOLE = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FORMAT_FILE = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logger(name: str = "photoprint", log_file: str | None = None, level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name (usually module name or 'photoprint' for root)
        log_file: Optional path to log file. If None, only console logging.
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        LOG_FORMAT_CONSOLE,
        datefmt=DATE_FORMAT
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            LOG_FORMAT_FILE,
            datefmt=DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. Creates it if it doesn't exist.
    
    Args:
        name: Logger name (usually __name__ from calling module)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default log file path
DEFAULT_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "photoprint.log")

# Setup root logger
_root_logger = setup_logger("photoprint", log_file=DEFAULT_LOG_FILE)
