"""
Common logging configuration for the Agentic RAG system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


# Default log format with colors
DEFAULT_LOG_FORMAT = f"{Colors.GREEN}%(asctime)s{Colors.RESET} - {Colors.CYAN}%(name)s{Colors.RESET} - %(levelname)s - %(message)s"  # noqa: E501
DEFAULT_LOG_LEVEL = logging.DEBUG

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(
    name: str,
    log_level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        log_level: Default log level for both file and console handlers
        log_file: Optional file to log to (relative to logs/ directory)
        file_level: Log level for file handler (overrides log_level if provided)
        console_level: Log level for console handler (overrides log_level if provided)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level, let handlers filter

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level or log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        log_path = LOG_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(file_level or log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create a default logger instance
default_logger = get_logger("agentic_rag")
