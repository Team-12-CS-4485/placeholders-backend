"""
logging.py - Centralized Logging Configuration

Provides a singleton logging setup that initializes once with a stdout stream handler.
Uses a consistent format: "timestamp | LEVEL | module | message"
All modules should use get_logger(__name__) to obtain a named logger instance.
"""

import logging
import sys


def setup_logging():
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def get_logger(name):
    setup_logging()
    return logging.getLogger(name)
