"""
logging.py - Centralized Logging Configuration

Emits structured JSON to stdout so logs are queryable in Render, GCP, or any
log aggregator.  Each line is a self-contained JSON object:

  {"timestamp": "2026-04-06T14:23:01Z", "level": "INFO",
   "module": "pipeline_service", "message": "PIPELINE_COMPLETE",
   "elapsed_s": 47.3, "videos_indexed": 48}

Log level is controlled by the LOG_LEVEL env var (default: INFO).
All modules should call get_logger(__name__) to obtain a named logger.
"""

import json
import logging
import os
import sys


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def setup_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger.setLevel(getattr(logging, level, logging.INFO))
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
