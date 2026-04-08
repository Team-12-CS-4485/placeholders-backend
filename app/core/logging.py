"""
logging.py - Centralized Logging Configuration

Emits structured JSON to stdout so logs are queryable in Render, GCP, or any
log aggregator.  Each line is a self-contained JSON object:

  {"timestamp": "2026-04-06T14:23:01Z", "level": "INFO",
   "module": "pipeline_service", "request_id": "a1b2c3d4e5f6",
   "message": "PIPELINE_COMPLETE elapsed_s=47.3"}

Log level is controlled by the LOG_LEVEL env var (default: INFO).
All modules should call get_logger(__name__) to obtain a named logger.
"""

import json
import logging
import os
import sys
import warnings
from contextvars import ContextVar

# Set by the HTTP middleware for each request; defaults to "-" for background tasks
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "module": record.name,
            "request_id": request_id_var.get(),
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

    # Silence noisy third-party loggers — WARNING keeps rate limit alerts visible
    for noisy in (
        "httpx",
        "httpcore",
        "google_genai",
        "google.ai",
        "botocore",
        "boto3",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Fully suppress — these have no actionable warnings for us
    for silent in (
        "sentence_transformers",
        "huggingface_hub",
        "transformers",
        "transformers_modules",
    ):
        logging.getLogger(silent).setLevel(logging.ERROR)

    # Suppress Python warnings from third-party libraries (umap, sklearn, hf)
    warnings.filterwarnings("ignore", category=UserWarning, module="umap")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings(
        "ignore", message=".*unauthenticated.*", category=UserWarning
    )
    warnings.filterwarnings("ignore", message=".*All keys matched.*")

    # Stop HuggingFace from printing directly to stderr
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
