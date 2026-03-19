"""
rate_limit.py - Retry and quota utilities for YouTube API calls.

Provides:
  - QuotaExhaustedError  — raised when the daily API quota is consumed.
                           The scheduler catches this to abort the run cleanly.
  - retry_api_call       — decorator that retries transient errors (429 / 5xx)
                           with exponential backoff, and converts quota errors
                           to QuotaExhaustedError.
"""

import functools
import logging
import time

from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# Seconds to wait before retry attempts — long enough for YouTube's rate limiter to reset
_BACKOFF_DELAYS = [60, 120, 300]

# HTTP status codes that are safe to retry
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class QuotaExhaustedError(Exception):
    """Raised when the YouTube Data API daily quota is consumed (HTTP 403 quotaExceeded)."""


def retry_api_call(fn):
    """
    Decorator for YouTube API calls.

    - Retries on transient HTTP errors (429, 5xx) with exponential backoff.
    - Raises QuotaExhaustedError immediately on 403 quotaExceeded (non-retryable).
    - Re-raises any other HttpError unchanged.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt, delay in enumerate([0] + _BACKOFF_DELAYS, start=1):
            if delay:
                logger.warning(
                    f"[retry] Waiting {delay}s before attempt {attempt} of {fn.__name__}"
                )
                time.sleep(delay)
            try:
                return fn(*args, **kwargs)
            except HttpError as exc:
                status = exc.resp.status
                if status == 403:
                    reason = _extract_reason(exc)
                    if reason in ("quotaExceeded", "dailyLimitExceeded"):
                        raise QuotaExhaustedError(
                            f"YouTube API daily quota exhausted during {fn.__name__}. "
                            "Run will resume on the next scheduled cycle."
                        ) from exc
                    # Other 403 (e.g. forbidden video) — not retryable, re-raise as-is
                    raise
                if status in _RETRYABLE_STATUSES:
                    logger.warning(
                        f"[retry] HTTP {status} on {fn.__name__} "
                        f"(attempt {attempt}/{len(_BACKOFF_DELAYS) + 1}): {exc}"
                    )
                    last_exc = exc
                    continue
                # Non-retryable status — re-raise immediately
                raise
        # All retries exhausted
        raise last_exc
    return wrapper


def retry_transcript(fn):
    """
    Decorator for yt-dlp transcript fetches.

    yt-dlp raises yt_dlp.utils.DownloadError (not HttpError) on 429s.
    Retries with the same backoff schedule when a 429 / Too Many Requests
    string is detected in the error message; other errors propagate normally.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt, delay in enumerate([0] + _BACKOFF_DELAYS, start=1):
            if delay:
                logger.warning(
                    f"[retry] yt-dlp rate limited — waiting {delay}s before attempt {attempt}"
                )
                time.sleep(delay)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "too many requests" in msg:
                    logger.warning(
                        f"[retry] yt-dlp 429 on {fn.__name__} "
                        f"(attempt {attempt}/{len(_BACKOFF_DELAYS) + 1})"
                    )
                    last_exc = exc
                    continue
                raise
        logger.error(f"[retry] yt-dlp still rate-limited after all retries — skipping video")
        return None
    return wrapper


def _extract_reason(exc: HttpError) -> str:
    """Pull the error reason string out of a GoogleAPI HttpError response body."""
    try:
        import json
        body = json.loads(exc.content.decode())
        errors = body.get("error", {}).get("errors", [])
        if errors:
            return errors[0].get("reason", "")
    except Exception:
        pass
    return ""
