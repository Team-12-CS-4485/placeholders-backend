"""
cache.py - Simple In-Memory TTL Cache

Thread-safe dict-based cache for read endpoints whose data only changes
when the pipeline runs. Call invalidate_all() after a successful pipeline
run to flush all entries.

Usage:
    from app.core.cache import get_cached, invalidate_all

    result = get_cached("trends:heat_score:None", ttl=300, fetch_fn=lambda: ...)
"""

import threading
import time
from typing import Any, Callable

_cache: dict[str, tuple[Any, float]] = {}
_lock = threading.Lock()


def get_cached(key: str, ttl: int, fetch_fn: Callable) -> Any:
    """
    Return cached value if still fresh, otherwise call fetch_fn(),
    store the result, and return it.

    Args:
        key:      Unique string identifying this cached value.
        ttl:      Time-to-live in seconds.
        fetch_fn: Zero-argument callable that produces the value.
    """
    now = time.time()
    with _lock:
        entry = _cache.get(key)
        if entry is not None:
            result, expiry = entry
            if now < expiry:
                return result

    result = fetch_fn()

    with _lock:
        _cache[key] = (result, now + ttl)

    return result


def invalidate_all() -> None:
    """Clear all cached entries. Call after a successful pipeline run."""
    with _lock:
        _cache.clear()
