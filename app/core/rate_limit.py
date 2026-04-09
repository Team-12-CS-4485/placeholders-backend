"""
rate_limit.py - Shared slowapi limiter instance

Defined here (not in main.py) to avoid circular imports when
router files need to reference the same limiter object.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
