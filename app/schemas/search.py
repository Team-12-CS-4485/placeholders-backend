"""
search.py - Pydantic schemas for the Search API
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.schemas.video import VideoItem


class SearchResultItem(VideoItem):
    score: float
    excerpt: str


class SearchResponse(BaseModel):
    query: str
    limit: int
    total: int
    results: list[SearchResultItem]
