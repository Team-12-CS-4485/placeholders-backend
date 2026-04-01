"""
narrative.py - Pydantic schemas for Narrative API responses

Narratives are the editorial/story side of clusters — headline, summary,
claims, and the videos that belong to them. Metrics live in /api/trends.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.schemas.trend import ClassifiedClaims, WeekData
from app.schemas.video import VideoItem

# ── GET /api/narratives — lean list ──────────────────────────────────────────


class NarrativeListItem(BaseModel):
    cluster_id: int
    label: str
    category: str
    narrative_headline: Optional[str] = None
    narrative_summary: Optional[str] = None
    top_topics: list[str]
    video_count: int
    dominant_sentiment: str


class NarrativeListResponse(BaseModel):
    narratives: list[NarrativeListItem]
    total: int


# ── GET /api/narratives/{id} — full detail ────────────────────────────────────


class NarrativeDetail(BaseModel):
    cluster_id: int
    label: str
    category: str
    narrative_headline: Optional[str] = None
    narrative_summary: Optional[str] = None
    top_topics: list[str]
    top_claims: list[str]
    video_count: int
    channel_count: int
    breaking_count: int
    dominant_sentiment: str
    channels: list[str]
    week_data: list[WeekData]
    avg_clickbait_rating: Optional[float] = None
    thumbnail_tone_breakdown: dict[str, int] = {}


# ── GET /api/narratives/{id}/claims ──────────────────────────────────────────


class NarrativeClaims(BaseModel):
    cluster_id: int
    claims: ClassifiedClaims


# ── GET /api/narratives/{id}/videos ──────────────────────────────────────────


class NarrativeVideosResponse(BaseModel):
    cluster_id: int
    items: list[VideoItem]
    total_returned: int
    next_cursor: Optional[str] = None
