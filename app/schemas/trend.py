"""
trend.py - Pydantic schemas for Trend API responses
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


# ── Week-level data ───────────────────────────────────────────────────────────

class WeekData(BaseModel):
    week: str
    video_count: int
    channel_count: int
    view_count: int
    breaking_count: int
    sentiment_breakdown: dict[str, int]


# ── Claims schemas ────────────────────────────────────────────────────────────

class ConsensusClaim(BaseModel):
    claim: str
    sources: list[str]
    source_count: int
    video_ids: list[str]
    transcript_excerpt: str


class DebatedClaimPerspective(BaseModel):
    channel: str
    sentiment: str
    video_id: str
    video_title: str
    transcript_excerpt: str


class DebatedClaim(BaseModel):
    claim: str
    perspectives: list[DebatedClaimPerspective]
    source_count: int
    framing_divergence: float


class UniqueClaim(BaseModel):
    claim: str
    channel: str
    video_id: str
    video_title: str
    transcript_excerpt: str


class ClassifiedClaims(BaseModel):
    consensus: list[ConsensusClaim]
    debated: list[DebatedClaim]
    unique: list[UniqueClaim]


# ── GET /api/trends — lean list ───────────────────────────────────────────────

class TrendListItem(BaseModel):
    cluster_id: int
    label: str
    category: str
    trend_type: str
    metric_badge: str
    heat_score: float
    video_count: int
    channel_count: int
    view_count_total: int
    breaking_count: int
    sentiment_label: str
    recent_sentiment_label: str
    dominant_sentiment: str
    top_topics: list[str]


class TrendListResponse(BaseModel):
    trends: list[TrendListItem]
    total: int


# ── GET /api/trends/{id} — full detail ───────────────────────────────────────

class TrendDetail(BaseModel):
    cluster_id: int
    label: str
    category: str
    trend_type: str
    metric_badge: str
    heat_score: float

    video_count: int
    channel_count: int
    view_count_total: int
    total_likes: int
    total_comments: int
    engagement_index: float
    breaking_count: int

    sentiment_breakdown: dict[str, int]
    sentiment_label: str
    recent_sentiment_label: str
    dominant_sentiment: str

    channels: list[str]
    week_data: list[WeekData]
    top_claims: list[str]
    top_topics: list[str]


# ── GET /api/trends/{id}/sentiment ───────────────────────────────────────────

class WeekSentiment(BaseModel):
    week: str
    sentiment_breakdown: dict[str, int]
    dominant_sentiment: str


class TrendSentiment(BaseModel):
    cluster_id: int
    sentiment_breakdown: dict[str, int]
    sentiment_label: str
    recent_sentiment_label: str
    dominant_sentiment: str
    by_week: list[WeekSentiment]


# ── GET /api/trends/{id}/claims ──────────────────────────────────────────────

class TrendClaims(BaseModel):
    cluster_id: int
    claims: ClassifiedClaims
