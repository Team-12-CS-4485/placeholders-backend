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
    channel: str = ""
    sources: list[str]
    source_count: int
    video_ids: list[str]
    transcript_excerpt: str
    risk_score: float = 0.0


class DebatedClaimPerspective(BaseModel):
    channel: str
    sentiment: str
    video_id: str
    video_title: str
    transcript_excerpt: str


class DebatedClaim(BaseModel):
    claim: str
    channel: str = ""
    perspectives: list[DebatedClaimPerspective]
    source_count: int
    framing_divergence: float
    risk_score: float = 0.0


class UniqueClaim(BaseModel):
    claim: str
    channel: str
    video_id: str
    video_title: str
    transcript_excerpt: str
    risk_score: float = 0.0


class ClassifiedClaims(BaseModel):
    consensus: list[ConsensusClaim]
    debated: list[DebatedClaim]
    unique: list[UniqueClaim]


class CreatorRisk(BaseModel):
    name: str
    riskScore: float
    riskLevel: str
    claimCount: int


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
    dominant_public_sentiment: str = "neutral"
    sentiment_divergence: bool = False
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

    public_sentiment_breakdown: dict[str, int] = {}
    dominant_public_sentiment: str = "neutral"
    avg_public_sentiment_score: float = 0.0
    sentiment_divergence: bool = False

    channels: list[str]
    week_data: list[WeekData]
    top_claims: list[str]
    top_topics: list[str]
    creator_risk: list[CreatorRisk] = []
    avg_clickbait_rating: Optional[float] = None
    thumbnail_tone_breakdown: dict[str, int] = {}


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


# ── GET /api/weeks ────────────────────────────────────────────────────────────


class WeekSummary(BaseModel):
    week: str
    total_videos: int
    total_views: int
    active_clusters: int
    breaking_count: int
    dominant_sentiment: str


class WeeksResponse(BaseModel):
    weeks: list[WeekSummary]
    total: int
