"""
trend.py - Pydantic schemas for Trend API responses

Shapes match the Figma Trends Archive views:
- TrendListResponse  → list view (screenshot 1) + grid view (screenshot 2)
- TrendHeaderStats   → header bar (active narratives, total volume, new signals)
- TrendItem          → individual trend card/row
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class WeekData(BaseModel):
    """Per-week metrics for a single trend (powers sparklines + bar charts)."""

    week: str  # "week1", "week2", ...
    video_count: int  # unique videos this week
    channel_count: int  # distinct channels covering it
    view_count: int  # total views this week
    breaking_count: int  # is_breaking videos
    sentiment_breakdown: dict[str, int]  # {"positive": 2, "negative": 5, "neutral": 1}


# ── Classified Claims schemas ────────────────────────────────────────────


class ConsensusClaim(BaseModel):
    """Claim reported by 3+ channels — high confidence factual statement."""

    claim: str
    sources: list[str]  # channel names
    source_count: int
    video_ids: list[str]
    transcript_excerpt: str


class DebatedClaimPerspective(BaseModel):
    """One channel's framing of a debated claim."""

    channel: str
    sentiment: str
    video_id: str
    video_title: str
    transcript_excerpt: str


class DebatedClaim(BaseModel):
    """Claim reported by 2 channels with different framing."""

    claim: str
    perspectives: list[DebatedClaimPerspective]
    source_count: int
    framing_divergence: float  # 0.0 = identical framing, 1.0 = completely different


class UniqueClaim(BaseModel):
    """Claim reported by only one channel — exclusive angle or scoop."""

    claim: str
    channel: str
    video_id: str
    video_title: str
    transcript_excerpt: str


class ClassifiedClaims(BaseModel):
    """Three types of claims per narrative cluster."""

    consensus: list[ConsensusClaim]
    debated: list[DebatedClaim]
    unique: list[UniqueClaim]


class TrendItem(BaseModel):
    """
    Single trend/narrative — powers both list rows and grid cards.

    List view uses: label, description, category, metric_badge, trend_type
    Grid view adds: week_data, sentiment_breakdown, channels
    """

    cluster_id: int
    label: str  # cluster_label — e.g. "Maritime Security"
    description: str  # top key_claim or summary snippet
    category: str  # dominant_category — e.g. "Middle East Conflict"
    trend_type: str  # "surging" | "emerging" | "dominant" | "fading" | "holding"
    metric_badge: str  # display string — "+40% Vol", "High Impact", "Fading"

    # Volume & engagement
    video_count: int  # total videos across all weeks
    channel_count: int  # total distinct channels
    view_count_total: int  # total views across all weeks
    total_likes: int  # total YouTube likes across all videos
    total_comments: int  # total YouTube comments across all videos
    engagement_index: float  # (likes + comments) / views × 10000 — reaction intensity
    breaking_count: int  # total is_breaking videos
    heat_score: float  # composite ranking score

    # Sentiment
    sentiment_breakdown: dict[str, int]
    sentiment_label: str  # e.g. "Overwhelmingly Critical", "Polarized", "Measured"
    recent_sentiment_label: (
        str  # latest week — may include shift prefix like "Sentiment Reversal —"
    )
    dominant_sentiment: str

    # Channels covering this trend
    channels: list[str]

    # Week-over-week data (powers sparklines, bar charts)
    week_data: list[WeekData]

    # Top claims from this cluster (flat list — legacy, kept for backwards compat)
    top_claims: list[str]

    # Classified claims — consensus, debated, unique (from claim analysis service)
    claims: ClassifiedClaims

    # Top topics within the cluster
    top_topics: list[str]


class TrendHeaderStats(BaseModel):
    """Header bar stats — top of the Trends Archive page."""

    active_narratives: int  # number of clusters (excluding noise)
    total_volume: int  # sum of all view_counts
    new_signals_pct: float  # week-over-week total volume % change


class TrendListResponse(BaseModel):
    """Full response for GET /api/trends/list."""

    header: TrendHeaderStats
    trends: list[TrendItem]
    generated_at: str  # ISO timestamp
