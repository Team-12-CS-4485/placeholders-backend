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
    week: str                       # "week1", "week2", ...
    video_count: int                # unique videos this week
    channel_count: int              # distinct channels covering it
    view_count: int                 # total views this week
    breaking_count: int             # is_breaking videos
    sentiment_breakdown: dict[str, int]  # {"positive": 2, "negative": 5, "neutral": 1}


class TrendItem(BaseModel):
    """
    Single trend/narrative — powers both list rows and grid cards.

    List view uses: label, description, category, metric_badge, trend_type
    Grid view adds: week_data, sentiment_breakdown, channels
    """
    cluster_id: int
    label: str                      # cluster_label — e.g. "Maritime Security"
    description: str                # top key_claim or summary snippet
    category: str                   # dominant_category — e.g. "Middle East Conflict"
    trend_type: str                 # "rising" | "emerging" | "dominant" | "declining" | "stable"
    metric_badge: str               # display string — "+40% Vol", "68% Neg", "High Impact"

    # Volume & engagement
    video_count: int                # total videos across all weeks
    channel_count: int              # total distinct channels
    view_count_total: int           # total views across all weeks
    breaking_count: int             # total is_breaking videos
    heat_score: float               # composite ranking score

    # Sentiment
    sentiment_breakdown: dict[str, int]
    dominant_sentiment: str

    # Channels covering this trend
    channels: list[str]

    # Week-over-week data (powers sparklines, bar charts)
    week_data: list[WeekData]

    # Top claims from this cluster (powers description + detail page)
    top_claims: list[str]

    # Top topics within the cluster
    top_topics: list[str]


class TrendHeaderStats(BaseModel):
    """Header bar stats — top of the Trends Archive page."""
    active_narratives: int          # number of clusters (excluding noise)
    total_volume: int               # sum of all view_counts
    new_signals_pct: float          # week-over-week total volume % change


class TrendListResponse(BaseModel):
    """Full response for GET /api/trends/list."""
    header: TrendHeaderStats
    trends: list[TrendItem]
    generated_at: str               # ISO timestamp