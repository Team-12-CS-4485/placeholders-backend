"""
trends.py - Trend API Endpoints

GET /api/trends                  → lean cluster list (optional ?week=week1&sort_by=heat_score)
GET /api/trends/{id}             → full cluster detail
GET /api/trends/{id}/sentiment   → sentiment breakdown + per-week
GET /api/trends/{id}/claims      → classified claims (consensus/debated/unique)
GET /api/weeks                   → available weeks with aggregate stats
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.cache import get_cached
from app.schemas.trend import (
    TrendListResponse,
    TrendDetail,
    TrendSentiment,
    TrendClaims,
    WeeksResponse,
)
from app.services.trend_service import TrendService

router = APIRouter(prefix="/api/trends", tags=["trends"])
weeks_router = APIRouter(prefix="/api/weeks", tags=["weeks"])


@router.get("", response_model=TrendListResponse)
def get_trends(
    sort_by: str = Query(
        default="heat_score",
        description="Sort field: heat_score, video_count, view_count_total, channel_count, engagement_index",
    ),
    week: Optional[str] = Query(
        default=None,
        description="Filter to a specific week (e.g. week1, week2). Returns metrics scoped to that week.",
    ),
):
    return get_cached(
        f"trends:{sort_by}:{week}",
        ttl=300,
        fetch_fn=lambda: TrendService().get_trends(sort_by=sort_by, week=week),
    )


@router.get("/{cluster_id}/sentiment", response_model=TrendSentiment)
def get_trend_sentiment(cluster_id: int):
    try:
        return get_cached(
            f"trend_sentiment:{cluster_id}",
            ttl=300,
            fetch_fn=lambda: TrendService().get_trend_sentiment(cluster_id),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Trend {cluster_id} not found")


@router.get("/{cluster_id}/claims", response_model=TrendClaims)
def get_trend_claims(cluster_id: int):
    try:
        return get_cached(
            f"trend_claims:{cluster_id}",
            ttl=300,
            fetch_fn=lambda: TrendService().get_trend_claims(cluster_id),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Trend {cluster_id} not found")


@router.get("/{cluster_id}", response_model=TrendDetail)
def get_trend_detail(cluster_id: int):
    try:
        return get_cached(
            f"trend_detail:{cluster_id}",
            ttl=300,
            fetch_fn=lambda: TrendService().get_trend_detail(cluster_id),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Trend {cluster_id} not found")


@weeks_router.get("", response_model=WeeksResponse)
def get_weeks():
    """List all available weeks with aggregate stats across all clusters."""
    return get_cached("weeks", ttl=300, fetch_fn=lambda: TrendService().get_weeks())
