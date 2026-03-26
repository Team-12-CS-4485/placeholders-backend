"""
trends.py - Trend API Endpoints

GET /api/trends                  → lean cluster list
GET /api/trends/{id}             → full cluster detail
GET /api/trends/{id}/sentiment   → sentiment breakdown + per-week
GET /api/trends/{id}/claims      → classified claims (consensus/debated/unique)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.trend import TrendListResponse, TrendDetail, TrendSentiment, TrendClaims
from app.services.trend_service import TrendService

router = APIRouter(prefix="/api/trends", tags=["trends"])


@router.get("", response_model=TrendListResponse)
def get_trends(
    sort_by: str = Query(
        default="heat_score",
        description="Sort field: heat_score, video_count, view_count_total, channel_count, engagement_index",
    ),
):
    service = TrendService()
    return service.get_trends(sort_by=sort_by)


@router.get("/{cluster_id}/sentiment", response_model=TrendSentiment)
def get_trend_sentiment(cluster_id: int):
    service = TrendService()
    try:
        return service.get_trend_sentiment(cluster_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Trend {cluster_id} not found")


@router.get("/{cluster_id}/claims", response_model=TrendClaims)
def get_trend_claims(cluster_id: int):
    service = TrendService()
    try:
        return service.get_trend_claims(cluster_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Trend {cluster_id} not found")


@router.get("/{cluster_id}", response_model=TrendDetail)
def get_trend_detail(cluster_id: int):
    service = TrendService()
    try:
        return service.get_trend_detail(cluster_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Trend {cluster_id} not found")
