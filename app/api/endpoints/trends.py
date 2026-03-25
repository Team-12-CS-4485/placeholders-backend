"""
trends.py - Trend API Endpoints

GET /api/trends          → full trend list (powers list + grid views)
GET /api/trends/{id}     → single trend detail by cluster_id
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.trend import TrendListResponse, TrendItem
from app.services.trend_service import TrendService

router = APIRouter(prefix="/api/trends", tags=["trends"])


@router.get("", response_model=TrendListResponse)
def get_trends(
    sort_by: str = Query(
        default="heat_score",
        description="Sort field: heat_score, video_count, view_count_total, channel_count",
    ),
):
    """
    Returns all active trends/narratives with metrics.
    Powers both the list view and grid view on the Trends Archive page.
    """
    service = TrendService()
    return service.get_trends(sort_by=sort_by)


@router.get("/{cluster_id}", response_model=TrendItem)
def get_trend_detail(cluster_id: int):
    """
    Returns a single trend by cluster_id.
    Powers the trend detail / article page.
    """
    service = TrendService()
    data = service.get_trends()

    for trend in data["trends"]:
        if trend["cluster_id"] == cluster_id:
            return trend

    raise HTTPException(
        status_code=404, detail=f"Trend with cluster_id={cluster_id} not found"
    )
