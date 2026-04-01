"""
stats.py - Stats API Endpoint

GET /api/stats : Top-level aggregate summary across all clusters and weeks.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.trend_service import TrendService

router = APIRouter(prefix="/api/stats", tags=["stats"])


class StatsResponse(BaseModel):
    total_videos: int
    total_clusters: int
    total_weeks: int
    breaking_count: int


@router.get("", response_model=StatsResponse)
def get_stats():
    """Aggregate stats derived from narrative-clusters — no extra reads."""
    service = TrendService()
    return service.get_stats()
