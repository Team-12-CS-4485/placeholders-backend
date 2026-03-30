"""
narratives.py - Narrative API Endpoints

GET /api/narratives                  → lean narrative list (optional ?week=week1)
GET /api/narratives/{id}             → full narrative detail
GET /api/narratives/{id}/claims      → classified claims (consensus/debated/unique)
GET /api/narratives/{id}/videos      → paginated videos belonging to this cluster
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.narrative import (
    NarrativeListResponse,
    NarrativeDetail,
    NarrativeClaims,
    NarrativeVideosResponse,
)
from app.services.trend_service import TrendService
from app.services.dynamo_service import DynamoService

router = APIRouter(prefix="/api/narratives", tags=["narratives"])


@router.get("", response_model=NarrativeListResponse)
def get_narratives(
    week: Optional[str] = Query(
        default=None,
        description="Filter to a specific week (e.g. week1, week2).",
    ),
):
    service = TrendService()
    return service.get_narratives(week=week)


@router.get("/{cluster_id}/claims", response_model=NarrativeClaims)
def get_narrative_claims(cluster_id: int):
    service = TrendService()
    try:
        return service.get_narrative_claims(cluster_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Narrative {cluster_id} not found")


@router.get("/{cluster_id}/videos", response_model=NarrativeVideosResponse)
def get_narrative_videos(
    cluster_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: Optional[str] = Query(default=None),
):
    service = DynamoService()
    result = service.scan_videos(limit=limit, cursor=cursor, cluster_id=cluster_id)
    return {
        "cluster_id": cluster_id,
        "items": result["items"],
        "total_returned": result["total_returned"],
        "next_cursor": result["next_cursor"],
    }


@router.get("/{cluster_id}", response_model=NarrativeDetail)
def get_narrative_detail(cluster_id: int):
    service = TrendService()
    try:
        return service.get_narrative_detail(cluster_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Narrative {cluster_id} not found")
