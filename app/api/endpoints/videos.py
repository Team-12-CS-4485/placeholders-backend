from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.video import VideoDetailItem, VideoListResponse
from app.services.storage_service import StorageService

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.get("", response_model=VideoListResponse)
def list_videos(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str = Query(default=None),
):
    try:
        service = StorageService()
        return service.list_videos(limit=limit, cursor=cursor)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list videos: {exc}"
        ) from exc


@router.get("/by-id", response_model=VideoDetailItem)
def get_video_by_id(
    video_id: str = Query(..., min_length=1),
):
    try:
        service = StorageService()
        item = service.get_video_by_id(video_id=video_id)
        if not item:
            detail = f"Video not found for video_id='{video_id}'"
            raise HTTPException(status_code=404, detail=detail)
        return item
    except HTTPException:
        raise
    except Exception as exc:
        scope = f"video_id='{video_id}'"
        raise HTTPException(
            status_code=500,
            detail=("Failed to fetch video for " f"{scope}: {exc}"),
        ) from exc
