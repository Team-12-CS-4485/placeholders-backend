from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.video import VideoListResponse
from app.services.dynamo_service import DynamoService

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.get("", response_model=VideoListResponse)
def list_videos(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str = Query(default=None),
    week: Optional[str] = Query(default=None),
):
    try:
        service = DynamoService()
        return service.scan_videos(limit=limit, cursor=cursor, week=week)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list videos: {exc}"
        ) from exc
