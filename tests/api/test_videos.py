from unittest.mock import patch, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.main import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_VIDEO = {
    "video_id": "abc123",
    "channel": "CNBC",
    "title": "Market Update",
    "published_at": "2026-03-10T12:00:00+00:00",
    "view_count": 142000,
    "like_count": 3200,
    "comment_count": 410,
    "week": "week1",
    "topics": ["inflation", "federal reserve"],
    "category": "Wall Street & Markets",
    "sentiment": "negative",
    "key_claims": ["Fed expected to hold rates"],
    "is_breaking": False,
    "cluster_id": 4,
    "cluster_label": "Federal Reserve Policy",
    "thumbnail_url": "https://img.youtube.com/vi/abc123/maxresdefault.jpg",
    "thumbnail_tone": "urgency",
    "thumbnail_clickbait_score": 3,
    "thumbnail_insight": "Bold overlay with alarming colors",
    "thumbnail_brand_consistent": True,
}

SCAN_RESULT = {
    "items": [SAMPLE_VIDEO],
    "total_returned": 1,
    "next_cursor": None,
}


# ── GET /api/videos ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_videos_returns_items():
    with patch(
        "app.api.endpoints.videos.DynamoService.scan_videos",
        return_value=SCAN_RESULT,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/videos")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["total_returned"] == 1
    assert data["items"][0]["video_id"] == "abc123"
    assert data["items"][0]["channel"] == "CNBC"


@pytest.mark.asyncio
async def test_list_videos_passes_week_filter():
    mock_scan = MagicMock(return_value=SCAN_RESULT)
    with patch("app.api.endpoints.videos.DynamoService.scan_videos", mock_scan):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await ac.get("/api/videos?week=week1")

    mock_scan.assert_called_once()
    _, kwargs = mock_scan.call_args
    assert kwargs.get("week") == "week1"


@pytest.mark.asyncio
async def test_list_videos_default_limit():
    mock_scan = MagicMock(return_value=SCAN_RESULT)
    with patch("app.api.endpoints.videos.DynamoService.scan_videos", mock_scan):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await ac.get("/api/videos")

    _, kwargs = mock_scan.call_args
    assert kwargs.get("limit") == 20


@pytest.mark.asyncio
async def test_list_videos_cursor_forwarded():
    mock_scan = MagicMock(return_value=SCAN_RESULT)
    with patch("app.api.endpoints.videos.DynamoService.scan_videos", mock_scan):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await ac.get("/api/videos?cursor=abc&limit=5")

    _, kwargs = mock_scan.call_args
    assert kwargs.get("cursor") == "abc"
    assert kwargs.get("limit") == 5


# ── GET /api/videos/by-id ────────────────────────────────────────────────────

SAMPLE_DETAIL = {
    **SAMPLE_VIDEO,
    "description": "A detailed description.",
    "transcript": "Transcript text here.",
    "top_comments": [{"author": "user1", "text": "Great!", "likes": 10}],
}


@pytest.mark.asyncio
async def test_get_video_by_id_found():
    with patch("app.api.endpoints.videos.StorageService") as MockStorage:
        MockStorage.return_value.get_video_by_id.return_value = SAMPLE_DETAIL
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/videos/by-id?video_id=abc123")

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "abc123"
    assert "transcript" in data
    assert "top_comments" in data


@pytest.mark.asyncio
async def test_get_video_by_id_not_found():
    with patch("app.api.endpoints.videos.StorageService") as MockStorage:
        MockStorage.return_value.get_video_by_id.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/videos/by-id?video_id=missing")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_video_by_id_missing_param():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/api/videos/by-id")
    assert response.status_code == 422
