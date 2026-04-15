from unittest.mock import patch, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.main import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_NARRATIVE = {
    "cluster_id": 8,
    "label": "Epstein Investigation",
    "category": "Congress & Legislation",
    "narrative_headline": "Epstein Files Expose Network of Powerful Connections",
    "narrative_summary": "Newly unsealed documents link Epstein to 36 public figures.",
    "top_topics": ["epstein", "government"],
    "video_count": 9,
    "dominant_sentiment": "negative",
}

SAMPLE_DETAIL = {
    **SAMPLE_NARRATIVE,
    "top_claims": ["Documents released by court order"],
    "channel_count": 5,
    "breaking_count": 2,
    "channels": ["CNBC", "FoxNews"],
    "week_data": [
        {
            "week": "week1",
            "video_count": 5,
            "channel_count": 3,
            "view_count": 200000,
            "breaking_count": 1,
            "sentiment_breakdown": {"negative": 4, "neutral": 1},
            "public_sentiment_breakdown": {},
        }
    ],
    "creator_risk": [],
    "avg_clickbait_rating": 3.2,
    "thumbnail_tone_breakdown": {"urgency": 5, "neutral": 4},
}

SAMPLE_CLAIMS = {
    "cluster_id": 8,
    "claims": {
        "consensus": [],
        "debated": [],
        "unique": [],
    },
}

SAMPLE_VIDEO = {
    "video_id": "abc123",
    "channel": "CNBC",
    "title": "Epstein Docs",
    "published_at": "2026-03-10T12:00:00+00:00",
    "view_count": 50000,
    "like_count": 1000,
    "comment_count": 200,
    "week": "week1",
    "topics": None,
    "category": None,
    "sentiment": None,
    "key_claims": None,
    "is_breaking": None,
    "cluster_id": 8,
    "cluster_label": "Epstein Investigation",
    "thumbnail_url": "https://img.youtube.com/vi/abc123/maxresdefault.jpg",
    "thumbnail_tone": None,
    "thumbnail_clickbait_score": None,
    "thumbnail_insight": None,
    "thumbnail_brand_consistent": None,
}


# ── GET /api/narratives ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_narratives():
    with patch(
        "app.api.endpoints.narratives.TrendService.get_narratives",
        return_value={"narratives": [SAMPLE_NARRATIVE], "total": 1},
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["narratives"][0]["cluster_id"] == 8
    assert data["narratives"][0]["label"] == "Epstein Investigation"


@pytest.mark.asyncio
async def test_list_narratives_passes_week():
    mock_get = MagicMock(return_value={"narratives": [], "total": 0})
    with patch("app.api.endpoints.narratives.TrendService.get_narratives", mock_get):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await ac.get("/api/narratives?week=week2&sort_by=label")

    mock_get.assert_called_once_with(week="week2", sort_by="label")


# ── GET /api/narratives/{id} ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_narrative_detail():
    with patch(
        "app.api.endpoints.narratives.TrendService.get_narrative_detail",
        return_value=SAMPLE_DETAIL,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/8")

    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] == 8
    assert data["avg_clickbait_rating"] == 3.2
    assert len(data["week_data"]) == 1
    assert "public_sentiment_breakdown" in data["week_data"][0]


@pytest.mark.asyncio
async def test_get_narrative_detail_not_found():
    with patch(
        "app.api.endpoints.narratives.TrendService.get_narrative_detail",
        side_effect=KeyError(99),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/99")

    assert response.status_code == 404


# ── GET /api/narratives/{id}/claims ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_narrative_claims():
    with patch(
        "app.api.endpoints.narratives.TrendService.get_narrative_claims",
        return_value=SAMPLE_CLAIMS,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/8/claims")

    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] == 8
    assert "consensus" in data["claims"]
    assert "debated" in data["claims"]
    assert "unique" in data["claims"]


# ── GET /api/narratives/{id}/videos ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_narrative_videos():
    scan_result = {"items": [SAMPLE_VIDEO], "total_returned": 1, "next_cursor": None}
    with patch(
        "app.api.endpoints.narratives.DynamoService.scan_videos",
        return_value=scan_result,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/8/videos")

    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] == 8
    assert data["total_returned"] == 1


@pytest.mark.asyncio
async def test_get_narrative_videos_week_filter():
    mock_scan = MagicMock(
        return_value={"items": [], "total_returned": 0, "next_cursor": None}
    )
    with patch("app.api.endpoints.narratives.DynamoService.scan_videos", mock_scan):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await ac.get("/api/narratives/8/videos?week=week2")

    _, kwargs = mock_scan.call_args
    assert kwargs.get("week") == "week2"
    assert kwargs.get("cluster_id") == 8
