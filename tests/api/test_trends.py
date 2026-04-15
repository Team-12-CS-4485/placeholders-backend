from unittest.mock import patch, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.main import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

WEEK_DATA = [
    {
        "week": "week1",
        "video_count": 8,
        "channel_count": 4,
        "view_count": 400000,
        "breaking_count": 2,
        "sentiment_breakdown": {"negative": 6, "neutral": 2},
        "public_sentiment_breakdown": {"negative": 5, "neutral": 3},
    },
    {
        "week": "week2",
        "video_count": 12,
        "channel_count": 6,
        "view_count": 800000,
        "breaking_count": 3,
        "sentiment_breakdown": {"negative": 10, "neutral": 2},
        "public_sentiment_breakdown": {"negative": 8, "neutral": 4},
    },
]

SAMPLE_TREND_ITEM = {
    "cluster_id": 9,
    "label": "Iran Conflict",
    "category": "Middle East Conflict",
    "trend_type": "dominant",
    "metric_badge": "High Impact",
    "heat_score": 38.6,
    "video_count": 20,
    "channel_count": 7,
    "view_count_total": 1200000,
    "breaking_count": 5,
    "sentiment_label": "Negative",
    "recent_sentiment_label": "Negative",
    "dominant_sentiment": "negative",
    "dominant_public_sentiment": "negative",
    "sentiment_divergence": False,
    "top_topics": ["iran", "israel", "missiles"],
}

SAMPLE_TREND_DETAIL = {
    **SAMPLE_TREND_ITEM,
    "total_likes": 48000,
    "total_comments": 12000,
    "engagement_index": 50.0,
    "sentiment_breakdown": {"negative": 18, "neutral": 2},
    "public_sentiment_breakdown": {"negative": 15, "neutral": 5},
    "avg_public_sentiment_score": -0.6,
    "channels": ["CNBC", "FoxNews"],
    "week_data": WEEK_DATA,
    "top_claims": ["Iran launched missiles toward Israel"],
    "creator_risk": [],
    "avg_clickbait_rating": 3.8,
    "thumbnail_tone_breakdown": {"urgency": 12, "fear": 4},
}

SAMPLE_SENTIMENT = {
    "cluster_id": 9,
    "sentiment_breakdown": {"negative": 18, "neutral": 2},
    "sentiment_label": "Negative",
    "recent_sentiment_label": "Negative",
    "dominant_sentiment": "negative",
    "by_week": [
        {
            "week": "week1",
            "sentiment_breakdown": {"negative": 6, "neutral": 2},
            "dominant_sentiment": "negative",
        }
    ],
}

SAMPLE_CLAIMS = {
    "cluster_id": 9,
    "claims": {"consensus": [], "debated": [], "unique": []},
}


# ── GET /api/trends ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_trends():
    with patch(
        "app.api.endpoints.trends.TrendService.get_trends",
        return_value={"trends": [SAMPLE_TREND_ITEM], "total": 1},
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    item = data["trends"][0]
    assert item["cluster_id"] == 9
    assert item["heat_score"] == 38.6
    assert item["trend_type"] == "dominant"


@pytest.mark.asyncio
async def test_list_trends_passes_sort_and_week():
    mock_get = MagicMock(return_value={"trends": [], "total": 0})
    with patch("app.api.endpoints.trends.TrendService.get_trends", mock_get):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await ac.get("/api/trends?sort_by=video_count&week=week1")

    mock_get.assert_called_once_with(sort_by="video_count", week="week1")


# ── GET /api/trends/{id} ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_trend_detail():
    with patch(
        "app.api.endpoints.trends.TrendService.get_trend_detail",
        return_value=SAMPLE_TREND_DETAIL,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends/9")

    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] == 9
    assert data["engagement_index"] == 50.0
    assert len(data["week_data"]) == 2
    assert "public_sentiment_breakdown" in data["week_data"][0]


@pytest.mark.asyncio
async def test_get_trend_detail_not_found():
    with patch(
        "app.api.endpoints.trends.TrendService.get_trend_detail",
        side_effect=KeyError(99),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends/99")

    assert response.status_code == 404


# ── GET /api/trends/{id}/sentiment ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_trend_sentiment():
    with patch(
        "app.api.endpoints.trends.TrendService.get_trend_sentiment",
        return_value=SAMPLE_SENTIMENT,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends/9/sentiment")

    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] == 9
    assert data["dominant_sentiment"] == "negative"
    assert len(data["by_week"]) == 1


# ── GET /api/trends/{id}/claims ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_trend_claims():
    with patch(
        "app.api.endpoints.trends.TrendService.get_trend_claims",
        return_value=SAMPLE_CLAIMS,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends/9/claims")

    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] == 9
    assert "consensus" in data["claims"]


# ── GET /api/weeks ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_weeks():
    weeks_data = {
        "weeks": [
            {
                "week": "week1",
                "total_videos": 60,
                "total_views": 4200000,
                "active_clusters": 10,
                "breaking_count": 8,
                "dominant_sentiment": "Negative",
            }
        ],
        "total": 1,
    }
    with patch(
        "app.api.endpoints.trends.TrendService.get_weeks",
        return_value=weeks_data,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/weeks")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["weeks"][0]["week"] == "week1"


# ── GET /api/weeks/{week} ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_week_narratives():
    week_narratives = {
        "week": "week1",
        "narratives": [
            {
                "cluster_id": 9,
                "cluster_label": "Iran Conflict",
                "narrative_headline": "Iran Launches Missiles",
                "narrative_summary": "Escalation continues.",
                "week_overview": "This week saw...",
                "top_topics": ["iran"],
                "top_claims": [],
                "video_count": 8,
                "view_count": 400000,
                "breaking_count": 2,
                "dominant_sentiment": "negative",
            }
        ],
        "total": 1,
    }
    with patch(
        "app.api.endpoints.trends.TrendService.get_week_narratives",
        return_value=week_narratives,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/weeks/week1")

    assert response.status_code == 200
    data = response.json()
    assert data["week"] == "week1"
    assert data["total"] == 1
    assert data["narratives"][0]["cluster_label"] == "Iran Conflict"
