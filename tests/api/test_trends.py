from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.main import app

TREND_LIST_FIXTURE = {
    "trends": [
        {
            "cluster_id": 9,
            "label": "Iran-Israel Energy Security Crisis",
            "category": "Middle East Conflict",
            "trend_type": "dominant",
            "metric_badge": "High Impact",
            "heat_score": 106.33,
            "video_count": 37,
            "channel_count": 7,
            "view_count_total": 2732519,
            "breaking_count": 29,
            "sentiment_label": "Negative",
            "recent_sentiment_label": "Negative",
            "dominant_sentiment": "negative",
            "top_topics": ["Middle East Conflict", "Oil Markets", "US Foreign Policy"],
        }
    ],
    "total": 1,
}

TREND_DETAIL_FIXTURE = {
    "cluster_id": 9,
    "label": "Iran-Israel Energy Security Crisis",
    "category": "Middle East Conflict",
    "trend_type": "dominant",
    "metric_badge": "High Impact",
    "heat_score": 106.33,
    "video_count": 37,
    "channel_count": 7,
    "view_count_total": 2732519,
    "total_likes": 5000,
    "total_comments": 1200,
    "engagement_index": 190.5,
    "breaking_count": 29,
    "sentiment_breakdown": {"negative": 32, "neutral": 5},
    "sentiment_label": "Negative",
    "recent_sentiment_label": "Negative",
    "dominant_sentiment": "negative",
    "channels": ["BBCNews", "CBSNews", "FoxNews"],
    "week_data": [
        {
            "week": "week1",
            "video_count": 16,
            "channel_count": 5,
            "view_count": 776779,
            "breaking_count": 14,
            "sentiment_breakdown": {"neutral": 3, "negative": 13},
        },
        {
            "week": "week2",
            "video_count": 8,
            "channel_count": 5,
            "view_count": 924101,
            "breaking_count": 7,
            "sentiment_breakdown": {"negative": 8},
        },
    ],
    "top_claims": [
        "The US destroyed 16 Iranian mine-laying boats.",
        "Over 1M passengers affected.",
    ],
    "top_topics": ["Middle East Conflict", "Oil Markets"],
    "avg_clickbait_rating": None,
    "thumbnail_tone_breakdown": {},
}


@pytest.mark.asyncio
async def test_get_trends_returns_200():
    with patch("app.api.endpoints.trends.TrendService") as mock_svc:
        mock_svc.return_value.get_trends.return_value = TREND_LIST_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_trends_response_shape():
    with patch("app.api.endpoints.trends.TrendService") as mock_svc:
        mock_svc.return_value.get_trends.return_value = TREND_LIST_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends")
    data = response.json()
    assert "trends" in data
    assert "total" in data
    item = data["trends"][0]
    assert isinstance(item["cluster_id"], int)
    assert isinstance(item["label"], str)
    assert isinstance(item["category"], str)
    assert isinstance(item["heat_score"], float)
    assert isinstance(item["sentiment_label"], str)
    assert isinstance(item["recent_sentiment_label"], str)
    assert isinstance(item["top_topics"], list)


@pytest.mark.asyncio
async def test_get_trend_detail_returns_200():
    with patch("app.api.endpoints.trends.TrendService") as mock_svc:
        mock_svc.return_value.get_trend_detail.return_value = TREND_DETAIL_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends/9")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_trend_detail_shape():
    with patch("app.api.endpoints.trends.TrendService") as mock_svc:
        mock_svc.return_value.get_trend_detail.return_value = TREND_DETAIL_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/trends/9")
    data = response.json()
    assert isinstance(data["cluster_id"], int)
    assert isinstance(data["heat_score"], float)
    assert isinstance(data["engagement_index"], float)
    assert isinstance(data["top_claims"], list)
    assert isinstance(data["week_data"], list)
    week = data["week_data"][0]
    assert isinstance(week["week"], str)
    assert isinstance(week["video_count"], int)
    assert isinstance(week["view_count"], int)
    assert isinstance(week["breaking_count"], int)
