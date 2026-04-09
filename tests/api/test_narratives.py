from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.main import app

NARRATIVE_LIST_FIXTURE = {
    "narratives": [
        {
            "cluster_id": 9,
            "label": "Iran-Israel Energy Security Crisis",
            "category": "Middle East Conflict",
            "narrative_headline": "US-Israel Strikes Trigger Global Energy Crisis",
            "narrative_summary": "Following hostilities, 32 countries released oil reserves.",
            "top_topics": ["Middle East Conflict", "Oil Markets", "US Foreign Policy"],
            "video_count": 37,
            "dominant_sentiment": "negative",
        }
    ],
    "total": 1,
}

NARRATIVE_DETAIL_FIXTURE = {
    "cluster_id": 9,
    "label": "Iran-Israel Energy Security Crisis",
    "category": "Middle East Conflict",
    "narrative_headline": "US-Israel Strikes Trigger Global Energy Crisis",
    "narrative_summary": "Following hostilities, 32 countries released oil reserves.",
    "top_topics": ["Middle East Conflict", "Oil Markets"],
    "top_claims": [
        "The US destroyed 16 Iranian mine-laying boats.",
        "Over 1M passengers affected.",
    ],
    "video_count": 37,
    "channel_count": 7,
    "breaking_count": 29,
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
        }
    ],
    "avg_clickbait_rating": None,
    "thumbnail_tone_breakdown": {},
}

NARRATIVE_CLAIMS_FIXTURE = {
    "cluster_id": 9,
    "claims": {
        "consensus": [
            {
                "claim": "The US destroyed 16 Iranian mine-laying boats.",
                "sources": ["BBCNews", "CBSNews"],
                "source_count": 2,
                "video_ids": ["abc123def"],
                "transcript_excerpt": "The US military reported destroying 16 boats.",
            }
        ],
        "debated": [
            {
                "claim": "Airspace disruptions forced longer airline routes.",
                "perspectives": [
                    {
                        "channel": "FoxNews",
                        "sentiment": "negative",
                        "video_id": "def456ghi",
                        "video_title": "Airlines Rerouting Flights",
                        "transcript_excerpt": "Airlines are adding hours to transatlantic flights.",
                    }
                ],
                "source_count": 2,
                "framing_divergence": 0.54,
            }
        ],
        "unique": [
            {
                "claim": "US struck Popular Mobilization Forces south of Baghdad.",
                "channel": "aljazeeraenglish",
                "video_id": "ghi789jkl",
                "video_title": "US Airstrikes Baghdad",
                "transcript_excerpt": "US airstrikes targeted PMF positions.",
            }
        ],
    },
}


@pytest.mark.asyncio
async def test_get_narratives_returns_200():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narratives.return_value = NARRATIVE_LIST_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_narratives_response_shape():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narratives.return_value = NARRATIVE_LIST_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives")
    data = response.json()
    assert "narratives" in data
    assert "total" in data
    item = data["narratives"][0]
    assert isinstance(item["cluster_id"], int)
    assert isinstance(item["label"], str)
    assert isinstance(item["category"], str)
    assert isinstance(item["top_topics"], list)
    assert isinstance(item["video_count"], int)
    assert isinstance(item["dominant_sentiment"], str)


@pytest.mark.asyncio
async def test_get_narratives_week_filter_passes_param():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narratives.return_value = NARRATIVE_LIST_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives?week=week1")
    assert response.status_code == 200
    mock_svc.return_value.get_narratives.assert_called_once_with(
        week="week1", sort_by="video_count"
    )


@pytest.mark.asyncio
async def test_get_narrative_detail_returns_200():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narrative_detail.return_value = (
            NARRATIVE_DETAIL_FIXTURE
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/9")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_narrative_detail_shape():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narrative_detail.return_value = (
            NARRATIVE_DETAIL_FIXTURE
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/9")
    data = response.json()
    assert isinstance(data["cluster_id"], int)
    assert isinstance(data["label"], str)
    assert isinstance(data["category"], str)
    assert isinstance(data["top_claims"], list)
    assert isinstance(data["week_data"], list)
    assert isinstance(data["channels"], list)
    week = data["week_data"][0]
    assert isinstance(week["week"], str)
    assert isinstance(week["video_count"], int)
    assert isinstance(week["view_count"], int)


@pytest.mark.asyncio
async def test_get_narrative_claims_returns_200():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narrative_claims.return_value = (
            NARRATIVE_CLAIMS_FIXTURE
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/9/claims")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_narrative_claims_shape():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narrative_claims.return_value = (
            NARRATIVE_CLAIMS_FIXTURE
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/9/claims")
    claims = response.json()["claims"]
    assert "consensus" in claims
    assert "debated" in claims
    assert "unique" in claims
    assert isinstance(claims["consensus"], list)
    assert isinstance(claims["debated"], list)
    assert isinstance(claims["unique"], list)
    c = claims["consensus"][0]
    assert isinstance(c["claim"], str)
    assert isinstance(c["sources"], list)
    assert isinstance(c["video_ids"], list)
    assert isinstance(c["transcript_excerpt"], str)
    u = claims["unique"][0]
    assert isinstance(u["claim"], str)
    assert isinstance(u["channel"], str)
    assert isinstance(u["video_id"], str)
    assert isinstance(u["transcript_excerpt"], str)
