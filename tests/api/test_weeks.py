from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.main import app

WEEKS_FIXTURE = {
    "weeks": [
        {
            "week": "week1",
            "total_videos": 37,
            "total_views": 2737594,
            "active_clusters": 7,
            "breaking_count": 19,
            "dominant_sentiment": "negative",
        }
    ],
    "total": 1,
}


@pytest.mark.asyncio
async def test_get_weeks_returns_200():
    with patch("app.api.endpoints.trends.TrendService") as mock_svc:
        mock_svc.return_value.get_weeks.return_value = WEEKS_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/weeks")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_weeks_response_shape():
    with patch("app.api.endpoints.trends.TrendService") as mock_svc:
        mock_svc.return_value.get_weeks.return_value = WEEKS_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/weeks")
    data = response.json()
    assert "weeks" in data
    assert "total" in data
    assert isinstance(data["total"], int)
    assert len(data["weeks"]) > 0


@pytest.mark.asyncio
async def test_get_weeks_item_fields():
    with patch("app.api.endpoints.trends.TrendService") as mock_svc:
        mock_svc.return_value.get_weeks.return_value = WEEKS_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/weeks")
    week = response.json()["weeks"][0]
    assert isinstance(week["week"], str)
    assert isinstance(week["total_videos"], int)
    assert isinstance(week["total_views"], int)
    assert isinstance(week["active_clusters"], int)
    assert isinstance(week["breaking_count"], int)
    assert isinstance(week["dominant_sentiment"], str)
