from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

# Mocking the services before importing the app to avoid heavy initialization/downloads
with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.main import app


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
