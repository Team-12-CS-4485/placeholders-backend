from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.main import app

# ── GET /api/stats ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_stats():
    stats = {
        "total_videos": 191,
        "total_clusters": 12,
        "total_weeks": 4,
        "breaking_count": 24,
    }
    with patch(
        "app.api.endpoints.stats.TrendService.get_stats",
        return_value=stats,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["total_videos"] == 191
    assert data["total_clusters"] == 12
    assert data["total_weeks"] == 4
    assert data["breaking_count"] == 24


# ── GET /api/articles ─────────────────────────────────────────────────────────

SAMPLE_ARTICLE_ITEM = {
    "article_id": "article-a1b2c3d4",
    "cluster_id": 8,
    "cluster_label": "Epstein Investigation",
    "week_number": 1,
    "week_start_date": "2026-03-02",
    "title": "Epstein Files Expose Network",
    "overview": "Newly unsealed documents link Epstein to 36 public figures.",
    "created_at": "2026-03-10T09:00:00+00:00",
}

SAMPLE_ARTICLE_DETAIL = {
    **SAMPLE_ARTICLE_ITEM,
    "body": "Full article prose goes here...",
    "updated_at": "2026-03-10T09:00:00+00:00",
}


@pytest.mark.asyncio
async def test_list_articles():
    with patch(
        "app.api.endpoints.articles.ArticleService.get_articles",
        return_value=[SAMPLE_ARTICLE_ITEM],
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/articles")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    item = data["articles"][0]
    assert item["article_id"] == "article-a1b2c3d4"
    assert item["cluster_id"] == 8
    assert "body" not in item


@pytest.mark.asyncio
async def test_get_article_by_id():
    with patch(
        "app.api.endpoints.articles.ArticleService.get_article_by_id",
        return_value=SAMPLE_ARTICLE_DETAIL,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/articles/article-a1b2c3d4")

    assert response.status_code == 200
    data = response.json()
    assert data["article_id"] == "article-a1b2c3d4"
    assert "body" in data
    assert data["body"] == "Full article prose goes here..."


@pytest.mark.asyncio
async def test_get_article_by_id_not_found():
    with patch(
        "app.api.endpoints.articles.ArticleService.get_article_by_id",
        return_value=None,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/articles/missing-id")

    assert response.status_code == 404
