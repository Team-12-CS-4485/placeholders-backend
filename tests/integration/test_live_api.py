"""
Live smoke tests — hit the deployed backend and verify response shapes.
Run manually: pytest tests/integration/ -m live -v
Never runs in regular CI (excluded by -m "not live").
"""

import pytest
import httpx

BASE_URL = "https://placeholders-backend.onrender.com"

# Populated by test_narratives_list_live; reused by detail endpoint tests
_cluster_id: int = 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_weeks_live():
    """Fetches /api/weeks and validates every field the frontend consumes."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/api/weeks")
    assert response.status_code == 200
    data = response.json()
    assert "weeks" in data
    assert "total" in data
    assert len(data["weeks"]) > 0
    week = data["weeks"][0]
    for field in ("week", "dominant_sentiment"):
        assert isinstance(week[field], str), f"Expected str for {field}"
    for field in ("total_videos", "total_views", "active_clusters", "breaking_count"):
        assert isinstance(week[field], int), f"Expected int for {field}"


@pytest.mark.live
@pytest.mark.asyncio
async def test_narratives_list_live():
    """Fetches /api/narratives and records a cluster_id for downstream tests."""
    global _cluster_id
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/api/narratives")
    assert response.status_code == 200
    data = response.json()
    assert "narratives" in data
    assert len(data["narratives"]) > 0
    item = data["narratives"][0]
    _cluster_id = item["cluster_id"]
    assert isinstance(item["cluster_id"], int)
    assert isinstance(item["label"], str)
    assert isinstance(item["category"], str)
    assert isinstance(item["top_topics"], list)
    assert isinstance(item["video_count"], int)
    assert isinstance(item["dominant_sentiment"], str)


@pytest.mark.live
@pytest.mark.asyncio
async def test_narratives_week_filter_live():
    """Fetches /api/narratives?week=week1 and verifies the list shape."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/api/narratives?week=week1")
    assert response.status_code == 200
    data = response.json()
    assert "narratives" in data
    assert "total" in data
    assert isinstance(data["narratives"], list)


@pytest.mark.live
@pytest.mark.asyncio
async def test_narrative_detail_live():
    """Fetches /api/narratives/{id} and verifies the detail shape."""
    assert _cluster_id != 0, "Run test_narratives_list_live first to populate _cluster_id"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/api/narratives/{_cluster_id}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["cluster_id"], int)
    assert isinstance(data["label"], str)
    assert isinstance(data["top_claims"], list)
    assert isinstance(data["week_data"], list)
    assert isinstance(data["channels"], list)
    if data["week_data"]:
        week = data["week_data"][0]
        assert isinstance(week["week"], str)
        assert isinstance(week["video_count"], int)
        assert isinstance(week["view_count"], int)


@pytest.mark.live
@pytest.mark.asyncio
async def test_narrative_claims_live():
    """Fetches /api/narratives/{id}/claims and verifies classified claims shape."""
    assert _cluster_id != 0, "Run test_narratives_list_live first to populate _cluster_id"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/api/narratives/{_cluster_id}/claims")
    assert response.status_code == 200
    claims = response.json()["claims"]
    assert "consensus" in claims
    assert "debated" in claims
    assert "unique" in claims
    assert isinstance(claims["consensus"], list)
    assert isinstance(claims["debated"], list)
    assert isinstance(claims["unique"], list)


@pytest.mark.live
@pytest.mark.asyncio
async def test_trends_list_live():
    """Fetches /api/trends and verifies list shape."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/api/trends")
    assert response.status_code == 200
    data = response.json()
    assert "trends" in data
    assert len(data["trends"]) > 0
    item = data["trends"][0]
    assert isinstance(item["cluster_id"], int)
    assert isinstance(item["label"], str)
    assert isinstance(item["heat_score"], float)
    assert isinstance(item["sentiment_label"], str)
    assert isinstance(item["recent_sentiment_label"], str)
    assert isinstance(item["top_topics"], list)


@pytest.mark.live
@pytest.mark.asyncio
async def test_trend_detail_live():
    """Fetches /api/trends/{id} and verifies detail shape."""
    assert _cluster_id != 0, "Run test_narratives_list_live first to populate _cluster_id"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/api/trends/{_cluster_id}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["heat_score"], float)
    assert isinstance(data["week_data"], list)
    assert isinstance(data["top_claims"], list)
    if data["week_data"]:
        week = data["week_data"][0]
        assert isinstance(week["week"], str)
        assert isinstance(week["video_count"], int)
        assert isinstance(week["view_count"], int)
