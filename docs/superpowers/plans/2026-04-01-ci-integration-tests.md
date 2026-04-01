# CI Integration Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated contract tests for every frontend-consumed API endpoint, adapter unit tests for the frontend data layer, and a manually-triggered live smoke suite against the deployed backend.

**Architecture:** Backend contract tests use FastAPI's ASGI transport with mocked `TrendService` — no DynamoDB, no network. Frontend adapter tests use vitest to exercise pure adapter functions with fixture data. Live smoke tests share the same assertion patterns but hit `placeholders-backend.onrender.com` directly, triggered only via `workflow_dispatch`.

**Tech Stack:** pytest + httpx (backend, already installed), vitest (frontend, new dev dep), GitHub Actions `workflow_dispatch`

**Spec:** `docs/superpowers/specs/2026-04-01-ci-integration-tests-design.md`

---

## File Map

**Created:**
- `placeholders-backend/tests/integration/__init__.py`
- `placeholders-backend/tests/api/test_weeks.py`
- `placeholders-backend/tests/api/test_narratives.py`
- `placeholders-backend/tests/api/test_trends.py`
- `placeholders-backend/tests/integration/test_live_api.py`
- `placeholders-backend/.github/workflows/live-smoke.yml`
- `placeholders-frontend/vitest.config.ts`
- `placeholders-frontend/src/lib/adapters.test.ts`

**Modified:**
- `placeholders-backend/pytest.ini` — add `live` marker
- `placeholders-backend/.github/workflows/ci.yml` — exclude live tests from pytest run
- `placeholders-frontend/package.json` — add vitest dep + test script
- `placeholders-frontend/.github/workflows/ci.yml` — add test step

---

## Task 1: Backend — Register `live` marker and integration package

**Files:**
- Modify: `placeholders-backend/pytest.ini`
- Create: `placeholders-backend/tests/integration/__init__.py`

- [ ] **Step 1: Add the `live` marker to `pytest.ini`**

Open `placeholders-backend/pytest.ini` and replace its contents with:

```ini
[pytest]
asyncio_mode = strict
pythonpath = .
markers =
    live: marks tests as live integration tests requiring network access (deselect with -m "not live")
```

- [ ] **Step 2: Create the integration package file**

Create `placeholders-backend/tests/integration/__init__.py` as an empty file.

- [ ] **Step 3: Verify pytest recognises the marker**

```bash
cd placeholders-backend
pytest --markers | grep live
```

Expected output contains:
```
live: marks tests as live integration tests requiring network access
```

- [ ] **Step 4: Commit**

```bash
cd placeholders-backend
git add pytest.ini tests/integration/__init__.py
git commit -m "test: register live marker and integration test package"
```

---

## Task 2: Backend — `test_weeks.py`

**Files:**
- Create: `placeholders-backend/tests/api/test_weeks.py`

- [ ] **Step 1: Write the test file**

Create `placeholders-backend/tests/api/test_weeks.py`:

```python
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
```

- [ ] **Step 2: Run the tests and verify they pass**

```bash
cd placeholders-backend
pytest tests/api/test_weeks.py -v
```

Expected output:
```
tests/api/test_weeks.py::test_get_weeks_returns_200 PASSED
tests/api/test_weeks.py::test_get_weeks_response_shape PASSED
tests/api/test_weeks.py::test_get_weeks_item_fields PASSED
3 passed
```

- [ ] **Step 3: Commit**

```bash
git add tests/api/test_weeks.py
git commit -m "test: add contract tests for GET /api/weeks"
```

---

## Task 3: Backend — `test_narratives.py`

**Files:**
- Create: `placeholders-backend/tests/api/test_narratives.py`

- [ ] **Step 1: Write the test file**

Create `placeholders-backend/tests/api/test_narratives.py`:

```python
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
    "top_claims": ["The US destroyed 16 Iranian mine-laying boats.", "Over 1M passengers affected."],
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
        mock_svc.return_value.get_narrative_detail.return_value = NARRATIVE_DETAIL_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/9")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_narrative_detail_shape():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narrative_detail.return_value = NARRATIVE_DETAIL_FIXTURE
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
        mock_svc.return_value.get_narrative_claims.return_value = NARRATIVE_CLAIMS_FIXTURE
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/narratives/9/claims")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_narrative_claims_shape():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narrative_claims.return_value = NARRATIVE_CLAIMS_FIXTURE
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
```

- [ ] **Step 2: Run the tests and verify they pass**

```bash
cd placeholders-backend
pytest tests/api/test_narratives.py -v
```

Expected output:
```
tests/api/test_narratives.py::test_get_narratives_returns_200 PASSED
tests/api/test_narratives.py::test_get_narratives_response_shape PASSED
tests/api/test_narratives.py::test_get_narratives_week_filter_passes_param PASSED
tests/api/test_narratives.py::test_get_narrative_detail_returns_200 PASSED
tests/api/test_narratives.py::test_get_narrative_detail_shape PASSED
tests/api/test_narratives.py::test_get_narrative_claims_returns_200 PASSED
tests/api/test_narratives.py::test_get_narrative_claims_shape PASSED
7 passed
```

- [ ] **Step 3: Commit**

```bash
git add tests/api/test_narratives.py
git commit -m "test: add contract tests for narrative endpoints"
```

---

## Task 4: Backend — `test_trends.py`

**Files:**
- Create: `placeholders-backend/tests/api/test_trends.py`

- [ ] **Step 1: Write the test file**

Create `placeholders-backend/tests/api/test_trends.py`:

```python
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
    "top_claims": ["The US destroyed 16 Iranian mine-laying boats.", "Over 1M passengers affected."],
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
```

- [ ] **Step 2: Run the tests and verify they pass**

```bash
cd placeholders-backend
pytest tests/api/test_trends.py -v
```

Expected output:
```
tests/api/test_trends.py::test_get_trends_returns_200 PASSED
tests/api/test_trends.py::test_get_trends_response_shape PASSED
tests/api/test_trends.py::test_get_trend_detail_returns_200 PASSED
tests/api/test_trends.py::test_get_trend_detail_shape PASSED
4 passed
```

- [ ] **Step 3: Run the full backend test suite (excluding live) to confirm no regressions**

```bash
cd placeholders-backend
pytest tests/ -m "not live" -v
```

Expected: all tests pass, including `test_health.py`.

- [ ] **Step 4: Commit**

```bash
git add tests/api/test_trends.py
git commit -m "test: add contract tests for trend endpoints"
```

---

## Task 5: Backend — Update `ci.yml` to exclude live tests

**Files:**
- Modify: `placeholders-backend/.github/workflows/ci.yml`

- [ ] **Step 1: Update the pytest command**

In `placeholders-backend/.github/workflows/ci.yml`, find the pytest step and change it:

```yaml
      - name: Test with pytest
        run: pytest tests/ -m "not live"
```

The full `lint-and-test` job after the change:

```yaml
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.head_ref }}
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      - name: Format with black
        run: black .

      - name: Commit black formatting changes
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "style: auto-format with black"

      - name: Test with pytest
        run: pytest tests/ -m "not live"
```

- [ ] **Step 2: Commit**

```bash
cd placeholders-backend
git add .github/workflows/ci.yml
git commit -m "ci: exclude live-marked tests from required pipeline"
```

---

## Task 6: Live smoke tests + workflow

**Files:**
- Create: `placeholders-backend/tests/integration/test_live_api.py`
- Create: `placeholders-backend/.github/workflows/live-smoke.yml`

- [ ] **Step 1: Write the live smoke test file**

Create `placeholders-backend/tests/integration/test_live_api.py`:

```python
"""
Live smoke tests — hit the deployed backend and verify response shapes.
Run manually: pytest tests/integration/ -m live -v
Never runs in regular CI (excluded by -m "not live").
"""

import pytest
import httpx

BASE_URL = "https://placeholders-backend.onrender.com"

# Populated by test_weeks_live; reused by detail endpoint tests
_cluster_id: int = 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_weeks_live():
    """Fetches /api/weeks and validates every field the frontend consumes."""
    global _cluster_id
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
```

- [ ] **Step 2: Verify live tests are excluded from regular runs**

```bash
cd placeholders-backend
pytest tests/ -m "not live" --collect-only 2>&1 | grep "test_live"
```

Expected: no output (live tests not collected).

- [ ] **Step 3: Write the live-smoke workflow**

Create `placeholders-backend/.github/workflows/live-smoke.yml`:

```yaml
name: Live API Smoke Tests

on:
  workflow_dispatch:

jobs:
  smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run live smoke tests
        run: pytest tests/integration/ -m live -v
```

- [ ] **Step 4: Commit**

```bash
cd placeholders-backend
git add tests/integration/test_live_api.py .github/workflows/live-smoke.yml
git commit -m "test: add live smoke tests and manual workflow"
```

---

## Task 7: Frontend — Vitest setup

**Files:**
- Modify: `placeholders-frontend/package.json`
- Create: `placeholders-frontend/vitest.config.ts`

- [ ] **Step 1: Install vitest**

```bash
cd placeholders-frontend
npm install --save-dev vitest
```

- [ ] **Step 2: Create `vitest.config.ts`**

Create `placeholders-frontend/vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
```

- [ ] **Step 3: Add the `test` script to `package.json`**

In `placeholders-frontend/package.json`, add `"test": "vitest run"` to the `scripts` section:

```json
{
  "name": "placeholders-frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "test": "vitest run",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.2.0",
    "react-dom": "^19.2.0"
  },
  "devDependencies": {
    "@eslint/js": "^9.39.1",
    "@types/node": "^24.10.1",
    "@types/react": "^19.2.7",
    "@types/react-dom": "^19.2.3",
    "@vitejs/plugin-react": "^5.1.1",
    "eslint": "^9.39.1",
    "eslint-plugin-react-hooks": "^7.0.1",
    "eslint-plugin-react-refresh": "^0.4.24",
    "globals": "^16.5.0",
    "typescript": "~5.9.3",
    "typescript-eslint": "^8.48.0",
    "vite": "^7.3.1",
    "vitest": "<installed version>"
  }
}
```

> Note: the `"vitest"` version will be whatever `npm install` resolved. Do not hardcode it — leave what npm wrote.

- [ ] **Step 4: Verify vitest runs (no tests yet, expect zero collected)**

```bash
cd placeholders-frontend
npm run test
```

Expected output:
```
No test files found, exiting with code 1
```

> This exit code 1 is expected — vitest exits 1 when no tests are found. It confirms the runner is wired up.

- [ ] **Step 5: Commit**

```bash
cd placeholders-frontend
git add vitest.config.ts package.json package-lock.json
git commit -m "chore: add vitest for adapter unit tests"
```

---

## Task 8: Frontend — `adapters.test.ts`

**Files:**
- Create: `placeholders-frontend/src/lib/adapters.test.ts`

- [ ] **Step 1: Write the test file**

Create `placeholders-frontend/src/lib/adapters.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import {
  adaptWeeks,
  adaptNarrativesList,
  adaptNarrativeDetail,
  adaptClaims,
  adaptTrendsList,
  adaptTrendDetail,
  generateTrendAlerts,
} from './adapters';
import type {
  BackendWeeksResponse,
  BackendNarrativeListItem,
  BackendNarrativeDetail,
  BackendNarrativeClaims,
  BackendTrendListItem,
  BackendTrendDetail,
} from '../services/api';

// ── Fixtures ──────────────────────────────────────────────────────────────────

const WEEKS_RESPONSE: BackendWeeksResponse = {
  weeks: [
    {
      week: 'week1',
      total_videos: 37,
      total_views: 2737594,
      active_clusters: 7,
      breaking_count: 19,
      dominant_sentiment: 'negative',
    },
  ],
  total: 1,
};

const NARRATIVE_LIST_ITEM: BackendNarrativeListItem = {
  cluster_id: 9,
  label: 'Iran-Israel Energy Crisis',
  category: 'Middle East Conflict',
  narrative_headline: 'US-Israel Strikes Trigger Energy Crisis',
  top_topics: ['Middle East', 'Oil Markets'],
  video_count: 37,
  dominant_sentiment: 'negative',
};

const NARRATIVE_DETAIL: BackendNarrativeDetail = {
  cluster_id: 9,
  label: 'Iran-Israel Energy Crisis',
  category: 'Middle East Conflict',
  narrative_headline: 'US-Israel Strikes Trigger Energy Crisis',
  narrative_summary: 'Following hostilities, 32 countries released oil reserves.',
  top_topics: ['Middle East', 'Oil Markets'],
  top_claims: ['Claim one.', 'Claim two.'],
  video_count: 37,
  channel_count: 7,
  breaking_count: 29,
  dominant_sentiment: 'negative',
  channels: ['BBCNews', 'CBSNews'],
  week_data: [
    {
      week: 'week1',
      video_count: 16,
      channel_count: 5,
      view_count: 776779,
      breaking_count: 14,
      sentiment_breakdown: { neutral: 3, negative: 13 },
    },
  ],
  creator_risk: [],
  avg_clickbait_rating: null,
  thumbnail_tone_breakdown: {},
};

const CLAIMS_RESPONSE: BackendNarrativeClaims = {
  cluster_id: 9,
  claims: {
    consensus: [
      {
        claim: 'The US destroyed 16 Iranian mine-laying boats.',
        channel: 'BBCNews',
        sources: ['BBCNews', 'CBSNews'],
        source_count: 2,
        video_ids: ['abc123'],
        transcript_excerpt: 'The US military reported destroying 16 boats.',
        risk_score: 0.25,
      },
    ],
    debated: [
      {
        claim: 'Airspace disruptions forced longer airline routes.',
        channel: 'FoxNews',
        perspectives: [{ text: 'Airlines are adding hours to routes.' }],
        source_count: 2,
        framing_divergence: 0.54,
        risk_score: 0.54,
      },
    ],
    unique: [
      {
        claim: 'US struck PMF forces south of Baghdad.',
        channel: 'aljazeeraenglish',
        video_id: 'ghi789',
        video_title: 'US Airstrikes Baghdad',
        transcript_excerpt: 'US airstrikes targeted PMF positions.',
        risk_score: 0.85,
      },
    ],
  },
};

const TREND_LIST_ITEM: BackendTrendListItem = {
  cluster_id: 9,
  label: 'Iran-Israel Energy Security Crisis',
  category: 'Middle East Conflict',
  trend_type: 'dominant',
  metric_badge: 'High Impact',
  heat_score: 106.33,
  video_count: 37,
  channel_count: 7,
  view_count_total: 2732519,
  breaking_count: 29,
  sentiment_label: 'Negative',
  recent_sentiment_label: 'Negative',
  dominant_sentiment: 'negative',
  dominant_public_sentiment: 'negative',
  sentiment_divergence: false,
  top_topics: ['Middle East Conflict', 'Oil Markets'],
};

const TREND_DETAIL: BackendTrendDetail = {
  ...TREND_LIST_ITEM,
  total_likes: 5000,
  total_comments: 1200,
  engagement_index: 190.5,
  sentiment_breakdown: { negative: 32, neutral: 5 },
  public_sentiment_breakdown: { negative: 20, neutral: 10 },
  avg_public_sentiment_score: -0.6,
  channels: ['BBCNews', 'CBSNews'],
  week_data: [
    { week: 'week1', video_count: 16, channel_count: 5, view_count: 776779, breaking_count: 14, sentiment_breakdown: { negative: 13, neutral: 3 } },
    { week: 'week2', video_count: 8, channel_count: 5, view_count: 924101, breaking_count: 7, sentiment_breakdown: { negative: 8 } },
    { week: 'week3', video_count: 6, channel_count: 4, view_count: 913364, breaking_count: 4, sentiment_breakdown: { negative: 6 } },
  ],
  top_claims: ['Claim one.', 'Claim two.'],
  creator_risk: [
    { name: 'FoxNews', riskScore: 0.76, riskLevel: 'high', claimCount: 6 },
  ],
  avg_clickbait_rating: null,
  thumbnail_tone_breakdown: {},
};

// ── adaptWeeks ────────────────────────────────────────────────────────────────

describe('adaptWeeks', () => {
  it('maps week identifier to id and formatted weekName', () => {
    const result = adaptWeeks(WEEKS_RESPONSE);
    expect(result[0].id).toBe('week1');
    expect(result[0].weekName).toBe('Week 1');
  });

  it('starts with an empty narratives array', () => {
    const result = adaptWeeks(WEEKS_RESPONSE);
    expect(result[0].narratives).toEqual([]);
  });

  it('includes breaking_count in summary headline', () => {
    const result = adaptWeeks(WEEKS_RESPONSE);
    expect(result[0].summary.headline).toContain('19');
  });

  it('capitalises dominant_sentiment in summary headline', () => {
    const result = adaptWeeks(WEEKS_RESPONSE);
    expect(result[0].summary.headline).toContain('Negative');
  });

  it('returns one entry per week in the response', () => {
    const result = adaptWeeks(WEEKS_RESPONSE);
    expect(result).toHaveLength(1);
  });
});

// ── adaptNarrativesList ───────────────────────────────────────────────────────

describe('adaptNarrativesList', () => {
  it('converts cluster_id to a string id', () => {
    const result = adaptNarrativesList([NARRATIVE_LIST_ITEM], 'week1');
    expect(result[0].id).toBe('9');
  });

  it('passes weekId through unchanged', () => {
    const result = adaptNarrativesList([NARRATIVE_LIST_ITEM], 'week1');
    expect(result[0].weekId).toBe('week1');
  });

  it('sets trendIds to [cluster_id.toString()]', () => {
    const result = adaptNarrativesList([NARRATIVE_LIST_ITEM], 'week1');
    expect(result[0].trendIds).toEqual(['9']);
  });

  it('starts with an empty claims array', () => {
    const result = adaptNarrativesList([NARRATIVE_LIST_ITEM], 'week1');
    expect(result[0].claims).toEqual([]);
  });

  it('uses narrative_headline when present', () => {
    const result = adaptNarrativesList([NARRATIVE_LIST_ITEM], 'week1');
    expect(result[0].headline).toBe('US-Israel Strikes Trigger Energy Crisis');
  });

  it('falls back to label when narrative_headline is null', () => {
    const item = { ...NARRATIVE_LIST_ITEM, narrative_headline: null };
    const result = adaptNarrativesList([item], 'week1');
    expect(result[0].headline).toBe('Iran-Israel Energy Crisis');
  });

  it('assigns pageNumber starting at 1 based on index', () => {
    const result = adaptNarrativesList([NARRATIVE_LIST_ITEM, NARRATIVE_LIST_ITEM], 'week1');
    expect(result[0].pageNumber).toBe(1);
    expect(result[1].pageNumber).toBe(2);
  });
});

// ── adaptNarrativeDetail ──────────────────────────────────────────────────────

describe('adaptNarrativeDetail', () => {
  it('puts narrative_summary as the first fullText entry', () => {
    const result = adaptNarrativeDetail(NARRATIVE_DETAIL, CLAIMS_RESPONSE, 'week1');
    expect(result.fullText[0]).toBe('Following hostilities, 32 countries released oil reserves.');
  });

  it('appends each top_claim as a subsequent fullText entry', () => {
    const result = adaptNarrativeDetail(NARRATIVE_DETAIL, CLAIMS_RESPONSE, 'week1');
    expect(result.fullText[1]).toBe('Claim one.');
    expect(result.fullText[2]).toBe('Claim two.');
  });

  it('falls back to label when narrative_summary is null and top_claims is empty', () => {
    const detail = { ...NARRATIVE_DETAIL, narrative_summary: null, top_claims: [] };
    const emptyClaims: BackendNarrativeClaims = {
      cluster_id: 9,
      claims: { consensus: [], debated: [], unique: [] },
    };
    const result = adaptNarrativeDetail(detail, emptyClaims, 'week1');
    expect(result.fullText).toEqual(['Iran-Israel Energy Crisis']);
  });

  it('populates claims from the claims response', () => {
    const result = adaptNarrativeDetail(NARRATIVE_DETAIL, CLAIMS_RESPONSE, 'week1');
    expect(result.claims.length).toBeGreaterThan(0);
  });
});

// ── adaptClaims ───────────────────────────────────────────────────────────────

describe('adaptClaims', () => {
  it('flattens all three claim types into a single array', () => {
    const result = adaptClaims(CLAIMS_RESPONSE, '9');
    expect(result).toHaveLength(3);
  });

  it('maps consensus video_ids[0] to a YouTube watch URL', () => {
    const result = adaptClaims(CLAIMS_RESPONSE, '9');
    const c = result.find(r => r.id.includes('con'));
    expect(c?.videoUrl).toBe('https://www.youtube.com/watch?v=abc123');
  });

  it('uses # when consensus video_ids is empty', () => {
    const noVideo: BackendNarrativeClaims = {
      ...CLAIMS_RESPONSE,
      claims: {
        ...CLAIMS_RESPONSE.claims,
        consensus: [{ ...CLAIMS_RESPONSE.claims.consensus[0], video_ids: [] }],
      },
    };
    const result = adaptClaims(noVideo, '9');
    const c = result.find(r => r.id.includes('con'));
    expect(c?.videoUrl).toBe('#');
  });

  it('maps unique video_id to a YouTube watch URL', () => {
    const result = adaptClaims(CLAIMS_RESPONSE, '9');
    const u = result.find(r => r.id.includes('unq'));
    expect(u?.videoUrl).toBe('https://www.youtube.com/watch?v=ghi789');
  });

  it('derives creatorInitials from the first two words of the creator name', () => {
    const result = adaptClaims(CLAIMS_RESPONSE, '9');
    const c = result.find(r => r.id.includes('con'));
    expect(c?.creatorInitials).toBe('BB'); // BBCNews → B + B
  });

  it('passes risk_score through to the claim', () => {
    const result = adaptClaims(CLAIMS_RESPONSE, '9');
    const c = result.find(r => r.id.includes('con'));
    expect(c?.riskScore).toBe(0.25);
  });
});

// ── adaptTrendsList ───────────────────────────────────────────────────────────

describe('adaptTrendsList', () => {
  it('maps cluster_id to string id', () => {
    const result = adaptTrendsList([TREND_LIST_ITEM]);
    expect(result[0].id).toBe('9');
  });

  it('maps label to name', () => {
    const result = adaptTrendsList([TREND_LIST_ITEM]);
    expect(result[0].name).toBe('Iran-Israel Energy Security Crisis');
  });

  it('maps heat_score to totalEngagement', () => {
    const result = adaptTrendsList([TREND_LIST_ITEM]);
    expect(result[0].totalEngagement).toBe(106.33);
  });

  it('produces at least 2 engagementData points for MiniGraph', () => {
    const result = adaptTrendsList([TREND_LIST_ITEM]);
    expect(result[0].engagementData.length).toBeGreaterThanOrEqual(2);
  });

  it('initialises creatorRisks as an empty array', () => {
    const result = adaptTrendsList([TREND_LIST_ITEM]);
    expect(result[0].creatorRisks).toEqual([]);
  });
});

// ── adaptTrendDetail ──────────────────────────────────────────────────────────

describe('adaptTrendDetail', () => {
  it('maps week_data view_count to engagementData values', () => {
    const result = adaptTrendDetail(TREND_DETAIL);
    expect(result.engagementData[0].value).toBe(776779);
    expect(result.engagementData[1].value).toBe(924101);
  });

  it('formats week identifiers as Week N in engagementData', () => {
    const result = adaptTrendDetail(TREND_DETAIL);
    expect(result.engagementData[0].date).toBe('Week 1');
  });

  it('maps all week_data entries to barChartData 90 Days', () => {
    const result = adaptTrendDetail(TREND_DETAIL);
    expect(result.barChartData['90 Days']).toHaveLength(3);
  });

  it('maps only the last 2 week_data entries to barChartData 30 Days', () => {
    const result = adaptTrendDetail(TREND_DETAIL);
    expect(result.barChartData['30 Days']).toHaveLength(2);
    expect(result.barChartData['30 Days'][0].label).toBe('Week 2');
  });

  it('maps top_claims to detailedAnalysis', () => {
    const result = adaptTrendDetail(TREND_DETAIL);
    expect(result.detailedAnalysis).toEqual(['Claim one.', 'Claim two.']);
  });

  it('uppercases creator riskLevel "high" to "HIGH"', () => {
    const result = adaptTrendDetail(TREND_DETAIL);
    expect(result.creatorRisks[0].riskLevel).toBe('HIGH');
  });

  it('normalises "medium" riskLevel to "MED"', () => {
    const detail: BackendTrendDetail = {
      ...TREND_DETAIL,
      creator_risk: [{ name: 'TestChannel', riskScore: 0.5, riskLevel: 'medium', claimCount: 3 }],
    };
    const result = adaptTrendDetail(detail);
    expect(result.creatorRisks[0].riskLevel).toBe('MED');
  });

  it('maps creator_risk name to channelId', () => {
    const result = adaptTrendDetail(TREND_DETAIL);
    expect(result.creatorRisks[0].channelId).toBe('FoxNews');
  });
});

// ── generateTrendAlerts ───────────────────────────────────────────────────────

describe('generateTrendAlerts', () => {
  it('assigns SHIFT type when sentiment_divergence is true', () => {
    const item: BackendTrendListItem = { ...TREND_LIST_ITEM, sentiment_divergence: true, breaking_count: 5 };
    const result = generateTrendAlerts([item]);
    expect(result[0].type).toBe('SHIFT');
  });

  it('assigns WARNING type when breaking_count > 15 and no divergence', () => {
    const item: BackendTrendListItem = { ...TREND_LIST_ITEM, sentiment_divergence: false, breaking_count: 29 };
    const result = generateTrendAlerts([item]);
    expect(result[0].type).toBe('WARNING');
  });

  it('assigns NEW type for qualifying trends with low breaking count', () => {
    const item: BackendTrendListItem = { ...TREND_LIST_ITEM, sentiment_divergence: false, breaking_count: 11, heat_score: 55 };
    const result = generateTrendAlerts([item]);
    expect(result[0].type).toBe('NEW');
  });

  it('caps results at 3 alerts', () => {
    const items: BackendTrendListItem[] = Array.from({ length: 10 }, (_, i) => ({
      ...TREND_LIST_ITEM,
      cluster_id: i,
      heat_score: 60,
    }));
    const result = generateTrendAlerts(items);
    expect(result.length).toBeLessThanOrEqual(3);
  });

  it('excludes trends that do not meet any alert threshold', () => {
    const item: BackendTrendListItem = {
      ...TREND_LIST_ITEM,
      heat_score: 30,
      breaking_count: 5,
      sentiment_divergence: false,
    };
    const result = generateTrendAlerts([item]);
    expect(result).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run the adapter tests and verify they all pass**

```bash
cd placeholders-frontend
npm run test
```

Expected output:
```
 ✓ src/lib/adapters.test.ts (X tests)
Test Files  1 passed (1)
Tests       X passed (X)
```

All tests should pass since the adapters are already implemented.

- [ ] **Step 3: Commit**

```bash
cd placeholders-frontend
git add src/lib/adapters.test.ts
git commit -m "test: add unit tests for all adapter functions"
```

---

## Task 9: Frontend — Add test step to CI

**Files:**
- Modify: `placeholders-frontend/.github/workflows/ci.yml`

- [ ] **Step 1: Add `npm run test` between lint and build**

Replace the contents of `placeholders-frontend/.github/workflows/ci.yml` with:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Test
        run: npm run test

      - name: Build
        run: npm run build
```

- [ ] **Step 2: Verify the full pipeline locally**

```bash
cd placeholders-frontend
npm run lint && npm run test && npm run build
```

Expected: all three commands exit 0.

- [ ] **Step 3: Commit**

```bash
cd placeholders-frontend
git add .github/workflows/ci.yml
git commit -m "ci: add adapter unit tests to required pipeline"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Backend contract tests for all 6 frontend-consumed endpoints (weeks, narratives list, narratives week-filter, narrative detail, narrative claims, trends list, trend detail)
- ✅ `pytest.ini` updated with `live` marker
- ✅ Backend `ci.yml` excludes live tests
- ✅ Frontend vitest setup (config + package.json + script)
- ✅ All 7 adapter functions tested (`adaptWeeks`, `adaptNarrativesList`, `adaptNarrativeDetail`, `adaptClaims`, `adaptTrendsList`, `adaptTrendDetail`, `generateTrendAlerts`)
- ✅ Frontend `ci.yml` updated with test step
- ✅ Live smoke tests for all 7 endpoints
- ✅ `live-smoke.yml` with `workflow_dispatch`

**Type consistency:**
- Backend fixture field names match Pydantic model names exactly (verified against `app/schemas/trend.py` and `app/schemas/narrative.py`)
- Frontend fixture types use `BackendWeeksResponse`, `BackendNarrativeListItem`, etc. from `../services/api` — same types the adapter functions accept
- `adaptClaims` called with `(CLAIMS_RESPONSE, '9')` in both `adaptNarrativeDetail` tests and standalone — consistent with function signature `(res: BackendNarrativeClaims, narrativeId: string)`
