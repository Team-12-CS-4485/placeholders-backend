# CI Integration Tests — Design Spec

**Date:** 2026-04-01
**Status:** Approved
**Scope:** Backend contract tests + frontend adapter unit tests + manual live smoke tests

---

## Problem

The frontend–backend integration was completed and verified manually. There is no automated safety net to catch regressions if either side changes — a renamed schema field, a dropped response key, or a broken adapter mapping would only surface at runtime.

## Goal

Automated tests that run on every PR/push and fail fast when the integration contract is broken, plus a manually-triggered smoke suite that validates the live deployed backend.

---

## Architecture

Three test layers, each with a distinct scope:

```
placeholders-backend/
├── tests/
│   ├── api/
│   │   ├── test_health.py          (existing)
│   │   ├── test_narratives.py      NEW
│   │   ├── test_trends.py          NEW
│   │   └── test_weeks.py           NEW
│   └── integration/
│       └── test_live_api.py        NEW  (manual only)
├── pytest.ini                      MODIFY
└── .github/workflows/
    ├── ci.yml                      MODIFY
    └── live-smoke.yml              NEW

placeholders-frontend/
├── src/lib/
│   └── adapters.test.ts            NEW
├── vitest.config.ts                NEW
├── package.json                    MODIFY
└── .github/workflows/
    └── ci.yml                      MODIFY
```

**Layer responsibilities:**

| Layer | Where | Triggered | Tests |
|---|---|---|---|
| Backend contract tests | `tests/api/` | Every PR/push (required) | API response shapes match frontend `api.ts` type expectations |
| Frontend adapter tests | `src/lib/adapters.test.ts` | Every PR/push (required) | Adapter functions map backend fixtures to correct frontend types |
| Live smoke tests | `tests/integration/` | `workflow_dispatch` (manual) | Real deployed endpoints return expected shapes with real data |

---

## Backend Contract Tests

### Pattern

Follows the existing `test_health.py` approach: mock heavy services before app import, then mock `TrendService` methods at the endpoint module level per test. Uses ASGI transport — no network, no DynamoDB.

```python
with patch("app.services.pipeline_service.PipelineService"), \
     patch("app.services.storage_service.StorageService"):
    from app.main import app

@pytest.mark.asyncio
async def test_get_narratives_returns_required_fields():
    with patch("app.api.endpoints.narratives.TrendService") as mock_svc:
        mock_svc.return_value.get_narratives.return_value = NARRATIVE_LIST_FIXTURE
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/narratives")
    assert response.status_code == 200
    item = response.json()["narratives"][0]
    assert isinstance(item["cluster_id"], int)
    assert isinstance(item["label"], str)
    # ... all fields frontend api.ts expects
```

### Fixtures

Each test file defines a module-level `FIXTURE` dict — minimal but structurally complete, containing every field the frontend's `api.ts` type declarations reference. Fixtures are plain dicts (not Pydantic instances) to match what the serialized JSON response looks like.

### Coverage per file

**`test_weeks.py`**
- `GET /api/weeks` → 200, response has `weeks[]` and `total`
- Each week item has: `week` (str), `total_videos` (int), `total_views` (int), `active_clusters` (int), `breaking_count` (int), `dominant_sentiment` (str)

**`test_narratives.py`**
- `GET /api/narratives` → 200, has `narratives[]` and `total`
- `GET /api/narratives?week=week1` → 200, same shape
- `GET /api/narratives/{id}` → 200, detail has `top_claims[]`, `week_data[]`, `channels[]`, nullable `narrative_headline`, nullable `narrative_summary`
- `GET /api/narratives/{id}/claims` → 200, `claims` has `consensus[]`, `debated[]`, `unique[]`; each consensus item has `claim`, `sources[]`, `video_ids[]`, `transcript_excerpt`, `risk_score`; each unique item has `claim`, `channel`, `video_id`, `transcript_excerpt`, `risk_score`

**`test_trends.py`**
- `GET /api/trends` → 200, each item has `cluster_id`, `label`, `category`, `heat_score` (float), `sentiment_label`, `recent_sentiment_label`, `top_topics[]`
- `GET /api/trends/{id}` → 200, detail adds `week_data[]` (each with `week`, `video_count`, `view_count`), `creator_risk[]` (each with `name`, `riskScore`, `riskLevel`), `top_claims[]`

### pytest.ini change

```ini
[pytest]
asyncio_mode = strict
pythonpath = .
markers =
    live: marks tests as live integration tests (deselect with -m "not live")
```

### ci.yml change (backend)

```yaml
- name: Test with pytest
  run: pytest tests/ -m "not live"   # was: pytest tests/
```

---

## Frontend Adapter Unit Tests

### Setup

```bash
npm install --save-dev vitest
```

**`vitest.config.ts`:**
```ts
import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: { include: ['src/**/*.test.ts'] },
});
```

**`package.json` addition:**
```json
"test": "vitest run"
```

### Test file: `src/lib/adapters.test.ts`

One `describe` block per exported function. Each test passes a minimal fixture (only fields the function actually reads) and asserts the output shape.

| Function | Key assertions |
|---|---|
| `adaptWeeks` | `id === "week1"`, `weekName === "Week 1"`, `narratives === []`, `summary.headline` contains breaking count and sentiment |
| `adaptNarrativesList` | `id === cluster_id.toString()`, `weekId` passed through, `trendIds === [id]`, `claims === []`, `headline` falls back to `label` when `narrative_headline` is null |
| `adaptNarrativeDetail` | `fullText` includes `narrative_summary` and each `top_claims` entry; `claims` comes from second arg |
| `adaptClaims` | consensus + debated + unique all flattened into one array; `video_ids[0]` → `https://www.youtube.com/watch?v=...`; `video_id` on unique claim → watch URL; empty `video_ids` → `"#"`; `creatorInitials` derived from name |
| `adaptTrendsList` | `name === label`, `totalEngagement === heat_score`, `engagementData.length >= 2`, `creatorRisks === []` |
| `adaptTrendDetail` | `week_data` maps to `engagementData` and `barChartData['90 Days']`; `barChartData['30 Days']` is last 2 entries; `creator_risk[].riskLevel` uppercased; `"medium"` → `"MED"`; `top_claims` → `detailedAnalysis` |
| `generateTrendAlerts` | `sentiment_divergence: true` → type `"SHIFT"`; `breaking_count > 15` → `"WARNING"`; others → `"NEW"`; capped at 3 results |

### ci.yml change (frontend)

```yaml
- name: Lint
  run: npm run lint

- name: Test        # NEW step
  run: npm run test

- name: Build
  run: npm run build
```

---

## Live Smoke Tests

### `tests/integration/test_live_api.py`

Uses `httpx.AsyncClient` hitting `https://placeholders-backend.onrender.com` directly. All tests marked `@pytest.mark.live`.

The first test (`test_weeks_live`) fetches `/api/weeks`, validates the response, and extracts `cluster_id` values for use by subsequent tests — no hardcoded IDs.

| Test | Endpoint | Asserts |
|---|---|---|
| `test_weeks_live` | `GET /api/weeks` | 200, `weeks` non-empty, all required fields present |
| `test_narratives_list_live` | `GET /api/narratives` | 200, `narratives` non-empty, required fields on first item |
| `test_narratives_week_filter_live` | `GET /api/narratives?week=week1` | 200, returns valid list structure |
| `test_narrative_detail_live` | `GET /api/narratives/{id}` | 200, has `week_data`, `top_claims`, `channels` |
| `test_narrative_claims_live` | `GET /api/narratives/{id}/claims` | 200, `claims` has all three keys |
| `test_trends_list_live` | `GET /api/trends` | 200, `trends` non-empty, has `heat_score`, `sentiment_label` |
| `test_trend_detail_live` | `GET /api/trends/{id}` | 200, has `week_data[]`, `creator_risk[]` |

### `live-smoke.yml`

```yaml
name: Live API Smoke Tests
on:
  workflow_dispatch:

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/integration/ -m live -v
```

---

## What This Does Not Cover

- Value assertions (exact content of returned data) — intentionally deferred; shape tests are stable, value tests can be layered on later
- Component rendering tests — out of scope; the adapter layer is the integration boundary
- Pipeline endpoints — not used by the frontend, not tested here
