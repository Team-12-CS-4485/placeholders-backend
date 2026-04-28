# Pipeline Parallelization

## Overview

Both the ingest and article generation stages fan out across **10 parallel workers**, each holding a subset of the available Gemini API keys. This reduces wall-clock time by ~10x for large batches.

**Claim analysis makes zero Gemini calls** — it uses local nomic embeddings and is already fast.

---

## Ingest (PipelineService)

```
POST /api/pipeline/ingest  →  returns { job_id, status: "running" } immediately

background thread:
  Phase 1: Load S3 objects (sequential)
  Phase 2: Filter already-indexed videos (Qdrant check)
  Phase 3: Gemini fan-out — 10 workers in parallel
           worker_0 (keys 0–2):  chunk → intelligence → thumbnail → DynamoDB write
           worker_1 (keys 3–5):  chunk → intelligence → thumbnail → DynamoDB write
           ...
           [exhausted workers → retry_queue]
  Phase 4: Retry queue (sequential, full key pool)
  Phase 5: Embed chunks + Qdrant upsert (sequential — sentence-transformers not thread-safe)
  Phase 6: sync_missing (catches any Qdrant gaps)
```

Workers **fail fast** on key exhaustion (no 60s sleep) — the failed video goes to the retry queue.

---

## Article Generation (ArticleService)

```
POST /api/articles/generate

  Phase 1: Load existing articles (pre-flight dedup check)
  Phase 2: Gemini fan-out — 10 workers in parallel
           worker_0 (keys 0–2):  build prompt → Gemini → parse → DynamoDB write
           worker_1 (keys 3–5):  build prompt → Gemini → parse → DynamoDB write
           ...
           [exhausted workers → retry_queue]
  Phase 3: Retry queue (sequential, full key pool)
```

---

## Key Services

| File | Role |
|------|------|
| `app/services/gemini_service.py` | `KeysExhaustedError` + `fail_on_exhaustion` flag per worker |
| `app/services/job_service.py` | In-memory job store with `threading.Lock` — tracks status/result by UUID |
| `app/api/endpoints/pipeline.py` | `/run` and `/ingest` spawn background threads, return `job_id` immediately |

---

## Speedup

| Scenario | Sequential | Parallel (10 workers) |
|----------|------------|----------------------|
| 20 videos | ~3 min | ~25 sec |
| 50 videos | ~8 min | ~1 min |
| 15 articles | ~2.5 min | ~20 sec |

Poll `GET /api/pipeline/jobs/{job_id}` for status after submitting an async job.
