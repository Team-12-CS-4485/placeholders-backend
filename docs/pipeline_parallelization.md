# Pipeline Parallelization

## Overview

The ingest pipeline currently processes videos **sequentially** — one Gemini call per video, one after another. With 50 videos at ~10s per call, that's ~8 minutes of blocking. Article generation has the same problem — one Gemini call per (cluster × week).

With 30 API keys, we run **10 parallel workers × 3 keys each**. Workers fail fast on key exhaustion (no 60s sleep), failed work goes to a retry queue, and `sync_missing` runs at the end to catch any Qdrant gaps. The API endpoints return a job ID immediately so the frontend can poll for status.

**Claim analysis makes zero Gemini calls** — pure local embeddings + numpy math, no changes needed.

---

## Expected Speedup

| Scenario | Before | After |
|---|---|---|
| 20 videos, 10 workers | ~3 min | ~25 sec |
| 50 videos, 10 workers | ~8 min | ~1 min |
| 15 articles, 10 workers | ~2.5 min | ~20 sec |

---

## Architecture

```
POST /api/pipeline/ingest
  └─ returns { job_id, status: "running" } immediately
       │
       └─ background thread:
            ├─ Phase 1: Load S3 objects (sequential, fast)
            ├─ Phase 2: Filter already-indexed videos (Qdrant check, sequential)
            ├─ Phase 3: Gemini fan-out (10 workers in parallel)
            │    worker_0 (keys 0,1,2): chunk → Gemini → DynamoDB write
            │    worker_1 (keys 3,4,5): chunk → Gemini → DynamoDB write
            │    ...
            │    [any exhausted workers → retry_queue]
            ├─ Phase 4: Retry queue (sequential, full 30-key pool)
            ├─ Phase 5: Embed + Qdrant upsert (sequential, not thread-safe)
            └─ Phase 6: sync_missing (catches any Qdrant gaps)

GET /api/pipeline/jobs/{job_id}
  └─ { status, progress, total, failed_videos, result }
```

---

## Files to Change

| File | Change |
|---|---|
| `app/services/gemini_service.py` | Add `KeysExhaustedError` + `fail_on_exhaustion` flag |
| `app/services/pipeline_service.py` | Extract `_process_single_video()`; 10-worker fan-out; retry queue; sync_missing |
| `app/services/article_service.py` | Extract `_generate_single_article()`; 10-worker fan-out; retry queue |
| `app/services/job_service.py` | **New file** — in-memory job store with `threading.Lock` |
| `app/api/endpoints/pipeline.py` | `/ingest` + `/run` return job_id immediately; add `GET /jobs/{job_id}` |
| `app/schemas/pipeline.py` | Add `JobSubmitResponse`, `JobStatusResponse` |

See `docs/article_worker_parallelization.md` for the article service changes in isolation.

---

## Step 1 — `gemini_service.py`

This is a prerequisite for everything else. Two small changes.

### Add `KeysExhaustedError` (module level, after imports)

```python
class KeysExhaustedError(Exception):
    """Raised when all keys in a worker's pool are rate-limited."""
    pass
```

### Add `fail_on_exhaustion` to `__init__`

```python
def __init__(self, client=None, chunker=None, api_keys=None,
             fail_on_exhaustion: bool = False):
    ...
    self.fail_on_exhaustion = fail_on_exhaustion
```

### Update the `ALL_KEYS_EXHAUSTED` block in both `_gemini()` and `_gemini_vision()`

Find (appears twice):
```python
logger.warning("ALL_KEYS_EXHAUSTED resetting to key 0 and waiting 60s")
self.current_key_index = 0
self.client = genai.Client(api_key=self.api_keys[0])
time.sleep(60)
continue
```

Replace with:
```python
if self.fail_on_exhaustion:
    raise KeysExhaustedError(
        f"All {len(self.api_keys)} keys exhausted for this worker"
    )
logger.warning("ALL_KEYS_EXHAUSTED resetting to key 0 and waiting 60s")
self.current_key_index = 0
self.client = genai.Client(api_key=self.api_keys[0])
time.sleep(60)
continue
```

> Default is `False` — no existing callers break.

---

## Step 2 — `pipeline_service.py`

### 2a. Add imports + constant

```python
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.gemini_service import KeysExhaustedError

NUM_WORKERS = 10
```

### 2b. Add `_make_worker_gemini_services()` to `PipelineService`

```python
def _make_worker_gemini_services(self) -> list[GeminiService]:
    """Split all keys into NUM_WORKERS groups of 3 (one GeminiService per group)."""
    keys = self.gemini_service.api_keys
    group_size = max(1, len(keys) // NUM_WORKERS)
    services = []
    for i in range(0, len(keys), group_size):
        services.append(GeminiService(
            api_keys=keys[i : i + group_size],
            fail_on_exhaustion=True,
        ))
        if len(services) == NUM_WORKERS:
            break
    return services
```

### 2c. Extract `_process_single_video()` from the main loop

Pull the inner body of the `for video in videos` loop (current lines 324–488) into:

```python
def _process_single_video(
    self,
    video: dict,
    source_key: str,
    gemini_service: GeminiService,
) -> dict:
    """
    Runs for one video:
      1. Qdrant already-indexed check (skip if exists)
      2. chunk_text
      3. extract_full_video_intelligence (Gemini call via caller-supplied service)
      4. _write_intelligence_to_dynamodb
      5. _write_thumbnail_to_dynamodb

    Returns a result dict: { video_id, chunk_count, chunks, intelligence, thumbnail, error }
    NOTE: chunks are returned so the caller can embed them sequentially after all futures complete.
    Raises KeysExhaustedError if the worker's key pool is fully exhausted.
    """
```

**Important:** Do NOT call `embed_chunks` or `upsert_transcript_chunks` inside this method — `sentence-transformers` is not thread-safe. Return `chunks` in the result dict and embed them sequentially after the thread pool finishes.

### 2d. Replace the main video loop in `run_s3_transcript_analysis()`

```python
# --- Build pending list ---
pending = []  # list of (video_dict, source_key)
for source in source_objects:
    source_key = source.get("key", "")
    for video in source.get("videos", []):
        # Qdrant already-indexed check can stay here or move into _process_single_video
        pending.append((video, source_key))

if job_id:
    job_service.update_total(job_id, len(pending))

# --- Parallel Gemini phase ---
worker_services = self._make_worker_gemini_services()
retry_queue = []     # (video, source_key) — workers that ran out of keys
gemini_results = {}  # video_id → result dict

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
    futures = {
        pool.submit(
            self._process_single_video,
            video, source_key,
            worker_services[i % NUM_WORKERS],
        ): (video, source_key)
        for i, (video, source_key) in enumerate(pending)
    }

    for future in as_completed(futures):
        video, source_key = futures[future]
        try:
            result = future.result()
            gemini_results[video["videoId"]] = result
        except KeysExhaustedError:
            retry_queue.append((video, source_key))
            self.logger.warning(
                f"WORKER_EXHAUSTED videoId={video['videoId']} → retry queue"
            )
        except Exception as exc:
            gemini_results[video["videoId"]] = {"status": "failed", "error": str(exc)}
            self.logger.error(f"VIDEO_FAILED videoId={video['videoId']} error={exc}")

        if job_id:
            job_service.increment_progress(job_id)

# --- Retry pass: sequential, full 30-key pool ---
if retry_queue:
    self.logger.info(f"RETRY_PASS videos={len(retry_queue)}")
    for video, source_key in retry_queue:
        try:
            result = self._process_single_video(
                video, source_key, self.gemini_service  # full pool, fail_on_exhaustion=False
            )
            gemini_results[video["videoId"]] = result
        except Exception as exc:
            gemini_results[video["videoId"]] = {"status": "failed", "error": str(exc)}

# --- Sequential embed + Qdrant upsert ---
for video_id, result in gemini_results.items():
    if result.get("status") == "failed" or not result.get("chunks"):
        continue
    vectors = self.embedding_service.embed_chunks(result["chunks"])
    self.vector_service.upsert_transcript_chunks(
        transcript_key=result["transcript_key"],
        source_key=result["source_key"],
        transcript_index=video_id,
        chunks=result["chunks"],
        vectors=vectors,
        extra_metadata=None,
    )

# --- sync_missing ---
self._run_sync_missing()
```

### 2e. Add `_run_sync_missing()` method

```python
def _run_sync_missing(self):
    """
    After ingest: find videos in DynamoDB that are missing from Qdrant
    and re-index them. Catches any Qdrant write failures from the parallel phase.
    """
    from scripts.sync_missing import main as sync_main
    self.logger.info("SYNC_MISSING_START")
    try:
        sync_main()
        self.logger.info("SYNC_MISSING_COMPLETE")
    except Exception as exc:
        self.logger.error(f"SYNC_MISSING_FAILED error={exc}")
```

---

## Step 3 — `article_service.py`

See `docs/article_worker_parallelization.md` for the full step-by-step.

Summary of changes:
- Add `_make_worker_gemini_services()` (same logic as pipeline)
- Extract `_generate_single_article(cluster, week_slice, wk, gemini_service, force) → dict`
- Replace the `for c, week_slice, wk in deduped_jobs` loop with `ThreadPoolExecutor` fan-out + retry pass

---

## Step 4 — `app/services/job_service.py` (new file)

```python
"""
job_service.py - In-memory async job tracker

Stores pipeline job state so endpoints can return immediately and
clients can poll for status. State is lost on server restart (acceptable
for pipeline jobs — re-submit if needed).
"""

import threading
import uuid
from datetime import datetime, timezone

_store: dict = {}
_lock = threading.Lock()


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _store[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "total": 0,
            "errors": [],
            "failed_videos": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "result": None,
        }
    return job_id


def update_total(job_id: str, total: int) -> None:
    with _lock:
        job = _store.get(job_id)
        if job:
            job["total"] = total


def increment_progress(job_id: str, failed_video: str = None) -> None:
    with _lock:
        job = _store.get(job_id)
        if not job:
            return
        job["progress"] += 1
        if failed_video:
            job["failed_videos"].append(failed_video)


def complete_job(job_id: str, result: dict) -> None:
    with _lock:
        job = _store.get(job_id)
        if not job:
            return
        job["status"] = "complete"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["result"] = result


def fail_job(job_id: str, error: str) -> None:
    with _lock:
        job = _store.get(job_id)
        if not job:
            return
        job["status"] = "failed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["errors"].append(error)


def get_job(job_id: str) -> dict | None:
    return _store.get(job_id)
```

---

## Step 5 — `app/api/endpoints/pipeline.py`

### Add imports

```python
import threading
from app.services import job_service
```

### Change `POST /api/pipeline/ingest`

```python
@router.post("/ingest", response_model=JobSubmitResponse)
def run_ingest(request: PipelineRunRequest):
    """
    Starts ingest in the background. Returns job_id immediately.
    Poll GET /api/pipeline/jobs/{job_id} for status.
    dry_run still runs synchronously (no Gemini calls, fast).
    """
    if request.dry_run:
        result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix, limit=request.limit, dry_run=True
        )
        return {"job_id": None, "status": "complete", "result": result}

    job_id = job_service.create_job()
    threading.Thread(
        target=_run_ingest_background,
        args=(request, job_id),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running"}


def _run_ingest_background(request: PipelineRunRequest, job_id: str) -> None:
    try:
        result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix,
            limit=request.limit,
            job_id=job_id,
        )
        job_service.complete_job(job_id, result)
    except Exception as exc:
        job_service.fail_job(job_id, str(exc))
```

### Change `POST /api/pipeline/run` (same pattern)

```python
@router.post("/run", response_model=JobSubmitResponse)
def run_full_pipeline(request: PipelineRunRequest):
    if request.dry_run:
        # ... existing dry_run logic, return synchronously
        pass

    job_id = job_service.create_job()
    threading.Thread(
        target=_run_full_pipeline_background,
        args=(request, job_id),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running"}


def _run_full_pipeline_background(request: PipelineRunRequest, job_id: str) -> None:
    try:
        # Steps 1-5 in sequence, each updating job progress
        ingest_result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix, limit=request.limit, job_id=job_id
        )
        ClusteringService().run_clustering()
        ClaimAnalysisService().run_claim_analysis()
        article_result = ArticleService().run_article_generation()
        job_service.complete_job(job_id, {
            "ingestion": ingest_result,
            "articles": article_result,
        })
    except Exception as exc:
        job_service.fail_job(job_id, str(exc))
```

### Add `GET /api/pipeline/jobs/{job_id}` (new endpoint)

```python
@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Poll this after submitting a pipeline run to check progress."""
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
```

---

## Step 6 — `app/schemas/pipeline.py`

Add to the existing file:

```python
from typing import Optional

class JobSubmitResponse(BaseModel):
    job_id: Optional[str]
    status: str   # "running" | "complete"

class JobStatusResponse(BaseModel):
    job_id: str
    status: str           # "running" | "complete" | "failed"
    progress: int         # videos/articles processed so far
    total: int            # total to process
    errors: list[str]
    failed_videos: list[str]
    started_at: str
    completed_at: Optional[str]
    result: Optional[dict]
```

---

## Key Exhaustion Flow

```
Worker 3 holds keys [9, 10, 11]:

  video X → key_9 hits 429 → rotate to key_10
  video X → key_10 hits 429 → rotate to key_11
  video X → key_11 hits 429 → ALL EXHAUSTED
    → fail_on_exhaustion=True → raise KeysExhaustedError
    → thread is freed immediately (no 60s sleep)
    → video X added to retry_queue

After all 10 workers finish:
  retry_queue processed sequentially using full 30-key GeminiService
  (fail_on_exhaustion=False → existing behavior: rotate all 30, wait 60s if needed)
  If still fails → video_id stored in failed_videos in job result
```

---

## Thread Safety Notes

| Component | Thread-safe? | Notes |
|---|---|---|
| `GeminiService` per worker | Yes | Each worker has its own instance, no shared state |
| `GeminiService` retry pass | Yes | Single-threaded sequential |
| boto3 DynamoDB | Yes | Safe for concurrent reads/writes |
| `qdrant-client` | Yes | Thread-safe, but kept sequential anyway |
| `sentence-transformers` (nomic) | **No** | Must run sequentially — kept after all futures complete |
| `job_service` dict | Yes | Protected by `threading.Lock` |
| `retry_queue` list | Yes | Only written in `as_completed` loop (single thread) |

---

## What Does NOT Change

- `claim_analysis_service.py` — zero Gemini calls, no changes needed
- `clustering_service.py` — no Gemini calls in the hot path
- All existing API response schemas for `/cluster`, `/claims` endpoints
- `dry_run` behavior on all endpoints — still synchronous and fast
- GeminiService retry logic for 429/503 — preserved within each worker instance

---

## Verification Checklist

- [ ] `POST /api/pipeline/ingest` returns `{ job_id, status: "running" }` in < 1 second
- [ ] `GET /api/pipeline/jobs/{job_id}` shows `progress` incrementing over time
- [ ] Logs show `GEMINI_COMBINED_START key=#1`, `key=#4`, `key=#7` firing near-simultaneously (not one-after-another)
- [ ] `dry_run=true` still returns synchronously with no job_id
- [ ] Simulate key exhaustion on one worker → video appears in `failed_videos`, no 60s hang in logs
- [ ] After full run: DynamoDB video count matches Qdrant vector count (sync_missing confirms 0 gaps)
- [ ] Article generation logs show multiple cluster IDs completing in parallel
