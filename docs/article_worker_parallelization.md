# Article Worker Parallelization

## What This Does

`article_service.py` currently generates articles **sequentially** — one Gemini call per (cluster × week), blocking the full time. With 30 API keys available, we can run **10 parallel workers** each holding **3 keys**, reducing total article generation time by ~10x.

When a worker's 3 keys are all rate-limited, it **fails fast** (no 60s sleep) and puts that job in a retry queue. The retry queue is then handled sequentially using the full 30-key pool.

---

## Files You Will Touch

| File | What to do |
|---|---|
| `app/services/gemini_service.py` | Add `fail_on_exhaustion` flag (1 change, ~10 lines) |
| `app/services/article_service.py` | Extract method + add thread pool (main work) |

---

## Step 1 — `app/services/gemini_service.py`

### 1a. Add `KeysExhaustedError` at the top of the file (after imports)

```python
class KeysExhaustedError(Exception):
    """Raised when all keys in a worker's pool are rate-limited."""
    pass
```

### 1b. Add `fail_on_exhaustion` parameter to `__init__`

```python
def __init__(
    self,
    client=None,
    chunker=None,
    api_keys=None,
    fail_on_exhaustion: bool = False,   # ← add this
):
    ...
    self.fail_on_exhaustion = fail_on_exhaustion   # ← add this
```

### 1c. Update the `ALL_KEYS_EXHAUSTED` block in both `_gemini()` and `_gemini_vision()`

Find this block (appears in both methods):
```python
logger.warning(
    "ALL_KEYS_EXHAUSTED resetting to key 0 and waiting 60s"
)
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
logger.warning(
    "ALL_KEYS_EXHAUSTED resetting to key 0 and waiting 60s"
)
self.current_key_index = 0
self.client = genai.Client(api_key=self.api_keys[0])
time.sleep(60)
continue
```

> The `fail_on_exhaustion=False` default means nothing breaks for existing callers.

---

## Step 2 — `app/services/article_service.py`

### 2a. Add imports at the top

```python
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.gemini_service import KeysExhaustedError
```

### 2b. Add `NUM_WORKERS` constant (module level, below imports)

```python
NUM_WORKERS = 10
```

### 2c. Add `_make_worker_gemini_services()` method to `ArticleService`

```python
def _make_worker_gemini_services(self) -> list[GeminiService]:
    """
    Split all API keys into NUM_WORKERS groups of 3 (or fewer if < 30 keys).
    Each worker gets its own GeminiService with fail_on_exhaustion=True so it
    fails fast rather than blocking its thread for 60s on quota exhaustion.
    """
    keys = self._gemini.api_keys
    group_size = max(1, len(keys) // NUM_WORKERS)
    services = []
    for i in range(0, len(keys), group_size):
        services.append(
            GeminiService(
                api_keys=keys[i : i + group_size],
                fail_on_exhaustion=True,
            )
        )
        if len(services) == NUM_WORKERS:
            break
    return services
```

### 2d. Extract `_generate_single_article()` method

Pull the body of the `for c, week_slice, wk in deduped_jobs` loop out into its own method.

```python
def _generate_single_article(
    self,
    cluster: dict,
    week_slice: dict,
    wk: str,
    gemini_service: GeminiService,
    force: bool,
) -> dict:
    """
    Generate and persist one article for a (cluster, week) combination.

    Returns a result dict:
      { status: "generated"|"skipped"|"failed", article_id, week, headline, error }

    Raises KeysExhaustedError if the worker's key pool is exhausted.
    Raises any other exception on Gemini or DynamoDB failure (caller catches).
    """
    cid = int(cluster.get("cluster_id", -1))
    wk_num = _week_number(wk)
    label = cluster.get("cluster_label", f"Cluster {cid}")

    # Skip if article already exists (unless force=True)
    if not force and self._dynamo.article_exists(cid, wk_num):
        logger.info(f"ARTICLE_SKIP cluster={cid} week={wk} (already exists)")
        return {"status": "skipped", "week": wk}

    if force:
        self._dynamo.delete_articles(cid, wk_num)

    video_titles = self._dynamo.get_video_titles_for_cluster(cid)
    prompt = self._build_prompt(cluster, wk, week_slice, video_titles)

    # gemini_service is the caller-supplied worker instance
    raw = gemini_service._gemini(prompt)
    headline, overview, body = self._parse_response(raw)

    now = datetime.now(timezone.utc).isoformat()
    article_id = f"article-{uuid.uuid4().hex[:8]}"

    self._dynamo.save_article({
        "article_id": article_id,
        "cluster_id": _dec(cid),
        "cluster_label": label,
        "week": wk,
        "week_number": _dec(wk_num),
        "title": headline,
        "overview": overview,
        "body": body,
        "created_at": now,
        "updated_at": now,
    })

    logger.info(f"ARTICLE_GENERATED cluster={cid} week={wk} id={article_id}")
    return {
        "status": "generated",
        "article_id": article_id,
        "week": wk,
        "headline": headline,
    }
```

### 2e. Replace the main `for` loop in `run_article_generation()` with the parallel version

Find and replace the loop starting at `for c, week_slice, wk in deduped_jobs:` (and everything inside it) with:

```python
worker_services = self._make_worker_gemini_services()
retry_jobs: list[tuple[dict, dict, str]] = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
    futures = {
        pool.submit(
            self._generate_single_article,
            c, week_slice, wk,
            worker_services[i % NUM_WORKERS],
            force,
        ): (c, week_slice, wk)
        for i, (c, week_slice, wk) in enumerate(deduped_jobs)
    }

    for future in as_completed(futures):
        c, week_slice, wk = futures[future]
        cid = int(c.get("cluster_id", -1))
        wk_num = _week_number(wk)
        label = c.get("cluster_label", f"Cluster {cid}")
        key = f"{cid}:{wk}"
        weeks_seen.add(wk)

        try:
            result = future.result()
            per_cluster[key] = result
            if result["status"] == "generated":
                generated += 1
            else:
                skipped += 1
        except KeysExhaustedError:
            # Worker ran out of keys — defer to retry pass
            retry_jobs.append((c, week_slice, wk))
            logger.warning(
                f"ARTICLE_WORKER_EXHAUSTED cluster={cid} week={wk} → retry queue"
            )
        except Exception as exc:
            failed += 1
            per_cluster[key] = {"status": "failed", "week": wk, "error": str(exc)}
            logger.error(f"ARTICLE_FAILED cluster={cid} week={wk} error={exc}")

# Retry pass — sequential, full 30-key GeminiService (fail_on_exhaustion=False)
if retry_jobs:
    logger.info(f"ARTICLE_RETRY_PASS jobs={len(retry_jobs)}")
    for c, week_slice, wk in retry_jobs:
        cid = int(c.get("cluster_id", -1))
        key = f"{cid}:{wk}"
        weeks_seen.add(wk)
        try:
            result = self._generate_single_article(
                c, week_slice, wk, self._gemini, force
            )
            per_cluster[key] = result
            if result["status"] == "generated":
                generated += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            per_cluster[key] = {"status": "failed", "week": wk, "error": str(exc)}
            logger.error(f"ARTICLE_RETRY_FAILED cluster={cid} week={wk} error={exc}")
```

---

## How It Works

```
10 workers run in parallel:

worker_0 (keys 0,1,2)  → cluster 1, week1 → Gemini → DynamoDB
worker_1 (keys 3,4,5)  → cluster 2, week1 → Gemini → DynamoDB
worker_2 (keys 6,7,8)  → cluster 3, week1 → Gemini → DynamoDB
...

If worker_0's keys 0,1,2 all hit 429:
  → raises KeysExhaustedError immediately (no 60s sleep)
  → cluster 1 goes to retry_jobs

After all workers done:
  retry_jobs processed one by one using the main GeminiService (all 30 keys)
  → existing retry logic (rotates through all 30, waits 60s if needed)
```

---

## What Does NOT Change

- `claim_analysis_service.py` — makes zero Gemini calls, no changes needed
- `clustering_service.py` — no Gemini calls in the hot path
- Existing `run_article_generation()` signature — same params, same return shape
- `dry_run` path — still returns immediately before touching the thread pool
- All existing API endpoints — no contract changes

---

## Testing

1. Run `POST /api/pipeline/articles` (or trigger via CLI `--articles-only`)
2. Check logs — you should see `ARTICLE_GENERATED` entries firing near-simultaneously from multiple cluster IDs, not one-after-another
3. Confirm article count in DynamoDB matches expected number
4. Test with `dry_run=True` first — should still work fine (skips the thread pool entirely)
