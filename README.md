# placeholders-backend

FastAPI backend for **Newsify — The Investigative Archive**. Ingests YouTube video data from S3, runs AI analysis via Gemini, clusters videos into narrative trends, and serves a REST API consumed by the frontend.

---

## Project structure

```text
app/
├── api/endpoints/
│   ├── articles.py       GET/POST /api/articles/*
│   ├── narratives.py     GET /api/narratives/*
│   ├── pipeline.py       POST /api/pipeline/*
│   ├── search.py         GET /api/search
│   ├── stats.py          GET /api/stats
│   ├── trends.py         GET /api/trends/*, GET /api/weeks
│   └── videos.py         GET /api/videos/*
├── core/
│   ├── config.py         Environment variable loading (settings singleton)
│   └── logging.py        Shared logger
├── schemas/
│   ├── article.py        Pydantic models for /api/articles
│   ├── narrative.py      Pydantic models for /api/narratives
│   ├── pipeline.py       Pydantic models for /api/pipeline
│   ├── search.py         Pydantic models for /api/search
│   ├── trend.py          Pydantic models for /api/trends + /api/weeks
│   └── video.py          Pydantic models for /api/videos
├── services/
│   ├── article_service.py            Gemini article generation → DynamoDB articles table
│   ├── claim_analysis_service.py     Claim embedding, grouping, classification → DynamoDB
│   ├── chunk_labeling_service.py     Semantic text chunking (nomic model)
│   ├── cluster_labeling_service.py   Gemini cluster labeling + stable ID matching
│   ├── clustering_service.py         UMAP + HDBSCAN clustering → DynamoDB
│   ├── dynamo_service.py             Primary read layer for all API endpoints
│   ├── embedding_service.py          Gemini video intelligence + thumbnail analysis
│   ├── gemini_service.py             Gemini client with API key rotation
│   ├── job_service.py                In-memory background job tracker
│   ├── pipeline_service.py           Full ingest pipeline orchestration
│   ├── storage_service.py            S3 reads + DynamoDB video detail joins
│   ├── trend_service.py              Derived metrics (heat score, sentiment labels, etc.)
│   └── vector_service.py             Qdrant collection management + similarity search
└── workers/
    └── pipeline_worker.py            CLI pipeline runner (alternative to API endpoints)
```

---

## API endpoints

### Videos
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/videos` | Paginated video list. Optional `?week=`, `?cursor=` |
| GET | `/api/videos/by-id?video_id=` | Single video with transcript + comments |
| GET | `/api/videos/by-channel?channel=` | Paginated videos filtered by channel name |

### Narratives (editorial view)
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/narratives` | Lean narrative list. Optional `?week=`, `?sort_by=` |
| GET | `/api/narratives/{id}` | Full narrative detail — headline, summary, claims, channels |
| GET | `/api/narratives/{id}/claims` | Classified claims (consensus / debated / unique) |
| GET | `/api/narratives/{id}/videos` | Paginated videos in this cluster |

### Trends (metrics view)
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/trends` | Cluster list with metrics. Optional `?week=`, `?sort_by=` |
| GET | `/api/trends/{id}` | Full cluster detail — heat score, engagement, sentiment |
| GET | `/api/trends/{id}/sentiment` | Sentiment breakdown + per-week sentiment |
| GET | `/api/trends/{id}/claims` | Classified claims (consensus / debated / unique) |

### Weeks, Search, Stats
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/weeks` | All weeks with aggregate stats (drives Archives view) |
| GET | `/api/weeks/{week}` | All clusters active in a given week with per-week headlines |
| GET | `/api/search?q=` | Semantic search — VideoItem-shaped results with score + excerpt |
| GET | `/api/stats` | Aggregate: total_videos, total_clusters, total_weeks, breaking_count |

### Articles
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/articles/generate` | Generate articles for clusters × weeks via Gemini. Accepts `week`, `cluster_id`, `force`, `dry_run` |
| GET | `/api/articles` | List articles. Optional `?cluster_id=`, `?week=`, `?limit=` |
| GET | `/api/articles/{id}` | Full article detail including body text |

### Pipeline
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/pipeline/run` | **Async** — full pipeline: ingest → cluster → claims → articles. Returns `{job_id}` immediately |
| POST | `/api/pipeline/ingest` | **Async** — ingest only: S3 → Gemini → DynamoDB → Qdrant |
| POST | `/api/pipeline/cluster` | Cluster only — UMAP/HDBSCAN → stable match → DynamoDB |
| POST | `/api/pipeline/claims` | Claim analysis only → DynamoDB |
| POST | `/api/pipeline/articles` | Article generation only → DynamoDB. Accepts `week`, `dry_run` |
| GET | `/api/pipeline/jobs/{job_id}` | Poll async job status — returns `{job_id, status, result, error}` |
| POST | `/api/pipeline/search` | Raw Qdrant vector search (internal use) |
| GET | `/health` | Health check |

All pipeline write endpoints accept `dry_run: bool` to preview without writing.

---

## Pipeline flow

```
POST /api/pipeline/run  (async background job)
│
├── 1. Ingest        Load JSON from S3 (youtube-data/ prefix)
├── 2. Intelligence  Gemini per video → topics, sentiment, key_claims, is_breaking
├── 3. Thumbnail     Gemini Vision per video → thumbnail_tone, clickbait_score
├── 4. Embed         Local nomic embeddings → Qdrant (transcript chunks)
│
├── 5. Cluster       UMAP (768d→15d) + HDBSCAN → clusters
│   ├── Label        Gemini generates cluster_label, narrative_headline, narrative_summary
│   ├── Match        Cosine similarity matching against existing stable cluster IDs
│   │                (threshold 0.65 full-text, 0.70 topics-only fallback)
│   ├── Write        cluster_id/label → youtube-videos, summaries → narrative-clusters
│   ├── Snapshots    Per-week snapshots → cluster-weeks (current + historical backfill)
│   ├── Lifecycle    active → declining → inactive → purge (with ghost cleanup)
│   └── Renumber     Sequential IDs (0,1,2...) across all 4 tables
│
├── 6. Claims        Embed claims, group by similarity, classify consensus/debated/unique
│                    Writes classified_claims + top_claims + creator_risk → narrative-clusters
│
└── 7. Articles      Orphan cleanup → Gemini article per cluster×week → DynamoDB articles
```

> **Read APIs pull exclusively from DynamoDB.** Qdrant is only used during pipeline steps 4–5 and `POST /api/pipeline/search`.

---

## DynamoDB tables

**`youtube-videos`** — PK: `PartitionKey` (channel), SK: `SortKey` (videoId)
- GSI: `cluster-index` on `cluster_id` — used for efficient per-cluster video lookups
- Key fields: `title`, `publishedAt`, `viewCount`, `likeCount`, `commentCount`, `transcript`, `topics[]`, `sentiment`, `category`, `key_claims[]`, `is_breaking`, `cluster_id`, `cluster_label`, `cluster_confidence`, `thumbnail_tone`, `thumbnail_clickbait_score`, `week`, `source_key`

**`narrative-clusters`** — PK: `cluster_id`
- Key fields: `cluster_label`, `video_count`, `channel_count`, `channels[]`, `top_topics[]`, `dominant_sentiment`, `sentiment_breakdown`, `week_data[]`, `top_claims[]`, `narrative_headline`, `narrative_summary`, `classified_claims {consensus[], debated[], unique[]}`, `creator_risk[]`, `avg_clickbait_rating`, `thumbnail_tone_breakdown`, `status` (new/active/declining/inactive)

**`cluster-weeks`** — Composite PK: `cluster_id` + `week`
- Per-week snapshots: `narrative_headline`, `narrative_summary`, `week_overview`, `top_claims[]`, `top_topics[]`, `video_count`, `view_count`, `channel_count`, `breaking_count`, `dominant_sentiment`
- Written by pipeline each run; used by `GET /api/weeks/{week}` and article title substitution

**`articles`** — PK: `article_id`
- Key fields: `cluster_id`, `cluster_label`, `week_number`, `week`, `title`, `overview`, `body`, `created_at`, `updated_at`

---

## Environment variables

```bash
# AWS
AWS_REGION=us-east-2
S3_BUCKET=your-bucket-name
S3_PREFIX=youtube-data/
S3_OBJECT_LIMIT=100
DYNAMODB_TABLE=youtube-videos          # youtube-videos table name

# Gemini (supports up to 40 keys for quota rotation via GENAI_API_KEY through GENAI_API_KEY_40)
GENAI_API_KEY=your-key
GENAI_API_KEY_2=your-key-2
# ...up to GENAI_API_KEY_40
GEMINI_MODEL_ID=gemini-3-flash-preview

# Embeddings (runs locally, no API key needed)
EMBEDDING_MODEL_ID=nomic-ai/nomic-embed-text-v1.5

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=transcript_chunks

# Optional tuning
CHUNK_SIZE_CHARS=6000
CHUNK_OVERLAP_CHARS=400
SEMANTIC_SIMILARITY_THRESHOLD=0.48
```

Copy `.env.example` as a starting point:
```bash
cp .env.example .env
```

---

## Running locally

1. Install dependencies:
```bash
pip install -r requirements.txt
pre-commit install   # enables black on every git commit
```

2. Copy and fill in environment variables:
```bash
cp .env.example .env
```

3. Start Qdrant:
```bash
docker compose -f docker-compose.qdrant.yml up -d
```

4. Start the API server:
```bash
uvicorn app.main:app --reload
# Docs at http://localhost:8000/docs
```

5. Run the full pipeline (dry run first to verify config):
```bash
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

6. Poll job status:
```bash
curl http://localhost:8000/api/pipeline/jobs/<job_id>
```

7. Semantic search:
```bash
curl "http://localhost:8000/api/search?q=iran+conflict&limit=5"
```

### CLI alternative (pipeline_worker.py)

```bash
# Full pipeline
python -m app.workers.pipeline_worker

# Clustering + claims + articles only (skip ingest)
python -m app.workers.pipeline_worker --clustering-only

# Articles only
python -m app.workers.pipeline_worker --articles-only --week week8

# Force-regenerate articles for one cluster
python -m app.workers.pipeline_worker --articles-only --cluster-id 5 --force-articles
```

---

## Scripts

Utility and diagnostic scripts in `scripts/`:

| Script | Purpose |
|--------|---------|
| `sync_missing.py` | Sync videos in DynamoDB but missing from Qdrant |
| `run_clustering.py` | Run clustering step standalone |
| `run_claim_analysis.py` | Run claim analysis step standalone |
| `articles.py` | Generate articles standalone |
| `backfill_cluster_weeks.py` | Regenerate per-week headlines via Gemini for cluster-weeks table |
| `audit_pipeline.py` | Post-pipeline data quality audit (calls Claude for review) |
| `audit_empty_claims.py` | Diagnose clusters with missing claims |
| `dump_all_tables.py` | Full snapshot of all 4 tables with cross-table consistency checks |
| `check_missing_intel.py` | Find videos missing Gemini intelligence fields |
| `cleanup_orphaned_articles.py` | Remove articles for non-existent clusters |
| `delete_clusters.py` | Admin utility to delete specific clusters |

Historical one-off migration scripts are in `scripts/archive/`.

---

## Tests

```bash
pytest
# or with coverage
pytest --cov=app
```

Tests cover API read endpoints and the DynamoDB/trend service read layers. Pipeline write paths (ingest, clustering, claims, articles) are validated via `dry_run=true` on the live endpoints rather than unit tests.

---

## CI/CD

- **Push to any branch** → flake8 lint + black auto-format (commits back if changed) + pytest
- **Merge to `main`** → all of the above, then Docker build + deploy to **GCP Cloud Run**
- Deployment is triggered exclusively by GitHub Actions after CI passes — do not enable auto-deploy directly on Cloud Run
