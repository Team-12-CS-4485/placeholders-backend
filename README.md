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
│   ├── article_service.py        Gemini article generation → DynamoDB articles table
│   ├── claim_analysis_service.py Claim embedding, grouping, classification → DynamoDB
│   ├── chunking_service.py       Semantic text chunking (nomic model)
│   ├── clustering_service.py     UMAP + HDBSCAN clustering → DynamoDB
│   ├── dynamo_service.py         Primary read layer for all API endpoints
│   ├── embedding_service.py      Gemini video intelligence + thumbnail analysis
│   ├── pipeline_service.py       Full pipeline orchestration (ingest → cluster → claims)
│   ├── storage_service.py        S3 reads + DynamoDB video detail joins
│   ├── trend_service.py          Derived metrics (heat score, sentiment labels, etc.)
│   └── vector_service.py         Qdrant collection management + similarity search
└── workers/
    └── pipeline_worker.py        Background pipeline runner
```

---

## API endpoints

### Videos
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/videos` | Paginated video list. Optional `?week=`, `?cursor=` |
| GET | `/api/videos/by-id?video_id=` | Single video with transcript + comments |

### Narratives (editorial view)
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/narratives` | Lean narrative list. Optional `?week=`, `?sort_by=` |
| GET | `/api/narratives/{id}` | Full narrative detail — headline, summary, claims, channels, week presence |
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
| GET | `/api/search?q=` | Semantic search — VideoItem-shaped results with score + excerpt |
| GET | `/api/stats` | Aggregate: total_videos, total_clusters, total_weeks, breaking_count |

### Articles
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/articles/generate` | Generate articles for clusters × weeks via Gemini |
| GET | `/api/articles` | List articles. Optional `?cluster_id=`, `?week=`, `?limit=` |
| GET | `/api/articles/{id}` | Full article detail including body text |

### Pipeline
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/pipeline/run` | Full pipeline: ingest → cluster → claim analysis |
| POST | `/api/pipeline/ingest` | Ingest only — S3 → Gemini → DynamoDB → Qdrant |
| POST | `/api/pipeline/cluster` | Cluster only — UMAP/HDBSCAN → DynamoDB |
| POST | `/api/pipeline/claims` | Claim analysis only → DynamoDB |
| POST | `/api/pipeline/search` | Raw Qdrant vector search (internal use) |
| GET | `/health` | Health check |

All pipeline endpoints accept `dry_run: bool` to preview without writing.

---

## Pipeline flow

1. **Ingest** — load JSON from S3 (`youtube-data/` prefix), extract video + transcript data
2. **Intelligence** — Gemini call per video → `topics`, `category`, `sentiment`, `key_claims`, `is_breaking`
3. **Thumbnail** — Gemini Vision per video → `thumbnail_tone`, `thumbnail_clickbait_score`, etc.
4. **Embed** — local nomic embeddings (`nomic-ai/nomic-embed-text-v1.5`) → store chunks in Qdrant
5. **Cluster** — UMAP (768d → 15d) + HDBSCAN → Gemini labels each cluster → write to DynamoDB `narrative-clusters`
6. **Claim analysis** — embed claims, group by similarity, classify consensus/debated/unique → write `classified_claims` + `creator_risk` to DynamoDB
7. **Articles** (optional) — Gemini generates long-form article per cluster × week → write to DynamoDB `articles`

---

## DynamoDB tables

**`youtube-videos`** — PK: `PartitionKey` (channel), SK: `SortKey` (videoId)

Key attributes: `title`, `publishedAt`, `viewCount`, `likeCount`, `commentCount`, `transcript`, `topics[]`, `sentiment`, `category`, `key_claims[]`, `is_breaking`, `cluster_id`, `cluster_label`, `thumbnail_tone`, `thumbnail_clickbait_score`, `week`, `source_key`

**`narrative-clusters`** — PK: `cluster_id`

Key attributes: `cluster_label`, `video_count`, `channel_count`, `channels[]`, `top_topics[]`, `dominant_sentiment`, `sentiment_breakdown`, `week_data[]`, `top_claims[]`, `narrative_headline`, `narrative_summary`, `classified_claims`, `creator_risk[]`, `avg_clickbait_rating`, `thumbnail_tone_breakdown`

**`articles`** — PK: `article_id`

Key attributes: `cluster_id`, `cluster_label`, `week_number`, `title`, `overview`, `body`, `created_at`

> **Note:** All read APIs pull exclusively from DynamoDB. Qdrant is only used during the pipeline (vector storage, clustering) and `POST /api/pipeline/search`.

---

## Environment variables

```bash
# AWS
AWS_REGION=us-east-2
S3_BUCKET=your-bucket-name
S3_PREFIX=youtube-data/
S3_OBJECT_LIMIT=100
DYNAMODB_TABLE=youtube-videos

# Gemini (supports up to 20 keys for quota rotation)
GENAI_API_KEY=your-key
GENAI_API_KEY_2=your-key-2
# ...
GEMINI_MODEL_ID=gemini-2.0-flash

# Embeddings (local)
EMBEDDING_MODEL_ID=nomic-ai/nomic-embed-text-v1.5

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=transcript_chunks
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
cp config/.env.example .env
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

5. Run the full pipeline (dry run first):
```bash
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

6. Semantic search:
```bash
curl "http://localhost:8000/api/search?q=climate+policy&limit=5"
```

---

## CI/CD

- **Push to any branch** → flake8 lint + black auto-format (commits back if changed) + pytest
- **Merge to `main`** → all of the above, then auto-deploy to Render via deploy hook
- Render native auto-deploy should be **disabled** — deployment is triggered exclusively by GitHub Actions after CI passes
