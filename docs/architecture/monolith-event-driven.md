# Architecture: Current State & Design Notes

## Current State: Synchronous Monolith

All pipeline stages run synchronously within a single API request. The HTTP call blocks until the stage completes.

```
POST /api/pipeline/run
  └── PipelineService          (S3 → Gemini → DynamoDB → Qdrant)
  └── ClusteringService        (Qdrant + DynamoDB → UMAP/HDBSCAN → DynamoDB)
  └── ClaimAnalysisService     (DynamoDB → embed → classify → DynamoDB)
  └── ArticleService           (DynamoDB → Gemini → DynamoDB)
HTTP Response
```

Each stage can also be triggered individually:
- `POST /api/pipeline/ingest`
- `POST /api/pipeline/cluster`
- `POST /api/pipeline/claims`
- `POST /api/articles/generate`

**Tradeoffs:**
- Simple to run and debug — one process, one log stream
- HTTP timeouts are a real risk for large ingest runs (100+ videos × Gemini calls per video)
- Gemini API rate limits are managed by rotating across up to 20 API keys (`GENAI_API_KEY` through `GENAI_API_KEY_20`)
- Videos already in Qdrant are skipped automatically to preserve quota on re-runs

---

## Read vs Write Separation

All read APIs (GET endpoints) pull exclusively from **DynamoDB**. Qdrant is only touched by:
- `POST /api/pipeline/ingest` — vector upsert
- `POST /api/pipeline/cluster` — vector scroll for clustering
- `POST /api/pipeline/search` — raw vector search (internal/debug)
- `GET /api/search` — embed query → Qdrant → DynamoDB join

This means the frontend can function even if Qdrant is down, as long as the pipeline has been run previously.

---

## Data Flow (Read Path)

```
GET /api/trends
  └── DynamoService.get_all_clusters()
        └── narrative-clusters table (full scan, small table)
  └── TrendService._build_cluster_from_dynamo()
        └── computes heat_score, trend_type, metric_badge, sentiment_labels
  └── Response

GET /api/narratives?week=week1
  └── TrendService.get_narratives(week="week1")
        └── same cluster scan, filtered + editorial fields only

GET /api/videos
  └── DynamoService.scan_videos()
        └── youtube-videos table (paginated scan)

GET /api/search?q=...
  └── EmbeddingService.embed_query()     (local nomic model)
  └── VectorService.search_similar_chunks()  (Qdrant)
  └── DynamoService.get_video_by_id() per hit (DynamoDB)
```

---

## DynamoDB Tables

| Table | PK | SK | Written by |
|-------|----|----|------------|
| `youtube-videos` | `PartitionKey` (channel) | `SortKey` (videoId) | YouTube ingestion scripts + pipeline ingest stage |
| `narrative-clusters` | `cluster_id` | — | Clustering stage + claim analysis stage |
| `articles` | `article_id` | — | Article generation stage |

The `narrative-clusters` table is the central aggregation table. All `/api/trends`, `/api/narratives`, `/api/weeks`, and `/api/stats` responses are derived from it.

---

## Potential Future Direction: Event-Driven

The intended long-term direction would decouple ingestion from processing:

```
YouTube Ingestion (data_collection/)
  └── writes to S3 + DynamoDB

S3 Event / SQS
  └── Pipeline Worker
        ├── Gemini intelligence + thumbnail
        ├── nomic embedding
        └── Qdrant upsert
```

**Benefits:**
- Ingestion and processing scale independently
- Worker can retry failed videos without blocking the API
- API stays responsive — pipeline status would be polled rather than waited on

The pipeline worker (`app/workers/pipeline_worker.py`) exists but is not yet wired to any event source. Moving toward event-driven would require:
1. S3 event notifications or SQS queue triggered on new object uploads
2. Worker consuming from the queue instead of scanning S3 on demand
3. A status tracking mechanism so the API can report per-video processing state
