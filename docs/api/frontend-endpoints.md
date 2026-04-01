# Frontend API Endpoints

Base URL: `http://localhost:8000` (local) / proxied via nginx in production.

All read endpoints pull exclusively from DynamoDB. Qdrant is only used by the pipeline and `POST /api/pipeline/search`.

---

## Health

### `GET /health`

```json
{ "status": "ok" }
```

---

## Videos

### `GET /api/videos`

Paginated list of videos. Does not include transcripts or comments.

**Query Parameters**

| Parameter | Type   | Default | Description |
|-----------|--------|---------|-------------|
| `limit`   | int    | 20      | Max results (1–100) |
| `cursor`  | string | —       | Pagination token from previous response |
| `week`    | string | —       | Filter by week, e.g. `week1` or `1` |

**Response**
```json
{
  "items": [
    {
      "video_id": "abc123",
      "channel": "CNBC",
      "title": "Market Update",
      "published_at": "2026-03-10T12:00:00+00:00",
      "view_count": 142000,
      "like_count": 3200,
      "comment_count": 410,
      "week": "week1",
      "topics": ["inflation", "federal reserve"],
      "category": "Economy",
      "sentiment": "negative",
      "key_claims": ["Fed expected to hold rates"],
      "is_breaking": false,
      "cluster_id": 4,
      "cluster_label": "Inflation",
      "thumbnail_tone": "urgency",
      "thumbnail_clickbait_score": 6,
      "thumbnail_insight": "Bold text overlay with alarming color scheme",
      "thumbnail_brand_consistent": true
    }
  ],
  "total_returned": 20,
  "next_cursor": "eyJQYXJ0aXRpb25LZXkiOiAiQ05CQyJ9"
}
```

Pass `next_cursor` as `cursor` on the next request. When `next_cursor` is `null`, you've reached the last page.

> **Note:** Filtered scans (`?week=` or `?cluster_id=`) use Python-side pagination. The cursor format differs from unfiltered scans — do not mix them.

---

### `GET /api/videos/by-id`

Single video detail including transcript and top comments.

**Query Parameters**

| Parameter  | Type   | Required | Description |
|------------|--------|----------|-------------|
| `video_id` | string | Yes      | YouTube video ID |

**Response** — all fields from `VideoItem` plus:
```json
{
  "description": "Full video description...",
  "transcript": "Full transcript text...",
  "top_comments": [
    { "author": "user123", "text": "Great video!", "likes": 42 }
  ]
}
```

---

## Narratives

### `GET /api/narratives`

Lean editorial narrative list — no metrics.

**Query Parameters**

| Parameter | Type   | Default       | Description |
|-----------|--------|---------------|-------------|
| `week`    | string | —             | Filter to a specific week |
| `sort_by` | string | `video_count` | `video_count` or `label` |

**Response**
```json
{
  "narratives": [
    {
      "cluster_id": 8,
      "label": "Jeffrey Epstein Investigation",
      "category": "Politics",
      "narrative_headline": "Epstein Files Reveal New Connections to Political Figures",
      "narrative_summary": "Newly released documents implicate...",
      "top_topics": ["epstein", "government", "scandal"],
      "video_count": 16,
      "dominant_sentiment": "negative"
    }
  ],
  "total": 12
}
```

---

### `GET /api/narratives/{cluster_id}`

Full narrative detail — story fields, channels, week presence, clickbait/tone data.

**Response**
```json
{
  "cluster_id": 8,
  "label": "Jeffrey Epstein Investigation",
  "category": "Politics",
  "narrative_headline": "...",
  "narrative_summary": "...",
  "top_topics": ["epstein", "government"],
  "top_claims": ["Document reveals..."],
  "video_count": 16,
  "channel_count": 7,
  "breaking_count": 3,
  "dominant_sentiment": "negative",
  "channels": ["CNBC", "FoxNews", "BBCNews"],
  "week_data": [
    {
      "week": "week1",
      "video_count": 6,
      "channel_count": 4,
      "view_count": 310000,
      "breaking_count": 1,
      "sentiment_breakdown": { "negative": 5, "neutral": 1 }
    }
  ],
  "creator_risk": [
    { "name": "@channel", "riskScore": 0.82, "riskLevel": "high", "claimCount": 3 }
  ],
  "avg_clickbait_rating": 6.2,
  "thumbnail_tone_breakdown": { "urgency": 8, "neutral": 4 }
}
```

---

### `GET /api/narratives/{cluster_id}/claims`

Classified claims for this narrative.

**Response**
```json
{
  "cluster_id": 8,
  "claims": {
    "consensus": [
      {
        "claim": "Documents released by court order",
        "channel": "CNBC",
        "sources": ["CNBC", "BBCNews", "FoxNews"],
        "source_count": 3,
        "video_ids": ["abc", "def"],
        "transcript_excerpt": "...the documents released...",
        "risk_score": 0.08
      }
    ],
    "debated": [
      {
        "claim": "Epstein had government connections",
        "channel": "FoxNews",
        "perspectives": [
          { "channel": "FoxNews", "sentiment": "negative", "video_id": "abc", "video_title": "...", "transcript_excerpt": "..." }
        ],
        "source_count": 2,
        "framing_divergence": 0.61,
        "risk_score": 0.58
      }
    ],
    "unique": [
      {
        "claim": "CIA involved in cover-up",
        "channel": "@conspiracy_channel",
        "video_id": "xyz",
        "video_title": "...",
        "transcript_excerpt": "...",
        "risk_score": 0.85
      }
    ]
  }
}
```

---

### `GET /api/narratives/{cluster_id}/videos`

Paginated videos belonging to this cluster.

**Query Parameters:** `limit` (default 20, max 100), `cursor`

**Response**
```json
{
  "cluster_id": 8,
  "items": [ ...VideoItem ],
  "total_returned": 16,
  "next_cursor": null
}
```

---

## Trends

### `GET /api/trends`

Cluster list with computed metrics.

**Query Parameters**

| Parameter | Type   | Default      | Description |
|-----------|--------|--------------|-------------|
| `sort_by` | string | `heat_score` | `heat_score`, `video_count`, `view_count_total`, `channel_count`, `engagement_index` |
| `week`    | string | —            | Scope metrics to a specific week |

**Response**
```json
{
  "trends": [
    {
      "cluster_id": 9,
      "label": "Middle East Conflict",
      "category": "World",
      "trend_type": "dominant",
      "metric_badge": "High Impact",
      "heat_score": 38.6,
      "video_count": 17,
      "channel_count": 7,
      "view_count_total": 1660000,
      "breaking_count": 5,
      "sentiment_label": "Negative",
      "recent_sentiment_label": "Sentiment Shift — Negative",
      "dominant_sentiment": "negative",
      "dominant_public_sentiment": "negative",
      "sentiment_divergence": false,
      "top_topics": ["iran", "israel", "missiles"]
    }
  ],
  "total": 12
}
```

---

### `GET /api/trends/{cluster_id}`

Full trend detail.

**Response** — all list fields plus:
```json
{
  "total_likes": 48200,
  "total_comments": 12400,
  "engagement_index": 36.3,
  "sentiment_breakdown": { "negative": 14, "neutral": 2, "positive": 1 },
  "public_sentiment_breakdown": { "negative": 10, "neutral": 5 },
  "avg_public_sentiment_score": -0.42,
  "channels": ["CNBC", "FoxNews", "BBCNews"],
  "week_data": [ ...WeekData ],
  "top_claims": ["Iran launched drones toward..."],
  "top_topics": ["iran", "israel"],
  "creator_risk": [ ... ],
  "avg_clickbait_rating": 7.1,
  "thumbnail_tone_breakdown": { "urgency": 12, "fear": 3 }
}
```

---

### `GET /api/trends/{cluster_id}/sentiment`

Sentiment breakdown + per-week history.

**Response**
```json
{
  "cluster_id": 9,
  "sentiment_breakdown": { "negative": 14, "neutral": 2, "positive": 1 },
  "sentiment_label": "Negative",
  "recent_sentiment_label": "Negative",
  "dominant_sentiment": "negative",
  "by_week": [
    {
      "week": "week1",
      "sentiment_breakdown": { "negative": 5, "neutral": 1 },
      "dominant_sentiment": "negative"
    }
  ]
}
```

---

### `GET /api/trends/{cluster_id}/claims`

Same shape as `GET /api/narratives/{id}/claims`.

---

## Weeks

### `GET /api/weeks`

All available weeks with aggregate stats. Drives the Archives view.

**Response**
```json
{
  "weeks": [
    {
      "week": "week1",
      "total_videos": 60,
      "total_views": 4200000,
      "active_clusters": 10,
      "breaking_count": 8,
      "dominant_sentiment": "Negative"
    }
  ],
  "total": 3
}
```

Weeks are sorted chronologically (`week1` → `week2` → ...).

---

## Search

### `GET /api/search`

Semantic search over indexed transcript chunks. Returns VideoItem-shaped results ranked by relevance.

**Query Parameters**

| Parameter | Type   | Default | Description |
|-----------|--------|---------|-------------|
| `q`       | string | —       | Search query (required) |
| `limit`   | int    | 10      | Max results (1–50) |

**Response**
```json
{
  "query": "federal reserve rate hike",
  "limit": 10,
  "total": 4,
  "results": [
    {
      "video_id": "abc123",
      "channel": "CNBC",
      "title": "...",
      "score": 0.9142,
      "excerpt": "...the Fed is expected to hold rates steady...",
      ...VideoItem fields
    }
  ]
}
```

---

## Stats

### `GET /api/stats`

Aggregate summary across all clusters. Derived entirely from `narrative-clusters` — no extra reads.

**Response**
```json
{
  "total_videos": 191,
  "total_clusters": 12,
  "total_weeks": 3,
  "breaking_count": 24
}
```

---

## Articles

### `POST /api/articles/generate`

Trigger Gemini article generation for all active cluster × week combinations.

**Request Body**
```json
{
  "week": "week1",
  "cluster_id": 8,
  "force": false,
  "dry_run": false
}
```

All fields are optional. Omitting `week` and `cluster_id` processes all combinations. `force=true` overwrites existing articles. `dry_run=true` previews job count without writing.

**Response**
```json
{
  "articles_generated": 10,
  "articles_skipped": 2,
  "articles_failed": 0,
  "weeks_processed": ["week1", "week2"],
  "dry_run": false
}
```

---

### `GET /api/articles`

List articles (no body text).

**Query Parameters**

| Parameter    | Type | Default | Description |
|--------------|------|---------|-------------|
| `cluster_id` | int  | —       | Filter by cluster |
| `week`       | int  | —       | Filter by week number (e.g. `1`) |
| `limit`      | int  | 50      | Max results (1–200) |

**Response**
```json
{
  "articles": [
    {
      "article_id": "article-a1b2c3d4",
      "cluster_id": 8,
      "cluster_label": "Jeffrey Epstein Investigation",
      "week_number": 1,
      "title": "Epstein Files Expose Network of Powerful Connections",
      "overview": "Newly unsealed documents link Epstein to 36 public figures.",
      "created_at": "2026-03-10T09:00:00+00:00"
    }
  ],
  "total": 1
}
```

---

### `GET /api/articles/{article_id}`

Full article including body text.

**Response** — all list fields plus:
```json
{
  "body": "Full article prose...",
  "updated_at": "2026-03-10T09:00:00+00:00"
}
```

---

## Pipeline

### `POST /api/pipeline/run`

Runs the full pipeline in sequence: ingest → cluster → claim analysis → article generation.

**Request Body**
```json
{
  "prefix": "youtube-data/",
  "limit": 100,
  "dry_run": false
}
```

`dry_run=true` previews all stages without writing anything.

**Response**
```json
{
  "ingestion": {
    "objects_processed": 9,
    "videos_found": 180,
    "videos_indexed": 175,
    "total_chunks_stored": 458,
    "dry_run": false
  },
  "clustering": {
    "total_videos": 175,
    "cluster_count": 12,
    "noise_videos": 30,
    "total_chunks_patched": 145,
    "dry_run": false
  },
  "claim_analysis": {
    "clusters_processed": 12,
    "total_patched": 12,
    "dry_run": false
  },
  "articles": {
    "articles_generated": 24,
    "articles_skipped": 0,
    "articles_failed": 0,
    "weeks_processed": ["week1", "week2", "week3"],
    "dry_run": false
  }
}
```

---

### `POST /api/pipeline/ingest`

Ingest only — S3 → Gemini intelligence + thumbnail → DynamoDB → Qdrant.

**Request Body:** same as `/run` (`prefix`, `limit`, `dry_run`)

**Response:** `IngestionSummary` block from `/run` response above.

---

### `POST /api/pipeline/cluster`

Cluster only — reads from Qdrant + DynamoDB, runs UMAP/HDBSCAN, writes to DynamoDB.

**Request Body:** `{ "dry_run": false }`

**Response:** `ClusteringSummary` block from `/run` response above.

---

### `POST /api/pipeline/claims`

Claim analysis only — reads clustered videos from DynamoDB, classifies claims, writes results.

**Request Body:** `{ "dry_run": false }`

**Response:** `ClaimAnalysisSummary` block from `/run` response above.

---

### `POST /api/pipeline/search`

Raw vector search over Qdrant (internal/debug use). For frontend search use `GET /api/search`.

**Request Body**
```json
{ "query": "federal reserve interest rate", "limit": 5 }
```

**Response**
```json
{
  "query": "federal reserve interest rate",
  "limit": 5,
  "hits": [
    {
      "id": "...",
      "score": 0.91,
      "transcript_key": "youtube-data/week1/cnbc.json::abc123",
      "source_key": "youtube-data/week1/cnbc.json",
      "channel": "CNBC",
      "text": "...the Fed is expected to...",
      "chunk_index": 2
    }
  ]
}
```
