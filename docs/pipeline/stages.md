# Pipeline Stages

The pipeline is triggered via `POST /api/pipeline/run` (or individual stage endpoints) and runs synchronously end-to-end. All stages accept `dry_run=true` to preview counts without writing.

---

## Overview

```
S3 (raw JSON)
  → Stage 1: Load videos + transcripts
  → Stage 2: Chunk transcripts (nomic chunker)
  → Stage 3: Gemini intelligence (topics, sentiment, claims, is_breaking)
  → Stage 4: Write intelligence to DynamoDB (youtube-videos)
  → Stage 5: Gemini Vision thumbnail analysis
  → Stage 6: Write thumbnail fields to DynamoDB (youtube-videos)
  → Stage 7: Embed chunks locally (nomic-ai/nomic-embed-text-v1.5)
  → Stage 8: Upsert vectors to Qdrant

--- (separate pipeline step) ---

  → Stage 9:  UMAP + HDBSCAN clustering
  → Stage 10: Gemini cluster labeling + narrative generation
  → Stage 11: Write cluster summaries to DynamoDB (narrative-clusters)
  → Stage 12: Update video cluster assignments in DynamoDB

--- (separate pipeline step) ---

  → Stage 13: Embed claims (nomic, local)
  → Stage 14: Group claims by similarity, classify consensus/debated/unique
  → Stage 15: Write classified_claims + creator_risk to DynamoDB (narrative-clusters)

--- (separate pipeline step) ---

  → Stage 16: Gemini article generation per cluster × week
  → Stage 17: Write articles to DynamoDB (articles)
```

---

## Stage 1: Load from S3

**Service:** `StorageService.load_videos_from_prefix`

Scans S3 under a given prefix (default: `youtube-data/`) up to the configured object limit. Each JSON file contains one channel's videos:

```json
{ "channel": "CNBC", "videos": [{ "videoId": "...", "title": "...", "transcript": "...", "top_comments": [...] }] }
```

The week is extracted from the S3 key path (`youtube-data/week1/cnbc.json` → `"week1"`). Falls back to ISO week date parsing if no `weekN/` prefix is found.

Videos already indexed in Qdrant (checked by `transcript_index` filter) are skipped to preserve Gemini quota.

**Config:** `S3_PREFIX`, `S3_OBJECT_LIMIT`, `S3_BUCKET`, `AWS_REGION`

---

## Stage 2: Chunk

**Service:** `EmbeddingService.chunk_text`

Each transcript is split into overlapping character-level chunks using the nomic chunker.

**Config:** `CHUNK_SIZE_CHARS` (default: 6000), `CHUNK_OVERLAP_CHARS` (default: 400)

---

## Stage 3: Gemini Video Intelligence

**Service:** `EmbeddingService.extract_video_intelligence`

One Gemini call per video (not per chunk). Extracts:

| Field | Type | Description |
|-------|------|-------------|
| `topics` | list[str] | Subject tags |
| `category` | str | Fixed taxonomy category |
| `sentiment` | str | `positive`, `negative`, or `neutral` |
| `key_claims` | list[str] | Factual claims made in the video |
| `is_breaking` | bool | Urgent/breaking language detected |
| `public_sentiment` | str | Inferred audience sentiment from comments |
| `public_sentiment_score` | float | -1.0 to 1.0 |

If Gemini returns empty `topics` (e.g., 503 error), the video is skipped entirely.

**Config:** `GENAI_API_KEY` through `GENAI_API_KEY_20` (rotated on 429), `GEMINI_MODEL_ID`

---

## Stage 4: Write Intelligence to DynamoDB

**Service:** `PipelineService._write_intelligence_to_dynamodb`

Updates the existing `youtube-videos` item (written during ingestion) with intelligence fields. Only non-empty fields are written. Also writes `week`, `source_key`, `chunk_count`, `indexed_at`.

---

## Stage 5: Gemini Thumbnail Analysis

**Service:** `EmbeddingService.analyze_thumbnail`

One Gemini Vision call per video using the YouTube thumbnail URL. Extracts:

| Field | Type | Description |
|-------|------|-------------|
| `thumbnail_tone` | str | e.g. `urgency`, `fear`, `neutral` |
| `thumbnail_clickbait_score` | int | 1–10 |
| `thumbnail_brand_consistent` | bool | Consistent with channel branding |
| `thumbnail_visual` | str | Visual description |
| `thumbnail_insight` | str | Why it is or isn't clickbait |

If the result is all-defaults (Gemini 503 or missing image), the thumbnail write is skipped non-fatally.

---

## Stage 6: Write Thumbnail to DynamoDB

**Service:** `PipelineService._write_thumbnail_to_dynamodb`

Updates the `youtube-videos` item with thumbnail fields.

---

## Stage 7: Embed Chunks (Local)

**Service:** `EmbeddingService.embed_chunks`

Embeds all chunks using the local nomic model (`nomic-ai/nomic-embed-text-v1.5`, 768 dimensions). No API calls — runs fully locally.

**Config:** `EMBEDDING_MODEL_ID`

---

## Stage 8: Upsert to Qdrant

**Service:** `VectorService.upsert_transcript_chunks`

Upserts one point per chunk. Each point stores:
- The 768-dim vector
- Payload: `transcript_index` (video ID), `source_key`, `transcript_key`, `chunk_index`, `text`, `word_count`

The `transcript_index` field (= YouTube video ID) is the key used to deduplicate by video in search results.

**Config:** `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`

---

## Stages 9–12: Clustering (`POST /api/pipeline/cluster`)

**Service:** `ClusteringService.run_clustering`

1. Scrolls all vectors from Qdrant, grouped by `transcript_index` (video ID)
2. Pulls video metadata from DynamoDB (topics, category, sentiment, week, etc.)
3. Mean-pools chunks per video into one representative vector
4. UMAP (768d → 15d, cosine metric) + HDBSCAN (`min_cluster_size=7`, `min_samples=2`)
5. Gemini call per cluster → `label` (3–6 word desk tag), `narrative_headline`, `narrative_summary`
6. Stable cluster matching — HDBSCAN IDs are mapped to persistent stable IDs using Jaccard topic similarity; unmatched old clusters are marked `inactive`
7. Writes cluster summaries to `narrative-clusters`
8. Updates `cluster_id`, `cluster_label`, `cluster_confidence` on each video in `youtube-videos`

**Config:** `GEMINI_MODEL_ID`, clustering parameters passed via API request body or defaults

---

## Stages 13–15: Claim Analysis (`POST /api/pipeline/claims`)

**Service:** `ClaimAnalysisService.run_claim_analysis`

1. Scans `youtube-videos` for all videos with `cluster_id` and `key_claims`
2. Embeds all claims locally (nomic) and computes a similarity matrix
3. Groups claims with cosine similarity ≥ 0.75
4. Classifies groups:
   - **Consensus** — 3+ distinct channels agree (risk score low, decreases with more sources)
   - **Debated** — 2 channels with divergent framing (risk score based on framing divergence)
   - **Unique** — single channel, single video (risk score 0.7 baseline)
5. Computes `creator_risk` per channel: weighted average risk score across all claims
6. Writes `classified_claims` and `creator_risk` to `narrative-clusters`

---

## Stages 16–17: Article Generation (`POST /api/articles/generate`)

**Service:** `ArticleService.run_article_generation`

1. Scans `narrative-clusters` for active clusters with `week_data`
2. Builds one job per (cluster × week) combination; deduplicates and sorts by week → views
3. Skips existing articles unless `force=True`
4. Gemini call per job — generates `headline`, `overview`, `body` (400–600 words)
5. Saves to `articles` table: `article_id`, `cluster_id`, `cluster_label`, `week`, `week_number`, `title`, `overview`, `body`, `created_at`

Gemini API keys are rotated on 429 errors. Generation is skipped entirely when `dry_run=True`.

---

## Status Values (ingest stage)

| Status           | Meaning |
|------------------|---------|
| `success`        | All videos in the S3 object processed without error |
| `partial_failed` | At least one video failed, others succeeded |
| `failed`         | Object could not be loaded (S3/JSON error) |
