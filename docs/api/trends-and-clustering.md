# Trends & Clustering

## Pipeline Overview

```
YouTube videos (S3)
  → Chunk transcripts → Embed (nomic-ai/nomic-embed-text-v1.5, 768d) → Qdrant
  → Gemini: topics, category, sentiment, key_claims, is_breaking → DynamoDB
  → UMAP (768d → 15d) + HDBSCAN → 12 narrative clusters
  → Gemini: cluster label, headline, summary → DynamoDB (narrative-clusters)
  → Claim analysis: consensus / debated / unique → DynamoDB
  → Trend service derives metrics at read time → powers /api/trends + /api/narratives
```

---

## Data Sources

**Channels:** ABCNews, BBCNews, CBSNews, CNBC, FoxNews, NBCNews, NewYorkTimes, SkyNews, WashingtonPost, aljazeeraenglish

| Week  | Date Range           | ~Videos |
|-------|----------------------|---------|
| week1 | ~March 3–9, 2026    | 60      |
| week2 | ~March 10–16, 2026  | 80      |
| week3 | ~March 17–23, 2026  | 40      |

**Totals:** ~180 videos, ~458 chunks, ~145 clustered, ~35 noise

---

## Clustering

### Algorithm

Two-stage pipeline: UMAP dimensionality reduction followed by HDBSCAN density-based clustering. HDBSCAN cannot operate directly on 768d embeddings — distances converge in high-dimensional space and density differences disappear. UMAP compresses to 15d while preserving local structure.

### Configuration

| Parameter         | Value                          | Purpose |
|-------------------|--------------------------------|---------|
| `min_cluster_size`| 7                              | Minimum videos to form a cluster |
| `min_samples`     | 2                              | HDBSCAN density strictness |
| `umap_components` | 15                             | Target dimensions after reduction |
| `umap_neighbors`  | 15                             | Neighborhood size (higher = more global) |
| `umap_metric`     | cosine                         | Distance metric for text embeddings |
| `embedding_model` | nomic-ai/nomic-embed-text-v1.5 | 768-dim, local inference |

### Cluster Labeling

Each cluster is labeled by Gemini, which receives the top topics, top claims, and representative video titles. Gemini returns:
- `label` — 3–6 word desk tag in Title Case (e.g. "US-Iran Military Escalation")
- `narrative_headline` — full newspaper-style headline
- `narrative_summary` — one sentence with a specific stat or data point

TF-IDF scoring provides a fallback label if Gemini fails.

### Stable Cluster IDs

HDBSCAN assigns arbitrary integer labels on each run. After clustering, new labels are matched to existing `narrative-clusters` records using Jaccard similarity on topic lists (threshold: 0.3). Matched clusters keep their stable ID. Unmatched existing clusters are marked `status: inactive` and excluded from read APIs.

### Sample Results (12 Clusters)

| ID | Label | Videos | Channels | Views | Sentiment |
|----|-------|--------|----------|-------|-----------|
| 9  | Middle East Conflict        | 17 | 7 | 1.66M | negative |
| 8  | Jeffrey Epstein Investigation| 16 | 7 | 697K  | negative |
| 4  | Inflation                   | 21 | 8 | 545K  | positive |
| 5  | Oil Markets                 | 14 | 7 | 609K  | negative |
| 10 | Iran Conflict               | 14 | 6 | 494K  | negative |
| 7  | US Foreign Policy           | 13 | 7 | 428K  | negative |
| 0  | Hezbollah                   | 11 | 4 | 818K  | negative |
| 1  | Missing Persons             | 10 | 7 | 217K  | negative |
| 3  | Automotive Industry*        |  8 | 4 | 322K  | negative |
| 2  | Data Centers                |  7 | 1 | 475K  | negative |
| 6  | Maritime Security           |  7 | 6 | 479K  | negative |
| 11 | Nuclear Proliferation       |  7 | 4 | 1.45M | neutral  |

\* Catch-all clusters (3, 4) contain mixed/unrelated topics — improves with more data.

### Running Clustering

```bash
# Via API (recommended)
curl -X POST http://localhost:8000/api/pipeline/cluster \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'

# Requires: umap-learn, scikit-learn>=1.3
pip install umap-learn scikit-learn
```

Re-run after ingesting new weeks so new videos get cluster assignments.

---

## Trend Service

All metrics are computed at read time from DynamoDB — no separate aggregation step.

### Heat Score Formula

```
heat_score = (channel_count × 3) + (breaking_count × 2) + (total_views / 100,000)
```

Channel breadth is weighted highest — 8 out of 10 channels covering the same story signals a real trend regardless of view counts.

### Trend Types

| Value      | Trigger |
|------------|---------|
| `surging`  | Volume +30% week-over-week, or +20% with 7+ channels |
| `emerging` | Zero videos last week, appeared this week |
| `dominant` | 7+ channels covering it, volume stable or growing |
| `fading`   | Dropped to zero this week, or -30% volume |
| `holding`  | Within ±30% movement |

### Metric Badge Examples

| Badge        | Meaning |
|--------------|---------|
| `+40% Vol`   | Volume up 40% |
| `-83% Vol`   | Volume down 83% |
| `High Impact`| 7+ channels (dominant) |
| `71% Neg`    | 71% negative sentiment |
| `New`        | Emerging this week |
| `Fading`     | Dropped to zero |
| `Holding`    | No significant movement |

### Sentiment Labels

| Label                    | Condition |
|--------------------------|-----------|
| `Negative`               | ≥50% negative |
| `Positive`               | ≥50% positive |
| `Neutral`                | ≥50% neutral |
| `Polarized`              | Both negative and positive >30% |
| `Sentiment Reversal — X` | Direction flipped from previous week |
| `Sentiment Shift — X`    | Was mixed/neutral, now strongly one direction |

### Engagement Index

```
engagement_index = ((total_likes + total_comments) / total_views) × 10,000
```

Expressed as interactions per 10K views.

---

## API Endpoints

### `GET /api/trends`

Returns all clusters sorted by `heat_score` (default). Optional `?week=` scopes all metrics to that week slice.

### `GET /api/trends/{id}`

Full cluster detail including `engagement_index`, `sentiment_breakdown`, `public_sentiment_breakdown`, `week_data`, `creator_risk`, and thumbnail stats.

### `GET /api/trends/{id}/sentiment`

Sentiment breakdown + `by_week` array showing per-week sentiment history.

### `GET /api/trends/{id}/claims`

Classified claims: `consensus` (3+ sources), `debated` (2 sources, divergent framing), `unique` (single source). Each claim includes a `risk_score` (0–1).

### `GET /api/narratives`

Editorial view — same clusters but only story fields: `narrative_headline`, `narrative_summary`, `top_topics`, `dominant_sentiment`. No metrics.

### `GET /api/weeks`

Per-week aggregates derived from `narrative-clusters.week_data[]`. Returns `total_videos`, `total_views`, `active_clusters`, `breaking_count`, `dominant_sentiment` per week.

---

## File Reference

| File | Purpose |
|------|---------|
| `app/services/clustering_service.py` | UMAP + HDBSCAN + Gemini labeling + stable ID matching |
| `app/services/claim_analysis_service.py` | Claim embedding, grouping, classification, creator risk |
| `app/services/trend_service.py` | Heat score, trend type, sentiment labels, narrative/trend read APIs |
| `app/services/dynamo_service.py` | Primary DynamoDB read layer for all API endpoints |
| `app/schemas/trend.py` | Pydantic response models for trends + weeks |
| `app/schemas/narrative.py` | Pydantic response models for narratives |
| `app/api/endpoints/trends.py` | `/api/trends/*` and `/api/weeks` route handlers |
| `app/api/endpoints/narratives.py` | `/api/narratives/*` route handlers |
