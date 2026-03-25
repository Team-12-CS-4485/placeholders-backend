# Trends & Clustering Pipeline

## Pipeline Overview

```
180 YouTube videos (10 channels, 3 weeks)
    → Chunk transcripts (458 chunks)
    → Embed with nomic-embed-text-v1.5 (768 dims)
    → Gemini extracts: topics, category, sentiment, key_claims, is_breaking
    → Store in Qdrant
    → UMAP (768d → 15d) + HDBSCAN clustering → 12 narrative clusters
    → Trend service aggregates by week → powers frontend
```

---

## Data Sources

**Channels:** ABCNews, BBCNews, CBSNews, CNBC, FoxNews, NBCNews, NewYorkTimes, SkyNews, WashingtonPost, aljazeeraenglish

| Week | Date Range | Videos |
|------|-----------|--------|
| week1 | ~March 3–9, 2026 | ~60 |
| week2 | ~March 10–16, 2026 | ~80 |
| week3 | ~March 17–23, 2026 | ~40 |

**Totals:** 180 videos, 458 chunks, 145 clustered, 35 noise

---

## Clustering

### Algorithm

Two-stage pipeline: UMAP dimensionality reduction followed by HDBSCAN density-based clustering. This is needed because HDBSCAN fails in high-dimensional space (768 dims) due to the curse of dimensionality — all distances converge and density differences disappear.

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `min_cluster_size` | 7 | Minimum videos to form a cluster |
| `min_samples` | 2 | HDBSCAN density strictness |
| `umap_components` | 15 | Target dimensions after reduction |
| `umap_neighbors` | 15 | Neighborhood size (higher = more global structure) |
| `umap_metric` | cosine | Distance metric for text embeddings |
| `embedding_model` | nomic-ai/nomic-embed-text-v1.5 | 768-dim, free, local inference |

### Cluster Labeling

Labels are assigned using TF-IDF-style scoring — each cluster gets labeled by its most *distinctive* topic, not just the most frequent. This prevents every cluster from being labeled "Iran Conflict" since that topic appears globally. Uniqueness is enforced so no two clusters share a label.

### Results (12 Clusters)

| ID | Label | Videos | Channels | Views | Sentiment |
|----|-------|--------|----------|-------|-----------|
| 9 | Middle East Conflict | 17 | 7 | 1.66M | negative |
| 8 | Jeffrey Epstein Investigation | 16 | 7 | 697K | negative |
| 4 | Inflation | 21 | 8 | 545K | positive |
| 5 | Oil Markets | 14 | 7 | 609K | negative |
| 10 | Iran Conflict | 14 | 6 | 494K | negative |
| 7 | US Foreign Policy | 13 | 7 | 428K | negative |
| 0 | Hezbollah | 11 | 4 | 818K | negative |
| 1 | Missing Persons | 10 | 7 | 217K | negative |
| 3 | Automotive Industry* | 8 | 4 | 322K | negative |
| 2 | Data Centers | 7 | 1 | 475K | negative |
| 6 | Maritime Security | 7 | 6 | 479K | negative |
| 11 | Nuclear Proliferation | 7 | 4 | 1.45M | neutral |

\* Catch-all clusters (3, 4) contain mixed/unrelated topics. Improves with more data.

### Running Clustering

```bash
pip install umap-learn scikit-learn>=1.3
python -m scripts.run_clustering --min-cluster-size 7 --min-samples 2
```

Re-run after ingesting new weeks of data so new videos get assigned to clusters.

---

## Trend Service

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trends` | GET | Full trend list with metrics, week_data, claims |
| `/api/trends?sort_by=heat_score` | GET | Sort by: `heat_score`, `video_count`, `view_count_total`, `channel_count` |
| `/api/trends/{cluster_id}` | GET | Single trend detail by cluster ID |

### Response Structure

```
{
  "header": { active_narratives, total_volume, new_signals_pct },
  "trends": [ ...TrendItem ],
  "generated_at": "ISO timestamp"
}
```

### Header Fields

| Field | Description |
|-------|-------------|
| `active_narratives` | Number of clusters (excluding noise) |
| `total_volume` | Sum of view_count across all clustered videos |
| `new_signals_pct` | Week-over-week % change in total views |

### Trend Fields

| Field | Type | Description |
|-------|------|-------------|
| `cluster_id` | int | Unique cluster ID — use for `GET /api/trends/{id}` |
| `label` | string | Cluster name (TF-IDF derived) |
| `description` | string | Top key_claim as summary snippet |
| `category` | string | Most common category from fixed taxonomy |
| `trend_type` | string | Movement classification (see below) |
| `metric_badge` | string | Short display string for UI |
| `video_count` | int | Total unique videos across all weeks |
| `channel_count` | int | Distinct channels covering this (max 10) |
| `view_count_total` | int | Sum of YouTube views |
| `breaking_count` | int | Videos with urgent/breaking language |
| `heat_score` | float | Composite ranking score |
| `sentiment_breakdown` | object | `{positive: n, negative: n, neutral: n}` |
| `dominant_sentiment` | string | Most common sentiment |
| `channels` | array | Channel names covering this trend |
| `week_data` | array | Per-week metrics (see below) |
| `top_claims` | array | Up to 5 factual claims from transcripts |
| `top_topics` | array | 5 most frequent topic tags |

### Trend Types

| Value | Meaning | Trigger |
|-------|---------|---------|
| `rising` | Growing fast | Video count +30% week-over-week |
| `emerging` | Brand new | Zero last week, appeared this week |
| `dominant` | Widespread | 7+ channels covering it |
| `declining` | Losing coverage | Video count -30% week-over-week |
| `stable` | No significant change | Within ±30% |

### Metric Badge Examples

| Badge | Meaning |
|-------|---------|
| `+40% Vol` | Volume up 40% |
| `-83% Vol` | Volume down 83% |
| `High Impact` | 7+ channels (dominant) |
| `71% Neg` | 71% negative sentiment |
| `68% Pos` | 68% positive sentiment |
| `New` | Emerging this week |
| `Fading` | Dropped to zero |
| `Stable` | No movement |

### Heat Score Formula

```
heat_score = (channel_count × 3) + (breaking_count × 2) + (view_count_total / 100,000)
```

Channel breadth weighted highest — if 8 out of 10 channels cover the same story, that's a real trend regardless of view counts.

### Week Data (per entry)

| Field | Description |
|-------|-------------|
| `week` | Week identifier (`week1`, `week2`, `week3`) |
| `video_count` | Videos published that week in this cluster |
| `channel_count` | Channels covering it that week |
| `view_count` | Views from that week's videos |
| `breaking_count` | Breaking videos that week |
| `sentiment_breakdown` | Sentiment counts for that week |

---

## Known Issues

| Issue | Impact | Resolution |
|-------|--------|------------|
| Clusters 3 & 4 are catch-alls | Mixed unrelated topics in same cluster | Improves with more data; can tighten `min_cluster_size` |
| `new_signals_pct` skewed | Partial weeks (week4 has 2 videos) distort the % | Fix: ignore weeks below a minimum video threshold |
| Week extraction | Videos ingested with timestamp prefixes instead of `weekN/` | Handled — falls back to ISO week date parsing |

---

## File Reference

| File | Purpose |
|------|---------|
| `app/services/clustering_service.py` | UMAP + HDBSCAN clustering pipeline |
| `app/services/trend_service.py` | Aggregation, trend classification, heat scoring |
| `app/schemas/trend.py` | Pydantic response models |
| `app/api/endpoints/trends.py` | FastAPI route handlers |
| `scripts/run_clustering.py` | CLI runner for clustering |