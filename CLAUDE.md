# Newsify Backend — Dev Notes

## Key conventions

- DynamoDB numbers come back as Python `Decimal` — always convert with `_dec()` or `_clean_for_dynamo()` when writing
- `PartitionKey`/`SortKey` are the DynamoDB key names; map to `channel`/`video_id` in API responses
- Cluster IDs are stable across pipeline runs via cosine similarity matching (0.65 full-text, 0.70 topics-only fallback)
- `top_claims` on `narrative-clusters` is written by `claim_analysis_service`, not the labeling step — don't confuse the two
- Rate limiting via slowapi — the starlette `Request` param **must** be named exactly `request`
- Gemini keys rotate across `GENAI_API_KEY` through `GENAI_API_KEY_40` (`gemini_service.py`)
- Week extraction from `source_key` (e.g. `youtube-data/week1/cnbc.json` → `"week1"`) lives in `pipeline_service._extract_week()`

## Pipeline consistency guarantees

Each `POST /api/pipeline/run` leaves all 4 tables in a consistent state:
1. `_purge_inactive_clusters` removes the cluster row AND cleans ghost references from videos, cluster-weeks, and articles
2. `_renumber_clusters` remaps IDs across all 4 tables atomically
3. `cleanup_orphaned_articles` runs before article generation to catch any remaining orphans
4. `video_count` on cluster rows is a GSI COUNT query (accurate historical count, not just current-run count)

## DynamoDB tables

| Table | PK | Notes |
|-------|----|-------|
| `youtube-videos` | `PartitionKey` (channel) + `SortKey` (videoId) | GSI `cluster-index` on `cluster_id` |
| `narrative-clusters` | `cluster_id` | |
| `cluster-weeks` | `cluster_id` + `week` (composite) | Requires delete+put for renumber (PK includes cluster_id) |
| `articles` | `article_id` | |

## Deployment

GCP Cloud Run. Docker image built and deployed by GitHub Actions on merge to `main`.
Do not enable auto-deploy on Cloud Run directly.
