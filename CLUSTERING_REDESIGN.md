# Clustering Redesign — Handoff Document
**Date: April 10, 2026**

---

## Context

Newsify clusters ~50 YouTube videos per week into narrative clusters using HDBSCAN on Qdrant embeddings. The frontend shows a weekly archive (Week 1 → Week N) with cluster headlines and articles per week.

**The problem:** Headlines repeat verbatim across weeks. The system has no memory of what it said last week, so Gemini regenerates nearly identical headlines from the same static cluster data every run.

---

## What Was Already Changed (do NOT redo)

### `app/services/dynamo_service.py`
`get_video_titles_for_cluster` now accepts `week: str | None` param. When provided, filters videos by that week via FilterExpression and paginates fully.

### `app/services/article_service.py`
- `_generate_single_article` passes `week=wk` to `get_video_titles_for_cluster`
- `_build_prompt` restructured: week-specific video titles are primary, cluster-level `narrative_headline`/`narrative_summary` moved to "ONGOING STORY CONTEXT (background only)" section. Prompt explicitly instructs Gemini to write about THIS WEEK only.

### `app/services/cluster_labeling_service.py`
- `label_clusters` now tracks `latest_week`, `latest_titles`, `latest_claims` per cluster (from the most recent week bucket)
- Stores them in `cluster_stats`
- Gemini labeling prompt now has a "MOST RECENT WEEK — weight this most heavily" section passed as primary context
- Historical titles/claims labelled as "background"

These changes improve week-specificity but do NOT fix the core duplicate problem because Gemini still has no memory of prior weeks' headlines.

---

## Root Causes (from analysis of Weeks 1-6)

1. **No prior-week memory** — Gemini sees the same cluster data each week and writes the same headline from scratch. DHS cluster had "Lawmakers Clash" word-for-word identical across W2, W3, W5, W6.

2. **Cluster record gets overwritten** — `narrative_headline` and `narrative_summary` on `narrative-clusters` are a single value, replaced each run. No per-week snapshot exists.

3. **Articles for purged clusters get deleted** — `_purge_inactive_clusters` in `clustering_service.py` deletes articles. Archive breaks when a cluster goes inactive.

4. **Iran over-segmentation** — HDBSCAN splits one conflict story into 4-5 sub-clusters (US-Iran Military, Israel-Lebanon, Strait of Hormuz, US Homeland Security, Ceasefire) which all appear independently each week with near-identical headlines.

5. **No staleness/exit criteria** — Evergreen stories (Olaplex, convenience retail, Arm chip) run indefinitely with no signal that the story is no longer novel.

---

## What is Actually Bad (clarified April 10)

Only `narrative_headline` and `narrative_summary` on `narrative-clusters` are duplicated/stale.
`cluster_label` (desk label e.g. "US-Iran Military Escalation") is fine — do NOT touch it.
Article bodies are fine — do NOT touch articles.

---

## Proposed Architecture

### New DynamoDB Table: `cluster-weeks`

```
PK: cluster_id (N)
SK: week (S)  e.g. "week5"

Attributes:
  narrative_headline     S   -- per-week, never overwritten
  narrative_summary      S
  week_overview          S
  story_phase            S   -- emerging | developing | ongoing | stale | concluded (Phase 2)
  top_claims             L
  top_topics             L
  video_count            N
  channel_count          N
  view_count             N
  breaking_count         N
  dominant_sentiment     S
  created_at             S
```

This replaces the `week_data` array embedded in `narrative-clusters` as the source of truth for per-week headlines. `week_data` stays on `narrative-clusters` for backwards compat but is not the authority.

### Changes to Existing Tables

- `narrative-clusters` — keep as-is. `narrative_headline`/`narrative_summary` get patched to the latest week's value after backfill/each run.
- `articles` — **stop deleting on purge**. Remove article deletion from `_purge_inactive_clusters`.

---

## Implementation Plan — Phase 1

### 5 code changes + 1 backfill script, run in this order:

---

### Change 1 — Create `cluster-weeks` table
**File:** `scripts/update_dynamodb_schema.py`

Add constant: `CLUSTER_WEEKS_TABLE = "cluster-weeks"`

Add function `create_cluster_weeks_table(client)`:
```python
try:
    client.create_table(
        TableName="cluster-weeks",
        KeySchema=[
            {"AttributeName": "cluster_id", "KeyType": "HASH"},
            {"AttributeName": "week", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "cluster_id", "AttributeType": "N"},
            {"AttributeName": "week", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    _wait_for_table(client, "cluster-weeks")
except ClientError as e:
    if e.response["Error"]["Code"] == "ResourceInUseException":
        print("  Table 'cluster-weeks' already exists")
    else:
        raise
```
Call it as Step 3 in `main()`, after existing Step 2. Add `cluster-weeks` to `verify_tables()`.

---

### Change 2 — Add 3 methods to `DynamoService`
**File:** `app/services/dynamo_service.py`

Add in `__init__`: `self._cluster_weeks_table = self._dynamodb.Table("cluster-weeks")`

**`get_video_data_for_cluster(cluster_id, week, limit=10)`**
- Query `youtube-videos` via `cluster-index` GSI
- `KeyConditionExpression = Key("cluster_id").eq(Decimal(str(cluster_id)))`
- `FilterExpression = Attr("week").eq(week)`
- `ProjectionExpression = "title, key_claims, #wk"` with `ExpressionAttributeNames = {"#wk": "week"}`
- Paginate fully (same pattern as `get_video_titles_for_cluster`)
- Return `[{"title": ..., "key_claims": [...]}]` up to `limit` results
- Return `[]` gracefully on error

**`save_cluster_week_snapshot(cluster_id, week, data)`**
- `put_item` to `self._cluster_weeks_table`
- Item: `cluster_id` (Decimal), `week` (str), plus everything in `data`
- Run all numeric values through `Decimal(str(v))`, remove None values
- Idempotent — safe to rerun

**`get_cluster_week_headlines(cluster_id, last_n=4)`**
- Query `self._cluster_weeks_table` with `KeyConditionExpression = Key("cluster_id").eq(Decimal(str(cluster_id)))`
- Sort results by week number (parse int from "week3" → 3)
- Return last `n` as `[{"week": "week3", "headline": "..."}]` sorted oldest → newest
- Return `[]` gracefully if table missing or no results

---

### Change 3 — Inject prior headlines into Gemini prompt
**File:** `app/services/cluster_labeling_service.py`

**Signature change:**
```python
def label_clusters(self, video_ids, labels, meta_map, dry_run=False,
                   existing_clusters=None, cluster_weeks_table=None):
```

**Preliminary stable match (new, runs before the Gemini loop):**
After TF-IDF labels are computed (~line 236), if `existing_clusters` is provided:
- Build a minimal `tfidf_cluster_info` dict: `{cid: {"label": tfidf_labels[cid], "top_topics": [t[0] for t in cluster_topic_counts[cid].most_common(5)]}}` for each real cid
- Call `preliminary_id_map, _, _ = self.match_to_existing_clusters(tfidf_cluster_info, existing_clusters)`
- Store as `preliminary_id_map: dict[int, int]` (HDBSCAN cid → stable cid)
- This is headline-lookup only. The real stable match still runs in `clustering_service.py` after this method returns (unchanged).

**Before the Gemini call for each `cid`** (inside `for cid in real_cids`):
```python
from app.services.dynamo_service import DynamoService
stable_cid = preliminary_id_map.get(cid) if preliminary_id_map else None
prior_headlines = []
if stable_cid is not None:
    prior_headlines = DynamoService().get_cluster_week_headlines(stable_cid, last_n=4)
```

**Prompt injection** — add this block to the prompt string (only if `prior_headlines` is non-empty):
```
PRIOR HEADLINES FOR THIS CLUSTER (do NOT repeat or rephrase any of these):
- week4: "Tuchel Implements Strategic Split-Camp Structure..."
- week3: "Thomas Tuchel Implements Split Camp Strategy..."

The new headline MUST describe a different angle or development than all of the above.
If no new development exists, prefix with "Week N: " and describe what continued.
```

**Verbatim dupe check** — after `parsed = json.loads(raw)` succeeds:
```python
new_hl = parsed.get("headline", "").strip().lower()
recent = [p["headline"].strip().lower() for p in prior_headlines[-2:]]
if new_hl in recent and not getattr(self, "_dupe_retried_" + str(cid), False):
    setattr(self, "_dupe_retried_" + str(cid), True)
    # append to prompt and continue to next attempt in the retry loop
    prompt += "\n\nIMPORTANT: Your previous attempt returned the exact same headline as a prior week. You MUST write something completely different."
    continue
```
Max 1 extra retry per cluster (tracked by the flag above, cleared after each cluster).

---

### Change 4 — Write snapshots + remove article deletion
**File:** `app/services/clustering_service.py`

**4a — Move `load_existing_clusters` before `label_clusters` and pass it in:**
Currently `existing_clusters` is loaded after `label_clusters`. Move it before so the preliminary match inside `label_clusters` can use it.

In `run_clustering()`, change the step 5/6 block to:
```python
# 5a. Load existing clusters (needed for preliminary match inside label_clusters)
existing_clusters = self._labeling_service.load_existing_clusters(self._clusters_table)

# 5b. Label (preliminary stable match happens inside, Gemini uses prior headlines)
cluster_info = self._labeling_service.label_clusters(
    video_ids, labels, filtered_meta, dry_run=dry_run,
    existing_clusters=existing_clusters,
    cluster_weeks_table=self._dynamodb.Table("cluster-weeks"),
)

# 6. Real stable cluster matching (unchanged)
id_map, new_ids, declined_ids = self._labeling_service.match_to_existing_clusters(
    cluster_info, existing_clusters
)
```
Remove the old `existing_clusters = self._labeling_service.load_existing_clusters(...)` call from step 6.

**4b — Add `_write_cluster_week_snapshots(cluster_info, current_week)` method:**
After `_write_cluster_summaries()` in `run_clustering` (step 8), add:
```python
# 8b. Write per-week snapshots to cluster-weeks
if current_week:
    self._write_cluster_week_snapshots(cluster_info, current_week)
```
Note: move `current_week = self._detect_current_week(cluster_info)` to before step 8 so it's available.

The method body: for each `cid, info` in `cluster_info` (skip -1):
- Find the `week_data` entry where `wd["week"] == current_week`
- If found, call `DynamoService().save_cluster_week_snapshot(cid, current_week, {narrative_headline, narrative_summary, week_overview (from wd), top_claims, top_topics, video_count, channel_count (from wd), view_count (from wd), breaking_count (from wd), dominant_sentiment})`

**4c — Remove article deletion from `_purge_inactive_clusters`:**
Delete lines 343-356 (the `articles_deleted` scan + delete loop including `articles_table` reference at top of method).
Keep only the cluster record deletion. Articles are permanent historical records.

---

### Change 5 — Backfill script for weeks 1-6
**File:** `scripts/backfill_cluster_weeks.py` (new file)

Regenerates `narrative_headline` + `narrative_summary` for all existing clusters x weeks using the dedup-aware prompt. Writes to `cluster-weeks`. Patches `narrative-clusters` with the latest week's result.

**Script structure:**
```python
#!/usr/bin/env python3
"""
backfill_cluster_weeks.py - Regenerate narrative headlines for weeks 1-6

Reads all clusters from narrative-clusters, processes each week in order
(week1 → weekN), calls Gemini with a dedup-aware prompt, writes results
to cluster-weeks table, and patches narrative-clusters with the latest headline.

Usage:
    python scripts/backfill_cluster_weeks.py
    python scripts/backfill_cluster_weeks.py --cluster-id 7   # single cluster
    python scripts/backfill_cluster_weeks.py --dry-run         # print prompts, no writes
"""
```

**Algorithm:**
```
for each cluster in narrative-clusters (scan all, skip status='inactive'):
    print(f"[{cluster_id}] {cluster_label}")
    week_data = cluster["week_data"]  # existing embedded array
    weeks = sorted by int(week[4:])   # week1, week2, ... weekN
    prior_headlines = []              # grows as we process each week

    for wd in weeks:
        week_name = wd["week"]        # e.g. "week3"

        # Pull actual video data for this cluster+week
        videos = dynamo_svc.get_video_data_for_cluster(cluster_id, week_name)
        titles = [v["title"] for v in videos][:6]
        claims = list(dict.fromkeys(c for v in videos for c in v.get("key_claims", [])))[:6]

        # Build prior headlines block
        prior_block = "\n".join(f'- {p["week"]}: "{p["headline"]}"' for p in prior_headlines[-4:])

        # Build prompt (see template below)
        prompt = build_backfill_prompt(cluster_label, cluster_topics, week_name, wd, titles, claims, prior_block)

        # Call Gemini with same retry/key-rotation as cluster_labeling_service.py
        result = gemini_call(prompt)  # {"headline": ..., "summary": ..., "week_overview": ...}

        # Verbatim check
        recent_hls = [p["headline"].strip().lower() for p in prior_headlines[-2:]]
        if result["headline"].strip().lower() in recent_hls:
            result = gemini_call(prompt + "\n\nIMPORTANT: Return a headline completely different from all prior weeks listed above.")

        # Write to cluster-weeks (idempotent put_item)
        dynamo_svc.save_cluster_week_snapshot(cluster_id, week_name, {
            "narrative_headline": result["headline"],
            "narrative_summary": result["summary"],
            "week_overview": result.get("week_overview", ""),
            "top_claims": claims,
            "top_topics": cluster["top_topics"],
            "video_count": wd["video_count"],
            "channel_count": wd.get("channel_count", 0),
            "view_count": wd.get("view_count", 0),
            "breaking_count": wd.get("breaking_count", 0),
            "dominant_sentiment": max(wd.get("sentiment_breakdown", {"neutral": 1}),
                                      key=wd.get("sentiment_breakdown", {"neutral": 1}).get),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        prior_headlines.append({"week": week_name, "headline": result["headline"]})
        print(f"  {week_name}: {result['headline']}")

    # Patch narrative-clusters with latest week's headline
    if prior_headlines:
        latest = prior_headlines[-1]
        clusters_table.update_item(
            Key={"cluster_id": Decimal(str(cluster_id))},
            UpdateExpression="SET narrative_headline = :h, narrative_summary = :s",
            ExpressionAttributeValues={":h": latest_headline, ":s": latest_summary},
        )
```

**Prompt template:**
```
You are a news editor. Given YouTube video data for cluster "{cluster_label}" during {week_name},
write a narrative headline and summary covering ONLY this week's developments.

VIDEO TITLES THIS WEEK: {titles}
KEY CLAIMS THIS WEEK: {claims}
STATS: {video_count} videos, {view_count} views, dominant sentiment: {dominant_sentiment}
CLUSTER TOPICS: {top_topics}

[IF prior_headlines non-empty:]
PRIOR HEADLINES FOR THIS CLUSTER (do NOT repeat or rephrase any of these):
{prior_block}
The new headline MUST describe a different angle or development than all of the above.
If no genuinely new development exists, prefix with "Week N: " and describe what continued.
[END IF]

Return ONLY valid JSON, no markdown:
{"headline": "...", "summary": "...", "week_overview": "2 sentences: main development this week. Scale/sentiment context."}
```

Use same Gemini model (`settings.gemini_model_id`) + key-rotation pattern as `cluster_labeling_service.py`.
Add `--cluster-id N` arg to rerun a single cluster. Add `--dry-run` to print without writing.

---

## Run Order

1. `python scripts/update_dynamodb_schema.py --skip-gsis` — creates `cluster-weeks` table
2. `python scripts/backfill_cluster_weeks.py` — populates weeks 1-6, patches `narrative-clusters`
3. Deploy changes to `dynamo_service.py`, `cluster_labeling_service.py`, `clustering_service.py`
4. Next pipeline run — writes to `cluster-weeks` and uses prior headlines automatically

---

## What NOT to Touch

- `cluster_label` on `narrative-clusters` or `youtube-videos` — fine, do NOT change
- Article bodies — fine, do NOT change
- `dynamo_service.get_video_titles_for_cluster` — already has week filter, do not redo
- `article_service._build_prompt` — already week-specific, do not redo
- `week_data` array on `narrative-clusters` — keep for backwards compat, do not remove

---

## Phase 2 — Story phase + staleness (future session)

Add `story_phase` field to `cluster-weeks` items and to the Gemini labeling prompt output.
Phases: `emerging` | `developing` | `ongoing` | `stale` | `concluded`
Staleness rule: 3+ consecutive weeks `ongoing` + unchanged top_claims → auto-set `stale`.

## Phase 3 — Cluster merging / Iran over-segmentation (future session)

Post-HDBSCAN merge step: cosine similarity > 0.82 on label+topics → combine clusters.
Implement as `_merge_overlapping_clusters(cluster_info)` in `clustering_service.py`, called between step 5 (label) and step 6 (stable matching).

## Archive API (future session)

```
GET /api/weeks/{week}
  scan cluster-weeks WHERE week = "week5"
  return [{cluster_id, cluster_label, narrative_headline, narrative_summary, video_count, ...}]
```

---

## Key Numbers

- ~50 videos/week ingested, ~10-15 clusters per week, 6 weeks of history
- Verbatim dupe check: exact string match case-insensitive, max 1 retry
- Prior headlines window passed to Gemini: last 4 weeks
- Phase 3 merge threshold: cosine similarity > 0.82
