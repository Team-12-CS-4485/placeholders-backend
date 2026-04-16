"""
cluster_labeling_service.py - Cluster Labeling & Matching

Handles cluster intelligence:
- Aggregation of per-cluster stats from video metadata (build_cluster_stats)
- Gemini stable matching to existing story threads (match_to_existing_clusters)
- Gemini labeling with NEW vs CONTINUING STORY context (label_clusters)
- Cluster ID remapping from HDBSCAN labels to persistent IDs

Flow:
  build_cluster_stats → match_to_existing_clusters → label_clusters → remap_cluster_info

Matching runs first (on raw topics/titles) so labeling has authoritative
match context — MATCH clusters get prior headlines injected into the label
prompt as a CONTINUING STORY section; NEW clusters get a fresh prompt.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from typing import Optional  # noqa: F401

import numpy as np
from google import genai
from google.genai import types

from app.core.config import settings
from app.services.chunking_service import get_default_chunker

logger = logging.getLogger(__name__)


class ClusterLabelingService:

    def __init__(self, api_keys=None, chunker=None):
        self._genai_api_keys = api_keys or settings.genai_api_keys
        self._genai_key_index = 0
        self._genai_client = genai.Client(api_key=self._genai_api_keys[0])
        self._chunker = chunker or get_default_chunker()

    def _rotate_key(self) -> bool:
        """Rotate to the next API key. Returns True if a new key is available."""
        next_index = self._genai_key_index + 1
        if next_index >= len(self._genai_api_keys):
            logger.error("API_KEY_EXHAUSTED all keys have hit quota")
            return False
        self._genai_key_index = next_index
        self._genai_client = genai.Client(
            api_key=self._genai_api_keys[self._genai_key_index]
        )
        logger.warning(
            f"API_KEY_ROTATED key_index={self._genai_key_index}"
            f"/{len(self._genai_api_keys)-1}"
        )
        return True

    # ── Step 1: Aggregate per-cluster stats ──────────────────────────────────

    def build_cluster_stats(
        self,
        video_ids,
        labels,
        meta_map,
    ) -> dict:
        """
        Aggregate per-cluster stats from video metadata.

        Pure aggregation — no Gemini, no matching. Returns raw cluster_info
        with topics, titles, claims, week_data, and TF-IDF-ranked top_topics.
        label/narrative_headline/narrative_summary are None until label_clusters runs.
        """
        cluster_members: dict[int, list[str]] = {}
        for vid, label in zip(video_ids, labels):
            cluster_members.setdefault(int(label), []).append(vid)

        cluster_topic_counts: dict[int, Counter] = {}
        cluster_stats: dict[int, dict] = {}
        cluster_claims: dict[int, list[str]] = {}

        for cid, vids in cluster_members.items():
            tc = Counter()
            channels = set()
            categories = Counter()
            sentiments = Counter()
            public_sentiments = Counter()
            public_sentiment_scores = []
            breaking = views = likes = comments = 0
            claims = []
            week_buckets: dict[str, list[dict]] = defaultdict(list)

            for vid in vids:
                m = meta_map.get(vid, {})
                for t in m.get("topics", []):
                    tc[t] += 1
                channels.add(m.get("channel", ""))
                categories[m.get("category", "Other")] += 1
                sentiments[m.get("sentiment", "neutral")] += 1
                public_sentiments[m.get("public_sentiment", "neutral")] += 1
                public_sentiment_scores.append(m.get("public_sentiment_score", 0.0))
                if m.get("is_breaking"):
                    breaking += 1
                views += m.get("view_count", 0)
                likes += m.get("like_count", 0)
                comments += m.get("comment_count", 0)
                claims.extend(m.get("key_claims", []))
                week_buckets[m.get("week", "unknown")].append(m)

            # Build week_data from buckets
            week_data = []
            for week_name in sorted(
                week_buckets,
                key=lambda w: (
                    int(w[4:]) if w.startswith("week") and w[4:].isdigit() else 9999
                ),
            ):
                wvids = week_buckets[week_name]
                week_channels = {v.get("channel", "") for v in wvids}
                week_sentiments = Counter(v.get("sentiment", "neutral") for v in wvids)
                week_breaking = sum(1 for v in wvids if v.get("is_breaking"))
                week_views = sum(v.get("view_count", 0) for v in wvids)
                week_pub_sentiments = Counter(
                    v.get("public_sentiment", "neutral") for v in wvids
                )
                week_data.append(
                    {
                        "week": week_name,
                        "video_count": len(wvids),
                        "channel_count": len(week_channels),
                        "view_count": week_views,
                        "breaking_count": week_breaking,
                        "sentiment_breakdown": dict(week_sentiments),
                        "public_sentiment_breakdown": dict(week_pub_sentiments),
                        "week_overview": "",  # populated by label_clusters
                    }
                )

            avg_public_score = (
                round(sum(public_sentiment_scores) / len(public_sentiment_scores), 3)
                if public_sentiment_scores
                else 0.0
            )

            dominant_creator = sentiments.most_common(1)[0][0] if sentiments else "neutral"
            dominant_public = (
                public_sentiments.most_common(1)[0][0] if public_sentiments else "neutral"
            )

            # Latest week's titles and claims for Gemini prompts
            latest_week_name = max(
                (w for w in week_buckets if w.startswith("week") and w[4:].isdigit()),
                key=lambda w: int(w[4:]),
                default=None,
            )
            if latest_week_name and latest_week_name in week_buckets:
                latest_vids = week_buckets[latest_week_name]
                latest_titles = [v["title"] for v in latest_vids if v.get("title")][:5]
                latest_claims = list(
                    dict.fromkeys(c for v in latest_vids for c in v.get("key_claims", []))
                )[:5]
            else:
                latest_week_name = None
                latest_titles = []
                latest_claims = []

            cluster_topic_counts[cid] = tc
            cluster_claims[cid] = list(dict.fromkeys(claims))[:5]
            cluster_stats[cid] = {
                "vids": vids,
                "channels": channels,
                "categories": categories,
                "sentiments": sentiments,
                "public_sentiments": public_sentiments,
                "avg_public_sentiment_score": avg_public_score,
                "sentiment_divergence": dominant_creator != dominant_public,
                "breaking": breaking,
                "views": views,
                "likes": likes,
                "comments": comments,
                "week_data": week_data,
                "latest_week": latest_week_name,
                "latest_titles": latest_titles,
                "latest_claims": latest_claims,
            }

        # TF-IDF scoring for discriminative top_topics selection
        real_cids = [c for c in cluster_members if c != -1]
        n_clusters = max(len(real_cids), 1)
        topic_cluster_count: Counter = Counter()
        for cid in real_cids:
            for topic in cluster_topic_counts[cid]:
                topic_cluster_count[topic] += 1

        cluster_scored: dict[int, list] = {}
        for cid in cluster_members:
            tc = cluster_topic_counts[cid]
            size = max(len(cluster_members[cid]), 1)
            scored = []
            for topic, count in tc.items():
                tf = count / size
                idf = (
                    math.log((n_clusters + 1) / (topic_cluster_count.get(topic, 0) + 1))
                    + 1
                )
                scored.append((topic, tf * idf))
            scored.sort(key=lambda x: x[1], reverse=True)
            cluster_scored[cid] = scored

        # Build cluster_info — label/headline/summary filled in by label_clusters
        cluster_info = {}
        for cid, vids in cluster_members.items():
            stats = cluster_stats[cid]
            top_topics = [t[0] for t in cluster_scored[cid][:5]]
            cluster_info[cid] = {
                "label": None,
                "narrative_headline": None,
                "narrative_summary": None,
                "video_count": len(vids),
                "video_ids": vids,
                "channels": sorted(stats["channels"]),
                "channel_count": len(stats["channels"]),
                "top_topics": top_topics,
                "dominant_category": (
                    stats["categories"].most_common(1)[0][0]
                    if stats["categories"]
                    else "Other"
                ),
                "dominant_sentiment": (
                    stats["sentiments"].most_common(1)[0][0]
                    if stats["sentiments"]
                    else "neutral"
                ),
                "sentiment_breakdown": dict(stats["sentiments"]),
                "public_sentiment_breakdown": dict(stats["public_sentiments"]),
                "avg_public_sentiment_score": stats["avg_public_sentiment_score"],
                "dominant_public_sentiment": (
                    stats["public_sentiments"].most_common(1)[0][0]
                    if stats["public_sentiments"]
                    else "neutral"
                ),
                "sentiment_divergence": stats["sentiment_divergence"],
                "breaking_count": stats["breaking"],
                "total_views": stats["views"],
                "total_likes": stats["likes"],
                "total_comments": stats["comments"],
                "week_data": stats["week_data"],
                "top_claims": cluster_claims.get(cid, []),
                # Prompt-only fields — not written to DynamoDB
                "latest_week": stats["latest_week"],
                "latest_titles": stats["latest_titles"],
                "latest_claims": stats["latest_claims"],
            }
            logger.debug(
                f"CLUSTER_STATS id={cid} videos={len(vids)} "
                f"top_topics={top_topics[:3]}"
            )

        return cluster_info

    # ── Step 2: Gemini stable matching ───────────────────────────────────────

    def load_existing_clusters(self, clusters_table) -> dict[int, dict]:
        """
        Scan narrative-clusters table and return
        {cluster_id: {label, created_at, status}}.
        """
        existing = {}
        scan_kwargs: dict = {}
        while True:
            resp = clusters_table.scan(**scan_kwargs)
            for item in resp["Items"]:
                cid = int(item["cluster_id"])
                existing[cid] = {
                    "label": item.get("cluster_label", ""),
                    "created_at": item.get("created_at"),
                    "status": item.get("status", "active"),
                }
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        logger.debug(f"STABLE_MATCH_LOADED existing_clusters={len(existing)}")
        return existing

    def match_to_existing_clusters(
        self,
        raw_cluster_info: dict[int, dict],
        existing_clusters: dict[int, dict],
        prior_headlines: dict[int, list[dict]] | None = None,
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """
        Match new HDBSCAN clusters to existing story threads using Gemini.

        Uses raw cluster stats (top_topics + latest_titles) — no label needed.
        Existing threads are described by their registry label + recent headlines.

        Week-1 fast-path: if no existing clusters, mint sequential IDs and skip Gemini.

        Returns:
            id_map: {hdbscan_label: stable_cluster_id}
            new_ids: stable IDs minted for genuinely new clusters
            declined_ids: existing cluster IDs that had no match this week
        """
        real_new = {cid: info for cid, info in raw_cluster_info.items() if cid != -1}

        if not existing_clusters:
            # Week-1 / first run — mint sequential IDs, no Gemini needed
            id_map = {cid: i for i, cid in enumerate(sorted(real_new.keys()))}
            logger.info(
                f"MATCH_FIRST_RUN no existing clusters — minting {len(id_map)} new IDs"
            )
            return id_map, list(id_map.values()), []

        prior_headlines = prior_headlines or {}
        matchable_old = {
            cid: info
            for cid, info in existing_clusters.items()
            if info.get("status") != "inactive"
        }

        # New cluster descriptions — topics + sample titles (no label yet)
        new_sections = []
        for hdbscan_id, info in sorted(real_new.items()):
            topics = ", ".join(info.get("top_topics", [])[:5])
            titles = info.get("latest_titles", [])[:2]
            titles_str = "; ".join(f'"{t}"' for t in titles) if titles else "none"
            new_sections.append(
                f'  Cluster {hdbscan_id}: topics=[{topics}] | titles=[{titles_str}]'
            )

        # Existing thread descriptions — registry label + prior headlines
        existing_sections = []
        for cid, info in sorted(matchable_old.items()):
            recent = prior_headlines.get(cid, [])
            recent_str = (
                ", ".join(f'{h["week"]}: "{h["headline"]}"' for h in recent[-2:])
                if recent
                else "no prior headlines"
            )
            existing_sections.append(
                f'  Thread {cid}: "{info["label"]}" — {recent_str}'
            )

        new_ids_list = sorted(real_new.keys())
        prompt = f"""You are matching this week's news clusters to existing ongoing story threads.

THIS WEEK'S CLUSTERS (described by topics and sample video titles):
{chr(10).join(new_sections)}

EXISTING STORY THREADS (described by their label and recent headlines):
{chr(10).join(existing_sections)}

TASK: For each cluster, decide:
- MATCH <thread_id> — if this cluster is clearly the same ongoing story as an existing thread
- NEW — if this is a genuinely new story not covered by any existing thread

Rules:
- Each cluster maps to at most one thread; each thread matches at most one cluster
- Be conservative: only MATCH if the story is clearly the same ongoing arc (same actors, same event)
- A cluster on a broad topic (e.g. "US Economy") is NOT a match for a specific thread (e.g. "Trump Tariff War") unless the titles confirm it's the same story

Return ONLY valid JSON, no markdown:
{{"matches": {{{", ".join(f'"{cid}": "MATCH <id> or NEW"' for cid in new_ids_list)}}}}}"""

        id_map: dict[int, int] = {}
        used_old: set[int] = set()

        for attempt in range(4 * len(self._genai_api_keys)):
            try:
                response = self._genai_client.models.generate_content(
                    model=settings.gemini_model_id,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    ),
                )
                time.sleep(0.5)
                raw = (getattr(response, "text", "") or str(response)).strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                parsed = json.loads(raw)
                matches = parsed.get("matches", {})

                for hdbscan_id in sorted(real_new.keys()):
                    decision = str(matches.get(str(hdbscan_id), "NEW")).strip()

                    if decision.upper().startswith("MATCH"):
                        parts = decision.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            old_cid = int(parts[1])
                            if old_cid in matchable_old and old_cid not in used_old:
                                id_map[hdbscan_id] = old_cid
                                used_old.add(old_cid)
                                logger.info(
                                    f"GEMINI_MATCH hdbscan={hdbscan_id} → stable={old_cid} "
                                    f"label={matchable_old[old_cid]['label']!r}"
                                )
                                continue
                    logger.info(
                        f"GEMINI_NEW_CLUSTER hdbscan={hdbscan_id} "
                        f"topics={real_new[hdbscan_id].get('top_topics', [])[:3]}"
                    )
                break

            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = (
                    "429" in exc_str
                    or "RESOURCE_EXHAUSTED" in exc_str
                    or getattr(exc, "code", None) == 429
                )
                if is_rate_limit and self._rotate_key():
                    continue
                logger.error(f"GEMINI_MATCH_ERROR attempt={attempt} error={exc}")
                if attempt >= 3:
                    break

        # Mint new stable IDs for unmatched clusters (no gap-filling)
        all_taken = set(existing_clusters.keys())
        new_ids = []
        next_id = (max(all_taken) + 1) if all_taken else 0
        for hdbscan_id in sorted(real_new.keys()):
            if hdbscan_id not in id_map:
                id_map[hdbscan_id] = next_id
                new_ids.append(next_id)
                all_taken.add(next_id)
                logger.info(
                    f"STABLE_NEW_ID hdbscan={hdbscan_id} → stable={next_id} "
                    f"topics={real_new[hdbscan_id].get('top_topics', [])[:3]}"
                )
                next_id += 1

        # Existing threads with no match this week → declining
        active_old = {
            cid
            for cid, info in existing_clusters.items()
            if info.get("status") in ("active", "new")
        }
        declined_ids = sorted(active_old - used_old)
        for cid in declined_ids:
            logger.info(
                f"STABLE_DECLINED cluster={cid} "
                f"label={existing_clusters[cid]['label']!r}"
            )

        return id_map, new_ids, declined_ids

    # ── Step 3: Gemini labeling with match context ────────────────────────────

    def label_clusters(
        self,
        raw_cluster_info: dict[int, dict],
        match_results: dict[int, dict],
        dry_run: bool = False,
    ) -> dict[int, dict]:
        """
        Gemini labeling with authoritative match context.

        match_results: {hdbscan_id: {"is_new": bool, "prior_headlines": [...]}}
          - MATCH clusters get prior headlines injected as CONTINUING STORY context
          - NEW clusters get a fresh prompt with no prior context

        Fills in label, narrative_headline, narrative_summary, and week_overviews
        on each week_data entry. Returns updated cluster_info.
        """
        cluster_info = {k: dict(v) for k, v in raw_cluster_info.items()}
        real_cids = [c for c in cluster_info if c != -1]

        if dry_run:
            for cid in real_cids:
                cluster_info[cid]["label"] = f"Cluster {cid}"
                cluster_info[cid]["narrative_headline"] = "[Gemini would generate a newspaper headline]"
                cluster_info[cid]["narrative_summary"] = "[Gemini would generate a one-sentence summary with stats]"
                for wd in cluster_info[cid]["week_data"]:
                    wd["week_overview"] = "[Gemini would generate a 2-sentence week overview]"
            logger.info(
                f"LABEL_DRY_RUN skipping Gemini for {len(real_cids)} clusters"
            )
            return cluster_info

        for cid in real_cids:
            info = cluster_info[cid]
            match_ctx = match_results.get(cid, {"is_new": True, "prior_headlines": []})
            is_new = match_ctx["is_new"]
            prior_headlines = match_ctx.get("prior_headlines", [])

            top_topics = info.get("top_topics", [])
            dominant_sentiment = info.get("dominant_sentiment", "neutral")
            latest_week = info.get("latest_week")
            latest_titles = info.get("latest_titles", [])
            latest_claims = info.get("latest_claims", [])

            # Week context for week_overviews
            week_context = []
            for wd in info["week_data"]:
                week_sentiments = wd.get("sentiment_breakdown", {})
                dominant_week_sentiment = (
                    max(week_sentiments, key=week_sentiments.get)
                    if week_sentiments
                    else "neutral"
                )
                week_context.append(
                    {
                        "week": wd["week"],
                        "video_count": wd["video_count"],
                        "view_count": wd["view_count"],
                        "breaking_count": wd["breaking_count"],
                        "dominant_sentiment": dominant_week_sentiment,
                    }
                )

            latest_week_section = ""
            if latest_week and (latest_titles or latest_claims):
                current_wd = next(
                    (wd for wd in info["week_data"] if wd["week"] == latest_week), {}
                )
                week_stats = (
                    f"{current_wd.get('video_count', 0)} videos, "
                    f"{current_wd.get('view_count', 0):,} views"
                )
                latest_week_section = f"""
THIS WEEK'S COVERAGE ({latest_week}) — write the headline and summary from THESE ONLY:
Titles: {latest_titles}
Claims: {latest_claims}
Stats: {week_stats}"""

            continuing_section = ""
            if not is_new and prior_headlines:
                lines = "\n".join(
                    f'  - {p["week"]}: "{p["headline"]}"' for p in prior_headlines
                )
                continuing_section = f"""
CONTINUING STORY — prior headlines for this thread:
{lines}

This is an ongoing story. The new headline should reflect the latest development,
NOT re-summarise the whole arc. If there is no new angle this week, start with
"Continuing:" and describe what is still unfolding. Do NOT repeat any prior headline verbatim."""

            prompt = f"""You are a news editor writing narrative labels for topic clusters.
Given topics, claims, and video titles from a YouTube cluster, generate:

1. label: A 3-6 word desk label in Title Case. NOT a headline or sentence — no verbs, no articles like "The". Think of it as a category tag on a news desk.
TOO BROAD: "Oil Markets", "Middle East Conflict"
TOO SPECIFIC: "Starmer Warned Over Mandelson Ties", "The Collapse Of Olaplex"
GOOD: "US-Iran Military Escalation", "UK Epstein Political Scandal", "England Squad Overhaul", "Olaplex Market Value Crisis", "Aviation Safety Funding Crisis"
If topics seem unrelated, focus on the dominant theme.
2. headline: A full newspaper headline, 8-14 words. Base it ONLY on this week's titles and claims — do not summarise the full story history.
3. summary: One sentence focused on this week's angle, include a specific stat or data point if available.
4. week_overviews: For each week listed in the week data below, write a 2-sentence plain-English
   overview of what was happening with this story that week. Sentence 1: the main development or
   focus of coverage. Sentence 2: scale or sentiment context (e.g. how many outlets covered it,
   whether coverage was growing or fading, the dominant tone). Only write overviews for weeks
   that have video_count > 0; set the value to "" for weeks with no coverage.
{continuing_section}
{latest_week_section}
All topics (for label only): {top_topics}
Dominant sentiment: {dominant_sentiment}
Week data (for week_overviews): {week_context}

Return ONLY valid JSON, no markdown:
{{"label": "...", "headline": "...", "summary": "...", "week_overviews": {{"week1": "...", "week2": "..."}}}}"""  # noqa: E501

            max_attempts = 6 * len(self._genai_api_keys)
            for attempt in range(max_attempts):
                try:
                    response = self._genai_client.models.generate_content(
                        model=settings.gemini_model_id,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level="low")
                        ),
                    )
                    time.sleep(0.5)
                    raw = getattr(response, "text", "") or str(response)
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw)

                    parsed = json.loads(raw)
                    fallback_label = top_topics[0] if top_topics else f"Cluster {cid}"
                    cluster_info[cid]["label"] = parsed.get("label", fallback_label)
                    cluster_info[cid]["narrative_headline"] = parsed.get("headline")
                    cluster_info[cid]["narrative_summary"] = parsed.get("summary")

                    week_overviews: dict = parsed.get("week_overviews") or {}
                    for wd in cluster_info[cid]["week_data"]:
                        overview = week_overviews.get(wd["week"], "")
                        wd["week_overview"] = str(overview) if overview else ""

                    logger.info(
                        f"GEMINI_LABEL cluster={cid} is_new={is_new} "
                        f"label={cluster_info[cid]['label']!r} "
                        f"week_overviews={list(week_overviews.keys())}"
                    )
                    break

                except Exception as exc:
                    exc_str = str(exc)
                    is_rate_limit = (
                        "429" in exc_str
                        or "RESOURCE_EXHAUSTED" in exc_str
                        or getattr(exc, "code", None) == 429
                    )
                    is_server_error = (
                        "503" in exc_str
                        or "500" in exc_str
                        or "UNAVAILABLE" in exc_str
                        or getattr(exc, "code", None) in (500, 503)
                    )

                    if is_rate_limit:
                        if self._rotate_key():
                            logger.warning(
                                f"GEMINI_KEY_ROTATED cluster={cid} "
                                f"attempt={attempt+1} retrying"
                            )
                            continue
                        logger.warning(
                            "ALL_KEYS_EXHAUSTED resetting to key 0 and waiting 60s"
                        )
                        self._genai_key_index = 0
                        self._genai_client = genai.Client(
                            api_key=self._genai_api_keys[0]
                        )
                        time.sleep(60)
                        continue

                    if is_server_error:
                        wait = min(5 * (attempt + 1), 30)
                        logger.warning(
                            f"GEMINI_LABEL_503 cluster={cid} "
                            f"attempt={attempt+1} waiting {wait}s before retry"
                        )
                        time.sleep(wait)
                        continue

                    logger.error(f"GEMINI_LABEL_FATAL cluster={cid} error={exc}")
                    break

        return cluster_info

    # ── ID remapping ─────────────────────────────────────────────────────────

    @staticmethod
    def remap_cluster_info(
        cluster_info: dict[int, dict],
        id_map: dict[int, int],
    ) -> dict[int, dict]:
        """Replace HDBSCAN labels with stable cluster IDs."""
        remapped = {}
        for hdbscan_id, info in cluster_info.items():
            if hdbscan_id == -1:
                remapped[-1] = info
                continue
            stable_id = id_map.get(hdbscan_id, hdbscan_id)
            remapped[stable_id] = info
        return remapped
