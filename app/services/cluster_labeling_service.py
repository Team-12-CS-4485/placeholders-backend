"""
cluster_labeling_service.py - Cluster Labeling & Matching

Handles cluster intelligence:
- TF-IDF fallback labeling from topic fields
- Gemini enrichment per cluster (label, headline, summary, week overviews)
- Stable cluster matching via embedding similarity
- Cluster ID remapping from HDBSCAN labels to persistent IDs

Extracted from clustering_service.py to separate labeling/matching
concerns from the UMAP/HDBSCAN pipeline and DynamoDB writes.
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

    # ── TF-IDF + Gemini labeling ────────────────────────────────────────────

    def label_clusters(
        self,
        video_ids,
        labels,
        meta_map,
        dry_run=False,
        existing_clusters=None,
        cluster_weeks_table=None,
    ):
        cluster_members: dict[int, list[str]] = {}
        for vid, label in zip(video_ids, labels):
            cluster_members.setdefault(int(label), []).append(vid)

        cluster_topic_counts: dict[int, Counter] = {}
        cluster_stats: dict[int, dict] = {}
        cluster_claims: dict[int, list[str]] = {}
        cluster_titles: dict[int, list[str]] = {}

        for cid, vids in cluster_members.items():
            tc = Counter()
            channels = set()
            categories = Counter()
            sentiments = Counter()
            public_sentiments = Counter()
            public_sentiment_scores = []
            breaking = views = likes = comments = 0
            claims = []
            titles = []
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
                if m.get("title"):
                    titles.append(m["title"])
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
                        # week_overview is populated below after Gemini call
                        "week_overview": "",
                    }
                )

            # Compute average public sentiment score for the cluster
            avg_public_score = (
                round(
                    sum(public_sentiment_scores) / len(public_sentiment_scores),
                    3,
                )
                if public_sentiment_scores
                else 0.0
            )

            # Detect creator vs audience divergence
            dominant_creator = (
                sentiments.most_common(1)[0][0] if sentiments else "neutral"
            )
            dominant_public = (
                public_sentiments.most_common(1)[0][0]
                if public_sentiments
                else "neutral"
            )
            sentiment_divergence = dominant_creator != dominant_public

            # Extract latest week's titles and claims for the Gemini prompt
            latest_week_name = max(
                (w for w in week_buckets if w.startswith("week") and w[4:].isdigit()),
                key=lambda w: int(w[4:]),
                default=None,
            )
            if latest_week_name and latest_week_name in week_buckets:
                latest_vids = week_buckets[latest_week_name]
                latest_titles = [v["title"] for v in latest_vids if v.get("title")][:5]
                latest_claims = list(
                    dict.fromkeys(
                        c for v in latest_vids for c in v.get("key_claims", [])
                    )
                )[:5]
            else:
                latest_week_name = None
                latest_titles = []
                latest_claims = []

            cluster_topic_counts[cid] = tc
            cluster_claims[cid] = list(dict.fromkeys(claims))[:5]
            cluster_titles[cid] = titles[:5]
            cluster_stats[cid] = {
                "vids": vids,
                "channels": channels,
                "categories": categories,
                "sentiments": sentiments,
                "public_sentiments": public_sentiments,
                "avg_public_sentiment_score": avg_public_score,
                "sentiment_divergence": sentiment_divergence,
                "breaking": breaking,
                "views": views,
                "likes": likes,
                "comments": comments,
                "week_data": week_data,
                "latest_week": latest_week_name,
                "latest_titles": latest_titles,
                "latest_claims": latest_claims,
            }

        # ── TF-IDF scoring (kept as fallback) ──
        real_cids = [c for c in cluster_members if c != -1]
        n_clusters = max(len(real_cids), 1)
        topic_cluster_count: Counter = Counter()
        for cid in real_cids:
            for topic in cluster_topic_counts[cid]:
                topic_cluster_count[topic] += 1

        cluster_scored: dict[int, list] = {}
        for cid in cluster_members:
            tc = cluster_topic_counts[cid]
            size = len(cluster_members[cid])
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

        # TF-IDF fallback labels
        used = set()
        tfidf_labels: dict[int, str] = {}
        if -1 in cluster_members:
            tfidf_labels[-1] = "Unclustered"

        for cid in sorted(
            real_cids, key=lambda c: len(cluster_members[c]), reverse=True
        ):
            chosen = None
            for topic, _ in cluster_scored[cid]:
                if topic not in used:
                    chosen = topic
                    break
            if not chosen:
                top_two = [t[0] for t in cluster_scored[cid][:2]]
                chosen = " & ".join(top_two) if len(top_two) == 2 else f"Cluster {cid}"
            tfidf_labels[cid] = chosen
            used.add(chosen)

        # ── Preliminary stable match for prior-headline lookup ──
        # Runs before Gemini so we can fetch prior week headlines from cluster-weeks.
        # The real stable match still runs in clustering_service.py after label_clusters returns.
        preliminary_id_map: dict[int, int] = {}
        if existing_clusters:
            tfidf_cluster_info = {
                cid: {
                    "label": tfidf_labels[cid],
                    "top_topics": [
                        t[0] for t in cluster_topic_counts[cid].most_common(5)
                    ],
                }
                for cid in real_cids
            }
            preliminary_id_map, _, _ = self.match_to_existing_clusters(
                tfidf_cluster_info, existing_clusters
            )

        # ── Gemini enrichment per cluster (skipped in dry_run) ──
        cluster_labels: dict[int, str] = dict(tfidf_labels)
        cluster_narratives: dict[int, dict] = {}
        dupe_retried_cids: set[int] = set()

        if dry_run:
            for cid in real_cids:
                cluster_narratives[cid] = {
                    "headline": "[Gemini would generate a newspaper headline]",
                    "summary": "[Gemini would generate a one-sentence summary with stats]",
                }
                for wd in cluster_stats[cid]["week_data"]:
                    wd["week_overview"] = (
                        "[Gemini would generate a 2-sentence week overview]"
                    )
            logger.info(
                f"LABEL_DRY_RUN skipping Gemini — using TF-IDF labels for {len(real_cids)} clusters"
            )

        for cid in real_cids if not dry_run else []:
            top_topics = [t[0] for t in cluster_topic_counts[cid].most_common(5)]
            claims = cluster_claims.get(cid, [])
            titles = cluster_titles.get(cid, [])
            dominant_sentiment = (
                cluster_stats[cid]["sentiments"].most_common(1)[0][0]
                if cluster_stats[cid]["sentiments"]
                else "neutral"
            )
            latest_week = cluster_stats[cid]["latest_week"]
            latest_titles = cluster_stats[cid]["latest_titles"]
            latest_claims = cluster_stats[cid]["latest_claims"]

            # Look up prior headlines for this cluster from cluster-weeks
            from app.services.dynamo_service import DynamoService

            stable_cid = preliminary_id_map.get(cid)
            prior_headlines: list[dict] = []
            if stable_cid is not None:
                prior_headlines = DynamoService().get_cluster_week_headlines(
                    stable_cid, last_n=4
                )

            # Build a compact week context to pass into the prompt
            week_context = []
            for wd in cluster_stats[cid]["week_data"]:
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
                latest_week_section = f"""
    MOST RECENT WEEK ({latest_week}) — weight this most heavily for the headline and summary:
    Titles: {latest_titles}
    Claims: {latest_claims}"""

            prior_headlines_section = ""
            if prior_headlines:
                lines = "\n".join(
                    f'- {p["week"]}: "{p["headline"]}"' for p in prior_headlines
                )
                prior_headlines_section = f"""
    PRIOR HEADLINES FOR THIS CLUSTER (do NOT repeat or rephrase any of these):
    {lines}

    The new headline MUST describe a different angle or development than all of the above.
    If no genuinely new development exists, describe what continued — do NOT prefix with "Week N:" or any week label."""

            prompt = f"""You are a news editor writing narrative labels for topic clusters.
    Given these topics, claims, and video titles from a cluster of YouTube videos, generate:

    1. label: A 3-6 word desk label in Title Case. NOT a headline or sentence — no verbs, no articles like "The". Think of it as a category tag on a news desk.
    TOO BROAD: "Oil Markets", "Middle East Conflict"
    TOO SPECIFIC: "Starmer Warned Over Mandelson Ties", "The Collapse Of Olaplex"
    GOOD: "US-Iran Military Escalation", "UK Epstein Political Scandal", "England Squad Overhaul", "Olaplex Market Value Crisis", "Aviation Safety Funding Crisis"
    If topics seem unrelated, focus on the dominant theme.
    2. headline: A full newspaper headline, 8-14 words. Must reflect the LATEST developments, not the full story history.
    3. summary: One sentence focused on the most recent angle, include a specific stat or data point if available.
    4. week_overviews: For each week listed in the week data below, write a 2-sentence plain-English
       overview of what was happening with this story that week. Sentence 1: the main development or
       focus of coverage. Sentence 2: scale or sentiment context (e.g. how many outlets covered it,
       whether coverage was growing or fading, the dominant tone). Only write overviews for weeks
       that have video_count > 0; set the value to "" for weeks with no coverage.
    {latest_week_section}
    {prior_headlines_section}
    All topics (for label only): {top_topics}
    Historical claims (background): {claims}
    Historical video titles (background): {titles}
    Dominant sentiment: {dominant_sentiment}
    Week data: {week_context}

    Return ONLY valid JSON, no markdown:
    {{"label": "...", "headline": "...", "summary": "...", "week_overviews": {{"week1": "...", "week2": "..."}}}}"""  # noqa: E501

            max_attempts = 6 * len(self._genai_api_keys)
            for attempt in range(max_attempts):
                try:
                    response = self._genai_client.models.generate_content(
                        model=settings.gemini_model_id,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(
                                thinking_level="low",
                            )
                        ),
                    )
                    time.sleep(0.5)
                    raw = getattr(response, "text", "") or str(response)
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw)

                    parsed = json.loads(raw)

                    # Verbatim dupe check — retry once if headline matches a prior week
                    new_hl = parsed.get("headline", "").strip().lower()
                    recent = [
                        p["headline"].strip().lower() for p in prior_headlines[-2:]
                    ]
                    if new_hl in recent and cid not in dupe_retried_cids:
                        dupe_retried_cids.add(cid)
                        prompt += (
                            "\n\nIMPORTANT: Your previous attempt returned the exact same "
                            "headline as a prior week. You MUST write something completely different."
                        )
                        continue

                    cluster_labels[cid] = parsed.get("label", tfidf_labels[cid])
                    cluster_narratives[cid] = {
                        "headline": parsed.get("headline"),
                        "summary": parsed.get("summary"),
                    }

                    week_overviews: dict = parsed.get("week_overviews") or {}
                    for wd in cluster_stats[cid]["week_data"]:
                        week_name = wd["week"]
                        overview = week_overviews.get(week_name, "")
                        wd["week_overview"] = str(overview) if overview else ""

                    logger.info(
                        f"GEMINI_LABEL cluster={cid} "
                        f"label={cluster_labels[cid]!r} "
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
                            "ALL_KEYS_EXHAUSTED resetting to key 0 " "and waiting 60s"
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
                            f"attempt={attempt+1} "
                            f"waiting {wait}s before retry"
                        )
                        time.sleep(wait)
                        continue

                    # Non-retryable error — log and use TF-IDF fallback
                    logger.error(f"GEMINI_LABEL_FATAL cluster={cid} error={exc}")
                    break

        # ── Build cluster_info ──
        cluster_info = {}
        for cid, vids in cluster_members.items():
            stats = cluster_stats[cid]
            top_topics = cluster_topic_counts[cid].most_common(5)
            narr = cluster_narratives.get(cid, {})
            cluster_info[cid] = {
                "label": cluster_labels[cid],
                "narrative_headline": narr.get("headline"),
                "narrative_summary": narr.get("summary"),
                "video_count": len(vids),
                "video_ids": vids,
                "channels": sorted(stats["channels"]),
                "channel_count": len(stats["channels"]),
                "top_topics": [t[0] for t in top_topics],
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
                # week_data now contains week_overview on each entry
                "week_data": stats["week_data"],
                "top_claims": cluster_claims.get(cid, []),
            }
            logger.debug(
                f"CLUSTER_LABELED id={cid} "
                f"label={cluster_labels[cid]!r} videos={len(vids)}"
            )

        return cluster_info

    # ── Stable cluster matching ────────────────────────────────────────────

    def load_existing_clusters(self, clusters_table) -> dict[int, dict]:
        """
        Scan narrative-clusters table and return
        {cluster_id: {top_topics, label, created_at, status}}.
        """
        existing = {}
        scan_kwargs: dict = {}
        while True:
            resp = clusters_table.scan(**scan_kwargs)
            for item in resp["Items"]:
                cid = int(item["cluster_id"])
                existing[cid] = {
                    "top_topics": list(item.get("top_topics", [])),
                    "label": item.get("cluster_label", ""),
                    "created_at": item.get("created_at"),
                    "status": item.get("status", "active"),
                }
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        logger.debug(f"STABLE_MATCH_LOADED existing_clusters={len(existing)}")
        return existing

    def embed_cluster_description(self, label: str, topics: list[str]) -> list[float]:
        """Embed a cluster's label + topics into a single vector."""
        text = f"{label} {' '.join(topics)}"
        return self._chunker.embed([text], is_query=False)[0]

    @staticmethod
    def cosine_similarity(vec_a, vec_b) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def match_to_existing_clusters(
        self,
        new_cluster_info: dict[int, dict],
        existing_clusters: dict[int, dict],
        threshold: float = 0.75,
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """
        Match new HDBSCAN clusters to existing stable clusters using
        embedding similarity on label + topics.

        Returns:
            id_map: {hdbscan_label: stable_cluster_id}
            new_ids: list of stable IDs assigned to genuinely new clusters
            declined_ids: list of existing cluster IDs that had no match
        """
        if not existing_clusters:
            # First run — use HDBSCAN labels as-is
            id_map = {cid: cid for cid in new_cluster_info if cid != -1}
            return id_map, list(id_map.values()), []

        real_new = {cid: info for cid, info in new_cluster_info.items() if cid != -1}

        # Match against active, new, and declining clusters
        matchable_old = {
            cid: info
            for cid, info in existing_clusters.items()
            if info.get("status") != "inactive"
        }
        old_embeddings = {}
        for old_cid, old_info in matchable_old.items():
            old_embeddings[old_cid] = self.embed_cluster_description(
                old_info["label"], old_info["top_topics"]
            )

        # Embed all new clusters
        new_embeddings = {}
        for new_cid, new_info in real_new.items():
            new_embeddings[new_cid] = self.embed_cluster_description(
                new_info["label"], new_info["top_topics"]
            )

        # Score all (new, existing) pairs
        scores: list[tuple[float, int, int]] = []
        for new_cid in real_new:
            for old_cid in matchable_old:
                sim = self.cosine_similarity(
                    new_embeddings[new_cid], old_embeddings[old_cid]
                )
                if sim >= threshold:
                    scores.append((sim, new_cid, old_cid))

        # Greedy 1-to-1 matching: best score first
        scores.sort(reverse=True)
        id_map: dict[int, int] = {}
        used_old: set[int] = set()
        used_new: set[int] = set()

        for sim, new_cid, old_cid in scores:
            if new_cid in used_new or old_cid in used_old:
                continue
            id_map[new_cid] = old_cid
            used_new.add(new_cid)
            used_old.add(old_cid)
            logger.debug(
                f"STABLE_MATCH hdbscan={new_cid} → stable={old_cid} "
                f"sim={sim:.2f} "
                f"label={existing_clusters[old_cid]['label']!r}"
            )

        # Assign new IDs for unmatched new clusters — fill lowest available gap
        all_taken = set(existing_clusters.keys())
        new_ids = []
        next_id = 0
        for new_cid in real_new:
            if new_cid not in id_map:
                while next_id in all_taken:
                    next_id += 1
                id_map[new_cid] = next_id
                new_ids.append(next_id)
                all_taken.add(next_id)
                logger.info(
                    f"STABLE_NEW_CLUSTER hdbscan={new_cid} "
                    f"→ stable={next_id} "
                    f"topics={real_new[new_cid]['top_topics']}"
                )
                next_id += 1

        # Existing clusters that weren't matched → declining
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
