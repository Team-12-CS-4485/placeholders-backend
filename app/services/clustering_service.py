"""
clustering_service.py - Narrative Clustering Service

Groups related videos into narrative clusters using HDBSCAN
on Qdrant embeddings. Writes results to DynamoDB.

Flow:
1. Pull vectors from Qdrant (search data only)
2. Pull metadata from DynamoDB (topics, channel, etc.)
3. Deduplicate to one representative per video (mean-pool vectors)
4. UMAP (768d → 15d) + HDBSCAN clustering
5. Label each cluster using TF-IDF on existing topic fields
6. Write cluster_id/label/confidence to DynamoDB youtube-videos
7. Write cluster summary to DynamoDB narrative-clusters table

Zero Gemini calls. Reads vectors from Qdrant, everything else from DynamoDB.

Usage:
    python -m scripts.run_clustering
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from google import genai
from google.genai import types
import json
import time
import numpy as np
import boto3

from app.core.config import settings
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_REGION", "us-east-2")


def _dec(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def _clean_for_dynamo(obj):
    """Recursively convert floats/ints to Decimal and remove None values."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return Decimal(str(round(obj, 4)))
    if isinstance(obj, int):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _clean_for_dynamo(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_clean_for_dynamo(i) for i in obj if i is not None]
    return obj


class ClusteringService:

    def __init__(self, vector_service: Optional[VectorService] = None):
        self.vector_service = vector_service or VectorService()
        self.collection_name = self.vector_service.collection_name
        self.client = self.vector_service.client

        self._dynamodb = boto3.resource("dynamodb", region_name=REGION)
        self._videos_table = self._dynamodb.Table(settings.dynamodb_table)
        self._clusters_table = self._dynamodb.Table("narrative-clusters")
        self._genai_api_keys = settings.genai_api_keys
        self._genai_key_index = 0
        self._genai_client = genai.Client(api_key=self._genai_api_keys[0])

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
            f"API_KEY_ROTATED key_index={self._genai_key_index}/{len(self._genai_api_keys)-1}"
        )
        return True

    # ── Step 1: Pull vectors from Qdrant ─────────────────────────────────────

    def _scroll_vectors(self) -> dict[str, list[dict]]:
        """
        Scroll Qdrant for vectors grouped by video.
        Returns {video_id: [{vector}, ...]}
        """
        video_chunks: dict[str, list[dict]] = {}
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_vectors=True,
                with_payload=["transcript_index"],
            )
            for point in results:
                vid = point.payload.get("transcript_index", "")
                if vid:
                    video_chunks.setdefault(vid, []).append(
                        {
                            "vector": point.vector,
                        }
                    )
            if offset is None:
                break

        logger.info(f"CLUSTER_SCROLL videos={len(video_chunks)}")
        return video_chunks

    # ── Step 2: Pull metadata from DynamoDB ──────────────────────────────────

    def _get_video_metadata(self) -> dict[str, dict]:
        """
        Scan DynamoDB for all videos with intelligence data.
        Returns {video_id: {channel, topics, category, sentiment, ...}}
        """
        meta = {}
        scan_kwargs = {
            "ProjectionExpression": "PartitionKey, SortKey, channel, topics, category, "
            "sentiment, is_breaking, viewCount, likeCount, commentCount, "
            "title, publishedAt, #wk, source_key, key_claims, "
            "public_sentiment, public_sentiment_score",
            "ExpressionAttributeNames": {"#wk": "week"},
        }

        while True:
            resp = self._videos_table.scan(**scan_kwargs)
            for item in resp["Items"]:
                vid = item.get("SortKey", "")
                if vid and item.get("topics"):
                    meta[vid] = {
                        "channel": item.get("channel") or item.get("PartitionKey", ""),
                        "topics": item.get("topics", []),
                        "category": item.get("category", "Other"),
                        "sentiment": item.get("sentiment", "neutral"),
                        "is_breaking": bool(item.get("is_breaking", False)),
                        "view_count": int(item.get("viewCount") or 0),
                        "like_count": int(item.get("likeCount") or 0),
                        "comment_count": int(item.get("commentCount") or 0),
                        "title": item.get("title", ""),
                        "week": item.get("week", "unknown"),
                        "source_key": item.get("source_key", ""),
                        "key_claims": item.get("key_claims", []),
                        "public_sentiment": item.get("public_sentiment", "neutral"),
                        "public_sentiment_score": float(
                            item.get("public_sentiment_score") or 0
                        ),
                    }
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        logger.info(f"CLUSTER_DYNAMO_META videos_with_intel={len(meta)}")
        return meta

    # ── Step 3: Build video representatives ──────────────────────────────────

    def _get_video_representatives(
        self,
        video_chunks: dict[str, list[dict]],
        meta_map: dict[str, dict],
    ) -> tuple[list[str], np.ndarray, dict[str, dict]]:
        video_ids = []
        vectors = []
        filtered_meta = {}

        for vid, chunks in video_chunks.items():
            if vid not in meta_map:
                continue
            chunk_vecs = [c["vector"] for c in chunks]
            mean_vec = np.mean(chunk_vecs, axis=0)
            vectors.append(mean_vec)
            video_ids.append(vid)
            filtered_meta[vid] = meta_map[vid]

        matrix = np.array(vectors, dtype=np.float32) if vectors else np.array([])
        logger.info(f"CLUSTER_REPRESENTATIVES shape={matrix.shape}")
        return video_ids, matrix, filtered_meta

    # ── Step 4: UMAP + HDBSCAN ──────────────────────────────────────────────

    def _reduce_dimensions(self, matrix, n_components=15, n_neighbors=15, min_dist=0.0):
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, len(matrix) - 1),
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(matrix)
        logger.info(f"UMAP_COMPLETE {matrix.shape[1]}d → {n_components}d")
        return reduced

    def _cluster_vectors(
        self,
        matrix,
        min_cluster_size=7,
        min_samples=2,
        umap_components=15,
        umap_neighbors=15,
    ):
        reduced = self._reduce_dimensions(
            matrix, n_components=umap_components, n_neighbors=umap_neighbors
        )
        try:
            from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

            clusterer = SklearnHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                store_centers="centroid",
            )
            clusterer.fit(reduced)
            labels = clusterer.labels_
            probabilities = getattr(clusterer, "probabilities_", np.ones(len(labels)))
        except ImportError:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
            )
            clusterer.fit(reduced)
            labels = clusterer.labels_
            probabilities = clusterer.probabilities_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        logger.info(f"HDBSCAN_COMPLETE clusters={n_clusters} noise={n_noise}")
        return labels, probabilities

    def _label_clusters(self, video_ids, labels, meta_map):
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
                    }
                )

            # Compute average public sentiment score for the cluster
            avg_public_score = (
                round(sum(public_sentiment_scores) / len(public_sentiment_scores), 3)
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

            cluster_topic_counts[cid] = tc
            cluster_claims[cid] = list(dict.fromkeys(claims))[:5]  # dedupe, top 5
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

        # ── Gemini enrichment per cluster ──
        cluster_labels: dict[int, str] = dict(tfidf_labels)
        cluster_narratives: dict[int, dict] = {}

        for cid in real_cids:
            top_topics = [t[0] for t in cluster_topic_counts[cid].most_common(5)]
            claims = cluster_claims.get(cid, [])
            titles = cluster_titles.get(cid, [])
            dominant_sentiment = (
                cluster_stats[cid]["sentiments"].most_common(1)[0][0]
                if cluster_stats[cid]["sentiments"]
                else "neutral"
            )

            prompt = f"""You are a news editor writing narrative labels for topic clusters.
    Given these topics, claims, and video titles from a cluster of YouTube videos, generate:

    1. label: A 3-6 word desk label in Title Case. NOT a headline or sentence — no verbs, no articles like "The". Think of it as a category tag on a news desk.
    TOO BROAD: "Oil Markets", "Middle East Conflict"
    TOO SPECIFIC: "Starmer Warned Over Mandelson Ties", "The Collapse Of Olaplex"
    GOOD: "US-Iran Military Escalation", "UK Epstein Political Scandal", "England Squad Overhaul", "Olaplex Market Value Crisis", "Aviation Safety Funding Crisis"
    If topics seem unrelated, focus on the dominant theme.
    2. headline: A full newspaper headline, 8-14 words
    3. summary: One sentence, include a specific stat or data point if available from the claims

    Topics: {top_topics}
    Claims: {claims}
    Video titles: {titles}
    Dominant sentiment: {dominant_sentiment}

    Return ONLY valid JSON, no markdown: {{"label": "...", "headline": "...", "summary": "..."}}"""

            for attempt in range(len(self._genai_api_keys) + 1):
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
                    time.sleep(0.5)  # pace RPM across keys
                    raw = getattr(response, "text", "") or str(response)
                    raw = raw.strip()
                    # strip markdown fences if present
                    if raw.startswith("```"):
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw)

                    parsed = json.loads(raw)
                    cluster_labels[cid] = parsed.get("label", tfidf_labels[cid])
                    cluster_narratives[cid] = {
                        "headline": parsed.get("headline"),
                        "summary": parsed.get("summary"),
                    }
                    logger.info(
                        f"GEMINI_LABEL cluster={cid} label={cluster_labels[cid]!r}"
                    )
                    break
                except Exception as exc:
                    is_rate_limit = (
                        "429" in str(exc) or getattr(exc, "code", None) == 429
                    )
                    if is_rate_limit and self._rotate_key():
                        logger.warning(
                            f"GEMINI_KEY_ROTATED cluster={cid} attempt={attempt+1} retrying"
                        )
                        continue
                    if is_rate_limit:
                        logger.warning(
                            "ALL_KEYS_EXHAUSTED resetting to key 0 and waiting 60s"
                        )
                        self._genai_key_index = 0
                        self._genai_client = genai.Client(
                            api_key=self._genai_api_keys[0]
                        )
                        time.sleep(60)
                        continue
                    logger.warning(
                        f"GEMINI_LABEL_FAILED cluster={cid} error={exc} "
                        f"falling_back_to={tfidf_labels[cid]!r}"
                    )
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
                "week_data": stats["week_data"],
                "top_claims": cluster_claims.get(cid, []),
            }
            logger.info(
                f"CLUSTER_LABELED id={cid} label={cluster_labels[cid]!r} videos={len(vids)}"
            )

        return cluster_info

    # ── Step 6: Write to DynamoDB ────────────────────────────────────────────

    def _write_clusters_to_dynamodb(
        self, video_ids, labels, probabilities, cluster_info, meta_map
    ):
        updated = 0
        for vid, label, prob in zip(video_ids, labels, probabilities):
            label_int = int(label)
            confidence = round(float(prob), 3)
            cluster_label = cluster_info.get(label_int, {}).get("label", "Unclustered")
            channel = meta_map.get(vid, {}).get("channel", "")
            if not channel:
                continue

            if label_int == -1:
                self._videos_table.update_item(
                    Key={"PartitionKey": channel, "SortKey": vid},
                    UpdateExpression="SET #clabel = :clabel, #cconf = :cconf REMOVE #cid",
                    ExpressionAttributeNames={
                        "#cid": "cluster_id",
                        "#clabel": "cluster_label",
                        "#cconf": "cluster_confidence",
                    },
                    ExpressionAttributeValues={
                        ":clabel": "Unclustered",
                        ":cconf": _dec(0),
                    },
                )
            else:
                self._videos_table.update_item(
                    Key={"PartitionKey": channel, "SortKey": vid},
                    UpdateExpression="SET #cid = :cid, #clabel = :clabel, #cconf = :cconf",
                    ExpressionAttributeNames={
                        "#cid": "cluster_id",
                        "#clabel": "cluster_label",
                        "#cconf": "cluster_confidence",
                    },
                    ExpressionAttributeValues={
                        ":cid": _dec(label_int),
                        ":clabel": cluster_label,
                        ":cconf": _dec(confidence),
                    },
                )
            updated += 1

        logger.info(f"DYNAMO_CLUSTER_WRITE videos={updated}")
        return updated

    # ── Stable cluster matching ────────────────────────────────────────────

    def _load_existing_clusters(self) -> dict[int, dict]:
        """
        Scan narrative-clusters table and return {cluster_id: {top_topics, label, created_at, status}}.
        """
        existing = {}
        scan_kwargs: dict = {}
        while True:
            resp = self._clusters_table.scan(**scan_kwargs)
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
        logger.info(f"STABLE_MATCH_LOADED existing_clusters={len(existing)}")
        return existing

    @staticmethod
    def _topic_jaccard(topics_a: list[str], topics_b: list[str]) -> float:
        """Jaccard similarity between two topic lists."""
        set_a = set(t.lower() for t in topics_a)
        set_b = set(t.lower() for t in topics_b)
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def _match_to_existing_clusters(
        self,
        new_cluster_info: dict[int, dict],
        existing_clusters: dict[int, dict],
        threshold: float = 0.3,
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """
        Match new HDBSCAN clusters to existing stable clusters by topic overlap.

        Returns:
            id_map: {hdbscan_label: stable_cluster_id}
            new_ids: list of stable IDs assigned to genuinely new clusters
            declined_ids: list of existing cluster IDs that had no match (inactive)
        """
        if not existing_clusters:
            # First run — use HDBSCAN labels as-is
            id_map = {cid: cid for cid in new_cluster_info if cid != -1}
            return id_map, list(id_map.values()), []

        real_new = {cid: info for cid, info in new_cluster_info.items() if cid != -1}

        # Score all (new, existing) pairs
        scores: list[tuple[float, int, int]] = []
        for new_cid, new_info in real_new.items():
            for old_cid, old_info in existing_clusters.items():
                if old_info.get("status") == "inactive":
                    continue
                sim = self._topic_jaccard(
                    new_info["top_topics"], old_info["top_topics"]
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
            logger.info(
                f"STABLE_MATCH hdbscan={new_cid} → stable={old_cid} "
                f"sim={sim:.2f} label={existing_clusters[old_cid]['label']!r}"
            )

        # Assign new IDs for unmatched new clusters
        max_existing_id = max(existing_clusters.keys()) if existing_clusters else -1
        next_id = max(max_existing_id, max(id_map.values(), default=-1)) + 1
        new_ids = []
        for new_cid in real_new:
            if new_cid not in id_map:
                id_map[new_cid] = next_id
                new_ids.append(next_id)
                logger.info(
                    f"STABLE_NEW_CLUSTER hdbscan={new_cid} → stable={next_id} "
                    f"topics={real_new[new_cid]['top_topics']}"
                )
                next_id += 1

        # Existing clusters that weren't matched → declining
        active_old = {
            cid
            for cid, info in existing_clusters.items()
            if info.get("status") != "inactive"
        }
        declined_ids = sorted(active_old - used_old)
        for cid in declined_ids:
            logger.info(
                f"STABLE_DECLINED cluster={cid} "
                f"label={existing_clusters[cid]['label']!r}"
            )

        return id_map, new_ids, declined_ids

    def _remap_cluster_info(
        self,
        cluster_info: dict[int, dict],
        id_map: dict[int, int],
    ) -> dict[int, dict]:
        """Replace HDBSCAN labels with stable cluster IDs in cluster_info."""
        remapped = {}
        for hdbscan_id, info in cluster_info.items():
            if hdbscan_id == -1:
                remapped[-1] = info
                continue
            stable_id = id_map.get(hdbscan_id, hdbscan_id)
            remapped[stable_id] = info
        return remapped

    def _mark_declined_clusters(self, declined_ids: list[int]) -> int:
        """Mark existing clusters that no longer have matching videos as inactive."""
        marked = 0
        for cid in declined_ids:
            self._clusters_table.update_item(
                Key={"cluster_id": _dec(cid)},
                UpdateExpression="SET #status = :s, #updated = :u",
                ExpressionAttributeNames={
                    "#status": "status",
                    "#updated": "updated_at",
                },
                ExpressionAttributeValues={
                    ":s": "inactive",
                    ":u": datetime.now(timezone.utc).isoformat(),
                },
            )
            marked += 1
            logger.info(f"DYNAMO_CLUSTER_DECLINED cluster={cid}")
        return marked

    def _write_cluster_summaries(self, cluster_info, existing_clusters=None):
        """
        Write cluster summaries to DynamoDB.
        Preserves created_at for matched (existing) clusters.
        """
        existing_clusters = existing_clusters or {}
        written = 0
        for cid, info in cluster_info.items():
            if cid == -1:
                continue
            now = datetime.now(timezone.utc).isoformat()
            # Preserve created_at if this is an existing cluster
            old = existing_clusters.get(cid)
            created_at = old["created_at"] if old and old.get("created_at") else now

            item = {
                "cluster_id": _dec(cid),
                "cluster_label": info["label"],
                "video_count": _dec(info["video_count"]),
                "channel_count": _dec(info["channel_count"]),
                "channels": info["channels"],
                "top_topics": info["top_topics"],
                "dominant_category": info["dominant_category"],
                "dominant_sentiment": info["dominant_sentiment"],
                "sentiment_breakdown": _clean_for_dynamo(info["sentiment_breakdown"]),
                "public_sentiment_breakdown": _clean_for_dynamo(
                    info.get("public_sentiment_breakdown", {})
                ),
                "avg_public_sentiment_score": _dec(
                    info.get("avg_public_sentiment_score", 0.0)
                ),
                "dominant_public_sentiment": info.get(
                    "dominant_public_sentiment", "neutral"
                ),
                "sentiment_divergence": info.get("sentiment_divergence", False),
                "breaking_count": _dec(info["breaking_count"]),
                "total_views": _dec(info["total_views"]),
                "total_likes": _dec(info["total_likes"]),
                "total_comments": _dec(info["total_comments"]),
                "narrative_headline": info.get("narrative_headline"),
                "narrative_summary": info.get("narrative_summary"),
                "week_data": _clean_for_dynamo(info["week_data"]),
                "top_claims": info["top_claims"],
                "status": "active",
                "created_at": created_at,
                "updated_at": now,
            }
            self._clusters_table.put_item(Item=item)
            written += 1
            logger.info(f"DYNAMO_CLUSTER_SUMMARY cluster={cid} label={info['label']!r}")

        return written

    # ── Public API ───────────────────────────────────────────────────────────

    def run_clustering(
        self,
        min_cluster_size=7,
        min_samples=2,
        umap_components=15,
        umap_neighbors=15,
        dry_run=False,
    ):
        # 1. Vectors from Qdrant
        video_chunks = self._scroll_vectors()
        if not video_chunks:
            return {"clusters": {}, "videos_updated": 0}

        # 2. Metadata from DynamoDB
        meta_map = self._get_video_metadata()
        if not meta_map:
            return {"clusters": {}, "videos_updated": 0}

        # 3. Representatives
        video_ids, matrix, filtered_meta = self._get_video_representatives(
            video_chunks, meta_map
        )
        if len(video_ids) < min_cluster_size:
            return {"clusters": {}, "videos_updated": 0}

        # 4. UMAP + HDBSCAN
        labels, probabilities = self._cluster_vectors(
            matrix, min_cluster_size, min_samples, umap_components, umap_neighbors
        )

        # 5. Label (uses HDBSCAN labels — temporary IDs)
        cluster_info = self._label_clusters(video_ids, labels, filtered_meta)

        # 6. Stable cluster matching — remap HDBSCAN IDs to stable IDs
        existing_clusters = self._load_existing_clusters()
        id_map, new_ids, declined_ids = self._match_to_existing_clusters(
            cluster_info, existing_clusters
        )
        cluster_info = self._remap_cluster_info(cluster_info, id_map)

        # Remap the per-video labels array to stable IDs
        stable_labels = np.array(
            [id_map.get(int(l), int(l)) if int(l) != -1 else -1 for l in labels]
        )

        if dry_run:
            logger.info("DRY_RUN skipping DynamoDB writes")
            videos_updated = 0
            clusters_written = 0
            declined_count = 0
        else:
            # 7. Write to DynamoDB youtube-videos (with stable IDs)
            videos_updated = self._write_clusters_to_dynamodb(
                video_ids, stable_labels, probabilities, cluster_info, filtered_meta
            )
            # 8. Write to DynamoDB narrative-clusters (preserving created_at)
            clusters_written = self._write_cluster_summaries(
                cluster_info, existing_clusters
            )
            # 9. Mark declined clusters as inactive
            declined_count = self._mark_declined_clusters(declined_ids)

        real_clusters = {k: v for k, v in cluster_info.items() if k != -1}
        noise_info = cluster_info.get(-1, {})

        return {
            "total_videos": len(video_ids),
            "videos_updated": videos_updated,
            "cluster_count": len(real_clusters),
            "clusters_written": clusters_written,
            "new_clusters": len(new_ids),
            "matched_clusters": len(real_clusters) - len(new_ids),
            "declined_clusters": declined_count if not dry_run else len(declined_ids),
            "noise_videos": noise_info.get("video_count", 0),
            "dry_run": dry_run,
            "clusters": {
                cid: {
                    "label": info["label"],
                    "video_count": info["video_count"],
                    "channel_count": info["channel_count"],
                    "channels": info["channels"],
                    "top_topics": info["top_topics"],
                    "dominant_category": info["dominant_category"],
                    "dominant_sentiment": info["dominant_sentiment"],
                    "breaking_count": info["breaking_count"],
                    "total_views": info["total_views"],
                    "week_data": info["week_data"],
                    "top_claims": info["top_claims"],
                    "is_new": cid in new_ids,
                }
                for cid, info in sorted(real_clusters.items())
            },
        }
