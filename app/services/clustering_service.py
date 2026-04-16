"""
clustering_service.py - Narrative Clustering Service

Groups related videos into narrative clusters using HDBSCAN
on Qdrant embeddings. Writes results to DynamoDB.

Flow:
1. Pull vectors from Qdrant (search data only)
2. Pull metadata from DynamoDB (topics, channel, etc.)
3. Deduplicate to one representative per video (mean-pool vectors)
4. UMAP (768d → 15d) + HDBSCAN clustering
5. Aggregate per-cluster stats (build_cluster_stats — no Gemini)
6. Fetch prior headlines for all existing story threads
7. Gemini match — map HDBSCAN IDs to stable story thread IDs (week-1: fast-path)
8. Gemini label — NEW clusters get fresh prompt; MATCH clusters get prior headlines
9. Write cluster_id/label/confidence to DynamoDB youtube-videos
10. Write cluster summary to DynamoDB narrative-clusters table

Labeling and matching logic lives in cluster_labeling_service.py.

Usage:
    python -m scripts.run_clustering
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import numpy as np
import boto3

from app.core.config import settings
from app.services.vector_service import VectorService
from app.services.cluster_labeling_service import ClusterLabelingService

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

        self._labeling_service = ClusterLabelingService()

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

        logger.debug(f"CLUSTER_SCROLL videos={len(video_chunks)}")
        return video_chunks

    # ── Step 2: Pull metadata from DynamoDB ──────────────────────────────────

    def _get_video_metadata(self, week: str | None = None) -> dict[str, dict]:
        """
        Scan DynamoDB for videos with intelligence data.
        If week is provided, only returns videos for that week.
        Returns {video_id: {channel, topics, category, sentiment, ...}}
        """
        from boto3.dynamodb.conditions import Attr

        meta = {}
        scan_kwargs: dict = {
            "ProjectionExpression": "PartitionKey, SortKey, channel, topics, category, "
            "sentiment, is_breaking, viewCount, likeCount, commentCount, "
            "title, publishedAt, #wk, source_key, key_claims, "
            "public_sentiment, public_sentiment_score",
            "ExpressionAttributeNames": {"#wk": "week"},
        }
        if week:
            scan_kwargs["FilterExpression"] = Attr("week").eq(week)

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

        logger.debug(
            f"CLUSTER_DYNAMO_META videos_with_intel={len(meta)} week={week or 'all'}"
        )
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
        logger.debug(f"CLUSTER_REPRESENTATIVES shape={matrix.shape}")
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
        logger.debug(f"UMAP_COMPLETE {matrix.shape[1]}d → {n_components}d")
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
                cluster_selection_method="leaf",
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
        logger.debug(f"HDBSCAN_COMPLETE clusters={n_clusters} noise={n_noise}")
        return labels, probabilities

    # ── Step 4b: Coherence filter ────────────────────────────────────────────

    def _filter_incoherent_clusters(
        self,
        video_ids: list[str],
        labels: np.ndarray,
        meta_map: dict[str, dict],
        hard_min_purity: float = 0.35,
        soft_min_purity: float = 0.55,
        max_categories_at_soft: int = 3,
    ) -> np.ndarray:
        """
        Reassign incoherent clusters to noise (-1) before Gemini labeling.

        A cluster is incoherent if:
          - purity < hard_min_purity  (dominant category < 35% of videos), OR
          - purity < soft_min_purity AND n_distinct_categories > max_categories_at_soft

        purity = dominant_category_count / cluster_size

        Returns a modified copy of labels.
        """
        from collections import Counter

        labels = labels.copy()

        # Build {hdbscan_id: [category, ...]}
        cluster_cats: dict[int, list[str]] = {}
        for vid, lbl in zip(video_ids, labels):
            lid = int(lbl)
            if lid == -1:
                continue
            cat = meta_map.get(vid, {}).get("category", "Other")
            cluster_cats.setdefault(lid, []).append(cat)

        for lid, cats in cluster_cats.items():
            n = len(cats)
            counter = Counter(cats)
            dominant_count = counter.most_common(1)[0][1]
            purity = dominant_count / n
            n_cats = len(counter)

            incoherent = purity < hard_min_purity or (
                purity < soft_min_purity and n_cats > max_categories_at_soft
            )
            if incoherent:
                mask = np.array([i for i, lb in enumerate(labels) if int(lb) == lid])
                labels[mask] = -1
                logger.info(
                    f"COHERENCE_FILTER cluster={lid} size={n} "
                    f"purity={purity:.2f} n_cats={n_cats} → noise"
                )

        return labels

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

        logger.debug(f"DYNAMO_CLUSTER_WRITE videos={updated}")
        return updated

    def _mark_declined_clusters(
        self, declined_ids: list[int], existing_clusters: dict
    ) -> int:
        """
        Mark existing clusters that no longer have matching videos.
        active → declining (first miss)
        declining → inactive (second consecutive miss)
        """
        marked = 0
        now = datetime.now(timezone.utc).isoformat()
        for cid in declined_ids:
            old_status = existing_clusters.get(cid, {}).get("status", "active")
            if old_status == "declining":
                new_status = "inactive"
            else:
                new_status = "declining"

            self._clusters_table.update_item(
                Key={"cluster_id": _dec(cid)},
                UpdateExpression="SET #status = :s, #updated = :u, #declined = :d",
                ExpressionAttributeNames={
                    "#status": "status",
                    "#updated": "updated_at",
                    "#declined": "declined_at",
                },
                ExpressionAttributeValues={
                    ":s": new_status,
                    ":u": now,
                    ":d": now,
                },
            )
            marked += 1
            logger.info(
                f"DYNAMO_CLUSTER_{new_status.upper()} cluster={cid} "
                f"label={existing_clusters.get(cid, {}).get('label', '?')!r}"
            )
        return marked

    def _purge_inactive_clusters(self, existing_clusters: dict) -> int:
        """
        Delete clusters with status='inactive' from narrative-clusters.
        Articles are permanent historical records and are NOT deleted.
        Called once per pipeline run after decline marking.
        """
        inactive_ids = [
            cid
            for cid, info in existing_clusters.items()
            if info.get("status") == "inactive"
        ]
        if not inactive_ids:
            return 0

        # Delete the cluster records
        for cid in inactive_ids:
            self._clusters_table.delete_item(Key={"cluster_id": _dec(cid)})
            logger.info(
                f"DYNAMO_CLUSTER_PURGED cluster={cid} "
                f"label={existing_clusters[cid].get('label', '?')!r}"
            )

        logger.info(f"CLUSTER_PURGE inactive={len(inactive_ids)}")
        return len(inactive_ids)

    def _write_cluster_week_snapshots(
        self, cluster_info: dict, current_week: str
    ) -> int:
        """
        Write per-week snapshots to cluster-weeks.

        cluster-weeks is the primary content table. narrative-clusters is registry-only.
        All content fields (stats, sentiment, claims, channels) live here keyed by week.

        Since clustering is week-scoped, week_data always has exactly one entry.
        """
        from app.services.dynamo_service import DynamoService

        dynamo_svc = DynamoService(self._dynamodb)
        written = 0
        now = datetime.now(timezone.utc).isoformat()

        for cid, info in cluster_info.items():
            if cid == -1:
                continue
            # week_data has exactly one entry when clustering is week-scoped
            wd = next(
                (w for w in info.get("week_data", []) if w["week"] == current_week),
                None,
            )
            if not wd:
                continue

            week_sentiments = wd.get("sentiment_breakdown", {})
            dominant_sentiment = (
                max(week_sentiments, key=week_sentiments.get)
                if week_sentiments
                else "neutral"
            )
            week_pub_sentiments = wd.get("public_sentiment_breakdown", {})
            dominant_public_sentiment = (
                max(week_pub_sentiments, key=week_pub_sentiments.get)
                if week_pub_sentiments
                else "neutral"
            )

            dynamo_svc.save_cluster_week_snapshot(
                cid,
                current_week,
                {
                    # if_not_exists — manual edits safe
                    "narrative_headline": info.get("narrative_headline"),
                    "narrative_summary": info.get("narrative_summary"),
                    "created_at": now,
                    # always overwritten
                    "updated_at": now,
                    "week_overview": wd.get("week_overview", ""),
                    "top_claims": info.get("top_claims", []),
                    "top_topics": info.get("top_topics", []),
                    "channels": info.get("channels", []),
                    "video_count": wd.get("video_count", 0),
                    "channel_count": wd.get("channel_count", 0),
                    "view_count": wd.get("view_count", 0),
                    "breaking_count": wd.get("breaking_count", 0),
                    "total_likes": info.get("total_likes", 0),
                    "total_comments": info.get("total_comments", 0),
                    "dominant_sentiment": dominant_sentiment,
                    "sentiment_breakdown": week_sentiments,
                    "dominant_public_sentiment": dominant_public_sentiment,
                    "public_sentiment_breakdown": week_pub_sentiments,
                    "avg_public_sentiment_score": info.get(
                        "avg_public_sentiment_score", 0.0
                    ),
                    "sentiment_divergence": info.get("sentiment_divergence", False),
                    "dominant_category": info.get("dominant_category", "Other"),
                },
            )
            written += 1
        logger.info(f"CLUSTER_WEEK_SNAPSHOTS written={written} week={current_week}")
        return written

    def _write_cluster_summaries(
        self, cluster_info, existing_clusters=None, new_ids=None
    ):
        """
        Write narrative-clusters registry entries.

        Writes cluster_label, status, timestamps (registry fields) plus
        week_data and top_topics so the backfill script can iterate weeks
        without querying youtube-videos for the week list.

        New clusters: create with status='new', created_at=now.
        Existing clusters: update label, status, updated_at, week_data, top_topics.
        """
        existing_clusters = existing_clusters or {}
        new_ids = set(new_ids or [])
        written = 0
        now = datetime.now(timezone.utc).isoformat()

        for cid, info in cluster_info.items():
            if cid == -1:
                continue

            is_new = cid in new_ids
            status = "new" if is_new else "active"
            week_data = _clean_for_dynamo(info.get("week_data", []))
            top_topics = info.get("top_topics", [])
            video_count = _dec(info.get("video_count", 0))
            total_views = _dec(info.get("total_views", 0))
            channel_count = _dec(info.get("channel_count", 0))
            dominant_category = info.get("dominant_category", "Other")
            dominant_sentiment = info.get("dominant_sentiment", "neutral")
            channels = info.get("channels", [])

            if is_new:
                self._clusters_table.update_item(
                    Key={"cluster_id": _dec(cid)},
                    UpdateExpression=(
                        "SET cluster_label = :label, #status = :status, "
                        "created_at = if_not_exists(created_at, :now), "
                        "updated_at = :now, "
                        "week_data = :week_data, top_topics = :top_topics, "
                        "video_count = :vc, total_views = :tv, "
                        "channel_count = :cc, dominant_category = :dcat, "
                        "dominant_sentiment = :dsent, channels = :channels"
                    ),
                    ExpressionAttributeNames={"#status": "status"},
                    ExpressionAttributeValues={
                        ":label": info["label"],
                        ":status": status,
                        ":now": now,
                        ":week_data": week_data,
                        ":top_topics": top_topics,
                        ":vc": video_count,
                        ":tv": total_views,
                        ":cc": channel_count,
                        ":dcat": dominant_category,
                        ":dsent": dominant_sentiment,
                        ":channels": channels,
                    },
                )
            else:
                self._clusters_table.update_item(
                    Key={"cluster_id": _dec(cid)},
                    UpdateExpression=(
                        "SET cluster_label = :label, #status = :status, "
                        "updated_at = :now, "
                        "week_data = :week_data, top_topics = :top_topics, "
                        "video_count = :vc, total_views = :tv, "
                        "channel_count = :cc, dominant_category = :dcat, "
                        "dominant_sentiment = :dsent, channels = :channels"
                    ),
                    ExpressionAttributeNames={"#status": "status"},
                    ExpressionAttributeValues={
                        ":label": info["label"],
                        ":status": status,
                        ":now": now,
                        ":week_data": week_data,
                        ":top_topics": top_topics,
                        ":vc": video_count,
                        ":tv": total_views,
                        ":cc": channel_count,
                        ":dcat": dominant_category,
                        ":dsent": dominant_sentiment,
                        ":channels": channels,
                    },
                )
            written += 1
            logger.debug(
                f"DYNAMO_CLUSTER_REGISTRY cluster={cid} label={info['label']!r} "
                f"status={status}"
            )

        return written

    # ── Sequential renumbering ───────────────────────────────────────────────

    def _renumber_clusters(self) -> int:
        """
        Renumber all active/declining clusters to sequential IDs (0, 1, 2...)
        sorted by created_at ascending. Updates narrative-clusters, youtube-videos,
        and articles so all foreign keys stay consistent.
        Called as the final step of every pipeline run.
        """
        articles_table = self._dynamodb.Table("articles")

        # Scan current clusters (active + declining only)
        all_clusters = []
        scan_kwargs: dict = {}
        while True:
            resp = self._clusters_table.scan(**scan_kwargs)
            all_clusters.extend(resp["Items"])
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        remappable = sorted(
            [
                i
                for i in all_clusters
                if i.get("status", "active") in ("active", "declining", "new")
            ],
            key=lambda i: i.get("created_at", ""),
        )

        # Build id_map: {old_id: new_id} — only for clusters that actually need to move
        id_map: dict[int, int] = {}
        for new_id, item in enumerate(remappable):
            old_id = int(item["cluster_id"])
            if old_id != new_id:
                id_map[old_id] = new_id

        if not id_map:
            logger.debug("RENUMBER_CLUSTERS no changes needed")
            return 0

        logger.info(f"RENUMBER_CLUSTERS remapping {len(id_map)} cluster IDs")

        # Update narrative-clusters (PK change = delete + put)
        for item in remappable:
            old_id = int(item["cluster_id"])
            if old_id not in id_map:
                continue
            new_id = id_map[old_id]
            new_item = dict(item)
            new_item["cluster_id"] = _dec(new_id)
            self._clusters_table.delete_item(Key={"cluster_id": _dec(old_id)})
            self._clusters_table.put_item(Item=new_item)
            logger.debug(
                f"RENUMBER cluster {old_id} → {new_id} [{item.get('cluster_label', '?')}]"
            )

        # Update youtube-videos
        old_ids_dec = {_dec(old_id) for old_id in id_map}
        video_scan_kwargs: dict = {
            "ProjectionExpression": "PartitionKey, SortKey, cluster_id",
            "FilterExpression": boto3.dynamodb.conditions.Attr("cluster_id").exists(),
        }
        while True:
            resp = self._videos_table.scan(**video_scan_kwargs)
            for item in resp["Items"]:
                if item.get("cluster_id") in old_ids_dec:
                    new_id = id_map[int(item["cluster_id"])]
                    self._videos_table.update_item(
                        Key={
                            "PartitionKey": item["PartitionKey"],
                            "SortKey": item["SortKey"],
                        },
                        UpdateExpression="SET cluster_id = :cid",
                        ExpressionAttributeValues={":cid": _dec(new_id)},
                    )
            if "LastEvaluatedKey" not in resp:
                break
            video_scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        # Update articles
        article_scan_kwargs: dict = {
            "ProjectionExpression": "article_id, cluster_id",
        }
        while True:
            resp = articles_table.scan(**article_scan_kwargs)
            for item in resp["Items"]:
                if item.get("cluster_id") in old_ids_dec:
                    new_id = id_map[int(item["cluster_id"])]
                    articles_table.update_item(
                        Key={"article_id": item["article_id"]},
                        UpdateExpression="SET cluster_id = :cid",
                        ExpressionAttributeValues={":cid": _dec(new_id)},
                    )
            if "LastEvaluatedKey" not in resp:
                break
            article_scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        logger.info(f"RENUMBER_COMPLETE remapped={len(id_map)}")
        return len(id_map)

    # ── Freshness helpers ────────────────────────────────────────────────────

    @staticmethod
    def _detect_current_week(cluster_info: dict) -> str | None:
        """Derive the current week from the highest week number."""
        max_week = None
        max_num = -1
        for info in cluster_info.values():
            for wd in info.get("week_data", []):
                wk = wd.get("week", "")
                if wk.startswith("week") and wk[4:].isdigit():
                    num = int(wk[4:])
                    if num > max_num:
                        max_num = num
                        max_week = wk
        return max_week

    def _find_stale_clusters(
        self,
        cluster_info: dict,
        existing_clusters: dict,
        current_week: str,
        already_declined: list[int],
    ) -> list[int]:
        """
        Find active clusters that matched HDBSCAN but have no videos
        in the current week — the story is alive historically but got
        no new coverage this week.
        """
        already = set(already_declined)
        stale = []
        for cid, info in cluster_info.items():
            if cid == -1 or cid in already:
                continue
            weeks_present = {wd["week"] for wd in info.get("week_data", [])}
            if current_week not in weeks_present:
                stale.append(cid)
                logger.info(
                    f"FRESHNESS_STALE cluster={cid} label={info.get('label', '?')!r} "
                    f"weeks={sorted(weeks_present)} missing={current_week}"
                )
        return stale

    # ── Public API ───────────────────────────────────────────────────────────

    def run_clustering(
        self,
        min_cluster_size=7,
        min_samples=3,
        umap_components=15,
        umap_neighbors=30,
        dry_run=False,
        week: str | None = None,
        verbose=False,
    ):
        """
        Run clustering for a specific week's videos only, or all weeks if week=None.

        If week is None, clusters all available videos across all weeks.
        Passing week="week8" scopes both the DynamoDB scan and Qdrant pull
        to that week only.

        verbose=True adds titles_by_week to each cluster in the output — a dict
        of {week: ["[Channel] Title", ...]} for verifying week-over-week coherence.
        """
        _start = time.time()

        # 1. Vectors from Qdrant (all — week filtering happens via meta_map join)
        video_chunks = self._scroll_vectors()
        if not video_chunks:
            return {"clusters": {}, "videos_updated": 0}

        logger.info(f"CLUSTER_START total_vectors={len(video_chunks)}")

        # 2. Metadata from DynamoDB — week-scoped if provided
        meta_map = self._get_video_metadata(week=week)
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

        # 4b. Filter incoherent clusters → noise before Gemini labeling
        labels = self._filter_incoherent_clusters(
            labels=labels, video_ids=video_ids, meta_map=filtered_meta
        )

        # 5a. Load existing clusters from narrative-clusters registry
        existing_clusters = self._labeling_service.load_existing_clusters(
            self._clusters_table
        )

        # 5b. Build raw cluster stats — pure aggregation, no Gemini
        from app.services.dynamo_service import DynamoService

        dynamo_svc = DynamoService(self._dynamodb)
        raw_cluster_info = self._labeling_service.build_cluster_stats(
            video_ids, labels, filtered_meta
        )

        # 5c. Fetch prior headlines for all existing clusters (match context)
        prior_headlines: dict[int, list[dict]] = {}
        for cid in existing_clusters:
            headlines = dynamo_svc.get_cluster_week_headlines(cid, last_n=3)
            if headlines:
                prior_headlines[cid] = headlines

        # 6. Gemini match — maps HDBSCAN IDs to stable cluster IDs
        #    Week-1 fast-path: no existing clusters → mint all as NEW, skip Gemini
        id_map, new_ids, declined_ids = (
            self._labeling_service.match_to_existing_clusters(
                raw_cluster_info, existing_clusters, prior_headlines=prior_headlines
            )
        )

        # 7. Build per-cluster match context for labeling
        #    MATCH clusters get their prior headlines; NEW clusters get empty list
        match_results: dict[int, dict] = {}
        for hdbscan_id in raw_cluster_info:
            if hdbscan_id == -1:
                continue
            stable_id = id_map.get(hdbscan_id)
            is_new = stable_id in new_ids
            ph = prior_headlines.get(stable_id, []) if stable_id is not None else []
            match_results[hdbscan_id] = {"is_new": is_new, "prior_headlines": ph}

        # 8. Gemini labeling — always runs, dry_run only gates DynamoDB writes
        cluster_info = self._labeling_service.label_clusters(
            raw_cluster_info, match_results, dry_run=False
        )

        # 9. Remap HDBSCAN IDs to stable cluster IDs
        cluster_info = self._labeling_service.remap_cluster_info(cluster_info, id_map)

        # Remap the per-video labels array to stable IDs
        stable_labels = np.array(
            [id_map.get(int(lb), int(lb)) if int(lb) != -1 else -1 for lb in labels]
        )

        if dry_run:
            logger.info("DRY_RUN skipping DynamoDB writes")
            videos_updated = 0
            clusters_written = 0
            declined_count = 0
        else:
            # Use passed week param or fall back to detecting from cluster_info
            current_week = week or self._detect_current_week(cluster_info)

            # 10. Write cluster_id/label to youtube-videos (stable IDs)
            videos_updated = self._write_clusters_to_dynamodb(
                video_ids, stable_labels, probabilities, cluster_info, filtered_meta
            )
            # 11. Write registry entries to narrative-clusters (label + status only)
            clusters_written = self._write_cluster_summaries(
                cluster_info, existing_clusters, new_ids
            )
            # 11b. Write per-week content snapshots to cluster-weeks.
            # Only runs when week-scoped (week param provided). All-time runs skip
            # this so the backfill script owns headline generation for all weeks
            # with proper prior-headline memory context.
            if week and current_week:
                self._write_cluster_week_snapshots(cluster_info, current_week)

            # 12. Mark declined clusters (active→declining→inactive)
            declined_count = self._mark_declined_clusters(
                declined_ids, existing_clusters
            )

            # 13. Freshness check — active clusters with no videos this week → stale
            if current_week:
                stale_ids = self._find_stale_clusters(
                    cluster_info, existing_clusters, current_week, declined_ids
                )
                if stale_ids:
                    stale_count = self._mark_declined_clusters(
                        stale_ids, existing_clusters
                    )
                    declined_count += stale_count
                    logger.info(
                        f"FRESHNESS_CHECK current_week={current_week} "
                        f"stale_clusters={stale_ids}"
                    )

            # 14. Purge inactive clusters
            self._purge_inactive_clusters(existing_clusters)
            # NOTE: _renumber_clusters intentionally removed — stable IDs must not
            # be reassigned, as cluster-weeks and youtube-videos both reference them.

        real_clusters = {k: v for k, v in cluster_info.items() if k != -1}
        noise_info = cluster_info.get(-1, {})

        logger.info(
            f"CLUSTER_COMPLETE clusters={len(real_clusters)} "
            f"new={len(new_ids)} matched={len(real_clusters) - len(new_ids)} "
            f"declined={declined_count if not dry_run else len(declined_ids)} "
            f"noise_videos={noise_info.get('video_count', 0)} "
            f"total_videos={len(video_ids)} "
            f"elapsed_s={time.time() - _start:.1f}"
        )

        def _titles_by_week(info: dict) -> dict[str, list[str]]:
            result: dict[str, list[str]] = {}
            for vid in info.get("video_ids", []):
                meta = filtered_meta.get(vid, {})
                wk = meta.get("week", "unknown")
                title = meta.get("title", "")
                channel = meta.get("channel", "")
                result.setdefault(wk, []).append(f"[{channel}] {title}")
            # Sort weeks numerically
            return dict(
                sorted(
                    result.items(),
                    key=lambda kv: (
                        int(kv[0][4:])
                        if kv[0].startswith("week") and kv[0][4:].isdigit()
                        else 0
                    ),
                )
            )

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
                    **({"titles_by_week": _titles_by_week(info)} if verbose else {}),
                }
                for cid, info in sorted(real_clusters.items())
            },
        }
