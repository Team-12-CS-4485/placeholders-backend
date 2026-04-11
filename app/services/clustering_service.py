"""
clustering_service.py - Narrative Clustering Service

Groups related videos into narrative clusters using HDBSCAN
on Qdrant embeddings. Writes results to DynamoDB.

Flow:
1. Pull vectors from Qdrant (search data only)
2. Pull metadata from DynamoDB (topics, channel, etc.)
3. Deduplicate to one representative per video (mean-pool vectors)
4. UMAP (768d → 15d) + HDBSCAN clustering
5. Label each cluster (via ClusterLabelingService)
6. Stable cluster matching (via ClusterLabelingService)
7. Write cluster_id/label/confidence to DynamoDB youtube-videos
8. Write cluster summary to DynamoDB narrative-clusters table

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

        logger.debug(f"CLUSTER_DYNAMO_META videos_with_intel={len(meta)}")
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
        """Write per-week headline/summary snapshots to the cluster-weeks table."""
        from app.services.dynamo_service import DynamoService

        dynamo_svc = DynamoService(self._dynamodb)
        written = 0
        for cid, info in cluster_info.items():
            if cid == -1:
                continue
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
            dynamo_svc.save_cluster_week_snapshot(
                cid,
                current_week,
                {
                    "narrative_headline": info.get("narrative_headline"),
                    "narrative_summary": info.get("narrative_summary"),
                    "week_overview": wd.get("week_overview", ""),
                    "top_claims": info.get("top_claims", []),
                    "top_topics": info.get("top_topics", []),
                    "video_count": wd.get("video_count", 0),
                    "channel_count": wd.get("channel_count", 0),
                    "view_count": wd.get("view_count", 0),
                    "breaking_count": wd.get("breaking_count", 0),
                    "dominant_sentiment": dominant_sentiment,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            written += 1
        logger.info(f"CLUSTER_WEEK_SNAPSHOTS written={written} week={current_week}")
        return written

    def _write_cluster_summaries(
        self, cluster_info, existing_clusters=None, new_ids=None
    ):
        """
        Write cluster summaries to DynamoDB.
        Preserves created_at for matched (existing) clusters.
        New clusters get status='new', existing ones stay 'active'.
        week_data entries include week_overview from the Gemini call.
        """
        existing_clusters = existing_clusters or {}
        new_ids = set(new_ids or [])
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
                # week_data now carries week_overview on each entry
                "week_data": _clean_for_dynamo(info["week_data"]),
                "top_claims": info["top_claims"],
                "status": "new" if cid in new_ids else "active",
                "created_at": created_at,
                "updated_at": now,
            }
            self._clusters_table.put_item(Item=item)
            written += 1
            logger.debug(
                f"DYNAMO_CLUSTER_SUMMARY cluster={cid} label={info['label']!r}"
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
    ):
        _start = time.time()

        # 1. Vectors from Qdrant
        video_chunks = self._scroll_vectors()
        if not video_chunks:
            return {"clusters": {}, "videos_updated": 0}

        logger.info(f"CLUSTER_START total_vectors={len(video_chunks)}")

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

        # 5a. Load existing clusters (needed for preliminary stable match inside label_clusters)
        existing_clusters = self._labeling_service.load_existing_clusters(
            self._clusters_table
        )

        # 5b. Label (preliminary stable match runs inside, Gemini uses prior headlines)
        cluster_info = self._labeling_service.label_clusters(
            video_ids,
            labels,
            filtered_meta,
            dry_run=dry_run,
            existing_clusters=existing_clusters,
            cluster_weeks_table=self._dynamodb.Table("cluster-weeks"),
        )

        # 6. Real stable cluster matching — remap HDBSCAN IDs to stable IDs (unchanged)
        id_map, new_ids, declined_ids = (
            self._labeling_service.match_to_existing_clusters(
                cluster_info, existing_clusters
            )
        )
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
            current_week = self._detect_current_week(cluster_info)

            # 7. Write to DynamoDB youtube-videos (with stable IDs)
            videos_updated = self._write_clusters_to_dynamodb(
                video_ids, stable_labels, probabilities, cluster_info, filtered_meta
            )
            # 8. Write to DynamoDB narrative-clusters (preserving created_at)
            clusters_written = self._write_cluster_summaries(
                cluster_info, existing_clusters, new_ids
            )
            # 8b. Write per-week snapshots to cluster-weeks
            if current_week:
                self._write_cluster_week_snapshots(cluster_info, current_week)

            # 9. Mark declined clusters (active→declining→inactive)
            declined_count = self._mark_declined_clusters(
                declined_ids, existing_clusters
            )

            # 10. Freshness check — clusters with no videos in the current week
            #     are also candidates for declining, even if HDBSCAN still matches them
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

            # 11. Purge inactive clusters and their articles
            self._purge_inactive_clusters(existing_clusters)

            # 12. Renumber all remaining clusters to sequential IDs
            self._renumber_clusters()

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
