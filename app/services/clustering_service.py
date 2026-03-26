"""
clustering_service.py - Narrative Clustering Service

Groups related transcript chunks into narrative clusters using HDBSCAN
on existing Qdrant embeddings. Zero Gemini calls — labels derived from
the most frequent topics already stored in each chunk's payload.

Flow:
1. Scroll all points from Qdrant (vectors + payload)
2. Deduplicate to one representative per video (transcript_index)
3. Run HDBSCAN clustering on the vectors
4. Label each cluster using most frequent topics across its members
5. Patch cluster_id, cluster_label, cluster_confidence back to ALL chunks
6. Create payload indexes for fast filtering

Usage:
    python -m scripts.run_clustering
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np

from app.core.config import settings
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class ClusteringService:

    def __init__(self, vector_service: Optional[VectorService] = None):
        self.vector_service = vector_service or VectorService()
        self.collection_name = self.vector_service.collection_name
        self.client = self.vector_service.client

    # ── Step 1: Pull everything from Qdrant ──────────────────────────────────

    def _scroll_all_points(self) -> list[dict]:
        """Scroll the entire collection. Returns list of {id, vector, payload}."""
        all_points = []
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            for point in results:
                all_points.append(
                    {
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload,
                    }
                )
            if offset is None:
                break

        logger.info(f"CLUSTER_SCROLL_COMPLETE total_points={len(all_points)}")
        return all_points

    # ── Step 2: Deduplicate to video-level representatives ───────────────────

    def _deduplicate_by_video(self, points: list[dict]) -> dict[str, list[dict]]:
        """
        Group points by transcript_index (videoId).
        Returns {video_id: [point, point, ...]}
        """
        video_groups: dict[str, list[dict]] = {}
        for point in points:
            video_id = point["payload"].get("transcript_index", "unknown")
            video_groups.setdefault(video_id, []).append(point)

        logger.info(
            f"CLUSTER_DEDUP videos={len(video_groups)} " f"from_chunks={len(points)}"
        )
        return video_groups

    def _get_video_representatives(
        self, video_groups: dict[str, list[dict]]
    ) -> tuple[list[str], np.ndarray, dict[str, dict]]:
        """
        For each video, compute the mean vector across its chunks.
        Returns:
            video_ids:  ordered list of video IDs
            matrix:     (n_videos, dims) numpy array of mean vectors
            meta_map:   {video_id: {topics, channel, category, ...}} for labeling
        """
        video_ids = []
        vectors = []
        meta_map = {}

        for video_id, chunks in video_groups.items():
            # Mean-pool all chunk vectors for this video
            chunk_vectors = [c["vector"] for c in chunks]
            mean_vec = np.mean(chunk_vectors, axis=0)
            vectors.append(mean_vec)
            video_ids.append(video_id)

            # Grab metadata from first chunk (same across all chunks of a video)
            payload = chunks[0]["payload"]
            meta_map[video_id] = {
                "topics": payload.get("topics", []),
                "channel": payload.get("channel", ""),
                "category": payload.get("category", ""),
                "sentiment": payload.get("sentiment", ""),
                "is_breaking": payload.get("is_breaking", False),
                "view_count": payload.get("view_count", 0),
                "title": payload.get("title", ""),
            }

        matrix = np.array(vectors, dtype=np.float32)
        logger.info(f"CLUSTER_REPRESENTATIVES shape={matrix.shape}")
        return video_ids, matrix, meta_map

    # ── Step 3: Dimensionality reduction + HDBSCAN clustering ──────────────

    def _reduce_dimensions(
        self,
        matrix: np.ndarray,
        n_components: int = 15,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
    ) -> np.ndarray:
        """
        UMAP dimensionality reduction: 768 dims → n_components.
        Essential for HDBSCAN — density-based clustering fails in high dimensions
        because distances become uniform (curse of dimensionality).

        Args:
            n_components: target dimensions (15-25 is ideal for clustering)
            n_neighbors:  local neighborhood size (higher = more global structure)
            min_dist:     how tightly UMAP packs points (0.0 = tight, best for clustering)
        """
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for dimensionality reduction. "
                "Install it: pip install umap-learn"
            )

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, len(matrix) - 1),  # can't exceed dataset size
            min_dist=min_dist,
            metric="cosine",  # cosine works best for normalized text embeddings
            random_state=42,  # reproducible results
        )
        reduced = reducer.fit_transform(matrix)
        logger.info(
            f"UMAP_COMPLETE input_dims={matrix.shape[1]} output_dims={n_components} "
            f"n_neighbors={min(n_neighbors, len(matrix) - 1)} min_dist={min_dist}"
        )
        return reduced

    def _cluster_vectors(
        self,
        matrix: np.ndarray,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        umap_components: int = 15,
        umap_neighbors: int = 15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        UMAP → HDBSCAN pipeline.
        Reduces 768-dim vectors to umap_components dims, then clusters.
        Returns (labels, probabilities) arrays — one entry per video.
        label == -1 means noise (unclustered).
        """
        # Dimensionality reduction first
        reduced = self._reduce_dimensions(
            matrix,
            n_components=umap_components,
            n_neighbors=umap_neighbors,
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
            try:
                import hdbscan

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                )
                clusterer.fit(reduced)
                labels = clusterer.labels_
                probabilities = clusterer.probabilities_

            except ImportError:
                raise ImportError(
                    "Neither sklearn>=1.3 HDBSCAN nor the hdbscan package is installed. "
                    "Install one: pip install scikit-learn>=1.3 or pip install hdbscan"
                )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        logger.info(
            f"HDBSCAN_COMPLETE clusters={n_clusters} noise_videos={n_noise} "
            f"min_cluster_size={min_cluster_size} min_samples={min_samples} "
            f"umap_dims={umap_components}"
        )
        return labels, probabilities

    # ── Step 4: Label clusters with distinctive topics (TF-IDF style) ───────

    def _label_clusters(
        self,
        video_ids: list[str],
        labels: np.ndarray,
        meta_map: dict[str, dict],
    ) -> dict[int, dict]:
        """
        Label each cluster using its most DISTINCTIVE topic, not just the most
        frequent. Uses TF-IDF-style scoring:
            score = (topic_freq_in_cluster / cluster_size) * log(total_clusters / clusters_with_topic)

        Also enforces uniqueness — no two clusters get the same label.
        If a cluster's top pick is taken, it falls back to the next best.

        Returns {cluster_id: {label, video_count, channels, top_topics, ...}}
        """
        import math

        cluster_info: dict[int, dict] = {}

        # ── First pass: group videos by cluster, collect raw stats ───────
        cluster_members: dict[int, list[str]] = {}
        for vid, label in zip(video_ids, labels):
            label = int(label)
            cluster_members.setdefault(label, []).append(vid)

        # Per-cluster topic counts + metadata
        cluster_topic_counts: dict[int, Counter] = {}
        cluster_stats: dict[int, dict] = {}

        for cluster_id, member_vids in cluster_members.items():
            topic_counter = Counter()
            channels = set()
            categories = Counter()
            sentiments = Counter()
            breaking_count = 0
            total_views = 0

            for vid in member_vids:
                meta = meta_map[vid]
                for topic in meta["topics"]:
                    topic_counter[topic] += 1
                channels.add(meta["channel"])
                categories[meta["category"]] += 1
                sentiments[meta["sentiment"]] += 1
                if meta["is_breaking"]:
                    breaking_count += 1
                total_views += meta.get("view_count", 0)

            cluster_topic_counts[cluster_id] = topic_counter
            cluster_stats[cluster_id] = {
                "member_vids": member_vids,
                "channels": channels,
                "categories": categories,
                "sentiments": sentiments,
                "breaking_count": breaking_count,
                "total_views": total_views,
            }

        # ── Compute IDF: how many clusters each topic appears in ─────────
        real_cluster_ids = [cid for cid in cluster_members if cid != -1]
        n_clusters = max(len(real_cluster_ids), 1)

        topic_cluster_count: Counter = Counter()
        for cid in real_cluster_ids:
            for topic in cluster_topic_counts[cid]:
                topic_cluster_count[topic] += 1

        # ── Score topics per cluster using TF-IDF ────────────────────────
        cluster_scored_topics: dict[int, list[tuple[str, float]]] = {}

        for cid in cluster_members:
            tc = cluster_topic_counts[cid]
            cluster_size = len(cluster_members[cid])
            scored = []

            for topic, count in tc.items():
                tf = count / cluster_size
                idf = (
                    math.log((n_clusters + 1) / (topic_cluster_count.get(topic, 0) + 1))
                    + 1
                )
                scored.append((topic, tf * idf))

            # Sort by TF-IDF score descending
            scored.sort(key=lambda x: x[1], reverse=True)
            cluster_scored_topics[cid] = scored

        # ── Assign unique labels (greedy, highest-scoring first) ─────────
        used_labels: set[str] = set()
        cluster_labels: dict[int, str] = {}

        # Noise cluster gets a fixed label
        if -1 in cluster_members:
            cluster_labels[-1] = "Unclustered"

        # Sort real clusters by size (largest first gets first pick)
        sorted_cids = sorted(
            real_cluster_ids,
            key=lambda cid: len(cluster_members[cid]),
            reverse=True,
        )

        for cid in sorted_cids:
            scored = cluster_scored_topics[cid]
            chosen = None

            for topic, score in scored:
                if topic not in used_labels:
                    chosen = topic
                    break

            if chosen is None:
                # All topics taken — combine top two for a unique label
                top_two = [t[0] for t in scored[:2]]
                chosen = " & ".join(top_two) if len(top_two) == 2 else f"Cluster {cid}"

            cluster_labels[cid] = chosen
            used_labels.add(chosen)

        # ── Build final cluster_info dict ────────────────────────────────
        for cluster_id, member_vids in cluster_members.items():
            stats = cluster_stats[cluster_id]
            top_topics_raw = cluster_topic_counts[cluster_id].most_common(5)

            cluster_info[cluster_id] = {
                "label": cluster_labels[cluster_id],
                "video_count": len(member_vids),
                "video_ids": member_vids,
                "channels": sorted(stats["channels"]),
                "channel_count": len(stats["channels"]),
                "top_topics": [t[0] for t in top_topics_raw],
                "dominant_category": (
                    stats["categories"].most_common(1)[0][0]
                    if stats["categories"]
                    else "Other"
                ),
                "sentiment_breakdown": dict(stats["sentiments"]),
                "dominant_sentiment": (
                    stats["sentiments"].most_common(1)[0][0]
                    if stats["sentiments"]
                    else "neutral"
                ),
                "breaking_count": stats["breaking_count"],
                "total_views": stats["total_views"],
            }

            logger.info(
                f"CLUSTER_LABELED id={cluster_id} "
                f"label={cluster_labels[cluster_id]!r} "
                f"videos={len(member_vids)} channels={len(stats['channels'])}"
            )

        return cluster_info

        return cluster_info

    # ── Step 5: Patch results back to Qdrant ─────────────────────────────────

    def _patch_qdrant(
        self,
        video_groups: dict[str, list[dict]],
        video_ids: list[str],
        labels: np.ndarray,
        probabilities: np.ndarray,
        cluster_info: dict[int, dict],
    ) -> int:
        """
        Write cluster_id, cluster_label, cluster_confidence back to
        every chunk in Qdrant. Returns total points patched.
        """
        total_patched = 0

        for vid, label, prob in zip(video_ids, labels, probabilities):
            label = int(label)
            confidence = round(float(prob), 3)

            # Get the cluster label (noise gets "Unclustered")
            if label == -1:
                cluster_label = "Unclustered"
            else:
                cluster_label = cluster_info[label]["label"]

            # Get all chunk point IDs for this video
            chunk_point_ids = [chunk["id"] for chunk in video_groups[vid]]

            self.client.set_payload(
                collection_name=self.collection_name,
                payload={
                    "cluster_id": label,
                    "cluster_label": cluster_label,
                    "cluster_confidence": confidence,
                },
                points=chunk_point_ids,
            )
            total_patched += len(chunk_point_ids)

        logger.info(f"CLUSTER_PATCH_COMPLETE total_points_patched={total_patched}")
        return total_patched

    # ── Step 6: Create payload indexes ───────────────────────────────────────

    def _create_indexes(self) -> None:
        """Add cluster_id (integer) and cluster_label (keyword) indexes."""
        from qdrant_client.http import models as qmodels

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="cluster_id",
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
            logger.info("PAYLOAD_INDEX_CREATED field=cluster_id type=integer")
        except Exception as exc:
            logger.warning(f"PAYLOAD_INDEX_SKIP field=cluster_id reason={exc}")

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="cluster_label",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            logger.info("PAYLOAD_INDEX_CREATED field=cluster_label type=keyword")
        except Exception as exc:
            logger.warning(f"PAYLOAD_INDEX_SKIP field=cluster_label reason={exc}")

    # ── Public API ───────────────────────────────────────────────────────────

    def run_clustering(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        umap_components: int = 15,
        umap_neighbors: int = 15,
    ) -> dict:
        """
        Full clustering pipeline. Returns summary dict with cluster info.

        Args:
            min_cluster_size: minimum videos to form a cluster (default 3)
            min_samples:      HDBSCAN density param (default 2)
            umap_components:  UMAP target dimensions (default 15)
            umap_neighbors:   UMAP neighborhood size (default 15)
        """
        # 1. Pull all points
        all_points = self._scroll_all_points()
        if not all_points:
            logger.warning("CLUSTER_ABORT no points in collection")
            return {"clusters": {}, "total_patched": 0}

        # 2. Deduplicate to video level
        video_groups = self._deduplicate_by_video(all_points)
        video_ids, matrix, meta_map = self._get_video_representatives(video_groups)

        if len(video_ids) < min_cluster_size:
            logger.warning(
                f"CLUSTER_ABORT only {len(video_ids)} videos, "
                f"need at least {min_cluster_size}"
            )
            return {"clusters": {}, "total_patched": 0}

        # 3. UMAP → HDBSCAN
        labels, probabilities = self._cluster_vectors(
            matrix,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            umap_components=umap_components,
            umap_neighbors=umap_neighbors,
        )

        # 4. Label
        cluster_info = self._label_clusters(video_ids, labels, meta_map)

        # 5. Patch back to Qdrant
        total_patched = self._patch_qdrant(
            video_groups, video_ids, labels, probabilities, cluster_info
        )

        # 6. Create indexes
        self._create_indexes()

        # Build summary (exclude noise from "real" clusters)
        real_clusters = {k: v for k, v in cluster_info.items() if k != -1}
        noise_info = cluster_info.get(-1, {})

        summary = {
            "total_videos": len(video_ids),
            "total_chunks_patched": total_patched,
            "cluster_count": len(real_clusters),
            "noise_videos": noise_info.get("video_count", 0),
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
                }
                for cid, info in sorted(real_clusters.items())
            },
        }

        logger.info(
            f"CLUSTERING_COMPLETE clusters={len(real_clusters)} "
            f"noise={summary['noise_videos']} patched={total_patched}"
        )
        return summary
