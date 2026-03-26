"""
claim_analysis_service.py - Claim Classification Service

For each narrative cluster, collects all claims with source attribution,
embeds them locally (nomic, free), groups by semantic similarity, then
classifies into three types:

- CONSENSUS: 3+ channels independently report the same claim
- CONTROVERSIAL: 2+ channels report the same claim with opposing sentiment
- UNIQUE: only one channel reports this claim

Results (top 2-3 of each type) are patched back to Qdrant as cluster-level
payload. Zero Gemini calls.

Usage:
    python -m scripts.run_claim_analysis
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from app.core.config import settings
from app.services.vector_service import VectorService
from app.services.chunking_service import get_default_chunker

logger = logging.getLogger(__name__)

# Minimum cosine similarity to consider two claims "the same"
CLAIM_SIMILARITY_THRESHOLD = 0.75


class ClaimAnalysisService:

    def __init__(self, vector_service: Optional[VectorService] = None):
        self.vector_service = vector_service or VectorService()
        self.client = self.vector_service.client
        self.collection_name = self.vector_service.collection_name
        self._chunker = get_default_chunker(model_name=settings.embedding_model_id)

    # ── Step 1: Pull all data from Qdrant ────────────────────────────────────

    def _scroll_all_points(self) -> list[dict]:
        """Pull all point payloads + IDs from Qdrant."""
        all_points = []
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                all_points.append({
                    "id": point.id,
                    "payload": point.payload,
                })
            if offset is None:
                break

        logger.info(f"CLAIM_SCROLL_COMPLETE total_chunks={len(all_points)}")
        return all_points

    # ── Step 2: Collect claims with source attribution ───────────────────────

    def _collect_attributed_claims(
        self, points: list[dict]
    ) -> dict[int, list[dict]]:
        """
        Group by cluster_id, deduplicate to video level, collect claims
        with full source attribution.

        Returns {cluster_id: [
            {claim, channel, video_id, video_title, sentiment, chunk_text},
            ...
        ]}
        """
        # Group points by cluster, then by video
        cluster_videos: dict[int, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for point in points:
            p = point["payload"]
            cluster_id = p.get("cluster_id")
            if cluster_id is None or cluster_id == -1:
                continue
            video_id = p.get("transcript_index", "")
            if video_id:
                cluster_videos[int(cluster_id)][video_id].append(p)

        # Extract claims per cluster with attribution
        cluster_claims: dict[int, list[dict]] = {}

        for cluster_id, videos in cluster_videos.items():
            claims = []
            seen_claims = set()  # deduplicate exact text within cluster

            for video_id, chunks in videos.items():
                # Use first chunk for video-level metadata
                first = chunks[0]
                channel = first.get("channel", "")
                sentiment = first.get("sentiment", "neutral")
                title = first.get("title", "")
                video_claims = first.get("key_claims", [])

                # Combine all chunk text for transcript excerpts
                full_text = " ".join(c.get("text", "") for c in chunks)

                for claim in video_claims:
                    if not claim or claim in seen_claims:
                        continue
                    seen_claims.add(claim)

                    # Find excerpt — window around claim keywords
                    excerpt = self._extract_excerpt(full_text, claim)

                    claims.append({
                        "claim": claim,
                        "channel": channel,
                        "video_id": video_id,
                        "video_title": title,
                        "sentiment": sentiment,
                        "transcript_excerpt": excerpt,
                    })

            cluster_claims[cluster_id] = claims
            logger.info(
                f"CLAIM_COLLECT cluster={cluster_id} "
                f"videos={len(videos)} claims={len(claims)}"
            )

        return cluster_claims

    def _extract_excerpt(self, full_text: str, claim: str, window: int = 80) -> str:
        """
        Find the best matching region in the transcript for a claim
        and return a ~window-word excerpt around it.
        """
        words = full_text.split()
        claim_words = claim.lower().split()

        if not words or not claim_words:
            return ""

        # Find the position with the most overlapping words
        best_pos = 0
        best_score = 0
        claim_word_set = set(claim_words)

        for i in range(len(words)):
            chunk = words[i : i + len(claim_words) + 10]
            chunk_lower = set(w.lower().strip(".,!?\"'") for w in chunk)
            overlap = len(claim_word_set & chunk_lower)
            if overlap > best_score:
                best_score = overlap
                best_pos = i

        # Extract window around best position
        start = max(0, best_pos - window // 4)
        end = min(len(words), best_pos + window)
        excerpt = " ".join(words[start:end])

        if start > 0:
            excerpt = "..." + excerpt
        if end < len(words):
            excerpt = excerpt + "..."

        return excerpt

    # ── Step 3: Embed claims and compute similarity matrix ───────────────────

    def _embed_claims(self, claim_texts: list[str]) -> np.ndarray:
        """Embed claim texts using local nomic model. Returns (n, 768) array."""
        if not claim_texts:
            return np.array([])

        vectors = self._chunker.embed(claim_texts, is_query=False)
        return np.array(vectors, dtype=np.float32)

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix. Embeddings are already L2-normalized by nomic."""
        if len(embeddings) == 0:
            return np.array([])
        return embeddings @ embeddings.T

    # ── Step 4: Group similar claims ─────────────────────────────────────────

    def _group_similar_claims(
        self,
        claims: list[dict],
        similarity_matrix: np.ndarray,
        threshold: float = CLAIM_SIMILARITY_THRESHOLD,
    ) -> list[list[int]]:
        """
        Greedy grouping: iterate claims, merge into existing group if
        similarity > threshold, otherwise start new group.
        Returns list of groups, each group is a list of claim indices.
        """
        n = len(claims)
        if n == 0:
            return []

        assigned = [-1] * n  # which group each claim belongs to
        groups: list[list[int]] = []

        for i in range(n):
            if assigned[i] != -1:
                continue

            # Start a new group with this claim
            group = [i]
            assigned[i] = len(groups)

            # Find all unassigned claims similar to this one
            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue
                if similarity_matrix[i][j] >= threshold:
                    group.append(j)
                    assigned[j] = len(groups)

            groups.append(group)

        logger.info(
            f"CLAIM_GROUPS total_claims={n} groups_formed={len(groups)}"
        )
        return groups

    # ── Step 5: Classify groups into consensus/debated/unique ────────────────

    def _compute_framing_divergence(self, group_claims: list[dict]) -> float:
        """
        Measure how differently channels frame the same claim by comparing
        their transcript excerpts. Lower cosine similarity between excerpts
        = higher framing divergence = more "debated."

        Returns 0.0 (identical framing) to 1.0 (completely different framing).
        """
        excerpts = [
            c["transcript_excerpt"]
            for c in group_claims
            if c.get("transcript_excerpt", "").strip()
        ]

        if len(excerpts) < 2:
            return 0.0

        vectors = self._embed_claims(excerpts)
        if len(vectors) < 2:
            return 0.0

        # Average pairwise cosine similarity
        sim_matrix = vectors @ vectors.T
        n = len(vectors)
        total_sim = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += float(sim_matrix[i][j])
                pairs += 1

        if pairs == 0:
            return 0.0

        avg_similarity = total_sim / pairs
        return round(1.0 - avg_similarity, 3)

    def _classify_groups(
        self,
        claims: list[dict],
        groups: list[list[int]],
    ) -> dict[str, list[dict]]:
        """
        Classify each claim group:
        - CONSENSUS: 3+ channels report the same claim
        - DEBATED: 2 channels report the same claim (scored by framing divergence)
        - UNIQUE: only one channel reports this claim

        Returns {"consensus": [...], "debated": [...], "unique": [...]}
        """
        consensus = []
        debated = []
        unique = []

        for group_indices in groups:
            group_claims = [claims[i] for i in group_indices]
            channels = set(c["channel"] for c in group_claims)

            representative = group_claims[0]

            if len(channels) >= 3:
                consensus.append({
                    "claim": representative["claim"],
                    "sources": sorted(channels),
                    "source_count": len(channels),
                    "video_ids": [c["video_id"] for c in group_claims],
                    "transcript_excerpt": representative["transcript_excerpt"],
                })

            elif len(channels) >= 2:
                # DEBATED — 2 channels cover the same claim
                divergence = self._compute_framing_divergence(group_claims)

                perspectives = []
                for c in group_claims:
                    perspectives.append({
                        "channel": c["channel"],
                        "sentiment": c["sentiment"],
                        "video_id": c["video_id"],
                        "video_title": c["video_title"],
                        "transcript_excerpt": c["transcript_excerpt"],
                    })

                debated.append({
                    "claim": representative["claim"],
                    "perspectives": perspectives,
                    "source_count": len(channels),
                    "framing_divergence": divergence,
                })

            elif len(channels) == 1 and len(group_indices) == 1:
                unique.append({
                    "claim": representative["claim"],
                    "channel": representative["channel"],
                    "video_id": representative["video_id"],
                    "video_title": representative["video_title"],
                    "transcript_excerpt": representative["transcript_excerpt"],
                })

        return {
            "consensus": consensus,
            "debated": debated,
            "unique": unique,
        }

    # ── Step 6: Rank and pick top 2-3 of each type ──────────────────────────

    def _select_top_claims(
        self,
        classified: dict[str, list[dict]],
        max_per_type: int = 3,
    ) -> dict[str, list[dict]]:
        """
        Rank and pick the best claims of each type.

        Consensus: ranked by source_count (more channels = stronger)
        Debated: ranked by framing_divergence (higher = more different framing = most interesting)
        Unique: ranked by transcript_excerpt length (richer context = better)
        """
        consensus = sorted(
            classified["consensus"],
            key=lambda c: c["source_count"],
            reverse=True,
        )[:max_per_type]

        debated = sorted(
            classified["debated"],
            key=lambda c: c.get("framing_divergence", 0),
            reverse=True,
        )[:max_per_type]

        unique = sorted(
            classified["unique"],
            key=lambda c: len(c.get("transcript_excerpt", "")),
            reverse=True,
        )[:max_per_type]

        return {
            "consensus": consensus,
            "debated": debated,
            "unique": unique,
        }

    # ── Step 7: Patch results back to Qdrant ─────────────────────────────────

    def _patch_qdrant(
        self,
        cluster_claims_results: dict[int, dict],
        all_points: list[dict],
    ) -> int:
        """
        Write classified claims to every chunk in each cluster.
        Returns total points patched.
        """
        # Build cluster_id → point_ids mapping
        cluster_point_ids: dict[int, list] = defaultdict(list)
        for point in all_points:
            cid = point["payload"].get("cluster_id")
            if cid is not None and cid != -1:
                cluster_point_ids[int(cid)].append(point["id"])

        total_patched = 0

        for cluster_id, claims_data in cluster_claims_results.items():
            point_ids = cluster_point_ids.get(cluster_id, [])
            if not point_ids:
                continue

            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"classified_claims": claims_data},
                points=point_ids,
            )
            total_patched += len(point_ids)

            c_count = len(claims_data.get("consensus", []))
            v_count = len(claims_data.get("controversial", []))
            u_count = len(claims_data.get("unique", []))
            logger.info(
                f"CLAIM_PATCH cluster={cluster_id} "
                f"consensus={c_count} controversial={v_count} unique={u_count} "
                f"points={len(point_ids)}"
            )

        logger.info(f"CLAIM_PATCH_COMPLETE total_patched={total_patched}")
        return total_patched

    # ── Public API ───────────────────────────────────────────────────────────

    def run_claim_analysis(self, max_per_type: int = 3) -> dict:
        """
        Full claim analysis pipeline. Returns summary dict.

        Args:
            max_per_type: how many claims to keep per type (default 3)
        """
        # 1. Pull all points
        all_points = self._scroll_all_points()
        if not all_points:
            logger.warning("CLAIM_ABORT no points in collection")
            return {"clusters_processed": 0, "total_patched": 0}

        # 2. Collect attributed claims per cluster
        cluster_claims = self._collect_attributed_claims(all_points)

        if not cluster_claims:
            logger.warning("CLAIM_ABORT no clusters found")
            return {"clusters_processed": 0, "total_patched": 0}

        # 3. Process each cluster
        cluster_results = {}
        summary_by_cluster = {}

        for cluster_id, claims in cluster_claims.items():
            if len(claims) < 2:
                logger.info(
                    f"CLAIM_SKIP cluster={cluster_id} "
                    f"reason=too_few_claims count={len(claims)}"
                )
                # Still store what we have as unique
                cluster_results[cluster_id] = {
                    "consensus": [],
                    "debated": [],
                    "unique": [
                        {
                            "claim": c["claim"],
                            "channel": c["channel"],
                            "video_id": c["video_id"],
                            "video_title": c["video_title"],
                            "transcript_excerpt": c["transcript_excerpt"],
                        }
                        for c in claims[:max_per_type]
                    ],
                }
                continue

            # 3a. Embed all claim texts
            claim_texts = [c["claim"] for c in claims]
            embeddings = self._embed_claims(claim_texts)

            if len(embeddings) == 0:
                continue

            # 3b. Similarity matrix
            sim_matrix = self._compute_similarity_matrix(embeddings)

            # 3c. Group similar claims
            groups = self._group_similar_claims(claims, sim_matrix)

            # 3d. Classify
            classified = self._classify_groups(claims, groups)

            # 3e. Select top N
            selected = self._select_top_claims(classified, max_per_type)

            cluster_results[cluster_id] = selected

            summary_by_cluster[cluster_id] = {
                "total_claims": len(claims),
                "groups_formed": len(groups),
                "consensus_found": len(classified["consensus"]),
                "debated_found": len(classified["debated"]),
                "unique_found": len(classified["unique"]),
                "consensus_selected": len(selected["consensus"]),
                "debated_selected": len(selected["debated"]),
                "unique_selected": len(selected["unique"]),
            }

        # 4. Patch back to Qdrant
        total_patched = self._patch_qdrant(cluster_results, all_points)

        summary = {
            "clusters_processed": len(cluster_results),
            "total_patched": total_patched,
            "per_cluster": summary_by_cluster,
        }

        logger.info(
            f"CLAIM_ANALYSIS_COMPLETE clusters={len(cluster_results)} "
            f"patched={total_patched}"
        )
        return summary