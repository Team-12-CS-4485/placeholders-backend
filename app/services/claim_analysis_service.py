"""
claim_analysis_service.py - Claim Classification Service

Reads video intelligence from DynamoDB (grouped by cluster_id),
embeds claims locally (nomic, free), groups by semantic similarity,
classifies into consensus/debated/unique, writes results to
DynamoDB narrative-clusters table.

Zero Gemini calls. Zero Qdrant reads.

Usage:
    python -m scripts.run_claim_analysis
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import numpy as np
import boto3

from app.core.config import settings
from app.services.chunking_service import get_default_chunker

logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_REGION", "us-east-2")
CLAIM_SIMILARITY_THRESHOLD = 0.75


def _dec(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def _clean_for_dynamo(obj):
    """Recursively convert floats to Decimal and remove None values."""
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


class ClaimAnalysisService:

    def __init__(self):
        self._chunker = get_default_chunker(model_name=settings.embedding_model_id)
        self._dynamodb = boto3.resource("dynamodb", region_name=REGION)
        self._videos_table = self._dynamodb.Table(settings.dynamodb_table)
        self._clusters_table = self._dynamodb.Table("narrative-clusters")

    # ── Step 1: Read video data from DynamoDB ────────────────────────────────

    def _get_clustered_videos(self) -> dict[int, list[dict]]:
        """
        Scan DynamoDB for videos with cluster_id and key_claims.
        Returns {cluster_id: [{video_id, channel, sentiment, key_claims, title, transcript}, ...]}
        """
        cluster_videos: dict[int, list[dict]] = defaultdict(list)

        scan_kwargs = {
            "ProjectionExpression": "SortKey, PartitionKey, channel, title, sentiment, "
            "key_claims, cluster_id, transcript",
        }

        while True:
            resp = self._videos_table.scan(**scan_kwargs)
            for item in resp["Items"]:
                cid = item.get("cluster_id")
                claims = item.get("key_claims", [])
                if cid is not None and claims:
                    cluster_videos[int(cid)].append({
                        "video_id": item.get("SortKey", ""),
                        "channel": item.get("channel") or item.get("PartitionKey", ""),
                        "title": item.get("title", ""),
                        "sentiment": item.get("sentiment", "neutral"),
                        "key_claims": claims,
                        "transcript": item.get("transcript", ""),
                    })
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        logger.info(
            f"CLAIM_READ clusters={len(cluster_videos)} "
            f"total_videos={sum(len(v) for v in cluster_videos.values())}"
        )
        return cluster_videos

    # ── Step 2: Collect attributed claims ─────────────────────────────────────

    def _collect_attributed_claims(
        self, cluster_videos: dict[int, list[dict]]
    ) -> dict[int, list[dict]]:
        """
        Flatten claims with attribution per cluster.
        Returns {cluster_id: [{claim, channel, video_id, sentiment, excerpt}, ...]}
        """
        cluster_claims: dict[int, list[dict]] = {}

        for cid, videos in cluster_videos.items():
            claims = []
            seen = set()

            for v in videos:
                transcript = v.get("transcript", "")
                for claim in v["key_claims"]:
                    if not claim or claim in seen:
                        continue
                    seen.add(claim)

                    excerpt = self._extract_excerpt(transcript, claim)
                    claims.append({
                        "claim": claim,
                        "channel": v["channel"],
                        "video_id": v["video_id"],
                        "video_title": v["title"],
                        "sentiment": v["sentiment"],
                        "transcript_excerpt": excerpt,
                    })

            cluster_claims[cid] = claims
            logger.info(f"CLAIM_COLLECT cluster={cid} claims={len(claims)}")

        return cluster_claims

    def _extract_excerpt(self, transcript: str, claim: str, window: int = 80) -> str:
        words = transcript.split()
        claim_words = claim.lower().split()
        if not words or not claim_words:
            return ""

        best_pos = 0
        best_score = 0
        claim_set = set(claim_words)

        for i in range(len(words)):
            chunk = words[i:i + len(claim_words) + 10]
            overlap = len(claim_set & set(w.lower().strip(".,!?\"'") for w in chunk))
            if overlap > best_score:
                best_score = overlap
                best_pos = i

        start = max(0, best_pos - window // 4)
        end = min(len(words), best_pos + window)
        excerpt = " ".join(words[start:end])

        if start > 0:
            excerpt = "..." + excerpt
        if end < len(words):
            excerpt = excerpt + "..."
        return excerpt

    # ── Step 3: Embed + similarity ───────────────────────────────────────────

    def _embed_claims(self, claim_texts: list[str]) -> np.ndarray:
        if not claim_texts:
            return np.array([])
        vectors = self._chunker.embed(claim_texts, is_query=False)
        return np.array(vectors, dtype=np.float32)

    def _group_similar_claims(self, claims, similarity_matrix, threshold=CLAIM_SIMILARITY_THRESHOLD):
        n = len(claims)
        if n == 0:
            return []

        assigned = [-1] * n
        groups = []

        for i in range(n):
            if assigned[i] != -1:
                continue
            group = [i]
            assigned[i] = len(groups)
            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue
                if similarity_matrix[i][j] >= threshold:
                    group.append(j)
                    assigned[j] = len(groups)
            groups.append(group)

        return groups

    # ── Step 4: Classify ─────────────────────────────────────────────────────

    def _compute_framing_divergence(self, group_claims):
        excerpts = [c["transcript_excerpt"] for c in group_claims if c.get("transcript_excerpt", "").strip()]
        if len(excerpts) < 2:
            return 0.0
        vectors = self._embed_claims(excerpts)
        if len(vectors) < 2:
            return 0.0
        sim = vectors @ vectors.T
        n = len(vectors)
        total = sum(float(sim[i][j]) for i in range(n) for j in range(i + 1, n))
        pairs = n * (n - 1) / 2
        return round(1.0 - (total / pairs if pairs else 0), 3)

    def _classify_groups(self, claims, groups):
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
                divergence = self._compute_framing_divergence(group_claims)
                perspectives = [{
                    "channel": c["channel"],
                    "sentiment": c["sentiment"],
                    "video_id": c["video_id"],
                    "video_title": c["video_title"],
                    "transcript_excerpt": c["transcript_excerpt"],
                } for c in group_claims]
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

        return {"consensus": consensus, "debated": debated, "unique": unique}

    def _select_top_claims(self, classified, max_per_type=3):
        consensus = sorted(classified["consensus"], key=lambda c: c["source_count"], reverse=True)[:max_per_type]
        debated = sorted(classified["debated"], key=lambda c: c.get("framing_divergence", 0), reverse=True)[:max_per_type]
        unique = sorted(classified["unique"], key=lambda c: len(c.get("transcript_excerpt", "")), reverse=True)[:max_per_type]
        return {"consensus": consensus, "debated": debated, "unique": unique}

    # ── Step 5: Write to DynamoDB ────────────────────────────────────────────

    def _write_claims_to_dynamodb(self, cluster_results: dict[int, dict]) -> int:
        """Update narrative-clusters items with classified_claims."""
        written = 0

        for cid, claims_data in cluster_results.items():
            clean_claims = _clean_for_dynamo(claims_data)

            try:
                self._clusters_table.update_item(
                    Key={"cluster_id": _dec(cid)},
                    UpdateExpression="SET #claims = :claims, #updated = :updated",
                    ExpressionAttributeNames={
                        "#claims": "classified_claims",
                        "#updated": "updated_at",
                    },
                    ExpressionAttributeValues={
                        ":claims": clean_claims,
                        ":updated": datetime.now(timezone.utc).isoformat(),
                    },
                )
                written += 1
                c = len(claims_data.get("consensus", []))
                d = len(claims_data.get("debated", []))
                u = len(claims_data.get("unique", []))
                logger.info(f"CLAIM_WRITTEN cluster={cid} consensus={c} debated={d} unique={u}")
            except Exception as exc:
                logger.error(f"CLAIM_WRITE_FAILED cluster={cid} error={exc}")

        return written

    # ── Public API ───────────────────────────────────────────────────────────

    def run_claim_analysis(self, max_per_type: int = 3) -> dict:
        # 1. Read from DynamoDB
        cluster_videos = self._get_clustered_videos()
        if not cluster_videos:
            return {"clusters_processed": 0, "total_written": 0}

        # 2. Collect attributed claims
        cluster_claims = self._collect_attributed_claims(cluster_videos)

        # 3. Process each cluster
        cluster_results = {}
        summary = {}

        for cid, claims in cluster_claims.items():
            if len(claims) < 2:
                cluster_results[cid] = {
                    "consensus": [],
                    "debated": [],
                    "unique": [{
                        "claim": c["claim"],
                        "channel": c["channel"],
                        "video_id": c["video_id"],
                        "video_title": c["video_title"],
                        "transcript_excerpt": c["transcript_excerpt"],
                    } for c in claims[:max_per_type]],
                }
                continue

            # Embed
            embeddings = self._embed_claims([c["claim"] for c in claims])
            if len(embeddings) == 0:
                continue

            # Similarity + group
            sim_matrix = embeddings @ embeddings.T
            groups = self._group_similar_claims(claims, sim_matrix)

            # Classify + select
            classified = self._classify_groups(claims, groups)
            selected = self._select_top_claims(classified, max_per_type)
            cluster_results[cid] = selected

            summary[cid] = {
                "total_claims": len(claims),
                "groups": len(groups),
                "consensus": len(classified["consensus"]),
                "debated": len(classified["debated"]),
                "unique": len(classified["unique"]),
                "selected_c": len(selected["consensus"]),
                "selected_d": len(selected["debated"]),
                "selected_u": len(selected["unique"]),
            }

        # 4. Write to DynamoDB
        written = self._write_claims_to_dynamodb(cluster_results)

        return {
            "clusters_processed": len(cluster_results),
            "total_written": written,
            "per_cluster": summary,
        }