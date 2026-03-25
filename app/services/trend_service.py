"""
trend_service.py - Trend Aggregation Service

Queries Qdrant for all indexed chunks, deduplicates by video (transcript_index),
groups by cluster_id, and computes per-cluster metrics including:
- Week-over-week volume, sentiment, channel coverage
- Heat score for ranking
- Trend type classification (rising, emerging, dominant, declining, stable)

Zero Gemini calls — everything computed from existing Qdrant payload fields.
Requires clustering to have been run first (cluster_id in payload).
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class TrendService:

    def __init__(self, vector_service: Optional[VectorService] = None):
        self.vector_service = vector_service or VectorService()
        self.client = self.vector_service.client
        self.collection_name = self.vector_service.collection_name

    # ── Data extraction ──────────────────────────────────────────────────────

    def _scroll_all_payloads(self) -> list[dict]:
        """Pull all point payloads from Qdrant (no vectors needed)."""
        all_payloads = []
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
                all_payloads.append(point.payload)
            if offset is None:
                break

        logger.info(f"TREND_SCROLL_COMPLETE total_chunks={len(all_payloads)}")
        return all_payloads

    def _extract_week(self, source_key: str) -> str:
        """
        Extract week identifier from source_key.
        e.g. 'youtube-data/week1/cnbc.json' → 'week1'
             'youtube-data/week2/bbcnews.json' → 'week2'
             'youtube-data/2026-03-23T21-43-24Z/aljazeera.json' → 'week3' (date-based fallback)
        Falls back to 'unknown' if no pattern matches.
        """
        # Try weekN pattern first
        match = re.search(r"(week\d+)", source_key, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Try timestamp pattern — map date to week number
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", source_key)
        if date_match:
            try:
                from datetime import datetime

                date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                # Map to week based on ISO week number relative to earliest data
                # For now, use iso calendar week modulo to assign
                iso_week = date.isocalendar()[1]
                # Map: week 10 = week1, week 11 = week2, etc.
                # Find offset from a known anchor (week1 starts ~2026-03-03)
                anchor_week = 10  # ISO week of week1 data
                week_num = iso_week - anchor_week + 1
                if week_num >= 1:
                    return f"week{week_num}"
            except (ValueError, Exception):
                pass

        return "unknown"

    def _deduplicate_to_videos(self, payloads: list[dict]) -> list[dict]:
        """
        Deduplicate chunks to one record per video using transcript_index.
        Takes the first chunk's metadata (same across all chunks of a video).
        """
        seen = {}
        for payload in payloads:
            video_id = payload.get("transcript_index", "")
            if video_id and video_id not in seen:
                seen[video_id] = payload
        logger.info(f"TREND_DEDUP videos={len(seen)} from_chunks={len(payloads)}")
        return list(seen.values())

    # ── Aggregation ──────────────────────────────────────────────────────────

    def _aggregate_by_cluster(self, videos: list[dict]) -> dict[int, dict]:
        """
        Group videos by cluster_id and compute all metrics per cluster.
        Returns {cluster_id: {metrics...}}
        """
        cluster_videos: dict[int, list[dict]] = defaultdict(list)

        for video in videos:
            cluster_id = video.get("cluster_id")
            if cluster_id is None or cluster_id == -1:
                continue  # skip unclustered noise
            cluster_videos[int(cluster_id)].append(video)

        clusters = {}

        for cluster_id, vids in cluster_videos.items():
            # ── Basic counts ─────────────────────────────────────────
            channels = set()
            sentiments = Counter()
            categories = Counter()
            breaking_count = 0
            total_views = 0
            all_claims = []
            all_topics = Counter()

            # ── Per-week buckets ─────────────────────────────────────
            week_buckets: dict[str, list[dict]] = defaultdict(list)

            for v in vids:
                channels.add(v.get("channel", ""))
                sentiments[v.get("sentiment", "neutral")] += 1
                categories[v.get("category", "Other")] += 1
                if v.get("is_breaking"):
                    breaking_count += 1
                total_views += v.get("view_count", 0)

                for claim in v.get("key_claims", []):
                    if claim and len(all_claims) < 10:
                        all_claims.append(claim)

                for topic in v.get("topics", []):
                    all_topics[topic] += 1

                week = self._extract_week(v.get("source_key", ""))
                week_buckets[week].append(v)

            # ── Build week_data ──────────────────────────────────────
            week_data = []
            for week_name in sorted(week_buckets.keys()):
                week_vids = week_buckets[week_name]
                week_channels = set(v.get("channel", "") for v in week_vids)
                week_sentiments = Counter(
                    v.get("sentiment", "neutral") for v in week_vids
                )
                week_breaking = sum(1 for v in week_vids if v.get("is_breaking"))
                week_views = sum(v.get("view_count", 0) for v in week_vids)

                week_data.append(
                    {
                        "week": week_name,
                        "video_count": len(week_vids),
                        "channel_count": len(week_channels),
                        "view_count": week_views,
                        "breaking_count": week_breaking,
                        "sentiment_breakdown": dict(week_sentiments),
                    }
                )

            # ── Trend classification ─────────────────────────────────
            trend_type, metric_badge = self._classify_trend(
                week_data, len(channels), total_views, sentiments
            )

            # ── Heat score ───────────────────────────────────────────
            heat_score = self._compute_heat_score(
                len(channels), breaking_count, total_views
            )

            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "label": vids[0].get("cluster_label", f"Cluster {cluster_id}"),
                "description": all_claims[0] if all_claims else "",
                "category": (
                    categories.most_common(1)[0][0] if categories else "Other"
                ),
                "trend_type": trend_type,
                "metric_badge": metric_badge,
                "video_count": len(vids),
                "channel_count": len(channels),
                "view_count_total": total_views,
                "breaking_count": breaking_count,
                "heat_score": round(heat_score, 2),
                "sentiment_breakdown": dict(sentiments),
                "dominant_sentiment": (
                    sentiments.most_common(1)[0][0] if sentiments else "neutral"
                ),
                "channels": sorted(channels),
                "week_data": week_data,
                "top_claims": all_claims[:5],
                "top_topics": [t[0] for t in all_topics.most_common(5)],
            }

        logger.info(f"TREND_AGGREGATE clusters={len(clusters)}")
        return clusters

    # ── Trend classification ─────────────────────────────────────────────────

    def _classify_trend(
        self,
        week_data: list[dict],
        channel_count: int,
        total_views: int,
        sentiments: Counter,
    ) -> tuple[str, str]:
        """
        Classify a trend and generate a human-readable metric badge.
        Returns (trend_type, metric_badge).
        """
        if len(week_data) < 2:
            # Only one week of data
            if week_data and week_data[0]["video_count"] > 0:
                return "emerging", "New"
            return "stable", "Stable"

        # Sort by week name to get chronological order
        sorted_weeks = sorted(week_data, key=lambda w: w["week"])
        latest = sorted_weeks[-1]
        previous = sorted_weeks[-2]

        latest_count = latest["video_count"]
        prev_count = previous["video_count"]
        latest_views = latest["view_count"]
        prev_views = previous["view_count"]

        # ── Emerging: zero in previous week ──────────────────────
        if prev_count == 0 and latest_count > 0:
            return "emerging", "New"

        # ── Declining: zero in latest week ───────────────────────
        if latest_count == 0 and prev_count > 0:
            return "declining", "Fading"

        # ── Volume change % ──────────────────────────────────────
        if prev_count > 0:
            vol_change = ((latest_count - prev_count) / prev_count) * 100
        else:
            vol_change = 0

        if prev_views > 0:
            view_change = ((latest_views - prev_views) / prev_views) * 100
        else:
            view_change = 0

        # ── Dominant: highest channel spread ─────────────────────
        if channel_count >= 7:
            if vol_change > 20:
                return "rising", f"+{int(vol_change)}% Vol"
            return "dominant", "High Impact"

        # ── Rising / Declining by volume ─────────────────────────
        if vol_change > 30:
            return "rising", f"+{int(vol_change)}% Vol"
        if vol_change < -30:
            return "declining", f"{int(vol_change)}% Vol"

        # ── Sentiment-based badge ────────────────────────────────
        total_sent = sum(sentiments.values())
        if total_sent > 0:
            neg_pct = int((sentiments.get("negative", 0) / total_sent) * 100)
            pos_pct = int((sentiments.get("positive", 0) / total_sent) * 100)
            if neg_pct >= 70:
                return "stable", f"{neg_pct}% Neg"
            if pos_pct >= 60:
                return "stable", f"{pos_pct}% Pos"

        return "stable", "Stable"

    def _compute_heat_score(
        self,
        channel_count: int,
        breaking_count: int,
        total_views: int,
    ) -> float:
        """
        Composite ranking score.
        channel_count weighted 3x (breadth = real trend signal)
        breaking_count weighted 2x (urgency)
        views normalized to 100k units
        """
        return (channel_count * 3) + (breaking_count * 2) + (total_views / 100_000)

    # ── Header stats ─────────────────────────────────────────────────────────

    def _compute_header_stats(
        self, clusters: dict[int, dict], all_videos: list[dict]
    ) -> dict:
        """Compute the three header metrics for the Trends Archive page."""
        active_narratives = len(clusters)
        total_volume = sum(c["view_count_total"] for c in clusters.values())

        # Week-over-week total volume change
        week_totals: dict[str, int] = defaultdict(int)
        for v in all_videos:
            week = self._extract_week(v.get("source_key", ""))
            week_totals[week] += v.get("view_count", 0)

        sorted_weeks = sorted(week_totals.keys())
        if len(sorted_weeks) >= 2:
            prev_total = week_totals[sorted_weeks[-2]]
            curr_total = week_totals[sorted_weeks[-1]]
            if prev_total > 0:
                new_signals_pct = round(
                    ((curr_total - prev_total) / prev_total) * 100, 1
                )
            else:
                new_signals_pct = 0.0
        else:
            new_signals_pct = 0.0

        return {
            "active_narratives": active_narratives,
            "total_volume": total_volume,
            "new_signals_pct": new_signals_pct,
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def get_trends(self, sort_by: str = "heat_score") -> dict:
        """
        Main entry point. Returns full trend data for the frontend.

        Args:
            sort_by: field to sort trends by (heat_score, video_count, view_count_total)

        Returns dict matching TrendListResponse schema.
        """
        # 1. Pull all payloads
        payloads = self._scroll_all_payloads()

        # 2. Deduplicate to video level
        videos = self._deduplicate_to_videos(payloads)

        # 3. Aggregate by cluster
        clusters = self._aggregate_by_cluster(videos)

        # 4. Sort
        valid_sort_fields = {
            "heat_score",
            "video_count",
            "view_count_total",
            "channel_count",
        }
        if sort_by not in valid_sort_fields:
            sort_by = "heat_score"

        sorted_trends = sorted(
            clusters.values(),
            key=lambda c: c.get(sort_by, 0),
            reverse=True,
        )

        # 5. Header stats
        header = self._compute_header_stats(clusters, videos)

        return {
            "header": header,
            "trends": sorted_trends,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
