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
from datetime import datetime
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
            total_likes = 0
            total_comments = 0
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
                total_likes += v.get("like_count", 0)
                total_comments += v.get("comment_count", 0)

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

            # ── Engagement index ──────────────────────────────────────
            # (likes + comments) / views × 10000 — measures reaction intensity
            if total_views > 0:
                engagement_index = round(
                    ((total_likes + total_comments) / total_views) * 10000, 1
                )
            else:
                engagement_index = 0.0

            # ── Sentiment labels ──────────────────────────────────────
            sentiment_label = self._compute_sentiment_label(sentiments)
            recent_sentiment_label = self._compute_recent_sentiment_label(
                week_data, sentiment_label
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
                "total_likes": total_likes,
                "total_comments": total_comments,
                "engagement_index": engagement_index,
                "breaking_count": breaking_count,
                "heat_score": round(heat_score, 2),
                "sentiment_breakdown": dict(sentiments),
                "sentiment_label": sentiment_label,
                "recent_sentiment_label": recent_sentiment_label,
                "dominant_sentiment": (
                    sentiments.most_common(1)[0][0] if sentiments else "neutral"
                ),
                "channels": sorted(channels),
                "week_data": week_data,
                "top_claims": all_claims[:5],
                "claims": vids[0].get(
                    "classified_claims",
                    {
                        "consensus": [],
                        "debated": [],
                        "unique": [],
                    },
                ),
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

        Status values: surging, emerging, dominant, fading, holding
        """
        if len(week_data) < 2:
            # Only one week of data
            if week_data and week_data[0]["video_count"] > 0:
                return "emerging", "New"
            return "holding", "Holding"

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

        # ── Fading: zero in latest week ──────────────────────────
        if latest_count == 0 and prev_count > 0:
            return "fading", "Fading"

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
                return "surging", f"+{int(vol_change)}% Vol"
            return "dominant", "High Impact"

        # ── Surging / Fading by volume ───────────────────────────
        if vol_change > 30:
            return "surging", f"+{int(vol_change)}% Vol"
        if vol_change < -30:
            return "fading", f"{int(vol_change)}% Vol"

        # ── Sentiment-based badge ────────────────────────────────
        total_sent = sum(sentiments.values())
        if total_sent > 0:
            neg_pct = int((sentiments.get("negative", 0) / total_sent) * 100)
            pos_pct = int((sentiments.get("positive", 0) / total_sent) * 100)
            if neg_pct >= 70:
                return "holding", f"{neg_pct}% Neg"
            if pos_pct >= 60:
                return "holding", f"{pos_pct}% Pos"

        return "holding", "Holding"

    # ── Sentiment labels ─────────────────────────────────────────────────────

    def _compute_sentiment_label(self, sentiments: Counter) -> str:
        """
        Generate a journalism-grade sentiment label from raw counts.

        Returns labels like:
        - "Overwhelmingly Critical" (90%+ negative)
        - "Heavily Critical" (70-89% negative)
        - "Leaning Critical" (50-69% negative)
        - "Strongly Favorable" (90%+ positive)
        - "Mostly Favorable" (70-89% positive)
        - "Leaning Favorable" (50-69% positive)
        - "Polarized" (both pos and neg > 30%)
        - "Contested" (no sentiment > 50%)
        - "Measured" (70%+ neutral)
        - "Neutral"
        """
        total = sum(sentiments.values())
        if total == 0:
            return "Neutral"

        neg = sentiments.get("negative", 0)
        pos = sentiments.get("positive", 0)
        neu = sentiments.get("neutral", 0)

        neg_pct = neg / total
        pos_pct = pos / total
        neu_pct = neu / total

        # Check for polarization first — both sides active
        if neg_pct > 0.30 and pos_pct > 0.30:
            return "Polarized"

        # Negative labels
        if neg_pct >= 0.90:
            return "Overwhelmingly Critical"
        if neg_pct >= 0.70:
            return "Heavily Critical"
        if neg_pct >= 0.50:
            return "Leaning Critical"

        # Positive labels
        if pos_pct >= 0.90:
            return "Strongly Favorable"
        if pos_pct >= 0.70:
            return "Mostly Favorable"
        if pos_pct >= 0.50:
            return "Leaning Favorable"

        # Neutral-heavy
        if neu_pct >= 0.70:
            return "Measured"

        # No clear majority
        return "Contested"

    def _compute_recent_sentiment_label(
        self, week_data: list[dict], overall_label: str
    ) -> str:
        """
        Compute sentiment label for the most recent week, with shift detection.

        If sentiment changed from previous week, prefix with:
        - "Sentiment Reversal —" (flipped direction)
        - "Sentiment Collapse —" (was mixed/neutral, now 90%+ one direction)
        - "Intensifying —" (same direction but stronger)
        """
        if not week_data:
            return overall_label

        sorted_weeks = sorted(week_data, key=lambda w: w["week"])
        latest = sorted_weeks[-1]
        latest_sentiment = Counter(latest.get("sentiment_breakdown", {}))
        latest_label = self._compute_sentiment_label(latest_sentiment)

        if len(sorted_weeks) < 2:
            return latest_label

        previous = sorted_weeks[-2]
        prev_sentiment = Counter(previous.get("sentiment_breakdown", {}))
        prev_label = self._compute_sentiment_label(prev_sentiment)

        # Detect shifts
        prev_is_critical = "Critical" in prev_label
        prev_is_favorable = "Favorable" in prev_label
        latest_is_critical = "Critical" in latest_label
        latest_is_favorable = "Favorable" in latest_label

        # Reversal — flipped direction
        if (prev_is_critical and latest_is_favorable) or (
            prev_is_favorable and latest_is_critical
        ):
            return f"Sentiment Reversal — {latest_label}"

        # Collapse — was mixed/neutral, now strongly one direction
        if prev_label in ("Contested", "Measured", "Polarized", "Neutral"):
            if "Overwhelmingly" in latest_label:
                return f"Sentiment Collapse — {latest_label}"

        # Intensifying — same direction but stronger
        if prev_is_critical and latest_is_critical:
            if "Overwhelmingly" in latest_label and "Overwhelmingly" not in prev_label:
                return f"Intensifying — {latest_label}"
        if prev_is_favorable and latest_is_favorable:
            if "Strongly" in latest_label and "Strongly" not in prev_label:
                return f"Intensifying — {latest_label}"

        # No shift — just return latest week's label
        return latest_label

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

    # ── Internal: full cluster computation ───────────────────────────────────

    def _get_all_clusters(self) -> dict[int, dict]:
        """Scroll Qdrant, deduplicate, and aggregate into a cluster dict."""
        payloads = self._scroll_all_payloads()
        videos = self._deduplicate_to_videos(payloads)
        return self._aggregate_by_cluster(videos)

    def _find_cluster(self, cluster_id: int) -> dict:
        """Return a single cluster dict or raise KeyError if not found."""
        clusters = self._get_all_clusters()
        if cluster_id not in clusters:
            raise KeyError(cluster_id)
        return clusters[cluster_id]

    # ── Public API ───────────────────────────────────────────────────────────

    def get_trends(self, sort_by: str = "heat_score") -> dict:
        """
        Lean cluster list for the trends list view.
        Returns only the fields needed for list/grid rendering.
        """
        clusters = self._get_all_clusters()

        valid_sort_fields = {
            "heat_score",
            "video_count",
            "view_count_total",
            "channel_count",
            "engagement_index",
        }
        if sort_by not in valid_sort_fields:
            sort_by = "heat_score"

        sorted_trends = sorted(
            clusters.values(),
            key=lambda c: c.get(sort_by, 0),
            reverse=True,
        )

        lean = [
            {
                "cluster_id": c["cluster_id"],
                "label": c["label"],
                "category": c["category"],
                "trend_type": c["trend_type"],
                "metric_badge": c["metric_badge"],
                "heat_score": c["heat_score"],
                "video_count": c["video_count"],
                "channel_count": c["channel_count"],
                "view_count_total": c["view_count_total"],
                "breaking_count": c["breaking_count"],
                "sentiment_label": c["sentiment_label"],
                "recent_sentiment_label": c["recent_sentiment_label"],
                "dominant_sentiment": c["dominant_sentiment"],
                "top_topics": c["top_topics"],
            }
            for c in sorted_trends
        ]

        return {"trends": lean, "total": len(lean)}

    def get_trend_detail(self, cluster_id: int) -> dict:
        """Full detail for a single cluster (everything except claims)."""
        c = self._find_cluster(cluster_id)
        return {
            "cluster_id": c["cluster_id"],
            "label": c["label"],
            "category": c["category"],
            "trend_type": c["trend_type"],
            "metric_badge": c["metric_badge"],
            "heat_score": c["heat_score"],
            "video_count": c["video_count"],
            "channel_count": c["channel_count"],
            "view_count_total": c["view_count_total"],
            "total_likes": c["total_likes"],
            "total_comments": c["total_comments"],
            "engagement_index": c["engagement_index"],
            "breaking_count": c["breaking_count"],
            "sentiment_breakdown": c["sentiment_breakdown"],
            "sentiment_label": c["sentiment_label"],
            "recent_sentiment_label": c["recent_sentiment_label"],
            "dominant_sentiment": c["dominant_sentiment"],
            "channels": c["channels"],
            "week_data": c["week_data"],
            "top_claims": c["top_claims"],
            "top_topics": c["top_topics"],
        }

    def get_trend_sentiment(self, cluster_id: int) -> dict:
        """Sentiment breakdown + per-week sentiment for a single cluster."""
        c = self._find_cluster(cluster_id)

        by_week = [
            {
                "week": w["week"],
                "sentiment_breakdown": w["sentiment_breakdown"],
                "dominant_sentiment": (
                    max(w["sentiment_breakdown"], key=w["sentiment_breakdown"].get)
                    if w["sentiment_breakdown"]
                    else "neutral"
                ),
            }
            for w in c["week_data"]
        ]

        return {
            "cluster_id": cluster_id,
            "sentiment_breakdown": c["sentiment_breakdown"],
            "sentiment_label": c["sentiment_label"],
            "recent_sentiment_label": c["recent_sentiment_label"],
            "dominant_sentiment": c["dominant_sentiment"],
            "by_week": by_week,
        }

    def get_trend_claims(self, cluster_id: int) -> dict:
        """Classified claims (consensus / debated / unique) for a single cluster."""
        c = self._find_cluster(cluster_id)
        return {
            "cluster_id": cluster_id,
            "claims": c.get(
                "claims",
                {"consensus": [], "debated": [], "unique": []},
            ),
        }
