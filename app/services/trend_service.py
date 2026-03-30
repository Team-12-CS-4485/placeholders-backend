"""
trend_service.py - Trend Aggregation Service

Reads cluster summaries from DynamoDB narrative-clusters table and computes
derived metrics (heat score, trend classification, sentiment labels, engagement
index) entirely in Python — zero Qdrant reads.

Requires clustering to have been run first so that narrative-clusters items
exist with sentiment_breakdown, week_data, top_claims, and classified_claims.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from app.services.dynamo_service import DynamoService

logger = logging.getLogger(__name__)


class TrendService:

    def __init__(self, dynamo_service: Optional[DynamoService] = None):
        self.dynamo_service = dynamo_service or DynamoService()

    # ── DynamoDB → cluster dict ───────────────────────────────────────────────

    def _build_cluster_from_dynamo(self, item: dict) -> dict:
        """Map one narrative-clusters DynamoDB item to the internal cluster dict."""
        channel_count = int(item.get("channel_count", 0))
        breaking_count = int(item.get("breaking_count", 0))
        total_views = int(item.get("total_views", 0))
        total_likes = int(item.get("total_likes", 0))
        total_comments = int(item.get("total_comments", 0))

        dominant_sentiment = item.get("dominant_sentiment", "neutral")
        week_data = item.get("week_data", [])
        classified_claims = item.get(
            "classified_claims", {"consensus": [], "debated": [], "unique": []}
        )

        # Use dominant_sentiment as a proxy Counter for trend classification
        sentiments = (
            Counter({dominant_sentiment: 1}) if dominant_sentiment else Counter()
        )

        heat_score = self._compute_heat_score(
            channel_count, breaking_count, total_views
        )
        trend_type, metric_badge = self._classify_trend(
            week_data, channel_count, total_views, sentiments
        )
        _label_map = {
            "negative": "Negative",
            "positive": "Positive",
            "neutral": "Neutral",
        }
        sentiment_label = _label_map.get(dominant_sentiment, "Neutral")
        recent_sentiment_label = self._compute_recent_sentiment_label(
            week_data, sentiment_label
        )

        if total_views > 0:
            engagement_index = round(
                ((total_likes + total_comments) / total_views) * 10000, 1
            )
        else:
            engagement_index = 0.0

        # top_claims: prefer consensus claim text, fall back to raw top_claims list
        consensus_claims = [
            c["claim"] for c in classified_claims.get("consensus", []) if c.get("claim")
        ]
        top_claims = (consensus_claims or list(item.get("top_claims", [])))[:5]

        return {
            "cluster_id": int(item["cluster_id"]),
            "label": item.get("cluster_label", ""),
            "category": item.get("dominant_category", "Other"),
            "trend_type": trend_type,
            "metric_badge": metric_badge,
            "heat_score": round(heat_score, 2),
            "video_count": int(item.get("video_count", 0)),
            "channel_count": channel_count,
            "view_count_total": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "engagement_index": engagement_index,
            "breaking_count": breaking_count,
            "sentiment_breakdown": dict(item.get("sentiment_breakdown") or {}),
            "sentiment_label": sentiment_label,
            "recent_sentiment_label": recent_sentiment_label,
            "dominant_sentiment": item.get("dominant_sentiment", "neutral"),
            "channels": list(item.get("channels", [])),
            "week_data": week_data,
            "top_claims": top_claims,
            "top_topics": list(item.get("top_topics", [])),
            "claims": classified_claims,
            "narrative_headline": item.get("narrative_headline"),
            "narrative_summary": item.get("narrative_summary"),
        }

    # ── Internal cluster fetching ─────────────────────────────────────────────

    def _get_all_clusters(self) -> dict[int, dict]:
        items = self.dynamo_service.get_all_clusters()
        return {
            int(item["cluster_id"]): self._build_cluster_from_dynamo(item)
            for item in items
            if item.get("cluster_id") is not None
        }

    def _find_cluster(self, cluster_id: int) -> dict:
        """Return a single cluster dict or raise KeyError if not found."""
        item = self.dynamo_service.get_cluster(cluster_id)
        return self._build_cluster_from_dynamo(item)

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
            if week_data and week_data[0]["video_count"] > 0:
                return "emerging", "New"
            return "holding", "Holding"

        sorted_weeks = sorted(
            week_data,
            key=lambda w: (
                int(w["week"][4:])
                if w["week"].startswith("week") and w["week"][4:].isdigit()
                else 9999
            ),
        )
        latest = sorted_weeks[-1]
        previous = sorted_weeks[-2]

        latest_count = latest["video_count"]
        prev_count = previous["video_count"]

        if prev_count == 0 and latest_count > 0:
            return "emerging", "New"

        if latest_count == 0 and prev_count > 0:
            return "fading", "Fading"

        vol_change = (
            ((latest_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
        )

        if channel_count >= 7:
            if vol_change > 20:
                return "surging", f"+{int(vol_change)}% Vol"
            return "dominant", "High Impact"

        if vol_change > 30:
            return "surging", f"+{int(vol_change)}% Vol"
        if vol_change < -30:
            return "fading", f"{int(vol_change)}% Vol"

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
        total = sum(sentiments.values())
        if total == 0:
            return "Neutral"

        neg = sentiments.get("negative", 0)
        pos = sentiments.get("positive", 0)
        neu = sentiments.get("neutral", 0)

        neg_pct = neg / total
        pos_pct = pos / total
        neu_pct = neu / total

        if neg_pct > 0.30 and pos_pct > 0.30:
            return "Polarized"

        if neg_pct >= 0.50:
            return "Negative"

        if pos_pct >= 0.50:
            return "Positive"

        if neu_pct >= 0.50:
            return "Neutral"

        return "Polarized"

    def _compute_recent_sentiment_label(
        self, week_data: list[dict], overall_label: str
    ) -> str:
        if not week_data:
            return overall_label

        sorted_weeks = sorted(
            week_data,
            key=lambda w: (
                int(w["week"][4:])
                if w["week"].startswith("week") and w["week"][4:].isdigit()
                else 9999
            ),
        )
        latest = sorted_weeks[-1]
        latest_sentiment = Counter(latest.get("sentiment_breakdown", {}))
        latest_label = self._compute_sentiment_label(latest_sentiment)

        if len(sorted_weeks) < 2:
            return latest_label

        previous = sorted_weeks[-2]
        prev_sentiment = Counter(previous.get("sentiment_breakdown", {}))
        prev_label = self._compute_sentiment_label(prev_sentiment)

        # Reversal — flipped direction
        if (prev_label == "Negative" and latest_label == "Positive") or (
            prev_label == "Positive" and latest_label == "Negative"
        ):
            return f"Sentiment Reversal — {latest_label}"

        # Collapse — was mixed/neutral, now strongly one direction
        if prev_label in ("Polarized", "Neutral") and latest_label in (
            "Negative",
            "Positive",
        ):
            return f"Sentiment Shift — {latest_label}"

        return latest_label

    # ── Heat score ───────────────────────────────────────────────────────────

    def _compute_heat_score(
        self,
        channel_count: int,
        breaking_count: int,
        total_views: int,
    ) -> float:
        return (channel_count * 3) + (breaking_count * 2) + (total_views / 100_000)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_trends(self, sort_by: str = "heat_score", week: str = None) -> dict:
        """
        Lean cluster list for the trends list view.
        Optional week filter returns only clusters active in that week,
        with metrics scoped to the week slice.
        """
        # Normalise ?week=2 → "week2"
        if week and week.isdigit():
            week = f"week{week}"

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

        # Apply week filter + metric override if requested
        if week:
            filtered = {}
            for cid, c in clusters.items():
                week_slice = next(
                    (w for w in c["week_data"] if w["week"] == week), None
                )
                if not week_slice:
                    continue
                # Shallow-copy and override with week-specific metrics
                c = dict(c)
                slice_sentiments = Counter(week_slice.get("sentiment_breakdown", {}))
                c["video_count"] = week_slice["video_count"]
                c["view_count_total"] = week_slice["view_count"]
                c["breaking_count"] = week_slice["breaking_count"]
                c["sentiment_breakdown"] = week_slice["sentiment_breakdown"]
                c["channel_count"] = week_slice["channel_count"]
                c["heat_score"] = round(
                    self._compute_heat_score(
                        week_slice["channel_count"],
                        week_slice["breaking_count"],
                        week_slice["view_count"],
                    ),
                    2,
                )
                c["sentiment_label"] = self._compute_sentiment_label(slice_sentiments)
                c["dominant_sentiment"] = (
                    slice_sentiments.most_common(1)[0][0]
                    if slice_sentiments
                    else "neutral"
                )
                filtered[cid] = c
            clusters = filtered

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

    # ── Narrative API ─────────────────────────────────────────────────────────

    def get_narratives(self, week: str = None) -> dict:
        """
        Lean narrative list — editorial fields only, no metrics.
        Optional week filter returns only clusters active in that week.
        """
        if week and week.isdigit():
            week = f"week{week}"

        clusters = self._get_all_clusters()

        result = []
        for c in clusters.values():
            if week:
                week_slice = next(
                    (w for w in c["week_data"] if w["week"] == week), None
                )
                if not week_slice:
                    continue

            result.append({
                "cluster_id": c["cluster_id"],
                "label": c["label"],
                "category": c["category"],
                "narrative_headline": c.get("narrative_headline"),
                "narrative_summary": c.get("narrative_summary"),
                "top_topics": c["top_topics"],
                "video_count": c["video_count"],
                "dominant_sentiment": c["dominant_sentiment"],
            })

        return {"narratives": result, "total": len(result)}

    def get_narrative_detail(self, cluster_id: int) -> dict:
        """Full narrative detail — story fields + channels, no metrics."""
        c = self._find_cluster(cluster_id)
        return {
            "cluster_id": c["cluster_id"],
            "label": c["label"],
            "category": c["category"],
            "narrative_headline": c.get("narrative_headline"),
            "narrative_summary": c.get("narrative_summary"),
            "top_topics": c["top_topics"],
            "video_count": c["video_count"],
            "channel_count": c["channel_count"],
            "breaking_count": c["breaking_count"],
            "dominant_sentiment": c["dominant_sentiment"],
            "channels": c["channels"],
        }

    def get_narrative_claims(self, cluster_id: int) -> dict:
        """Classified claims for a narrative cluster."""
        c = self._find_cluster(cluster_id)
        return {
            "cluster_id": cluster_id,
            "claims": c.get(
                "claims",
                {"consensus": [], "debated": [], "unique": []},
            ),
        }

    def get_weeks(self) -> dict:
        """
        Aggregate per-week summary across all clusters.
        Drives the Archives week selector on the frontend.
        No extra DynamoDB reads — derived from narrative-clusters.week_data[].
        """
        all_clusters = self.dynamo_service.get_all_clusters()

        week_summary: dict[str, dict] = {}

        for item in all_clusters:
            for wd in item.get("week_data", []):
                w = wd.get("week", "unknown")
                if w not in week_summary:
                    week_summary[w] = {
                        "week": w,
                        "total_videos": 0,
                        "total_views": 0,
                        "active_clusters": 0,
                        "breaking_count": 0,
                        "_sentiments": Counter(),
                    }
                week_summary[w]["total_videos"] += int(wd.get("video_count", 0))
                week_summary[w]["total_views"] += int(wd.get("view_count", 0))
                week_summary[w]["breaking_count"] += int(wd.get("breaking_count", 0))
                if int(wd.get("video_count", 0)) > 0:
                    week_summary[w]["active_clusters"] += 1
                for sent, count in wd.get("sentiment_breakdown", {}).items():
                    week_summary[w]["_sentiments"][sent] += int(count)

        # Resolve sentiment Counter → label; sort weeks chronologically
        weeks = []
        for _, data in sorted(
            week_summary.items(),
            key=lambda kv: (
                int(kv[0][4:])
                if kv[0].startswith("week") and kv[0][4:].isdigit()
                else 9999
            ),
        ):
            sentiments = data.pop("_sentiments")
            data["dominant_sentiment"] = self._compute_sentiment_label(sentiments)
            weeks.append(data)

        logger.info(f"WEEKS_SUMMARY weeks={len(weeks)}")
        return {"weeks": weeks, "total": len(weeks)}
