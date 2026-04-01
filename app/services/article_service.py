"""
article_service.py - News Article Generation Service

Reads narrative cluster data from DynamoDB (narrative-clusters table),
generates structured news articles via Gemini, and persists them to the
articles DynamoDB table.

Key design decisions:
- Uses classified_claims + top_claims already stored in narrative-clusters
  (zero additional S3/Qdrant reads)
- Optionally enriches with a few video titles from the cluster-index GSI
- Skips clusters / weeks that already have an article (idempotent by default)
- Rotates Gemini API keys on 429 errors, same pattern as other services

Articles table schema:
  PK: article_id  (string, e.g. "article-a1b2c3d4")
  Attributes: cluster_id (N), cluster_label, week_number (N),
              title, overview, body, created_at, updated_at

Usage:
    service = ArticleService()
    summary = service.run_article_generation(week="week1")
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError
from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

REGION = settings.aws_region
_WEEK_RE = re.compile(r"week(\d+)", re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dec(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def _deserialize(obj):
    """boto3 Decimal → int/float, set → list."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deserialize(i) for i in obj]
    if isinstance(obj, set):
        return sorted(_deserialize(i) for i in obj)
    return obj


def _normalize_week(week: Optional[str]) -> Optional[str]:
    """'1' → 'week1', 'Week2' → 'week2', None → None."""
    if week is None:
        return None
    if week.isdigit():
        return f"week{week}"
    return week.lower()


def _week_number(week: str) -> int:
    m = _WEEK_RE.search(week)
    return int(m.group(1)) if m else 9999


# ── Service ───────────────────────────────────────────────────────────────────


class ArticleService:

    def __init__(self):
        self._dynamodb = boto3.resource("dynamodb", region_name=REGION)
        self._clusters_table = self._dynamodb.Table("narrative-clusters")
        self._videos_table = self._dynamodb.Table(settings.dynamodb_table)
        self._articles_table = self._dynamodb.Table("articles")

        self._genai_api_keys = settings.genai_api_keys
        self._key_index = 0
        self._client = genai.Client(api_key=self._genai_api_keys[0])

    # ── Gemini helpers ────────────────────────────────────────────────────────

    def _rotate_key(self) -> bool:
        next_idx = self._key_index + 1
        if next_idx >= len(self._genai_api_keys):
            logger.error("ARTICLE_API_KEY_EXHAUSTED")
            return False
        self._key_index = next_idx
        self._client = genai.Client(api_key=self._genai_api_keys[self._key_index])
        logger.warning(
            f"ARTICLE_KEY_ROTATED index={self._key_index}/{len(self._genai_api_keys)-1}"
        )
        return True

    def _call_gemini(self, prompt: str, max_attempts: int = 8) -> str:
        last_exc = None
        for attempt in range(max_attempts):
            try:
                response = self._client.models.generate_content(
                    model=settings.gemini_model_id,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                        temperature=0.65,
                    ),
                )
                return (getattr(response, "text", None) or str(response)).strip()
            except Exception as exc:
                last_exc = exc
                s = str(exc)
                is_rate = (
                    "429" in s
                    or "RESOURCE_EXHAUSTED" in s
                    or getattr(exc, "code", None) == 429
                )
                if is_rate:
                    if self._rotate_key():
                        continue
                    # all keys exhausted — reset and back off
                    logger.warning("ARTICLE_ALL_KEYS_EXHAUSTED sleeping 60s")
                    self._key_index = 0
                    self._client = genai.Client(api_key=self._genai_api_keys[0])
                    time.sleep(60)
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _get_all_clusters(self) -> list[dict]:
        items: list[dict] = []
        scan_kwargs: dict = {}
        while True:
            resp = self._clusters_table.scan(**scan_kwargs)
            items.extend(_deserialize(i) for i in resp.get("Items", []))
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return items

    def _get_video_titles_for_cluster(
        self, cluster_id: int, limit: int = 6
    ) -> list[str]:
        """
        Pull a handful of video titles from youtube-videos via the cluster-index GSI.
        Returns an empty list gracefully if the GSI doesn't exist or the query fails.
        """
        try:
            resp = self._videos_table.query(
                IndexName="cluster-index",
                KeyConditionExpression=Key("cluster_id").eq(_dec(cluster_id)),
                ProjectionExpression="title",
                Limit=limit,
            )
            return [
                item["title"] for item in resp.get("Items", []) if item.get("title")
            ]
        except Exception as exc:
            logger.warning(
                f"ARTICLE_VIDEO_TITLES_FETCH_WARN cluster={cluster_id} error={exc}"
            )
            return []

    # ── Article existence check ───────────────────────────────────────────────

    def _article_exists(self, cluster_id: int, week_number: int) -> bool:
        """
        Scan articles table for an item with matching cluster_id + week_number.
<<<<<<< HEAD
        The table is tiny so a filtered scan is fine.

        Note: cluster_id and week_number are stored as Decimal in DynamoDB.
        The filter must use Decimal too — comparing a Python int against a stored
        Decimal never matches, which caused duplicates to be generated every run.
=======
        Paginates fully — Limit cannot be used with FilterExpression reliably.
>>>>>>> 4c1eb56 (Updated bugs)
        """
        scan_kwargs: dict = {
            "FilterExpression": Attr("cluster_id").eq(cluster_id)
            & Attr("week_number").eq(week_number),
            "ProjectionExpression": "article_id",
        }
        try:
<<<<<<< HEAD
            resp = self._articles_table.scan(
                FilterExpression=Attr("cluster_id").eq(_dec(cluster_id))
                & Attr("week_number").eq(_dec(week_number)),
                ProjectionExpression="article_id",
                Limit=1,
            )
            return len(resp.get("Items", [])) > 0
=======
            while True:
                resp = self._articles_table.scan(**scan_kwargs)
                if resp.get("Items"):
                    return True
                last_key = resp.get("LastEvaluatedKey")
                if not last_key:
                    break
                scan_kwargs["ExclusiveStartKey"] = last_key
            return False
>>>>>>> 4c1eb56 (Updated bugs)
        except Exception as exc:
            logger.warning(f"ARTICLE_EXISTS_CHECK_WARN error={exc}")
            return False

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        cluster: dict,
        week: str,
        week_slice: dict,
        video_titles: list[str],
    ) -> str:
        label = cluster.get("cluster_label", "")
        category = cluster.get("dominant_category", "Other")
        narrative_headline = cluster.get("narrative_headline") or ""
        narrative_summary = cluster.get("narrative_summary") or ""
        channels = cluster.get("channels", [])
        top_topics = cluster.get("top_topics", [])
        dominant_sentiment = cluster.get("dominant_sentiment", "neutral")

        video_count = week_slice.get("video_count", cluster.get("video_count", 0))
        view_count = week_slice.get("view_count", cluster.get("total_views", 0))
        breaking_count = week_slice.get(
            "breaking_count", cluster.get("breaking_count", 0)
        )
        week_sentiment = week_slice.get("sentiment_breakdown", {})

        # Build claims section
        classified_claims: dict = cluster.get("classified_claims") or {}
        consensus: list[dict] = classified_claims.get("consensus", [])
        debated: list[dict] = classified_claims.get("debated", [])
        top_claims: list[str] = cluster.get("top_claims", [])

        consensus_lines = []
        for c in consensus[:4]:
            claim = c.get("claim", "")
            sources = c.get("sources", [])[:4]
            sources_str = f" [Reported by: {', '.join(sources)}]" if sources else ""
            consensus_lines.append(f"• {claim}{sources_str}")

        # Fall back to top_claims when claim analysis hasn't been run
        if not consensus_lines and top_claims:
            consensus_lines = [f"• {c}" for c in top_claims[:5]]

        debated_lines = []
        for d in debated[:3]:
            claim = d.get("claim", "")
            perspectives = d.get("perspectives", [])[:3]
            persp_str = "; ".join(
                f"{p.get('channel', '')}: {p.get('sentiment', '')}"
                for p in perspectives
            )
            debated_lines.append(f"• {claim}\n  Perspectives: {persp_str}")

        # Format sentiment breakdown
        sentiment_parts = [f"{k}: {v}" for k, v in sorted(week_sentiment.items())]
        sentiment_str = (
            ", ".join(sentiment_parts) if sentiment_parts else dominant_sentiment
        )

        # Format view count
        view_str = (
            f"{view_count / 1_000_000:.1f}M"
            if view_count >= 1_000_000
            else (
                f"{view_count / 1_000:.0f}K" if view_count >= 1_000 else str(view_count)
            )
        )

        prompt = f"""You are a senior journalist writing analytical news articles for a digital media platform.

Generate a structured news article about the following story cluster.

═══════════════════════════════════════
STORY TOPIC:  {label}
CATEGORY:     {category}
WEEK:         {week}
═══════════════════════════════════════

COVERAGE METRICS
  Channels covering this story: {len(channels)}  |  Videos: {video_count}  |  Views: {view_str}
  Breaking reports: {breaking_count}
  Sentiment this week: {sentiment_str}
  Channels: {", ".join(channels[:8])}

KEY TOPICS:  {", ".join(top_topics)}

EDITORIAL CONTEXT (do NOT copy verbatim)
  Headline: {narrative_headline}
  Summary:  {narrative_summary}

═══════════════════════════════════════
VERIFIED FACTS  (confirmed by multiple sources)
{chr(10).join(consensus_lines) or "  No multi-source consensus claims on record."}

{"CONTESTED CLAIMS  (sources diverge on framing)" if debated_lines else ""}
{chr(10).join(debated_lines)}
═══════════════════════════════════════

REPRESENTATIVE VIDEO TITLES (for context only)
{chr(10).join(f"  - {t}" for t in video_titles) or "  (not available)"}

═══════════════════════════════════════
WRITING INSTRUCTIONS
1. Open with the single most newsworthy development — no filler openers.
2. Synthesise the verified facts into a coherent narrative (do not bullet-point them).
3. Where sources diverge, acknowledge the different framings in one sentence.
4. Include at least one specific number or statistic in the body.
5. Close with a single forward-looking sentence about implications or what to watch.
6. Tone: authoritative, neutral, past tense where describing events.
7. Length: 400–600 words in the body.

Return ONLY a valid JSON object with exactly these keys (no markdown fences):
{{
  "headline": "<compelling 8–15 word headline — different from the editorial context headline>",
  "overview": "<one concise sentence summary that includes a specific number or stat>",
  "body": "<full article text, prose paragraphs, no bullet points>"
}}"""

        return prompt

    # ── Parse Gemini response ─────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> tuple[str, str, str]:
        """
        Returns (headline, overview, body).
        Raises ValueError if parsing fails.
        """
        clean = raw.strip()
        # Strip markdown fences if present
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean).strip()

        parsed = json.loads(clean)

        headline = str(parsed.get("headline", "")).strip()
        overview = str(parsed.get("overview", "")).strip()
        body = str(parsed.get("body", "")).strip()

        if not headline or not body:
            raise ValueError("headline or body missing from parsed response")

        return headline, overview, body

    # ── Save to DynamoDB ──────────────────────────────────────────────────────

    def _save_article(
        self,
        cluster_id: int,
        cluster_label: str,
        week: str,
        week_number: int,
        title: str,
        overview: str,
        body: str,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        article_id = f"article-{uuid.uuid4().hex[:8]}"

        item = {
            "article_id": article_id,
            "cluster_id": _dec(cluster_id),
            "cluster_label": cluster_label,
            "week": week,
            "week_number": _dec(week_number),
            "title": title,
            "overview": overview,
            "body": body,
            "created_at": now,
            "updated_at": now,
        }

        self._articles_table.put_item(Item=item)
        return article_id

    def _delete_existing_articles(self, cluster_id: int, week_number: int) -> None:
        """Remove any existing articles for this cluster + week before re-generating."""
        try:
            resp = self._articles_table.scan(
                FilterExpression=Attr("cluster_id").eq(_dec(cluster_id))
                & Attr("week_number").eq(_dec(week_number)),
                ProjectionExpression="article_id",
            )
            for item in resp.get("Items", []):
                self._articles_table.delete_item(Key={"article_id": item["article_id"]})
        except Exception as exc:
            logger.warning(f"ARTICLE_DELETE_WARN cluster={cluster_id} error={exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_article_generation(
        self,
        week: Optional[str] = None,
        force: bool = False,
        cluster_id: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Generate articles for all active cluster × week combinations.

        Args:
            week:       Limit to a specific week (e.g. "week1" or "1").
            force:      Overwrite existing articles.
            cluster_id: Limit to a single cluster.
            dry_run:    Preview job count without generating or writing anything.

        Returns summary dict with counts and per-cluster details.
        """
        target_week = _normalize_week(week)
        clusters = self._get_all_clusters()

        if cluster_id is not None:
            clusters = [
                c for c in clusters if int(c.get("cluster_id", -1)) == cluster_id
            ]

        # Collect (cluster, week_slice, week_str) jobs
        jobs: list[tuple[dict, dict, str]] = []
        for c in clusters:
            for wd in c.get("week_data", []):
                w = wd.get("week", "")
                if not w or wd.get("video_count", 0) == 0:
                    continue
                if target_week and w != target_week:
                    continue
                jobs.append((c, wd, w))

        # De-duplicate: one job per (cluster_id, week)
        seen: set[tuple[int, str]] = set()
        deduped_jobs = []
        for c, wd, w in jobs:
            key = (int(c.get("cluster_id", -1)), w)
            if key not in seen:
                seen.add(key)
                deduped_jobs.append((c, wd, w))

        # Sort by week then by total views (descending)
        deduped_jobs.sort(
            key=lambda x: (_week_number(x[2]), -int(x[0].get("total_views", 0)))
        )

        generated = 0
        skipped = 0
        failed = 0
        weeks_seen: set[str] = set()
        per_cluster: dict[str, dict] = {}

        if dry_run:
            logger.info(
                f"ARTICLE_DRY_RUN jobs={len(deduped_jobs)} — skipping generation"
            )
            weeks_seen = {wk for _, _, wk in deduped_jobs}
            return {
                "articles_generated": 0,
                "articles_skipped": len(deduped_jobs),
                "articles_failed": 0,
                "weeks_processed": sorted(weeks_seen, key=_week_number),
                "per_cluster": {},
                "dry_run": True,
            }

        for c, week_slice, wk in deduped_jobs:
            cid = int(c.get("cluster_id", -1))
            wk_num = _week_number(wk)
            label = c.get("cluster_label", f"Cluster {cid}")
            weeks_seen.add(wk)

            logger.info(f"ARTICLE_JOB cluster={cid} week={wk} label={label!r}")

            if not force and self._article_exists(cid, wk_num):
                logger.info(f"ARTICLE_SKIP cluster={cid} week={wk} (already exists)")
                skipped += 1
                per_cluster[f"{cid}:{wk}"] = {"status": "skipped", "week": wk}
                continue

            if force:
                self._delete_existing_articles(cid, wk_num)

            video_titles = self._get_video_titles_for_cluster(cid)

            try:
                prompt = self._build_prompt(c, wk, week_slice, video_titles)
                raw = self._call_gemini(prompt)
                headline, overview, body = self._parse_response(raw)

                article_id = self._save_article(
                    cluster_id=cid,
                    cluster_label=label,
                    week=wk,
                    week_number=wk_num,
                    title=headline,
                    overview=overview,
                    body=body,
                )

                generated += 1
                per_cluster[f"{cid}:{wk}"] = {
                    "status": "generated",
                    "article_id": article_id,
                    "week": wk,
                    "headline": headline,
                }
                logger.info(
                    f"ARTICLE_GENERATED cluster={cid} week={wk} id={article_id}"
                )

                # Brief pause between Gemini calls to stay within rate limits
                time.sleep(3)

            except Exception as exc:
                failed += 1
                per_cluster[f"{cid}:{wk}"] = {
                    "status": "failed",
                    "week": wk,
                    "error": str(exc),
                }
                logger.error(f"ARTICLE_FAILED cluster={cid} week={wk} error={exc}")

        weeks_processed = sorted(weeks_seen, key=_week_number)
        logger.info(
            f"ARTICLE_RUN_COMPLETE generated={generated} "
            f"skipped={skipped} failed={failed}"
        )

        return {
            "articles_generated": generated,
            "articles_skipped": skipped,
            "articles_failed": failed,
            "weeks_processed": weeks_processed,
            "per_cluster": per_cluster,
            "dry_run": False,
        }

    # ── Read methods ──────────────────────────────────────────────────────────

    def get_articles(
        self,
        cluster_id: Optional[int] = None,
        week_number: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        List articles.  Optionally filter by cluster_id and/or week_number.
        Returns list of article metadata (no body).
        """
        filter_parts = []
        if cluster_id is not None:
            filter_parts.append(Attr("cluster_id").eq(cluster_id))
        if week_number is not None:
            filter_parts.append(Attr("week_number").eq(week_number))

        scan_kwargs: dict = {
            "ProjectionExpression": (
                "article_id, cluster_id, cluster_label, "
                "week_number, #wk, title, overview, created_at"
            ),
            "ExpressionAttributeNames": {"#wk": "week"},
        }

        if filter_parts:
            expr = filter_parts[0]
            for part in filter_parts[1:]:
                expr = expr & part
            scan_kwargs["FilterExpression"] = expr

        items: list[dict] = []
        while True:
            try:
                resp = self._articles_table.scan(**scan_kwargs)
            except ClientError as exc:
                logger.error(f"ARTICLE_LIST_ERROR error={exc}")
                break

            for raw in resp.get("Items", []):
                item = _deserialize(raw)
                items.append(
                    {
                        "article_id": item.get("article_id", ""),
                        "cluster_id": int(item.get("cluster_id", 0)),
                        "cluster_label": item.get("cluster_label", ""),
                        "week_number": int(item.get("week_number", 0)),
                        "title": item.get("title", ""),
                        "overview": item.get("overview", ""),
                        "created_at": item.get("created_at", ""),
                    }
                )
                if len(items) >= limit:
                    break

            if len(items) >= limit or "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        # Sort: newest week first, then by cluster_id
        items.sort(key=lambda x: (-x["week_number"], x["cluster_id"]))
        return items[:limit]

    def get_article_by_id(self, article_id: str) -> Optional[dict]:
        """
        Fetch a single article (including body) by article_id.
        Returns None if not found.
        """
        try:
            resp = self._articles_table.get_item(Key={"article_id": article_id})
        except ClientError as exc:
            logger.error(f"ARTICLE_GET_ERROR id={article_id} error={exc}")
            return None

        item = resp.get("Item")
        if not item:
            return None

        item = _deserialize(item)
        return {
            "article_id": item.get("article_id", ""),
            "cluster_id": int(item.get("cluster_id", 0)),
            "cluster_label": item.get("cluster_label", ""),
            "week_number": int(item.get("week_number", 0)),
            "title": item.get("title", ""),
            "overview": item.get("overview", ""),
            "body": item.get("body", ""),
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
        }
