"""
article_service.py - News Article Generation Service

Reads narrative cluster data from DynamoDB (via DynamoService), generates
structured news articles via Gemini (via GeminiService), and persists them
to the articles DynamoDB table.

Key design decisions:
- Uses classified_claims + top_claims already stored in narrative-clusters
  (zero additional S3/Qdrant reads)
- Optionally enriches with a few video titles from the cluster-index GSI
- Skips clusters / weeks that already have an article (idempotent by default)
- Delegates Gemini key rotation / retry to GeminiService

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
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.services.dynamo_service import DynamoService
from app.services.gemini_service import GeminiService, KeysExhaustedError

logger = get_logger(__name__)

_WEEK_RE = re.compile(r"week(\d+)", re.IGNORECASE)

NUM_WORKERS = 10


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

    def __init__(
        self,
        dynamo_service: Optional[DynamoService] = None,
        gemini_service: Optional[GeminiService] = None,
    ):
        self._dynamo = dynamo_service or DynamoService()
        self._gemini = gemini_service or GeminiService(api_keys=settings.genai_api_keys)

    # ── Worker helpers ────────────────────────────────────────────────────────

    def _make_worker_gemini_services(self) -> list[GeminiService]:
        """Split all API keys into NUM_WORKERS groups of ~3 keys each."""
        keys = self._gemini.api_keys
        group_size = max(1, len(keys) // NUM_WORKERS)
        services = []
        for i in range(0, len(keys), group_size):
            services.append(
                GeminiService(
                    api_keys=keys[i : i + group_size],
                    fail_on_exhaustion=True,
                )
            )
            if len(services) == NUM_WORKERS:
                break
        return services

    def _generate_single_article(
        self,
        cluster: dict,
        week_slice: dict,
        wk: str,
        gemini_service: GeminiService,
        force: bool,
        existing_set: set,
    ) -> dict:
        """
        Generate and persist one article for a (cluster, week) combination.

        Returns { status: "generated"|"skipped", article_id?, week, headline? }.
        Raises KeysExhaustedError if the worker's key pool is exhausted.
        """
        cid = int(cluster.get("cluster_id", -1))
        wk_num = _week_number(wk)
        label = cluster.get("cluster_label", f"Cluster {cid}")

        if not force and (cid, wk_num) in existing_set:
            logger.debug(f"ARTICLE_SKIP cluster={cid} week={wk} (already exists)")
            return {"status": "skipped", "week": wk}

        if force:
            self._dynamo.delete_articles(cid, wk_num)

        video_titles = self._dynamo.get_video_titles_for_cluster(cid, week=wk)
        prompt = self._build_prompt(cluster, wk, week_slice, video_titles)
        raw = gemini_service._gemini(prompt)
        headline, overview, body = self._parse_response(raw)

        now = datetime.now(timezone.utc).isoformat()
        article_id = f"article-{uuid.uuid4().hex[:8]}"

        self._dynamo.save_article(
            {
                "article_id": article_id,
                "cluster_id": _dec(cid),
                "cluster_label": label,
                "week": wk,
                "week_number": _dec(wk_num),
                "title": headline,
                "overview": overview,
                "body": body,
                "created_at": now,
                "updated_at": now,
            }
        )

        logger.info(f"ARTICLE_GENERATED cluster={cid} week={wk} id={article_id}")
        return {
            "status": "generated",
            "article_id": article_id,
            "week": wk,
            "headline": headline,
        }

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

        prompt = f"""You are a senior journalist writing analytical \
news articles for a digital media platform.

Write a news article covering what happened with this story DURING {week} ONLY.
Do not recap the full history — focus on what was new or developing this specific week.

═══════════════════════════════════════
STORY TOPIC:  {label}
CATEGORY:     {category}
WEEK:         {week}
═══════════════════════════════════════

THIS WEEK'S COVERAGE
  Channels covering this story: {len(channels)}  |  Videos this week: {video_count}  |  Views: {view_str}
  Breaking reports this week: {breaking_count}
  Sentiment this week: {sentiment_str}
  Channels: {", ".join(channels[:8])}

THIS WEEK'S VIDEO TITLES (what was actually published this week)
{chr(10).join(f"  - {t}" for t in video_titles) or "  (not available)"}

═══════════════════════════════════════
VERIFIED FACTS  (confirmed by multiple sources)
{chr(10).join(consensus_lines) or "  No multi-source consensus claims on record."}

{"CONTESTED CLAIMS  (sources diverge on framing)" if debated_lines else ""}
{chr(10).join(debated_lines)}
═══════════════════════════════════════

ONGOING STORY CONTEXT (background only — do NOT re-report this as new)
  {narrative_headline}
  {narrative_summary}

═══════════════════════════════════════
WRITING INSTRUCTIONS
1. Open with the single most newsworthy development FROM THIS WEEK — no filler openers.
2. Synthesise the verified facts into a coherent narrative (do not bullet-point them).
3. Where sources diverge, acknowledge the different framings in one sentence.
4. Include at least one specific number or statistic in the body.
5. Close with a single forward-looking sentence about what to watch next.
6. Tone: authoritative, neutral, past tense where describing events.
7. Length: 400–600 words in the body.
8. The headline must reflect what was NEW this week, not the overall story arc.

Return ONLY a valid JSON object with exactly these keys (no markdown fences):
{{
  "headline": "<compelling 8–15 word headline about what happened THIS WEEK specifically>",
  "overview": "<one concise sentence summary of this week's developments with a specific stat>",
  "body": "<full article text, prose paragraphs, no bullet points>"
}}"""

        return prompt

    # ── Parse Gemini response ─────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> tuple[str, str, str]:
        """
        Returns (headline, overview, body).
        Raises ValueError if parsing fails.

        Gemini often returns literal newlines inside JSON string values
        (the body is multi-paragraph prose). We try json.loads first,
        then fall back to regex extraction of each field.
        """
        clean = raw.strip()
        # Strip markdown fences if present
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean).strip()

        try:
            parsed = json.loads(clean)
            headline = str(parsed.get("headline", "")).strip()
            overview = str(parsed.get("overview", "")).strip()
            body = str(parsed.get("body", "")).strip()
        except json.JSONDecodeError:
            # Regex extraction: pull each field from the raw JSON text
            def _extract(key: str) -> str:
                pattern = rf'"{key}"\s*:\s*"(.*?)"(?:\s*[,}}])'
                m = re.search(pattern, clean, re.DOTALL)
                if not m:
                    return ""
                val = m.group(1)
                # Unescape any JSON escapes
                val = val.replace("\\n", "\n").replace('\\"', '"')
                return val.strip()

            headline = _extract("headline")
            overview = _extract("overview")
            # Body is last and longest — grab everything after "body": "
            body_match = re.search(r'"body"\s*:\s*"(.*)"', clean, re.DOTALL)
            body = ""
            if body_match:
                body = body_match.group(1)
                body = body.replace("\\n", "\n").replace('\\"', '"')
                body = body.strip()

        if not headline or not body:
            logger.error(f"ARTICLE_PARSE_EMPTY raw_start={raw[:300]}")
            raise ValueError("headline or body missing from parsed response")

        return headline, overview, body

    # ── Public API ────────────────────────────────────────────────────────────

    def cleanup_orphaned_articles(self) -> int:
        """
        Delete articles whose cluster_id no longer exists in narrative-clusters.
        Called before article generation each pipeline run to prevent accumulation
        of stale articles from purged or renumbered clusters.
        Returns count of deleted articles.
        """
        valid_cluster_ids = {
            int(c["cluster_id"])
            for c in self._dynamo.get_all_clusters()
            if c.get("cluster_id") is not None
        }
        articles_table = self._dynamo._articles_table()
        deleted = 0
        scan_kwargs: dict = {"ProjectionExpression": "article_id, cluster_id"}
        try:
            while True:
                resp = articles_table.scan(**scan_kwargs)
                for item in resp.get("Items", []):
                    cid = item.get("cluster_id")
                    if cid is not None and int(cid) not in valid_cluster_ids:
                        articles_table.delete_item(
                            Key={"article_id": item["article_id"]}
                        )
                        deleted += 1
                if "LastEvaluatedKey" not in resp:
                    break
                scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        except Exception as exc:
            logger.warning(f"ORPHAN_ARTICLE_CLEANUP_WARN error={exc}")

        if deleted:
            logger.info(f"ORPHAN_ARTICLES_DELETED count={deleted}")
        return deleted

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
            dry_run:    Preview job count without generating or writing.

        Returns summary dict with counts and per-cluster details.
        """
        target_week = _normalize_week(week)
        clusters = [_deserialize(c) for c in self._dynamo.get_all_clusters()]

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
            key=lambda x: (
                _week_number(x[2]),
                -int(x[0].get("total_views", 0)),
            )
        )

        # Load existing articles once — avoids N full-table scans (one per cluster×week).
        # Build a set of (cluster_id, week_number) pairs for O(1) existence checks.
        existing_articles = self._dynamo.get_articles(limit=10_000)
        existing_set: set[tuple[int, int]] = {
            (a["cluster_id"], a["week_number"]) for a in existing_articles
        }
        logger.info(f"ARTICLES_EXISTING_LOADED count={len(existing_set)}")

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

            # Build per-cluster preview — check which articles already exist
            dry_per_cluster: dict[str, dict] = {}
            would_generate = 0
            would_skip = 0
            for c, wd, wk in deduped_jobs:
                cid = str(int(c.get("cluster_id", -1)))
                label = c.get("cluster_label", "Unknown")
                exists = (int(cid), _week_number(wk)) in existing_set
                status = "would_skip" if exists else "would_generate"
                if exists:
                    would_skip += 1
                else:
                    would_generate += 1

                dry_per_cluster.setdefault(
                    cid,
                    {
                        "cluster_label": label,
                        "weeks": {},
                    },
                )
                dry_per_cluster[cid]["weeks"][wk] = {
                    "status": status,
                    "video_count": wd.get("video_count", 0),
                    "view_count": wd.get("view_count", 0),
                    "simulated_structure": (
                        None
                        if exists
                        else {
                            "title": f"[Gemini would generate headline for '{label}' in {wk}]",
                            "overview": "[Gemini would generate 2-3 sentence overview with key stats]",
                            "body": f"[Gemini would generate 400-600 word article covering {wd.get('video_count', 0)} videos across {wk}]",
                        }
                    ),
                }

            return {
                "articles_generated": 0,
                "articles_would_generate": would_generate,
                "articles_would_skip": would_skip,
                "articles_failed": 0,
                "weeks_processed": sorted(weeks_seen, key=_week_number),
                "per_cluster": dry_per_cluster,
                "dry_run": True,
            }

        _start = time.time()
        worker_services = self._make_worker_gemini_services()
        retry_jobs: list[tuple[dict, dict, str]] = []

        new_jobs = [(c, wd, wk) for c, wd, wk in deduped_jobs]
        logger.info(
            f"ARTICLES_START jobs={len(new_jobs)} skipping={len(deduped_jobs) - len(new_jobs)} workers={NUM_WORKERS}"
        )

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futures = {}
            for i, (c, week_slice, wk) in enumerate(deduped_jobs):
                worker_idx = i % NUM_WORKERS
                cid = int(c.get("cluster_id", -1))
                logger.info(f"WORKER_START worker={worker_idx} cluster={cid} week={wk}")
                fut = pool.submit(
                    self._generate_single_article,
                    c,
                    week_slice,
                    wk,
                    worker_services[worker_idx],
                    force,
                    existing_set,
                )
                futures[fut] = (c, week_slice, wk, worker_idx, time.time())

            for future in as_completed(futures):
                c, week_slice, wk, worker_idx, t0 = futures[future]
                cid = int(c.get("cluster_id", -1))
                key = f"{cid}:{wk}"
                weeks_seen.add(wk)
                elapsed_ms = int((time.time() - t0) * 1000)

                try:
                    result = future.result()
                    per_cluster[key] = result
                    if result["status"] == "generated":
                        generated += 1
                        logger.info(
                            f"WORKER_COMPLETE worker={worker_idx} cluster={cid} week={wk} elapsed_ms={elapsed_ms}"
                        )
                    else:
                        skipped += 1
                except KeysExhaustedError:
                    retry_jobs.append((c, week_slice, wk))
                    logger.warning(
                        f"WORKER_RETRY worker={worker_idx} cluster={cid} week={wk} reason=keys_exhausted"
                    )
                except Exception as exc:
                    failed += 1
                    per_cluster[key] = {
                        "status": "failed",
                        "week": wk,
                        "error": str(exc),
                    }
                    logger.error(
                        f"WORKER_FAILED worker={worker_idx} cluster={cid} week={wk} error={exc}"
                    )

        # Retry pass — sequential, full key pool (fail_on_exhaustion=False)
        if retry_jobs:
            logger.info(f"RETRY_START jobs={len(retry_jobs)}")
            for c, week_slice, wk in retry_jobs:
                cid = int(c.get("cluster_id", -1))
                key = f"{cid}:{wk}"
                weeks_seen.add(wk)
                t0 = time.time()
                try:
                    result = self._generate_single_article(
                        c, week_slice, wk, self._gemini, force, existing_set
                    )
                    per_cluster[key] = result
                    if result["status"] == "generated":
                        generated += 1
                        logger.info(
                            f"RETRY_COMPLETE cluster={cid} week={wk} elapsed_ms={int((time.time()-t0)*1000)}"
                        )
                    else:
                        skipped += 1
                except Exception as exc:
                    failed += 1
                    per_cluster[key] = {
                        "status": "failed",
                        "week": wk,
                        "error": str(exc),
                    }
                    logger.error(f"RETRY_FAILED cluster={cid} week={wk} error={exc}")

        weeks_processed = sorted(weeks_seen, key=_week_number)
        logger.info(
            f"ARTICLES_COMPLETE generated={generated} skipped={skipped} "
            f"failed={failed} retried={len(retry_jobs)} elapsed_s={time.time() - _start:.1f}"
        )

        return {
            "articles_generated": generated,
            "articles_skipped": skipped,
            "articles_failed": failed,
            "weeks_processed": weeks_processed,
            "per_cluster": per_cluster,
            "dry_run": False,
        }

    # ── Read methods (delegate to DynamoService) ─────────────────────────────

    def get_articles(
        self,
        cluster_id: Optional[int] = None,
        week_number: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List articles. Optionally filter by cluster_id / week_number."""
        return self._dynamo.get_articles(
            cluster_id=cluster_id,
            week_number=week_number,
            limit=limit,
        )

    def get_article_by_id(self, article_id: str) -> Optional[dict]:
        """Fetch a single article (including body) by article_id."""
        return self._dynamo.get_article_by_id(article_id)
