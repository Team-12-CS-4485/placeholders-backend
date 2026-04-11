"""
backfill_cluster_weeks.py - Regenerate narrative headlines for weeks 1-6

Reads all clusters from narrative-clusters, processes each week in order
(week1 → weekN), calls Gemini with a dedup-aware prompt, writes results
to cluster-weeks table, and patches narrative-clusters with the latest headline.

Usage:
    python scripts/backfill_cluster_weeks.py
    python scripts/backfill_cluster_weeks.py --cluster-id 7   # single cluster
    python scripts/backfill_cluster_weeks.py --dry-run         # print prompts, no writes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

import boto3
from google import genai
from google.genai import types

from app.core.config import settings
from app.services.dynamo_service import DynamoService

REGION = os.getenv("AWS_REGION", "us-east-2")


# ── Gemini helpers ─────────────────────────────────────────────────────────────


class GeminiClient:
    def __init__(self):
        self._keys = settings.genai_api_keys
        self._idx = 0
        self._client = genai.Client(api_key=self._keys[0])

    def _rotate(self) -> bool:
        next_idx = self._idx + 1
        if next_idx >= len(self._keys):
            return False
        self._idx = next_idx
        self._client = genai.Client(api_key=self._keys[self._idx])
        print(f"  [key rotated → {self._idx}]")
        return True

    def generate(self, prompt: str) -> dict:
        """Call Gemini with retry + key rotation. Returns parsed JSON dict."""
        max_attempts = 6 * len(self._keys)
        for attempt in range(max_attempts):
            try:
                response = self._client.models.generate_content(
                    model=settings.gemini_model_id,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    ),
                )
                time.sleep(0.5)
                raw = getattr(response, "text", "") or str(response)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                return json.loads(raw)
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = (
                    "429" in exc_str
                    or "RESOURCE_EXHAUSTED" in exc_str
                    or getattr(exc, "code", None) == 429
                )
                is_server_error = (
                    "503" in exc_str
                    or "500" in exc_str
                    or "UNAVAILABLE" in exc_str
                    or getattr(exc, "code", None) in (500, 503)
                )

                if is_rate_limit:
                    if self._rotate():
                        continue
                    print("  [all keys exhausted, waiting 60s]")
                    self._idx = 0
                    self._client = genai.Client(api_key=self._keys[0])
                    time.sleep(60)
                    continue

                if is_server_error:
                    wait = min(5 * (attempt + 1), 30)
                    print(f"  [server error, waiting {wait}s, attempt {attempt + 1}]")
                    time.sleep(wait)
                    continue

                is_bad_json = isinstance(exc, (json.JSONDecodeError, ValueError))
                if is_bad_json:
                    wait = min(5 * (attempt + 1), 20)
                    print(
                        f"  [bad/empty response, waiting {wait}s, attempt {attempt + 1}]"
                    )
                    time.sleep(wait)
                    continue

                print(f"  [gemini fatal error: {exc}]")
                raise

        raise RuntimeError("Gemini: max attempts exhausted")


# ── Prompt builder ─────────────────────────────────────────────────────────────


def build_backfill_prompt(
    cluster_label: str,
    top_topics: list[str],
    week_name: str,
    wd: dict,
    titles: list[str],
    claims: list[str],
    prior_block: str,
) -> str:
    week_sentiments = wd.get("sentiment_breakdown", {})
    dominant_sentiment = (
        max(week_sentiments, key=week_sentiments.get) if week_sentiments else "neutral"
    )

    prior_section = ""
    if prior_block:
        prior_section = f"""
PRIOR HEADLINES FOR THIS CLUSTER (do NOT repeat or rephrase any of these):
{prior_block}

The new headline MUST describe a different angle or development than all of the above.
If no genuinely new development exists, prefix with "Week N: " and describe what continued."""

    return f"""You are a news editor. Given YouTube video data for cluster "{cluster_label}" during {week_name}, write a narrative headline and summary covering ONLY this week's developments.

VIDEO TITLES THIS WEEK: {titles}
KEY CLAIMS THIS WEEK: {claims}
STATS: {wd.get("video_count", 0)} videos, {wd.get("view_count", 0)} views, dominant sentiment: {dominant_sentiment}
CLUSTER TOPICS: {top_topics}
{prior_section}
Do NOT start the headline with "Week N:" or any week prefix — the week is already known from context.
Return ONLY valid JSON, no markdown:
{{"headline": "...", "summary": "...", "week_overview": "2 sentences: main development this week. Scale/sentiment context."}}"""


# ── Main backfill logic ────────────────────────────────────────────────────────


def run_backfill(cluster_id_filter: int | None, dry_run: bool) -> None:
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    clusters_table = dynamodb.Table("narrative-clusters")
    dynamo_svc = DynamoService(dynamodb)
    gemini = GeminiClient()

    # Scan all clusters (include inactive so we backfill their history too)
    all_clusters = []
    scan_kwargs: dict = {}
    while True:
        resp = clusters_table.scan(**scan_kwargs)
        all_clusters.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    if cluster_id_filter is not None:
        all_clusters = [
            c for c in all_clusters if int(c["cluster_id"]) == cluster_id_filter
        ]
        if not all_clusters:
            print(f"Cluster {cluster_id_filter} not found.")
            return

    print(f"Backfilling {len(all_clusters)} cluster(s)...\n")

    for cluster in all_clusters:
        cluster_id = int(cluster["cluster_id"])
        cluster_label = cluster.get("cluster_label", f"Cluster {cluster_id}")
        top_topics = list(cluster.get("top_topics", []))
        week_data = list(cluster.get("week_data", []))

        if not week_data:
            print(f"[{cluster_id}] {cluster_label} — no week_data, skipping")
            continue

        print(f"[{cluster_id}] {cluster_label}")

        weeks = sorted(
            week_data,
            key=lambda w: (
                int(w["week"][4:])
                if w["week"].startswith("week") and w["week"][4:].isdigit()
                else 9999
            ),
        )

        prior_headlines: list[dict] = []
        latest_result: dict | None = None
        latest_week_name: str | None = None

        for wd in weeks:
            week_name = wd["week"]

            # Pull actual video data for this cluster+week
            videos = dynamo_svc.get_video_data_for_cluster(
                cluster_id, week_name, limit=6
            )
            titles = [v["title"] for v in videos if v.get("title")][:6]
            claims = list(
                dict.fromkeys(c for v in videos for c in v.get("key_claims", []))
            )[:6]

            # Build prior headlines block
            prior_block = "\n".join(
                f'- {p["week"]}: "{p["headline"]}"' for p in prior_headlines[-4:]
            )

            prompt = build_backfill_prompt(
                cluster_label, top_topics, week_name, wd, titles, claims, prior_block
            )

            if dry_run:
                print(f"  {week_name}: [dry-run, would call Gemini]")
                print(f"    prompt snippet: {prompt[:120].strip()}...")
                prior_headlines.append(
                    {"week": week_name, "headline": f"[dry-run {week_name}]"}
                )
                continue

            result = gemini.generate(prompt)

            # Verbatim dupe check — retry once
            recent_hls = [p["headline"].strip().lower() for p in prior_headlines[-2:]]
            if result.get("headline", "").strip().lower() in recent_hls:
                retry_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Return a headline completely different from all prior weeks listed above."
                )
                result = gemini.generate(retry_prompt)

            # Write to cluster-weeks (idempotent)
            week_sentiments = wd.get("sentiment_breakdown", {})
            dominant_sentiment = (
                max(week_sentiments, key=week_sentiments.get)
                if week_sentiments
                else "neutral"
            )
            dynamo_svc.save_cluster_week_snapshot(
                cluster_id,
                week_name,
                {
                    "narrative_headline": result.get("headline", ""),
                    "narrative_summary": result.get("summary", ""),
                    "week_overview": result.get("week_overview", ""),
                    "top_claims": claims,
                    "top_topics": top_topics,
                    "video_count": wd.get("video_count", 0),
                    "channel_count": wd.get("channel_count", 0),
                    "view_count": wd.get("view_count", 0),
                    "breaking_count": wd.get("breaking_count", 0),
                    "dominant_sentiment": dominant_sentiment,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            prior_headlines.append(
                {"week": week_name, "headline": result.get("headline", "")}
            )
            latest_result = result
            latest_week_name = week_name
            print(f"  {week_name}: {result.get('headline', '')}")

        # Patch narrative-clusters with the latest week's headline
        if not dry_run and latest_result and latest_week_name:
            clusters_table.update_item(
                Key={"cluster_id": Decimal(str(cluster_id))},
                UpdateExpression="SET narrative_headline = :h, narrative_summary = :s",
                ExpressionAttributeValues={
                    ":h": latest_result.get("headline", ""),
                    ":s": latest_result.get("summary", ""),
                },
            )
            print(f"  → patched narrative-clusters with {latest_week_name} headline")

        print()


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Backfill cluster-weeks table for weeks 1-6"
    )
    parser.add_argument(
        "--cluster-id",
        type=int,
        default=None,
        help="Rerun a single cluster by stable ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling Gemini or writing to DynamoDB",
    )
    args = parser.parse_args()
    run_backfill(args.cluster_id, args.dry_run)


if __name__ == "__main__":
    main()
