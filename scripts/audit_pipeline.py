#!/usr/bin/env python3
"""
audit_pipeline.py — Post-pipeline data quality audit

Pulls cluster data from DynamoDB, sends it to Claude for a semantic review
(are labels accurate? do headlines reflect the videos?), then writes two files:

  audit_report.md    — full report with Claude's review (upload as GH artifact)
  audit_context.md   — paste this into a new Claude chat to continue the discussion

Optionally also posts the review as a GitHub issue.

Usage:
    python scripts/audit_pipeline.py
    python scripts/audit_pipeline.py --pipeline-status failed --error "Clustering crashed"
    python scripts/audit_pipeline.py --dry-run    # skip Claude API + GitHub issue
    python scripts/audit_pipeline.py --out-dir /tmp/audit
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

REGION = os.getenv("AWS_REGION", "us-east-2")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "youtube-videos")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "")  # e.g. "owner/placeholders-backend"
MODEL = "claude-sonnet-4-6"

# ── DynamoDB helpers ──────────────────────────────────────────────────────────


def scan_all(table, **kwargs):
    items = []
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def load_clusters(dynamodb) -> dict[int, dict]:
    table = dynamodb.Table("narrative-clusters")
    items = scan_all(table)
    clusters = {}
    for item in items:
        cid = int(item["cluster_id"])
        clusters[cid] = {
            "cluster_id": cid,
            "label": item.get("cluster_label", "?"),
            "headline": item.get("narrative_headline", ""),
            "summary": item.get("narrative_summary", ""),
            "status": item.get("status", "active"),
            "week_data": item.get("week_data", []),
            "top_topics": [str(t) for t in item.get("top_topics", [])],
            "top_claims": item.get("top_claims", []),
        }
    return clusters


def load_cluster_week_headlines(dynamodb) -> dict[int, list[dict]]:
    """Per-week headlines from cluster-weeks table. Returns {} if table missing."""
    try:
        table = dynamodb.Table("cluster-weeks")
        items = scan_all(table)
        by_cluster: dict[int, list[dict]] = defaultdict(list)
        for item in items:
            cid = int(item["cluster_id"])
            by_cluster[cid].append(
                {
                    "week": item.get("week", ""),
                    "headline": item.get("narrative_headline", ""),
                }
            )
        for cid in by_cluster:
            by_cluster[cid].sort(
                key=lambda x: int(x["week"][4:]) if x["week"][4:].isdigit() else 0
            )
        return dict(by_cluster)
    except Exception:
        return {}


def load_videos_by_cluster(dynamodb) -> dict[int, list[dict]]:
    """
    Scan youtube-videos and group video titles by cluster_id.
    Returns {cluster_id: [{title, channel, week}, ...]} sorted newest week first.
    """
    table = dynamodb.Table(DYNAMODB_TABLE)
    items = scan_all(
        table,
        ProjectionExpression="SortKey, cluster_id, #wk, title, PartitionKey",
        ExpressionAttributeNames={"#wk": "week"},
    )
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        cid = item.get("cluster_id")
        if cid is None:
            continue
        by_cluster[int(cid)].append(
            {
                "title": item.get("title", ""),
                "channel": item.get("PartitionKey", ""),
                "week": item.get("week", "unknown"),
            }
        )
    for cid in by_cluster:
        by_cluster[cid].sort(
            key=lambda v: int(v["week"][4:]) if v["week"][4:].isdigit() else 0,
            reverse=True,
        )
    return dict(by_cluster)


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data quality reviewer for Newsify, a YouTube narrative intelligence pipeline.

The pipeline ingests YouTube videos, clusters them by topic using HDBSCAN embeddings, then uses Gemini to generate cluster labels, narrative headlines, and summaries.

Your job is to review the clustering output and flag issues:
1. Does the cluster label accurately describe the videos in it?
2. Does the narrative headline reflect the actual video content?
3. Are there videos that clearly don't belong in the cluster?
4. Are headlines repeating or nearly identical across weeks? (known bug we are tracking)
5. Are any labels too vague or too broad to be meaningful?

Use ✓ / ⚠️ / ✗ as status indicators. Be concise and specific. Name the problematic video or headline directly. Focus on actionable findings only."""


def build_cluster_data(
    clusters: dict[int, dict],
    videos_by_cluster: dict[int, list[dict]],
    cluster_week_headlines: dict[int, list[dict]],
) -> str:
    """Build the raw cluster data block used in both the prompt and the context file."""
    lines = []
    active = {
        cid: c
        for cid, c in clusters.items()
        if c["status"] in ("active", "new", "declining")
    }

    for cid in sorted(active):
        c = clusters[cid]
        videos = videos_by_cluster.get(cid, [])
        week_headlines = cluster_week_headlines.get(cid, [])

        weeks_present = sorted(
            {v["week"] for v in videos if v["week"] != "unknown"},
            key=lambda w: int(w[4:]) if w[4:].isdigit() else 0,
        )

        lines.append(
            f"CLUSTER {cid} — {c['label']} ({c['status']}, weeks: {', '.join(weeks_present) or 'none'})"
        )
        lines.append(f'Headline: "{c["headline"]}"')

        if c["summary"]:
            lines.append(f"Summary: {c['summary'][:200]}")

        if c["top_topics"]:
            lines.append(f"Topics: {', '.join(c['top_topics'][:8])}")

        if c["top_claims"]:
            claim_lines = []
            for claim in c["top_claims"][:4]:
                if isinstance(claim, dict):
                    ctype = claim.get("type", "")
                    ctext = claim.get("claim", "")
                    claim_lines.append(f'  - {ctype}: "{ctext}"')
                else:
                    claim_lines.append(f"  - {claim}")
            if claim_lines:
                lines.append("Claims:")
                lines.extend(claim_lines)

        if videos:
            lines.append("Videos:")
            for v in videos[:8]:
                lines.append(
                    f'  - [{v["week"]}] "{v["title"] or "(no title)"}" ({v["channel"]})'
                )
            if len(videos) > 8:
                lines.append(f"  ... and {len(videos) - 8} more")
        else:
            lines.append("Videos: none assigned in youtube-videos table")

        # Per-week headlines for drift detection
        if week_headlines:
            lines.append("Per-week headlines:")
            for wh in week_headlines[-6:]:
                lines.append(f'  - {wh["week"]}: "{wh["headline"]}"')
        elif len(weeks_present) > 1:
            overviews = [
                (wd.get("week", ""), wd.get("week_overview", ""))
                for wd in c["week_data"]
                if wd.get("week_overview")
            ]
            if overviews:
                lines.append("Week overviews (fallback — cluster-weeks not populated):")
                for wk, ov in sorted(
                    overviews,
                    key=lambda x: int(x[0][4:]) if x[0][4:].isdigit() else 0,
                )[-4:]:
                    lines.append(f"  - {wk}: {ov[:150]}")

        lines.append("")

    return "\n".join(lines)


def build_prompt(cluster_data: str, pipeline_status: str, error_msg: str | None) -> str:
    header = []
    if pipeline_status == "failed":
        header.append(f"⚠️ Pipeline run FAILED: {error_msg or 'unknown error'}")
        header.append(
            "The data below reflects the last known good state in DynamoDB.\n"
        )
    else:
        header.append("Pipeline run completed successfully.\n")

    header.append(
        "Review each cluster below. Flag label accuracy, headline accuracy, "
        "misfit videos, and duplicate headlines across weeks.\n"
    )
    header.append("---\n")
    return "\n".join(header) + cluster_data + "\n---"


# ── Claude API call ───────────────────────────────────────────────────────────


def call_claude(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Report writers ────────────────────────────────────────────────────────────


def write_report(
    out_path: Path,
    review: str,
    cluster_data: str,
    pipeline_status: str,
    error_msg: str | None,
    current_week: str,
    cluster_count: int,
    active_count: int,
    video_count: int,
    job_id: str | None,
    now: str,
) -> None:
    status_emoji = "✅" if pipeline_status == "success" else "❌"
    lines = [
        "# Newsify Pipeline Audit Report",
        f"**Generated:** {now}",
        f"**Status:** {status_emoji} {pipeline_status}",
        f"**Week:** {current_week}",
        f"**Clusters:** {active_count} active / {cluster_count} total",
        f"**Videos:** {video_count}",
    ]
    if job_id:
        lines.append(f"**Job ID:** {job_id}")
    if error_msg:
        lines.append(f"**Error:** {error_msg}")

    lines += [
        "",
        "---",
        "",
        "## Claude Review",
        "",
        review,
        "",
        "---",
        "",
        "<details>",
        "<summary>Raw cluster data</summary>",
        "",
        "```",
        cluster_data[:12000],
        "```",
        "",
        "</details>",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"Report written → {out_path}")


def write_context_file(
    out_path: Path,
    review: str,
    cluster_data: str,
    pipeline_status: str,
    current_week: str,
    now: str,
) -> None:
    """
    A file you can paste directly into a new Claude chat to continue the discussion.
    Contains the raw data + the initial review so Claude has full context immediately.
    """
    lines = [
        "# Newsify Pipeline Audit — Context File",
        f"Generated: {now} | Week: {current_week} | Status: {pipeline_status}",
        "",
        "Paste this file into a Claude conversation to continue reviewing the pipeline audit results.",
        "You can ask things like:",
        '- "Dig deeper into cluster X"',
        '- "Why would cluster Y have low coherence?"',
        '- "What should we fix first?"',
        '- "Compare clusters 3 and 8 — are they the same story?"',
        "",
        "---",
        "",
        "## Raw Cluster Data",
        "",
        "```",
        cluster_data,
        "```",
        "",
        "---",
        "",
        "## Initial Claude Review",
        "",
        review,
        "",
        "---",
        "",
        "## Background on the pipeline",
        "",
        "- Videos come from YouTube via S3 (weekly batches, ~50 videos/week)",
        "- HDBSCAN clusters videos by embedding similarity (Qdrant)",
        "- Gemini labels each cluster and writes narrative_headline + narrative_summary",
        "- Known issue: headlines were duplicating verbatim across weeks (fix in progress via cluster-weeks table)",
        "- Known issue: Iran-related content was over-segmented into 4-5 sub-clusters",
        "- cluster_label (desk label) is stable across weeks; narrative_headline changes per week",
    ]
    out_path.write_text("\n".join(lines))
    print(f"Context file written → {out_path}")


# ── GitHub issue poster ───────────────────────────────────────────────────────


def post_github_issue(title: str, body: str) -> str:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return ""
    payload = json.dumps(
        {"title": title, "body": body[:65000], "labels": ["audit", "pipeline"]}
    ).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPO}/issues",
        data=payload,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        return result.get("html_url", "")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-status", default="success", choices=["success", "failed"]
    )
    parser.add_argument("--error", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--out-dir", default=".", help="Directory to write output files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Claude API + GitHub issue, just print data",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to DynamoDB...")
    dynamodb = boto3.resource("dynamodb", region_name=REGION)

    print("Loading clusters...")
    clusters = load_clusters(dynamodb)
    print(f"  {len(clusters)} clusters")

    print("Loading per-week headlines from cluster-weeks...")
    cluster_week_headlines = load_cluster_week_headlines(dynamodb)
    print(f"  {len(cluster_week_headlines)} clusters have per-week headline data")

    print("Loading video titles by cluster...")
    videos_by_cluster = load_videos_by_cluster(dynamodb)
    total_videos = sum(len(v) for v in videos_by_cluster.values())
    print(f"  {total_videos} videos across {len(videos_by_cluster)} clusters")

    # Derive current week from data
    all_weeks = {
        v["week"]
        for vlist in videos_by_cluster.values()
        for v in vlist
        if v["week"] != "unknown"
    }
    current_week = (
        max(all_weeks, key=lambda w: int(w[4:]) if w[4:].isdigit() else 0)
        if all_weeks
        else "unknown"
    )

    active_count = sum(1 for c in clusters.values() if c["status"] in ("active", "new"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("Building cluster data block...")
    cluster_data = build_cluster_data(
        clusters, videos_by_cluster, cluster_week_headlines
    )

    if args.dry_run:
        print("\n--- Cluster data (dry run, no Claude call) ---")
        print(cluster_data)
        dry_path = out_dir / "cluster_data.txt"
        dry_path.write_text(cluster_data)
        print(f"\nCluster data written → {dry_path}")
        return

    print("Calling Claude for review...")
    prompt = build_prompt(cluster_data, args.pipeline_status, args.error)
    review = call_claude(prompt)
    print("  Review received.")

    # Write files
    report_path = out_dir / "audit_report.md"
    context_path = out_dir / "audit_context.md"

    write_report(
        report_path,
        review=review,
        cluster_data=cluster_data,
        pipeline_status=args.pipeline_status,
        error_msg=args.error,
        current_week=current_week,
        cluster_count=len(clusters),
        active_count=active_count,
        video_count=total_videos,
        job_id=args.job_id,
        now=now,
    )

    write_context_file(
        context_path,
        review=review,
        cluster_data=cluster_data,
        pipeline_status=args.pipeline_status,
        current_week=current_week,
        now=now,
    )

    # Post GitHub issue (optional — only if GITHUB_TOKEN + GITHUB_REPO set)
    if GITHUB_TOKEN and GITHUB_REPO:
        status_emoji = "✅" if args.pipeline_status == "success" else "❌"
        issue_title = f"{status_emoji} Pipeline Audit — {current_week} — {now[:10]}"
        if args.job_id:
            issue_title += f" (job {args.job_id[:8]})"

        issue_body = (
            f"## Pipeline Audit — {current_week}\n"
            f"**Generated:** {now} | **Status:** {args.pipeline_status}\n"
            f"**Clusters:** {active_count} active / {len(clusters)} total | **Videos:** {total_videos}\n"
            + (f"**Error:** {args.error}\n" if args.error else "")
            + f"\n---\n\n## Claude Review\n\n{review}\n\n---\n\n"
            f"<details><summary>Raw cluster data</summary>\n\n```\n{cluster_data[:10000]}\n```\n\n</details>"
        )

        print("Posting GitHub issue...")
        url = post_github_issue(issue_title, issue_body)
        if url:
            print(f"Issue: {url}")
    else:
        print("Skipping GitHub issue (GITHUB_TOKEN or GITHUB_REPO not set)")

    print("\nDone.")


if __name__ == "__main__":
    main()
