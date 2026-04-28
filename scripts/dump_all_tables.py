#!/usr/bin/env python3
"""
dump_all_tables.py — Unified snapshot of all four DynamoDB tables.

Reads youtube-videos, narrative-clusters, cluster-weeks, and articles,
cross-references them, and writes a single JSON file with:

  - summary: counts + cross-table consistency checks
  - clusters: full cluster data with their videos, week snapshots, and articles joined in
  - orphans: videos with no matching cluster, articles with no matching cluster, etc.

Read-only. Does not mutate anything.

Usage:
    python scripts/dump_all_tables.py
    python scripts/dump_all_tables.py --out /tmp/snapshot.json
    python scripts/dump_all_tables.py --cluster 3   # single cluster only
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-2")
VIDEOS_TABLE = os.getenv("DYNAMODB_TABLE", "youtube-videos")
CLUSTERS_TABLE = "narrative-clusters"
CLUSTER_WEEKS_TABLE = "cluster-weeks"
ARTICLES_TABLE = "articles"


# ── Helpers ───────────────────────────────────────────────────────────────────


def scan_all(table, **kwargs):
    items = []
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def to_jsonable(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_clusters(dynamodb) -> list[dict]:
    table = dynamodb.Table(CLUSTERS_TABLE)
    return scan_all(table)


def load_videos(dynamodb) -> list[dict]:
    table = dynamodb.Table(VIDEOS_TABLE)
    return scan_all(
        table,
        ProjectionExpression=(
            "PartitionKey, SortKey, title, #wk, cluster_id, cluster_label, "
            "cluster_confidence, topics, sentiment, key_claims, is_breaking, "
            "viewCount, likeCount, commentCount, publishedAt, "
            "thumbnail_tone, thumbnail_clickbait_score, indexed_at, chunk_count"
        ),
        ExpressionAttributeNames={"#wk": "week"},
    )


def load_cluster_weeks(dynamodb) -> list[dict]:
    table = dynamodb.Table(CLUSTER_WEEKS_TABLE)
    return scan_all(table)


def load_articles(dynamodb) -> list[dict]:
    table = dynamodb.Table(ARTICLES_TABLE)
    return scan_all(table)


# ── Cross-reference builder ───────────────────────────────────────────────────


def build_snapshot(
    clusters: list[dict],
    videos: list[dict],
    cluster_weeks: list[dict],
    articles: list[dict],
    filter_cid: int | None = None,
) -> dict:

    # Index everything
    clusters_by_id: dict[int, dict] = {}
    for c in clusters:
        cid = int(c["cluster_id"])
        clusters_by_id[cid] = c

    videos_by_cluster: dict[int, list[dict]] = defaultdict(list)
    unclustered_videos: list[dict] = []
    for v in videos:
        cid = v.get("cluster_id")
        if cid is None:
            unclustered_videos.append(v)
        else:
            videos_by_cluster[int(cid)].append(v)

    weeks_by_cluster: dict[int, list[dict]] = defaultdict(list)
    for w in cluster_weeks:
        cid = int(w["cluster_id"])
        weeks_by_cluster[cid].append(w)

    articles_by_cluster: dict[int, list[dict]] = defaultdict(list)
    orphan_articles: list[dict] = []
    for a in articles:
        cid = a.get("cluster_id")
        if cid is None:
            orphan_articles.append(a)
        else:
            articles_by_cluster[int(cid)].append(a)

    # ── Per-cluster rows ──
    cluster_rows = []
    all_cids = sorted(set(clusters_by_id) | set(videos_by_cluster))

    for cid in all_cids:
        if filter_cid is not None and cid != filter_cid:
            continue

        c = clusters_by_id.get(cid, {})
        vids = videos_by_cluster.get(cid, [])
        wks = sorted(weeks_by_cluster.get(cid, []), key=lambda w: w.get("week", ""))
        arts = articles_by_cluster.get(cid, [])

        classified = c.get("classified_claims") or {
            "consensus": [],
            "debated": [],
            "unique": [],
        }
        top_claims = c.get("top_claims") or []

        # Per-video signals
        vid_rows = []
        for v in sorted(vids, key=lambda x: x.get("week", "")):
            vid_rows.append(
                {
                    "video_id": v.get("SortKey", ""),
                    "channel": v.get("PartitionKey", ""),
                    "title": v.get("title", ""),
                    "week": v.get("week", ""),
                    "published_at": v.get("publishedAt", ""),
                    "cluster_confidence": to_jsonable(v.get("cluster_confidence")),
                    "has_topics": bool(v.get("topics")),
                    "has_key_claims": bool(v.get("key_claims")),
                    "key_claims_count": len(v.get("key_claims") or []),
                    "has_transcript_chunked": bool(v.get("chunk_count")),
                    "chunk_count": int(v.get("chunk_count") or 0),
                    "sentiment": v.get("sentiment", ""),
                    "is_breaking": bool(v.get("is_breaking", False)),
                    "thumbnail_tone": v.get("thumbnail_tone", ""),
                    "thumbnail_clickbait_score": to_jsonable(
                        v.get("thumbnail_clickbait_score")
                    ),
                    "view_count": int(v.get("viewCount") or 0),
                    "indexed_at": v.get("indexed_at", ""),
                }
            )

        # Consistency checks for this cluster
        issues = []
        if not c:
            issues.append("videos reference this cluster_id but no cluster row exists")
        if c and not c.get("narrative_headline"):
            issues.append(
                "narrative_headline is missing (Gemini labeling may have failed)"
            )
        if (
            c
            and not top_claims
            and not any(classified.get(k) for k in ("consensus", "debated", "unique"))
        ):
            issues.append("top_claims and classified_claims are both empty")
        elif c and not top_claims:
            issues.append(
                "top_claims is empty (claim analysis wrote classified_claims but top_claims not refreshed yet)"
            )
        if c:
            attr_count = int(c.get("video_count") or 0)
            actual_count = len(vids)
            if attr_count != actual_count:
                issues.append(
                    f"video_count mismatch: cluster says {attr_count}, actual videos assigned = {actual_count}"
                )
        cw_weeks = {w.get("week") for w in wks}
        cluster_week_data = c.get("week_data") or []
        wd_weeks = {wd.get("week") for wd in cluster_week_data}
        missing_snapshots = wd_weeks - cw_weeks
        if missing_snapshots:
            issues.append(
                f"cluster-weeks missing snapshots for: {sorted(missing_snapshots)}"
            )
        if arts and not c:
            issues.append("articles exist for this cluster_id but no cluster row")

        cluster_rows.append(
            {
                "cluster_id": cid,
                "cluster_label": c.get("cluster_label", "(no cluster row)"),
                "status": c.get("status", ""),
                "created_at": c.get("created_at", ""),
                "updated_at": c.get("updated_at", ""),
                # Counts
                "video_count_attr": int(c.get("video_count") or 0),
                "video_count_actual": len(vids),
                "channel_count": int(c.get("channel_count") or 0),
                "breaking_count": int(c.get("breaking_count") or 0),
                "total_views": int(c.get("total_views") or 0),
                # Editorial
                "narrative_headline": c.get("narrative_headline"),
                "narrative_summary": c.get("narrative_summary"),
                "dominant_sentiment": c.get("dominant_sentiment", ""),
                "dominant_category": c.get("dominant_category", ""),
                "avg_clickbait_rating": to_jsonable(c.get("avg_clickbait_rating")),
                # Claims
                "top_claims": top_claims,
                "top_claims_count": len(top_claims),
                "classified_consensus": len(classified.get("consensus") or []),
                "classified_debated": len(classified.get("debated") or []),
                "classified_unique": len(classified.get("unique") or []),
                # Week data
                "week_data": c.get("week_data") or [],
                "cluster_weeks_snapshots": [
                    {
                        "week": w.get("week"),
                        "narrative_headline": w.get("narrative_headline"),
                        "video_count": int(w.get("video_count") or 0),
                        "view_count": int(w.get("view_count") or 0),
                        "dominant_sentiment": w.get("dominant_sentiment", ""),
                    }
                    for w in wks
                ],
                # Articles
                "articles": [
                    {
                        "article_id": a.get("article_id"),
                        "title": a.get("title"),
                        "week_number": int(a.get("week_number") or 0),
                        "created_at": a.get("created_at"),
                    }
                    for a in arts
                ],
                # Videos
                "videos": vid_rows,
                # Issues
                "issues": issues,
            }
        )

    # ── Orphans ──
    # Videos whose cluster_id points to a non-existent cluster
    ghost_cids = set(videos_by_cluster) - set(clusters_by_id)
    ghost_video_rows = []
    for cid in sorted(ghost_cids):
        for v in videos_by_cluster[cid]:
            ghost_video_rows.append(
                {
                    "video_id": v.get("SortKey", ""),
                    "channel": v.get("PartitionKey", ""),
                    "title": v.get("title", ""),
                    "week": v.get("week", ""),
                    "cluster_id": cid,
                }
            )

    # cluster-weeks rows whose cluster_id has no cluster row
    ghost_week_rows = [
        {"cluster_id": int(w["cluster_id"]), "week": w.get("week")}
        for w in cluster_weeks
        if int(w["cluster_id"]) not in clusters_by_id
    ]

    # ── Summary ──
    total_issues = sum(len(r["issues"]) for r in cluster_rows)
    clusters_with_issues = [r["cluster_id"] for r in cluster_rows if r["issues"]]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": {
            "youtube_videos": len(videos),
            "narrative_clusters": len(clusters),
            "cluster_weeks": len(cluster_weeks),
            "articles": len(articles),
        },
        "videos_unclustered": len(unclustered_videos),
        "videos_with_ghost_cluster_id": len(ghost_video_rows),
        "cluster_weeks_with_ghost_cluster_id": len(ghost_week_rows),
        "orphan_articles": len(orphan_articles),
        "total_issues_found": total_issues,
        "clusters_with_issues": clusters_with_issues,
    }

    return {
        "summary": summary,
        "clusters": cluster_rows,
        "orphans": {
            "unclustered_videos_count": len(unclustered_videos),
            "videos_pointing_to_missing_cluster": ghost_video_rows,
            "cluster_weeks_pointing_to_missing_cluster": ghost_week_rows,
            "articles_with_no_cluster_id": orphan_articles,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="scripts/all_tables_snapshot.json")
    parser.add_argument(
        "--cluster", type=int, default=None, help="Dump a single cluster only"
    )
    args = parser.parse_args()

    print(f"Connecting to DynamoDB ({REGION})...")
    dynamodb = boto3.resource("dynamodb", region_name=REGION)

    print(f"Loading {CLUSTERS_TABLE}...")
    clusters = load_clusters(dynamodb)
    print(f"  {len(clusters)} clusters")

    print(f"Loading {VIDEOS_TABLE}...")
    videos = load_videos(dynamodb)
    print(f"  {len(videos)} videos")

    print(f"Loading {CLUSTER_WEEKS_TABLE}...")
    cluster_weeks = load_cluster_weeks(dynamodb)
    print(f"  {len(cluster_weeks)} cluster-week rows")

    print(f"Loading {ARTICLES_TABLE}...")
    articles = load_articles(dynamodb)
    print(f"  {len(articles)} articles")

    print("Building snapshot...")
    snapshot = build_snapshot(
        clusters, videos, cluster_weeks, articles, filter_cid=args.cluster
    )
    snapshot = to_jsonable(snapshot)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(snapshot, indent=2))
    print(f"\nSnapshot written → {out_path}")

    print(f"\nSummary:")
    for k, v in snapshot["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
