#!/usr/bin/env python3
"""
audit_empty_claims.py — Phase 1 diagnostic for narratives missing claims.

Scans `narrative-clusters` and `youtube-videos`, then for each cluster reports:
  - how many videos have transcript / topics / sentiment / key_claims populated
  - whether the cluster's `top_claims` and `classified_claims` are empty
  - a diagnosis bucket placing the cluster into one of:
      ingestion_problem        — videos lack transcripts
      gemini_partial_response  — transcripts + topics present, key_claims empty
      intelligence_failed      — transcripts present but topics empty (intel call broke)
      claim_analysis_skipped   — per-video key_claims exist but classified_claims empty
      ok                       — claims present end to end

Writes a JSON report. Read-only — does not mutate DynamoDB.

Usage:
    python scripts/audit_empty_claims.py
    python scripts/audit_empty_claims.py --out empty_claims_audit.json
    python scripts/audit_empty_claims.py --only-empty   # skip clusters that look fine
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


def load_clusters(dynamodb) -> dict[int, dict]:
    table = dynamodb.Table(CLUSTERS_TABLE)
    items = scan_all(table)
    out: dict[int, dict] = {}
    for item in items:
        cid = int(item["cluster_id"])
        classified = item.get(
            "classified_claims", {"consensus": [], "debated": [], "unique": []}
        )
        out[cid] = {
            "cluster_id": cid,
            "cluster_label": item.get("cluster_label", ""),
            "status": item.get("status", ""),
            "video_count_attr": int(item.get("video_count", 0) or 0),
            "top_claims_count": len(item.get("top_claims") or []),
            "classified_consensus": len(classified.get("consensus", []) or []),
            "classified_debated": len(classified.get("debated", []) or []),
            "classified_unique": len(classified.get("unique", []) or []),
        }
    return out


def load_video_signals(dynamodb) -> dict[int, list[dict]]:
    """
    For each video that has a cluster_id, return the signals we care about.
    Returns {cluster_id: [{video_id, channel, has_transcript, has_topics,
                          has_sentiment, has_key_claims}, ...]}
    """
    table = dynamodb.Table(VIDEOS_TABLE)
    items = scan_all(
        table,
        ProjectionExpression="PartitionKey, SortKey, cluster_id, transcript, "
        "topics, sentiment, key_claims",
    )
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        cid = item.get("cluster_id")
        if cid is None:
            continue
        transcript = item.get("transcript") or ""
        topics = item.get("topics") or []
        sentiment = item.get("sentiment") or ""
        key_claims = item.get("key_claims") or []
        by_cluster[int(cid)].append(
            {
                "video_id": item.get("SortKey", ""),
                "channel": item.get("PartitionKey", ""),
                "has_transcript": bool(transcript and len(str(transcript).strip()) > 0),
                "has_topics": bool(topics),
                "has_sentiment": bool(sentiment),
                "has_key_claims": bool(key_claims),
            }
        )
    return dict(by_cluster)


def diagnose(cluster: dict, videos: list[dict]) -> str:
    n = len(videos)
    if n == 0:
        return "no_videos_assigned"

    with_transcript = sum(1 for v in videos if v["has_transcript"])
    with_topics = sum(1 for v in videos if v["has_topics"])
    with_claims = sum(1 for v in videos if v["has_key_claims"])

    classified_total = (
        cluster["classified_consensus"]
        + cluster["classified_debated"]
        + cluster["classified_unique"]
    )

    if with_claims > 0 and classified_total == 0:
        return "claim_analysis_skipped"

    if with_claims == 0:
        if with_transcript == 0:
            return "ingestion_problem"
        if with_topics == 0:
            return "intelligence_failed"
        return "gemini_partial_response"

    if classified_total == 0 and cluster["top_claims_count"] == 0:
        return "claim_analysis_skipped"

    return "ok"


def build_report(clusters: dict[int, dict], videos_by_cluster: dict[int, list[dict]]):
    cluster_rows = []
    bucket_counts: dict[str, list[int]] = defaultdict(list)

    for cid in sorted(clusters):
        c = clusters[cid]
        videos = videos_by_cluster.get(cid, [])
        n = len(videos)

        with_transcript = sum(1 for v in videos if v["has_transcript"])
        with_topics = sum(1 for v in videos if v["has_topics"])
        with_sentiment = sum(1 for v in videos if v["has_sentiment"])
        with_claims = sum(1 for v in videos if v["has_key_claims"])

        diagnosis = diagnose(c, videos)
        bucket_counts[diagnosis].append(cid)

        affected = [
            {"video_id": v["video_id"], "channel": v["channel"]}
            for v in videos
            if not v["has_key_claims"]
        ]

        cluster_rows.append(
            {
                "cluster_id": cid,
                "cluster_label": c["cluster_label"],
                "status": c["status"],
                "video_count_actual": n,
                "video_count_attr": c["video_count_attr"],
                "videos_with_transcript": with_transcript,
                "videos_with_topics": with_topics,
                "videos_with_sentiment": with_sentiment,
                "videos_with_key_claims": with_claims,
                "top_claims_count": c["top_claims_count"],
                "classified_claims_total": (
                    c["classified_consensus"]
                    + c["classified_debated"]
                    + c["classified_unique"]
                ),
                "classified_consensus": c["classified_consensus"],
                "classified_debated": c["classified_debated"],
                "classified_unique": c["classified_unique"],
                "diagnosis": diagnosis,
                "videos_missing_key_claims": affected,
            }
        )

    summary = {
        "total_clusters": len(clusters),
        "clusters_by_diagnosis": {k: len(v) for k, v in bucket_counts.items()},
        "cluster_ids_by_diagnosis": dict(bucket_counts),
        "total_videos_assigned": sum(len(v) for v in videos_by_cluster.values()),
        "total_videos_missing_key_claims": sum(
            sum(1 for v in vids if not v["has_key_claims"])
            for vids in videos_by_cluster.values()
        ),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "clusters": cluster_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="empty_claims_audit.json", help="Output JSON path"
    )
    parser.add_argument(
        "--only-empty",
        action="store_true",
        help="Include only clusters whose diagnosis is not 'ok'",
    )
    args = parser.parse_args()

    print(f"Connecting to DynamoDB ({REGION})...")
    dynamodb = boto3.resource("dynamodb", region_name=REGION)

    print(f"Loading clusters from {CLUSTERS_TABLE}...")
    clusters = load_clusters(dynamodb)
    print(f"  {len(clusters)} clusters")

    print(f"Loading video signals from {VIDEOS_TABLE}...")
    videos_by_cluster = load_video_signals(dynamodb)
    total_assigned = sum(len(v) for v in videos_by_cluster.values())
    print(f"  {total_assigned} videos across {len(videos_by_cluster)} clusters")

    report = build_report(clusters, videos_by_cluster)

    if args.only_empty:
        report["clusters"] = [c for c in report["clusters"] if c["diagnosis"] != "ok"]

    out_path = Path(args.out)
    out_path.write_text(json.dumps(to_jsonable(report), indent=2))
    print(f"\nReport written → {out_path}")

    print("\nDiagnosis breakdown:")
    for bucket, count in sorted(report["summary"]["clusters_by_diagnosis"].items()):
        print(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()
