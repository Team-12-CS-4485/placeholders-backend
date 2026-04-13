"""
backfill_html_unescape_transcripts.py - Clean transcripts and claim excerpts

Fixes three targets:
  1. youtube-videos (DynamoDB) — strips VTT headers + HTML entities from `transcript`
  2. narrative-clusters (DynamoDB) — strips VTT headers + HTML entities from
     `classified_claims[consensus|debated|unique][*].transcript_excerpt`
  3. Qdrant — strips HTML entities from `text` chunk payload

Usage:
    python -m scripts.backfill_html_unescape_transcripts
    python -m scripts.backfill_html_unescape_transcripts --dry-run
    python -m scripts.backfill_html_unescape_transcripts --skip-videos
    python -m scripts.backfill_html_unescape_transcripts --skip-clusters
    python -m scripts.backfill_html_unescape_transcripts --skip-qdrant
"""

import argparse
import copy
import html
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import boto3
from qdrant_client import QdrantClient

from app.core.config import settings

REGION = os.getenv("AWS_REGION", "us-east-2")
QDRANT_SCROLL_BATCH = 250
ENTITY_MARKERS = ("&gt;", "&lt;", "&amp;", "&quot;", "&apos;", "&#")
VTT_RE = re.compile(
    r"^\s*Kind:\s*captions\s+Language:\s*[a-zA-Z-]+\s*", re.IGNORECASE
)


def clean_text(text: str) -> str:
    """Apply the same cleaning as StorageService._clean_transcript."""
    cleaned = html.unescape(text.strip())
    cleaned = VTT_RE.sub("", cleaned)
    return cleaned.strip()


def is_dirty(text: str) -> bool:
    if VTT_RE.match(text):
        return True
    return any(m in text for m in ENTITY_MARKERS)


# ── 1. youtube-videos transcripts ────────────────────────────────────────────

def backfill_videos(table, dry_run: bool):
    print("\n── youtube-videos (transcript field) ──")
    to_fix = []
    scan_kwargs = {"ProjectionExpression": "PartitionKey, SortKey, transcript"}

    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp["Items"]:
            t = item.get("transcript", "")
            if t and is_dirty(t):
                to_fix.append(
                    {
                        "channel": item["PartitionKey"],
                        "video_id": item["SortKey"],
                        "cleaned": clean_text(t),
                    }
                )
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    print(f"Found {len(to_fix)} transcripts needing cleanup")

    if dry_run:
        for item in to_fix[:5]:
            print(f"  would fix: {item['channel']}/{item['video_id']}")
        if len(to_fix) > 5:
            print(f"  ... and {len(to_fix) - 5} more")
        return 0, 0

    success = failed = 0
    for i, item in enumerate(to_fix):
        print(f"  [{i+1}/{len(to_fix)}] {item['channel']}/{item['video_id']}...", end=" ")
        try:
            table.update_item(
                Key={"PartitionKey": item["channel"], "SortKey": item["video_id"]},
                UpdateExpression="SET transcript = :t",
                ExpressionAttributeValues={":t": item["cleaned"]},
            )
            print("ok")
            success += 1
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed += 1
        time.sleep(0.05)

    return success, failed


# ── 2. narrative-clusters classified_claims excerpts ─────────────────────────

def backfill_clusters(clusters_table, dry_run: bool):
    print("\n── narrative-clusters (classified_claims excerpts) ──")
    to_fix = []

    resp = clusters_table.scan(
        ProjectionExpression="cluster_id, classified_claims"
    )
    for item in resp.get("Items", []):
        cid = item["cluster_id"]
        claims = item.get("classified_claims")
        if not claims:
            continue

        new_claims = copy.deepcopy(dict(claims))
        changed = False

        for claim_type in ("consensus", "debated", "unique"):
            for c in new_claims.get(claim_type, []):
                excerpt = c.get("transcript_excerpt", "")
                if excerpt and is_dirty(excerpt):
                    c["transcript_excerpt"] = clean_text(excerpt)
                    changed = True

        if changed:
            to_fix.append({"cluster_id": cid, "classified_claims": new_claims})

    print(f"Found {len(to_fix)} clusters with dirty excerpts")

    if dry_run:
        for item in to_fix[:5]:
            print(f"  would fix cluster: {item['cluster_id']}")
        if len(to_fix) > 5:
            print(f"  ... and {len(to_fix) - 5} more")
        return 0, 0

    success = failed = 0
    for i, item in enumerate(to_fix):
        cid = item["cluster_id"]
        print(f"  [{i+1}/{len(to_fix)}] cluster {cid}...", end=" ")
        try:
            clusters_table.update_item(
                Key={"cluster_id": cid},
                UpdateExpression="SET classified_claims = :cc",
                ExpressionAttributeValues={":cc": item["classified_claims"]},
            )
            print("ok")
            success += 1
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed += 1
        time.sleep(0.05)

    return success, failed


# ── 3. Qdrant chunk text ──────────────────────────────────────────────────────

def backfill_qdrant(client: QdrantClient, collection: str, dry_run: bool):
    print("\n── Qdrant (chunk text field) ──")
    to_fix = []
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=QDRANT_SCROLL_BATCH,
            offset=offset,
            with_payload=["text"],
            with_vectors=False,
        )
        for point in results:
            text = point.payload.get("text", "")
            if text and is_dirty(text):
                to_fix.append({"id": point.id, "text": clean_text(text)})
        if offset is None:
            break

    print(f"Found {len(to_fix)} Qdrant points needing cleanup")

    if dry_run:
        for p in to_fix[:5]:
            print(f"  would fix point: {p['id']}")
        if len(to_fix) > 5:
            print(f"  ... and {len(to_fix) - 5} more")
        return 0, 0

    success = failed = 0
    batch_size = 100
    for i in range(0, len(to_fix), batch_size):
        batch = to_fix[i : i + batch_size]
        end = min(i + batch_size, len(to_fix))
        print(f"  [{end}/{len(to_fix)}] updating {len(batch)} points...", end=" ")
        try:
            for point in batch:
                client.set_payload(
                    collection_name=collection,
                    payload={"text": point["text"]},
                    points=[point["id"]],
                )
            print("ok")
            success += len(batch)
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed += len(batch)
        time.sleep(0.1)

    return success, failed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-clusters", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no changes will be made\n")

    dynamodb = boto3.resource("dynamodb", region_name=REGION)

    videos_ok = videos_fail = 0
    clusters_ok = clusters_fail = 0
    qdrant_ok = qdrant_fail = 0

    if not args.skip_videos:
        table = dynamodb.Table(settings.dynamodb_table)
        videos_ok, videos_fail = backfill_videos(table, args.dry_run)

    if not args.skip_clusters:
        clusters_table = dynamodb.Table("narrative-clusters")
        clusters_ok, clusters_fail = backfill_clusters(clusters_table, args.dry_run)

    if not args.skip_qdrant:
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        qdrant_ok, qdrant_fail = backfill_qdrant(
            qdrant_client, settings.qdrant_collection, args.dry_run
        )

    print("\n=== Summary ===")
    if not args.skip_videos:
        print(f"youtube-videos    updated={videos_ok}  failed={videos_fail}")
    if not args.skip_clusters:
        print(f"narrative-clusters updated={clusters_ok}  failed={clusters_fail}")
    if not args.skip_qdrant:
        print(f"Qdrant            updated={qdrant_ok}  failed={qdrant_fail}")


if __name__ == "__main__":
    main()
