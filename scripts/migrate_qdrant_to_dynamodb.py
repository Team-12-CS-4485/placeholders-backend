"""
migrate_qdrant_to_dynamodb.py - Migrate intelligence data from Qdrant to DynamoDB

Reads all chunk payloads from Qdrant, deduplicates to video level, and:
1. Updates youtube-videos items with intelligence + cluster fields
2. Populates narrative-clusters table with cluster stats + classified claims

Does NOT touch Qdrant — read only. Run this before switching trend_service to DynamoDB.

Usage:
    python -m scripts.migrate_qdrant_to_dynamodb
    python -m scripts.migrate_qdrant_to_dynamodb --dry-run    # preview without writing
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import boto3
from app.core.config import settings
from app.services.vector_service import VectorService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_REGION", "us-east-2")
YOUTUBE_VIDEOS_TABLE = os.getenv("DYNAMODB_TABLE", "youtube-videos")
NARRATIVE_CLUSTERS_TABLE = "narrative-clusters"


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_decimal(obj):
    """Convert floats/ints to Decimal for DynamoDB compatibility."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return Decimal(str(round(obj, 4)))
    if isinstance(obj, int):
        return Decimal(str(obj))


def extract_week(source_key: str) -> str:
    """Extract week from source_key path."""
    match = re.search(r"(week\d+)", source_key, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Try timestamp fallback
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", source_key)
    if date_match:
        try:
            date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            anchor_week = 10
            week_num = date.isocalendar()[1] - anchor_week + 1
            if week_num >= 1:
                return f"week{week_num}"
        except ValueError:
            pass
    return "unknown"


# ── Step 1: Read from Qdrant ──────────────────────────────────────────────────

def read_qdrant_data():
    """Scroll all Qdrant chunks, deduplicate to video level."""
    vs = VectorService()
    all_payloads = []
    offset = None

    while True:
        results, offset = vs.client.scroll(
            collection_name=vs.collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            all_payloads.append(point.payload)
        if offset is None:
            break

    logger.info(f"Read {len(all_payloads)} chunks from Qdrant")

    # Deduplicate to video level
    videos = {}
    for payload in all_payloads:
        video_id = payload.get("transcript_index", "")
        if video_id and video_id not in videos:
            videos[video_id] = payload

    logger.info(f"Deduplicated to {len(videos)} unique videos")
    return videos


# ── Step 2: Update youtube-videos table ───────────────────────────────────────

def update_youtube_videos(videos: dict, dry_run: bool = False):
    """
    For each video, find its DynamoDB item by videoId and update with
    intelligence + cluster fields from Qdrant.
    """
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(YOUTUBE_VIDEOS_TABLE)

    # Build a lookup: videoId → (PartitionKey, SortKey) from DynamoDB
    logger.info("Scanning DynamoDB for existing items...")
    dynamo_keys = {}
    scan_kwargs = {"ProjectionExpression": "PartitionKey, SortKey"}
    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp["Items"]:
            video_id = item["SortKey"]
            dynamo_keys[video_id] = {
                "PartitionKey": item["PartitionKey"],
                "SortKey": item["SortKey"],
            }
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    logger.info(f"Found {len(dynamo_keys)} items in DynamoDB")

    updated = 0
    skipped = 0
    not_found = 0

    for video_id, payload in videos.items():
        key = dynamo_keys.get(video_id)
        if not key:
            logger.warning(f"Video {video_id} not found in DynamoDB — skipping")
            not_found += 1
            continue

        source_key = payload.get("source_key", "") or ""
        week = extract_week(source_key)

        # Build update fields
        update_fields = {}
        
        # GSI key fields — MUST always be set, never NULL
        update_fields["week"] = week if week else "unknown"
        update_fields["sentiment"] = payload.get("sentiment", "") or "neutral"

        # Intelligence fields
        topics = payload.get("topics", [])
        if topics:
            update_fields["topics"] = topics
        
        category = payload.get("category", "")
        if category:
            update_fields["category"] = category

        sentiment = payload.get("sentiment", "")
        if sentiment:
            update_fields["sentiment"] = sentiment

        key_claims = payload.get("key_claims", [])
        if key_claims:
            update_fields["key_claims"] = key_claims

        is_breaking = payload.get("is_breaking", False)
        update_fields["is_breaking"] = is_breaking

        if source_key:
            update_fields["source_key"] = source_key

        update_fields["week"] = week if week else "unknown"

        # Chunk count
        update_fields["chunk_count"] = to_decimal(payload.get("chunk_count", 0))

        # Cluster fields
        cluster_id = payload.get("cluster_id")
        if cluster_id is not None and cluster_id != -1:
            update_fields["cluster_id"] = to_decimal(cluster_id)
            update_fields["cluster_label"] = payload.get("cluster_label", "")
            update_fields["cluster_confidence"] = to_decimal(
                payload.get("cluster_confidence", 0)
            )

        update_fields["indexed_at"] = datetime.now(timezone.utc).isoformat()

        if not update_fields:
            skipped += 1
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would update {video_id}: {list(update_fields.keys())}")
            updated += 1
            continue

        # Build DynamoDB update expression
        expr_parts = []
        expr_values = {}
        expr_names = {}

        for field, value in update_fields.items():
            safe_name = f"#f_{field}"
            safe_val = f":v_{field}"
            expr_parts.append(f"{safe_name} = {safe_val}")
            expr_names[safe_name] = field
            expr_values[safe_val] = to_decimal(value)

        try:
            table.update_item(
                Key=key,
                UpdateExpression="SET " + ", ".join(expr_parts),
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
            )
            updated += 1
        except Exception as e:
            logger.error(f"FAILED updating {video_id}: {e}")
            logger.error(f"  Fields: {list(update_fields.keys())}")
            logger.error(f"  Week: {repr(update_fields.get('week'))}")
            logger.error(f"  Sentiment: {repr(update_fields.get('sentiment'))}")
            skipped += 1

    logger.info(
        f"youtube-videos: updated={updated} skipped={skipped} not_found={not_found}"
    )
    return updated


# ── Step 3: Populate narrative-clusters table ─────────────────────────────────

def populate_narrative_clusters(videos: dict, dry_run: bool = False):
    """
    Aggregate video data by cluster_id and write one item per cluster
    to the narrative-clusters table. Includes classified_claims if present.
    """
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(NARRATIVE_CLUSTERS_TABLE)

    # Group videos by cluster
    cluster_videos: dict[int, list[dict]] = defaultdict(list)
    for video_id, payload in videos.items():
        cluster_id = payload.get("cluster_id")
        if cluster_id is None or cluster_id == -1:
            continue
        cluster_videos[int(cluster_id)].append(payload)

    logger.info(f"Found {len(cluster_videos)} clusters to write")

    written = 0

    for cluster_id, vids in cluster_videos.items():
        # Aggregate stats
        channels = set()
        sentiments = Counter()
        categories = Counter()
        topics_counter = Counter()
        breaking_count = 0
        total_views = 0
        total_likes = 0
        total_comments = 0

        for v in vids:
            channels.add(v.get("channel", ""))
            sentiments[v.get("sentiment", "neutral")] += 1
            categories[v.get("category", "Other")] += 1
            for t in v.get("topics", []):
                topics_counter[t] += 1
            if v.get("is_breaking"):
                breaking_count += 1
            total_views += v.get("view_count", 0)
            total_likes += v.get("like_count", 0)
            total_comments += v.get("comment_count", 0)

        # Get classified_claims from first video (same on all chunks in cluster)
        classified_claims = vids[0].get("classified_claims", {
            "consensus": [],
            "debated": [],
            "unique": [],
        })

        cluster_item = {
            "cluster_id": to_decimal(cluster_id),
            "cluster_label": vids[0].get("cluster_label", f"Cluster {cluster_id}"),
            "video_count": to_decimal(len(vids)),
            "channel_count": to_decimal(len(channels)),
            "channels": sorted(channels),
            "top_topics": [t[0] for t in topics_counter.most_common(5)],
            "dominant_category": (
                categories.most_common(1)[0][0] if categories else "Other"
            ),
            "dominant_sentiment": (
                sentiments.most_common(1)[0][0] if sentiments else "neutral"
            ),
            "breaking_count": to_decimal(breaking_count),
            "total_views": to_decimal(total_views),
            "total_likes": to_decimal(total_likes),
            "total_comments": to_decimal(total_comments),
            "classified_claims": to_decimal(classified_claims),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if dry_run:
            logger.info(
                f"[DRY RUN] Would write cluster {cluster_id}: "
                f"{cluster_item['cluster_label']} ({len(vids)} videos)"
            )
            written += 1
            continue

        table.put_item(Item=cluster_item)
        written += 1
        logger.info(
            f"Wrote cluster {cluster_id}: {cluster_item['cluster_label']} "
            f"({len(vids)} videos, {len(channels)} channels)"
        )

    logger.info(f"narrative-clusters: wrote {written} items")
    return written


# ── Step 4: Verify ────────────────────────────────────────────────────────────

def verify_migration():
    """Spot-check a few items to confirm data landed correctly."""
    dynamodb = boto3.resource("dynamodb", region_name=REGION)

    # Check youtube-videos
    videos_table = dynamodb.Table(YOUTUBE_VIDEOS_TABLE)
    resp = videos_table.scan(Limit=3)
    print("\n=== youtube-videos spot check ===\n")
    for item in resp["Items"]:
        vid = item["SortKey"]
        has_intel = "topics" in item and item["topics"]
        has_cluster = "cluster_id" in item
        has_week = "week" in item
        print(
            f"  {vid}: "
            f"intel={'YES' if has_intel else 'NO'} "
            f"cluster={'YES' if has_cluster else 'NO'} "
            f"week={'YES' if has_week else 'NO'}"
        )

    # Check narrative-clusters
    clusters_table = dynamodb.Table(NARRATIVE_CLUSTERS_TABLE)
    resp = clusters_table.scan()
    print(f"\n=== narrative-clusters ({len(resp['Items'])} items) ===\n")
    for item in sorted(resp["Items"], key=lambda x: int(x["cluster_id"])):
        cid = item["cluster_id"]
        label = item.get("cluster_label", "?")
        vcount = item.get("video_count", 0)
        channels = len(item.get("channels", []))
        claims = item.get("classified_claims", {})
        c_count = len(claims.get("consensus", []))
        d_count = len(claims.get("debated", []))
        u_count = len(claims.get("unique", []))
        print(
            f"  Cluster {cid}: {label} "
            f"({vcount} videos, {channels} channels, "
            f"claims: {c_count}c/{d_count}d/{u_count}u)"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migrate Qdrant intelligence to DynamoDB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to DynamoDB",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing migration, don't write",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_migration()
        return

    print("=== Qdrant → DynamoDB Migration ===\n")

    # Step 1: Read from Qdrant
    print("Step 1: Reading intelligence data from Qdrant...")
    videos = read_qdrant_data()

    if not videos:
        print("No videos found in Qdrant. Aborting.")
        return

    # Step 2: Update youtube-videos
    print(f"\nStep 2: Updating {len(videos)} items in youtube-videos...")
    video_count = update_youtube_videos(videos, dry_run=args.dry_run)

    # Step 3: Populate narrative-clusters
    print(f"\nStep 3: Populating narrative-clusters table...")
    cluster_count = populate_narrative_clusters(videos, dry_run=args.dry_run)

    # Step 4: Verify
    if not args.dry_run:
        verify_migration()

    print(f"\n=== Migration Complete ===")
    print(f"  Videos updated: {video_count}")
    print(f"  Clusters written: {cluster_count}")

    if args.dry_run:
        print("\n  (DRY RUN — no data was written)")


if __name__ == "__main__":
    main()