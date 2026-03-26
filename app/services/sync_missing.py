"""
sync_missing.py - Post-pipeline sync job

1. Scans DynamoDB and Qdrant to find videos missing from Qdrant
2. Pulls video data from DynamoDB
3. Runs Gemini intelligence → writes to DynamoDB
4. Embeds chunks → writes search-only data to Qdrant

Usage:
    python -m scripts.sync_missing
"""

import re
import sys
import os
from datetime import datetime, timezone
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import boto3
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.storage_service import StorageService
from qdrant_client import QdrantClient


def _extract_week(source_key: str) -> str:
    if not source_key:
        return "unknown"
    match = re.search(r"(week\d+)", source_key, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", source_key)
    if date_match:
        try:
            dt = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            wk = dt.isocalendar()[1] - 10 + 1
            if wk >= 1:
                return f"week{wk}"
        except ValueError:
            pass
    return "unknown"


def _dec(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def get_dynamo_video_ids():
    dynamo = boto3.resource("dynamodb", region_name=settings.aws_region)
    table = dynamo.Table(settings.dynamodb_table)
    video_ids = set()
    response = table.scan(ProjectionExpression="PartitionKey, SortKey")
    for item in response["Items"]:
        video_ids.add(item["SortKey"])
    while "LastEvaluatedKey" in response:
        response = table.scan(
            ProjectionExpression="PartitionKey, SortKey",
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        for item in response["Items"]:
            video_ids.add(item["SortKey"])
    return video_ids


def get_qdrant_video_ids():
    qdrant = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
    )
    video_ids = set()
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=offset,
            with_payload=["transcript_index"],
        )
        for r in results:
            vid = r.payload.get("transcript_index")
            if vid:
                video_ids.add(vid)
        if offset is None:
            break
    return video_ids


def write_intelligence_to_dynamodb(
    table, channel, video_id, source_key, intelligence, chunk_count
):
    """Write Gemini intelligence fields to DynamoDB video item."""
    week = _extract_week(source_key)

    set_parts = []
    names = {}
    values = {}

    topics = intelligence.get("topics") or []
    if topics:
        set_parts.append("#topics = :topics")
        names["#topics"] = "topics"
        values[":topics"] = topics

    category = intelligence.get("category") or ""
    if category:
        set_parts.append("#category = :category")
        names["#category"] = "category"
        values[":category"] = category

    sentiment = intelligence.get("sentiment") or ""
    if sentiment:
        set_parts.append("#sentiment = :sentiment")
        names["#sentiment"] = "sentiment"
        values[":sentiment"] = sentiment

    key_claims = intelligence.get("key_claims") or []
    if key_claims:
        set_parts.append("#claims = :claims")
        names["#claims"] = "key_claims"
        values[":claims"] = key_claims

    set_parts.append("#breaking = :breaking")
    names["#breaking"] = "is_breaking"
    values[":breaking"] = bool(intelligence.get("is_breaking", False))

    if source_key:
        set_parts.append("#src = :src")
        names["#src"] = "source_key"
        values[":src"] = source_key

    set_parts.append("#week = :week")
    names["#week"] = "week"
    values[":week"] = week

    set_parts.append("#chunks = :chunks")
    names["#chunks"] = "chunk_count"
    values[":chunks"] = _dec(chunk_count)

    set_parts.append("#ts = :ts")
    names["#ts"] = "indexed_at"
    values[":ts"] = datetime.now(timezone.utc).isoformat()

    table.update_item(
        Key={"PartitionKey": channel, "SortKey": video_id},
        UpdateExpression="SET " + ", ".join(set_parts),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def build_s3_video_map() -> dict[str, str]:
    """
    Scan S3 and build a mapping: videoId → source_key.
    e.g. {"4En8G-3c7Sw": "youtube-data/week2/bbcnews.json", ...}
    """
    import json as _json

    s3 = boto3.client("s3", region_name=settings.aws_region)
    paginator = s3.get_paginator("list_objects_v2")
    video_map = {}

    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix="youtube-data/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                body = s3.get_object(Bucket=settings.s3_bucket, Key=key)["Body"].read()
                data = _json.loads(body)
                for v in data.get("videos", []):
                    vid = v.get("videoId")
                    if vid:
                        video_map[vid] = key
            except Exception:
                continue

    print(f"  S3 video map: {len(video_map)} videos across S3")
    return video_map


def index_videos(video_ids: list):
    dynamo = boto3.resource("dynamodb", region_name=settings.aws_region)
    table = dynamo.Table(settings.dynamodb_table)
    embedding_service = EmbeddingService(api_keys=settings.genai_api_keys)
    vector_service = VectorService()
    storage_service = StorageService()

    CHANNELS = [
        "ABCNews",
        "BBCNews",
        "CBSNews",
        "CNBC",
        "FoxNews",
        "NBCNews",
        "NewYorkTimes",
        "SkyNews",
        "WashingtonPost",
        "aljazeeraenglish",
    ]

    # Build S3 lookup for correct source_key per video
    print("  Building S3 video map...")
    s3_map = build_s3_video_map()

    indexed = 0
    failed = 0

    for video_id in video_ids:
        print(f"\nProcessing {video_id}...")

        # Direct lookup — try each channel until found
        item = None
        for channel_name in CHANNELS:
            try:
                resp = table.get_item(
                    Key={"PartitionKey": channel_name, "SortKey": video_id},
                )
                if "Item" in resp:
                    item = resp["Item"]
                    break
            except Exception:
                continue

        if not item:
            print(f"  NOT FOUND in DynamoDB: {video_id}")
            failed += 1
            continue

        channel = item.get("PartitionKey", "")
        title = item.get("title", "")
        transcript = item.get("transcript", "")
        published_at = item.get("publishedAt", "")
        view_count = int(item.get("viewCount", 0))
        like_count = int(item.get("likeCount", 0))
        comment_count = int(item.get("commentCount", 0))

        if not transcript:
            print(f"  NO TRANSCRIPT: {video_id}")
            failed += 1
            continue

        transcript = storage_service._clean_transcript(transcript)

        # Look up real source_key from S3, fallback to constructed path
        source_key = s3_map.get(
            video_id, f"youtube-data/unknown/{channel.lower()}.json"
        )
        transcript_key = f"{source_key}::{video_id}"
        week = _extract_week(source_key)
        print(f"  Source: {source_key} → {week}")

        try:
            # Step 1: Chunk
            chunks = embedding_service.chunk_text(transcript)
            print(f"  Chunks: {len(chunks)}")

            # Step 2: Gemini intelligence
            intelligence = embedding_service.extract_video_intelligence(
                chunks=chunks, title=title
            )

            # Guard: if Gemini failed (503, empty response), skip this video
            if not intelligence.get("topics"):
                print(f"  SKIPPED: Gemini returned empty results (likely 503)")
                failed += 1
                continue

            print(
                f"  Category: {intelligence['category']} | "
                f"Sentiment: {intelligence['sentiment']}"
            )
            print(f"  Topics: {intelligence['topics']}")

            # Step 3: Write intelligence to DynamoDB
            write_intelligence_to_dynamodb(
                table=table,
                channel=channel,
                video_id=video_id,
                source_key=source_key,
                intelligence=intelligence,
                chunk_count=len(chunks),
            )
            print(f"  DynamoDB: intelligence written")

            # Step 4: Embed chunks
            vectors = embedding_service.embed_chunks(chunks)

            # Step 5: Upsert to Qdrant — search fields only
            points = vector_service.upsert_transcript_chunks(
                transcript_key=transcript_key,
                source_key=source_key,
                transcript_index=video_id,
                chunks=chunks,
                vectors=vectors,
                extra_metadata=None,  # No extra metadata — DynamoDB has it
            )
            print(f"  Qdrant: {points} points indexed")
            indexed += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    return indexed, failed


def main():
    print("=== Sync Missing Videos ===\n")

    print("Scanning DynamoDB...")
    dynamo_ids = get_dynamo_video_ids()
    print(f"  DynamoDB videos: {len(dynamo_ids)}")

    print("Scanning Qdrant...")
    qdrant_ids = get_qdrant_video_ids()
    print(f"  Qdrant videos: {len(qdrant_ids)}")

    # Videos in DynamoDB but not in Qdrant — these need indexing
    missing = list(dynamo_ids - qdrant_ids)
    print(f"\nMissing from Qdrant: {len(missing)}")

    if not missing:
        print("Nothing to sync — all good!")
        return

    for vid in sorted(missing):
        print(f"  {vid}")

    print(f"\nIndexing {len(missing)} missing videos...")
    indexed, failed = index_videos(missing)
    print(f"\n=== Done. Indexed: {indexed} Failed: {failed} ===")


if __name__ == "__main__":
    main()
