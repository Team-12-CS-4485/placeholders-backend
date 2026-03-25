"""
sync_missing.py - Post-pipeline sync job

1. Scans S3, DynamoDB, and Qdrant to find missing videos
2. Pulls missing videos from DynamoDB
3. Runs them through Gemini (transcript + thumbnail) + embedding + Qdrant indexing

Run after the main pipeline to catch any videos that were ingested but not indexed,
or use --backfill-thumbnails to add thumbnail data to already-indexed videos that
predate the thumbnail integration.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import boto3
import json
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.storage_service import StorageService
from qdrant_client import QdrantClient


def get_s3_video_ids():
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    video_ids = set()
    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix="youtube-data/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                body = s3.get_object(Bucket=settings.s3_bucket, Key=key)["Body"].read()
                data = json.loads(body)
                for v in data.get("videos", []):
                    vid = v.get("videoId")
                    if vid:
                        video_ids.add(vid)
    return video_ids


def get_dynamo_video_ids():
    dynamo = boto3.resource("dynamodb")
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


def get_qdrant_videos_missing_thumbnail():
    """
    Return video IDs that are in Qdrant but have an empty thumbnail_tone —
    i.e. they were indexed before thumbnail analysis was added.
    """
    qdrant = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
    )
    from qdrant_client.http import models

    video_ids = set()
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=offset,
            with_payload=["transcript_index", "thumbnail_tone"],
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="thumbnail_tone",
                        match=models.MatchValue(value=""),
                    )
                ]
            ),
        )
        for r in results:
            vid = r.payload.get("transcript_index")
            if vid:
                video_ids.add(vid)
        if offset is None:
            break
    return video_ids


def index_videos(video_ids: list):
    """Full pipeline (transcript intelligence + thumbnail + embed + Qdrant upsert)."""
    dynamo = boto3.resource("dynamodb")
    table = dynamo.Table(settings.dynamodb_table)
    embedding_service = EmbeddingService(api_keys=settings.genai_api_keys)
    vector_service = VectorService()
    storage_service = StorageService()

    indexed = 0
    failed = 0

    for video_id in video_ids:
        print(f"\nProcessing {video_id}...")

        response = table.scan(
            FilterExpression="SortKey = :vid",
            ExpressionAttributeValues={":vid": video_id},
            ProjectionExpression="PartitionKey, SortKey, title, transcript, publishedAt, viewCount, likeCount, commentCount",
        )
        items = response.get("Items", [])
        if not items:
            print(f"  NOT FOUND in DynamoDB: {video_id}")
            failed += 1
            continue

        item = items[0]
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
        source_key = f"youtube-data/week2/{channel.lower()}.json"
        transcript_key = f"{source_key}::{video_id}"

        try:
            chunks = embedding_service.chunk_text(transcript)
            print(f"  Chunks: {len(chunks)}")

            # Transcript intelligence
            intelligence = embedding_service.extract_video_intelligence(
                chunks=chunks, title=title
            )
            print(
                f"  Category: {intelligence['category']} | Sentiment: {intelligence['sentiment']}"
            )
            print(f"  Topics: {intelligence['topics']}")

            # Thumbnail intelligence
            thumbnail_intel = embedding_service.analyze_thumbnail(
                video_id=video_id, title=title
            )
            print(
                f"  Thumbnail tone: {thumbnail_intel['thumbnail_tone']} "
                f"| Clickbait: {thumbnail_intel['thumbnail_clickbait_score']}"
            )

            vectors = embedding_service.embed_chunks(chunks)

            video_metadata = {
                "channel": channel,
                "title": title,
                "published_at": published_at,
                "view_count": view_count,
                "like_count": like_count,
                "comment_count": comment_count,
                **intelligence,
                **thumbnail_intel,
            }

            points = vector_service.upsert_transcript_chunks(
                transcript_key=transcript_key,
                source_key=source_key,
                transcript_index=video_id,
                chunks=chunks,
                vectors=vectors,
                extra_metadata=video_metadata,
            )
            print(f"  Indexed: {points} points")
            indexed += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    return indexed, failed


def backfill_thumbnails(video_ids: list):
    """
    For videos already in Qdrant, fetch + analyze thumbnails and patch the
    existing point payloads using set_payload (no re-embedding needed).
    """
    qdrant = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
    )
    from qdrant_client.http import models as qdrant_models

    embedding_service = EmbeddingService(api_keys=settings.genai_api_keys)
    dynamo = boto3.resource("dynamodb")
    table = dynamo.Table(settings.dynamodb_table)

    patched = 0
    failed = 0

    for video_id in video_ids:
        print(f"\nBackfilling thumbnail for {video_id}...")

        # Fetch title from DynamoDB for better prompt context
        title = ""
        try:
            resp = table.scan(
                FilterExpression="SortKey = :vid",
                ExpressionAttributeValues={":vid": video_id},
                ProjectionExpression="title",
            )
            items = resp.get("Items", [])
            if items:
                title = items[0].get("title", "")
        except Exception:
            pass

        try:
            thumbnail_intel = embedding_service.analyze_thumbnail(
                video_id=video_id, title=title
            )
            print(
                f"  Tone: {thumbnail_intel['thumbnail_tone']} "
                f"| Clickbait: {thumbnail_intel['thumbnail_clickbait_score']}"
            )

            # Find all point IDs for this video_id
            point_ids = []
            offset = None
            while True:
                results, offset = qdrant.scroll(
                    collection_name=settings.qdrant_collection,
                    limit=50,
                    offset=offset,
                    with_payload=False,
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="transcript_index",
                                match=qdrant_models.MatchValue(value=video_id),
                            )
                        ]
                    ),
                )
                point_ids.extend([r.id for r in results])
                if offset is None:
                    break

            if not point_ids:
                print(f"  No points found in Qdrant for {video_id}")
                failed += 1
                continue

            # Patch payload on all chunks for this video without re-uploading vectors
            qdrant.set_payload(
                collection_name=settings.qdrant_collection,
                payload=thumbnail_intel,
                points=point_ids,
            )
            print(f"  Patched {len(point_ids)} points")
            patched += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    return patched, failed


def main():
    parser = argparse.ArgumentParser(description="Sync missing videos into Qdrant")
    parser.add_argument(
        "--backfill-thumbnails",
        action="store_true",
        help="Patch thumbnail data onto already-indexed videos that are missing it",
    )
    args = parser.parse_args()

    if args.backfill_thumbnails:
        print("=== Backfill Thumbnails ===\n")
        print("Scanning Qdrant for videos with empty thumbnail_tone...")
        missing_thumb_ids = get_qdrant_videos_missing_thumbnail()
        print(f"  Found {len(missing_thumb_ids)} videos missing thumbnail data")

        if not missing_thumb_ids:
            print("All videos already have thumbnail data — nothing to do.")
            return

        # Also ensure the indexes exist on the collection
        vector_service = VectorService()
        vector_service.add_thumbnail_indexes()

        patched, failed = backfill_thumbnails(list(missing_thumb_ids))
        print(f"\n=== Done. Patched: {patched}  Failed: {failed} ===")
        return

    # Normal sync: find videos in DynamoDB but not in Qdrant
    print("=== Sync Missing Videos ===\n")

    print("Scanning S3...")
    s3_ids = get_s3_video_ids()
    print(f"  S3 videos: {len(s3_ids)}")

    print("Scanning DynamoDB...")
    dynamo_ids = get_dynamo_video_ids()
    print(f"  DynamoDB videos: {len(dynamo_ids)}")

    print("Scanning Qdrant...")
    qdrant_ids = get_qdrant_video_ids()
    print(f"  Qdrant videos: {len(qdrant_ids)}")

    missing = list(dynamo_ids - qdrant_ids)
    print(f"\nMissing from Qdrant: {len(missing)}")

    if not missing:
        print("Nothing to sync — all good!")
        return

    for vid in sorted(missing):
        print(f"  {vid}")

    print(f"\nIndexing {len(missing)} missing videos...")
    indexed, failed = index_videos(missing)
    print(f"\n=== Done. Indexed: {indexed}  Failed: {failed} ===")


if __name__ == "__main__":
    main()
