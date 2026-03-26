"""
backfill_thumbnails.py - Re-analyze thumbnails and write directly to DynamoDB

Since thumbnail data was never migrated to DynamoDB and Qdrant has been trimmed,
this script re-fetches each video's YouTube thumbnail, runs Gemini vision analysis,
and writes the 5 thumbnail fields to the youtube-videos DynamoDB table.

Fields written per video:
  thumbnail_visual           str   (≤20 word description of image contents)
  thumbnail_tone             str   (urgency | fear | anger | sadness | curiosity |
                                    hope | neutral | celebratory | alarming)
  thumbnail_clickbait_score  int   (1–5; 0 = unknown/unavailable)
  thumbnail_brand_consistent bool  (looks like professional broadcast news)
  thumbnail_insight          str   (one sentence on engagement strategy)

Guards:
  - Skips videos where YouTube thumbnail image is unavailable (grey tile < 5 KB)
  - Skips videos where Gemini returned a 503 / empty response (no tone + score=0)
  - Rate limit (429) handled automatically via EmbeddingService key rotation

Usage:
    python -m scripts.backfill_thumbnails
    python -m scripts.backfill_thumbnails --dry-run          # preview, no writes
    python -m scripts.backfill_thumbnails --limit 10         # test on first 10 videos
    python -m scripts.backfill_thumbnails --skip-existing    # skip videos already analyzed
    python -m scripts.backfill_thumbnails --verify-only      # check coverage only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import boto3
from app.core.config import settings
from app.services.embedding_service import EmbeddingService

REGION = os.getenv("AWS_REGION", "us-east-2")


def _dec(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def _thumbnail_is_empty(thumb: dict) -> bool:
    """
    Returns True if the result looks like a Gemini 503 failure or missing image.

    analyze_thumbnail() returns these safe defaults on ANY failure:
      thumbnail_visual           = ""
      thumbnail_tone             = ""
      thumbnail_clickbait_score  = 0
      thumbnail_brand_consistent = False
      thumbnail_insight          = ""

    A real successful result will always have a non-empty tone AND score >= 1.
    If both are missing/zero, Gemini either 503'd or the image wasn't available.
    """
    return (
        not thumb.get("thumbnail_visual")
        and not thumb.get("thumbnail_tone")
        and thumb.get("thumbnail_clickbait_score", 0) == 0
    )


# ── Step 1: Scan all videos from DynamoDB ─────────────────────────────────────

def get_all_videos(table, skip_existing: bool = False) -> list[dict]:
    """
    Scan youtube-videos for all items.
    If skip_existing=True, excludes videos that already have thumbnail_visual.
    """
    items = []
    scan_kwargs = {
        "ProjectionExpression": "PartitionKey, SortKey, title, thumbnail_visual",
    }

    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp["Items"]:
            video_id = item.get("SortKey", "")
            channel = item.get("PartitionKey", "")
            if not video_id or not channel:
                continue
            has_thumbnail = bool(item.get("thumbnail_visual"))
            if skip_existing and has_thumbnail:
                continue
            items.append({
                "channel": channel,
                "video_id": video_id,
                "title": item.get("title", ""),
                "has_thumbnail": has_thumbnail,
            })
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    return items


# ── Step 2: Write thumbnail fields to DynamoDB ────────────────────────────────

def write_thumbnail_to_dynamodb(table, channel: str, video_id: str, thumb: dict) -> bool:
    """
    Write 5 thumbnail fields to a single DynamoDB item.

    Fields:
      thumbnail_visual, thumbnail_tone, thumbnail_clickbait_score,
      thumbnail_brand_consistent, thumbnail_insight
    """
    set_parts = []
    names = {}
    values = {}

    if thumb.get("thumbnail_visual"):
        set_parts.append("#tvis = :tvis")
        names["#tvis"] = "thumbnail_visual"
        values[":tvis"] = str(thumb["thumbnail_visual"])

    if thumb.get("thumbnail_tone"):
        set_parts.append("#ttone = :ttone")
        names["#ttone"] = "thumbnail_tone"
        values[":ttone"] = str(thumb["thumbnail_tone"])

    # Always write score (0 = unknown is a valid stored value)
    set_parts.append("#tscore = :tscore")
    names["#tscore"] = "thumbnail_clickbait_score"
    values[":tscore"] = _dec(thumb.get("thumbnail_clickbait_score", 0))

    # Always write brand_consistent
    set_parts.append("#tbrand = :tbrand")
    names["#tbrand"] = "thumbnail_brand_consistent"
    values[":tbrand"] = bool(thumb.get("thumbnail_brand_consistent", False))

    if thumb.get("thumbnail_insight"):
        set_parts.append("#tins = :tins")
        names["#tins"] = "thumbnail_insight"
        values[":tins"] = str(thumb["thumbnail_insight"])

    if not set_parts:
        return False

    try:
        table.update_item(
            Key={"PartitionKey": channel, "SortKey": video_id},
            UpdateExpression="SET " + ", ".join(set_parts),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
        return True
    except Exception as e:
        print(f"  DYNAMO_WRITE_FAILED {video_id}: {e}")
        return False


# ── Step 3: Verify coverage ───────────────────────────────────────────────────

def verify(table):
    all_items = []
    scan_kwargs = {
        "ProjectionExpression": "SortKey, thumbnail_visual, thumbnail_tone, thumbnail_clickbait_score",
    }
    while True:
        resp = table.scan(**scan_kwargs)
        all_items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    total = len(all_items)
    has_visual = sum(1 for i in all_items if i.get("thumbnail_visual"))
    has_tone = sum(1 for i in all_items if i.get("thumbnail_tone"))
    has_score = sum(1 for i in all_items if i.get("thumbnail_clickbait_score") is not None)

    print(f"\n=== Thumbnail coverage ({total} total videos) ===")
    print(f"  thumbnail_visual:           {has_visual} / {total}")
    print(f"  thumbnail_tone:             {has_tone} / {total}")
    print(f"  thumbnail_clickbait_score:  {has_score} / {total}")
    print(f"  Missing:                    {total - has_visual}")

    for item in all_items:
        if item.get("thumbnail_visual"):
            print(f"\n  Sample ({item['SortKey']}):")
            print(f"    visual: {str(item.get('thumbnail_visual', ''))[:70]}...")
            print(f"    tone:   {item.get('thumbnail_tone')}")
            print(f"    score:  {item.get('thumbnail_clickbait_score')}")
            break


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-analyze thumbnails and write to DynamoDB"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--verify-only", action="store_true", help="Check coverage only")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have thumbnail_visual in DynamoDB",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between Gemini calls (default: 3.0)",
    )
    args = parser.parse_args()

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(settings.dynamodb_table)

    if args.verify_only:
        verify(table)
        return

    print("=== Thumbnail Backfill ===\n")

    print("Scanning DynamoDB for videos...")
    videos = get_all_videos(table, skip_existing=args.skip_existing)
    print(f"  Found {len(videos)} videos to process")

    if args.limit:
        videos = videos[: args.limit]
        print(f"  Limited to {len(videos)} (--limit {args.limit})")

    if not videos:
        print("Nothing to process.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would analyze thumbnails for {len(videos)} videos:")
        for v in videos[:10]:
            print(f"  {v['video_id']} ({v['channel']}) — {v['title'][:50]}")
        if len(videos) > 10:
            print(f"  ... and {len(videos) - 10} more")
        return

    print("\nInitializing Gemini client...")
    embedding_service = EmbeddingService(api_keys=settings.genai_api_keys)

    print(f"\nAnalyzing {len(videos)} thumbnails...\n")
    succeeded = 0
    skipped_503 = 0
    failed_error = 0

    for idx, video in enumerate(videos, 1):
        video_id = video["video_id"]
        channel = video["channel"]
        title = video["title"]

        print(f"[{idx}/{len(videos)}] {video_id} ({channel})")

        try:
            thumb = embedding_service.analyze_thumbnail(video_id=video_id, title=title)

            # Guard: Gemini 503 or image unavailable — both return all-default dict
            if _thumbnail_is_empty(thumb):
                print(f"  SKIPPED — Gemini returned empty result (503 or no image available)")
                skipped_503 += 1
                continue

            print(f"  tone={thumb['thumbnail_tone']}  score={thumb['thumbnail_clickbait_score']}")
            print(f"  visual={thumb['thumbnail_visual'][:60]}...")

            ok = write_thumbnail_to_dynamodb(table, channel, video_id, thumb)
            if ok:
                succeeded += 1
                print(f"  ✓ Written to DynamoDB")
            else:
                failed_error += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            failed_error += 1

        if idx < len(videos):
            time.sleep(args.sleep)

    print(f"\n=== Done ===")
    print(f"  Succeeded:                {succeeded}")
    print(f"  Skipped (503/no image):   {skipped_503}")
    print(f"  Failed (error):           {failed_error}")
    print(f"  Total:                    {len(videos)}")

    if skipped_503 > 0:
        print(f"\n  Tip: re-run with --skip-existing to retry only the {skipped_503} skipped videos.")

    print("\nVerifying coverage...")
    verify(table)


if __name__ == "__main__":
    main()