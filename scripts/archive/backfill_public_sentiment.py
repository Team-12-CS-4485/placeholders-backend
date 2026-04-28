"""
backfill_public_sentiment.py - Backfill public sentiment for existing videos

Scans DynamoDB for videos that have topComments but no public_sentiment.
Runs a lightweight Gemini call (comments only, no transcript) to classify
audience sentiment and writes public_sentiment + public_sentiment_score.

Usage:
    python -m scripts.backfill_public_sentiment
    python -m scripts.backfill_public_sentiment --dry-run
    python -m scripts.backfill_public_sentiment --limit 50
"""

import argparse
import json
import os
import re
import sys
import time
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import boto3
from google import genai
from google.genai import types
from app.core.config import settings

REGION = os.getenv("AWS_REGION", "us-east-2")


def _dec(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def get_videos_needing_backfill(table, limit=None):
    """
    Scan for videos that have topComments but no public_sentiment.
    Returns list of {channel, video_id, top_comments[]}.
    """
    videos = []
    scan_kwargs = {
        "ProjectionExpression": "PartitionKey, SortKey, topComments, public_sentiment",
    }

    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp["Items"]:
            # Skip if already has public_sentiment
            if item.get("public_sentiment"):
                continue

            # Skip if no comments
            comments_raw = item.get("topComments", [])
            if not comments_raw:
                continue

            # Extract comment text
            comment_texts = []
            for c in comments_raw[:5]:
                if isinstance(c, dict):
                    text = c.get("text", "")
                    if text.strip():
                        comment_texts.append(text)
                elif isinstance(c, str) and c.strip():
                    comment_texts.append(c)

            if not comment_texts:
                continue

            videos.append(
                {
                    "channel": item["PartitionKey"],
                    "video_id": item["SortKey"],
                    "top_comments": comment_texts,
                }
            )

            if limit and len(videos) >= limit:
                return videos

        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    return videos


def classify_comment_sentiment(client, comments: list[str], model: str) -> dict:
    """
    Lightweight Gemini call — just comments, no transcript.
    Returns {public_sentiment, public_sentiment_score}.
    """
    comments_text = "\n".join(f"- {c}" for c in comments)
    prompt = (
        "Below are top viewer comments from a YouTube news video.\n\n"
        f"{comments_text}\n\n"
        "Classify the overall audience sentiment.\n"
        "Return ONLY valid JSON:\n"
        '{"public_sentiment": "<positive|negative|neutral|mixed>", '
        '"public_sentiment_score": <float -1.0 to 1.0>}\n\n'
        "Rules:\n"
        "- positive: supportive, agreeing, enthusiastic\n"
        "- negative: critical, angry, disappointed\n"
        "- neutral: informational, factual, no strong opinion\n"
        "- mixed: divided opinions, both positive and negative\n"
        "- public_sentiment_score: -1.0 (very negative) to 1.0 (very positive)\n"
        "Return raw JSON only."
    )

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        ),
    )

    raw = getattr(response, "text", "") or str(response)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    sentiment = data.get("public_sentiment", "neutral")
    if sentiment not in ("positive", "negative", "neutral", "mixed"):
        sentiment = "neutral"

    score = data.get("public_sentiment_score", 0.0)
    try:
        score = max(-1.0, min(1.0, float(score)))
    except (ValueError, TypeError):
        score = 0.0

    return {"public_sentiment": sentiment, "public_sentiment_score": score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(settings.dynamodb_table)

    print("Scanning for videos needing public sentiment backfill...")
    videos = get_videos_needing_backfill(table, limit=args.limit)
    print(f"Found {len(videos)} videos to backfill")

    if not videos:
        print("Nothing to backfill!")
        return

    if args.dry_run:
        for v in videos[:10]:
            print(
                f"  {v['channel']}/{v['video_id']} — {len(v['top_comments'])} comments"
            )
        if len(videos) > 10:
            print(f"  ... and {len(videos) - 10} more")
        print("\nDry run — no changes made.")
        return

    # Set up Gemini client with key rotation
    api_keys = settings.genai_api_keys
    key_index = 0
    client = genai.Client(api_key=api_keys[0])
    model = settings.gemini_model_id

    MAX_RETRIES = len(api_keys) + 2  # rotate all keys + 2 cooldown retries
    success = 0
    failed = 0

    for i, video in enumerate(videos):
        vid = video["video_id"]
        channel = video["channel"]
        comments = video["top_comments"]

        print(
            f"[{i+1}/{len(videos)}] {channel}/{vid} ({len(comments)} comments)...",
            end=" ",
        )

        result = None
        for attempt in range(MAX_RETRIES):
            try:
                result = classify_comment_sentiment(client, comments, model)
                break
            except Exception as exc:
                err_str = str(exc)
                code = getattr(exc, "code", None)
                is_rate_limit = "429" in err_str or code == 429
                is_server_error = (
                    "503" in err_str or "500" in err_str or code in (500, 503)
                )

                if is_rate_limit:
                    key_index += 1
                    if key_index < len(api_keys):
                        client = genai.Client(api_key=api_keys[key_index])
                        print(
                            f"\n    429 — rotated to key {key_index + 1}/{len(api_keys)}, retrying...",
                            end=" ",
                        )
                        continue
                    else:
                        print(
                            f"\n    429 — all keys exhausted, resetting + waiting 60s...",
                            end=" ",
                        )
                        key_index = 0
                        client = genai.Client(api_key=api_keys[0])
                        time.sleep(60)
                        continue

                if is_server_error:
                    wait = min(5 * (attempt + 1), 30)
                    print(
                        f"\n    503 — waiting {wait}s, retrying (attempt {attempt + 1})...",
                        end=" ",
                    )
                    time.sleep(wait)
                    continue

                # Non-retryable error
                print(f"FAILED: {exc}")
                break

        if result:
            table.update_item(
                Key={"PartitionKey": channel, "SortKey": vid},
                UpdateExpression="SET #ps = :ps, #pss = :pss",
                ExpressionAttributeNames={
                    "#ps": "public_sentiment",
                    "#pss": "public_sentiment_score",
                },
                ExpressionAttributeValues={
                    ":ps": result["public_sentiment"],
                    ":pss": _dec(result["public_sentiment_score"]),
                },
            )
            print(
                f"{result['public_sentiment']} ({result['public_sentiment_score']:+.2f})"
            )
            success += 1
            time.sleep(0.5)  # pace at ~120 RPM to stay under 150 RPM (30 keys × 5 RPM)
        else:
            failed += 1

    print(f"\n=== Done. Success: {success} | Failed: {failed} ===")


if __name__ == "__main__":
    main()
