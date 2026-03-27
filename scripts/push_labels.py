"""
push_labels.py - Push cluster narratives to DynamoDB

Modes:
  --labels-only   Push labels from cluster_summary.json (no Gemini)
  --narratives    Generate headline + summary via Gemini and push (keeps existing labels)

Usage:
    python -m scripts.push_labels --labels-only
    python -m scripts.push_labels --narratives
"""

import argparse
import json
import logging
import re
import time
import boto3
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_REGION", "us-east-2")


def get_genai_keys():
    keys = []
    for key in ["GENAI_API_KEY", "GENAI_API_KEY_2", "GENAI_API_KEY_3"]:
        val = os.getenv(key)
        if val:
            keys.append(val)
    if not keys:
        # fall back to settings
        from app.core.config import settings

        keys = settings.genai_api_keys
    return keys


def push_labels():
    with open("cluster_summary.json") as f:
        data = json.load(f)

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table("narrative-clusters")

    clusters = data.get("clusters", {})
    updated = 0

    for cid, info in clusters.items():
        label = info["label"]
        try:
            table.update_item(
                Key={"cluster_id": int(cid)},
                UpdateExpression="SET cluster_label = :label",
                ExpressionAttributeValues={":label": label},
            )
            logger.info(f"UPDATED cluster={cid} label={label!r}")
            updated += 1
        except Exception as exc:
            logger.error(f"FAILED cluster={cid} error={exc}")

    print(f"\nUpdated {updated}/{len(clusters)} cluster labels in DynamoDB")


def push_narratives():
    with open("cluster_summary.json") as f:
        data = json.load(f)

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table("narrative-clusters")

    api_keys = get_genai_keys()
    key_index = 0
    client = genai.Client(api_key=api_keys[0])

    from app.core.config import settings

    model_id = settings.gemini_model_id

    clusters = data.get("clusters", {})
    updated = 0

    for cid, info in clusters.items():
        label = info["label"]
        top_topics = info.get("top_topics", [])
        top_claims = info.get("top_claims", [])
        sentiment = info.get("dominant_sentiment", "neutral")

        prompt = f"""Given this news cluster, generate:
1. headline: A full newspaper headline, 8-14 words
2. summary: One sentence, include a specific stat or data point if available from the claims

Cluster label: {label}
Topics: {top_topics}
Claims: {top_claims}
Dominant sentiment: {sentiment}

Return ONLY valid JSON, no markdown: {{"headline": "...", "summary": "..."}}"""

        for attempt in range(len(api_keys) + 1):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="low",
                        )
                    ),
                )
                raw = getattr(response, "text", "") or str(response)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)

                parsed = json.loads(raw)
                headline = parsed.get("headline")
                summary = parsed.get("summary")

                table.update_item(
                    Key={"cluster_id": int(cid)},
                    UpdateExpression="SET narrative_headline = :h, narrative_summary = :s",
                    ExpressionAttributeValues={
                        ":h": headline,
                        ":s": summary,
                    },
                )
                logger.info(f"UPDATED cluster={cid} headline={headline!r}")
                updated += 1
                break

            except Exception as exc:
                is_rate_limit = "429" in str(exc) or getattr(exc, "code", None) == 429
                if is_rate_limit:
                    key_index += 1
                    if key_index < len(api_keys):
                        client = genai.Client(api_key=api_keys[key_index])
                        logger.warning(f"KEY_ROTATED to key {key_index}")
                        continue
                    else:
                        logger.warning("ALL_KEYS_EXHAUSTED waiting 60s")
                        key_index = 0
                        client = genai.Client(api_key=api_keys[0])
                        time.sleep(60)
                        continue
                logger.error(f"FAILED cluster={cid} error={exc}")
                break

    print(f"\nUpdated {updated}/{len(clusters)} cluster narratives in DynamoDB")


def main():
    parser = argparse.ArgumentParser(description="Push cluster data to DynamoDB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--labels-only", action="store_true", help="Push labels only (no Gemini)"
    )
    group.add_argument(
        "--narratives",
        action="store_true",
        help="Generate and push headline + summary via Gemini",
    )
    args = parser.parse_args()

    if args.labels_only:
        push_labels()
    elif args.narratives:
        push_narratives()


if __name__ == "__main__":
    main()
