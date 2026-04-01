"""
generate_weekly_report.py - Batch Article Generator & DynamoDB Sync (Improved)

Enhancements:
- Adds 1-sentence overview
- Separates headline, overview, and body
- Removes hardcoded week data (dynamic from DB)
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
import re
import logging
from datetime import datetime, timezone
from decimal import Decimal

from dotenv import load_dotenv
import boto3
from boto3.dynamodb.conditions import Key
from google import genai
from google.genai import types

# ── Setup ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Helpers ──────────────────────────────────────────────────────────────────


def _deserialize(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deserialize(i) for i in obj]
    return obj


# ── DynamoDB ─────────────────────────────────────────────────────────────────


def article_exists(region, cluster_id, week_key):
    """
    Checks if an article already exists for this cluster + week.
    """
    dynamo = boto3.resource("dynamodb", region_name=region)
    table = dynamo.Table("articles")

    week_num = int(re.sub(r"\D", "", week_key))

    try:
        resp = table.query(
            KeyConditionExpression=Key("cluster_id").eq(f"cluster-{cluster_id}")
        )

        for item in resp.get("Items", []):
            if item.get("weekNumber") == week_num:
                return True

        return False

    except Exception as e:
        logger.error(f"Error checking existing article: {e}")
        return False


def save_article_to_dynamodb(region, cluster_id, week_key, title, overview, body):
    dynamo = boto3.resource("dynamodb", region_name=region)
    table = dynamo.Table("articles")

    week_num = int(re.sub(r"\D", "", week_key))
    now = datetime.now(timezone.utc).isoformat()

    article_id = f"article-{uuid.uuid4().hex[:8]}"

    item = {
        "cluster_id": f"cluster-{cluster_id}",
        "article_id": article_id,
        "title": title,
        "overview": overview,
        "body": body,
        "weekNumber": week_num,
        "createdAt": now,
        "updatedAt": now,
    }

    try:
        table.put_item(Item=item)
        return article_id
    except Exception as e:
        logger.error(f"DynamoDB save failed: {e}")
        return None


# ── Data Fetching ────────────────────────────────────────────────────────────


def load_all_clusters(region):
    """Load all clusters once and extract unique weeks dynamically."""
    dynamo = boto3.resource("dynamodb", region_name=region)
    table = dynamo.Table("narrative-clusters")

    items, scan_kwargs = [], {}
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(_deserialize(i) for i in resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    return items


def extract_weeks(clusters):
    """Dynamically find all week keys present."""
    weeks = set()
    for c in clusters:
        for w in c.get("week_data", []):
            if "week" in w:
                weeks.add(w["week"])
    return sorted(list(weeks))


def filter_clusters_by_week(clusters, week_key):
    active = []
    for c in clusters:
        week_slice = next(
            (w for w in c.get("week_data", []) if w.get("week") == week_key), None
        )
        if week_slice and week_slice.get("video_count", 0) > 0:
            active.append({**c, "_week_slice": week_slice})

    active.sort(key=lambda c: c["_week_slice"]["view_count"], reverse=True)
    return active


def fetch_cluster_videos(region, cluster_id, table_name):
    dynamo = boto3.resource("dynamodb", region_name=region)
    table = dynamo.Table(table_name)

    items, query_kwargs = [], {
        "IndexName": "cluster-index",
        "KeyConditionExpression": Key("cluster_id").eq(cluster_id),
        "ProjectionExpression": "videoId, title, transcript, viewCount",
    }

    while True:
        resp = table.query(**query_kwargs)
        items.extend(_deserialize(i) for i in resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        query_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    return [v for v in items if v.get("transcript")]


# ── Parsing ──────────────────────────────────────────────────────────────────
def parse_output(content: str):
    """
    Robust parsing even if model formatting is imperfect.
    """

    # Extract headline
    headline_match = re.search(r"Headline:\s*(.*)", content, re.IGNORECASE)
    headline = headline_match.group(1).strip() if headline_match else ""

    # Extract overview
    overview_match = re.search(r"Overview:\s*(.*)", content, re.IGNORECASE)
    overview = overview_match.group(1).strip() if overview_match else ""

    # Remove headline + overview sections completely
    body = content

    body = re.sub(r"Headline:.*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"Overview:.*", "", body, flags=re.IGNORECASE)

    # Remove "Body:" label if present
    body = re.sub(r"Body:\s*", "", body, flags=re.IGNORECASE)

    # Clean extra whitespace
    body = body.strip()

    return headline, overview, body


# ── Prompt ───────────────────────────────────────────────────────────────────


def build_prompt(cluster, videos, week_key):
    transcripts = ""
    for v in videos[:10]:
        transcripts += f"{v['title']}\n{v['transcript'][:1500]}\n---\n"

    return f"""
You are a senior journalist.

Write a structured news article.

WEEK: {week_key}
TOPIC: {cluster.get('cluster_label')}
SENTIMENT: {cluster.get('dominant_sentiment')}

CONTENT:
{transcripts}

FORMAT STRICTLY:

Headline: <clear title>

Overview: <ONE sentence summary>

Body:
<400-600 word article>

RULES:
- Do NOT repeat headline or overview inside body
- Be objective and neutral
"""


# ── Generation ───────────────────────────────────────────────────────────────


def generate_article(client, model_id, prompt):
    res = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return res.text


# ── Pipeline ─────────────────────────────────────────────────────────────────


def process_week(week_key, clusters, args, client, model_id):
    logger.info(f"Processing {week_key}")

    week_clusters = filter_clusters_by_week(clusters, week_key)

    for i, cluster in enumerate(week_clusters, 1):
        cid = int(cluster["cluster_id"])
        label = cluster.get("cluster_label", f"Cluster {cid}")

        logger.info(f"[{i}] {label}")

        if article_exists(args.region, cid, week_key):
            logger.info(f"Skipping cluster {cid} for {week_key} (already exists)")
            continue

        videos = fetch_cluster_videos(args.region, cid, args.videos_table)

        if not videos:
            continue

        try:
            prompt = build_prompt(cluster, videos, week_key)
            raw = generate_article(client, model_id, prompt)

            title, overview, body = parse_output(raw)

            art_id = save_article_to_dynamodb(
                args.region, cid, week_key, title, overview, body
            )

            if art_id:
                logger.info(f"Saved {art_id}")

            time.sleep(2)

        except Exception as e:
            logger.error(f"Error cluster {cid}: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--videos-table", default="youtube-videos")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")

    client = genai.Client(api_key=api_key)

    # 🔥 NEW: dynamic weeks
    clusters = load_all_clusters(args.region)
    weeks = extract_weeks(clusters)

    logger.info(f"Found weeks: {weeks}")

    for week in weeks:
        process_week(week, clusters, args, client, model_id)


if __name__ == "__main__":
    main()
