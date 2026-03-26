"""
migrate_v2.py - Simple Qdrant → DynamoDB migration

No complex expression builders. Just reads Qdrant, writes DynamoDB directly.

Usage:
    python -m scripts.migrate_v2
    python -m scripts.migrate_v2 --verify-only
"""

import os
import sys
import re
from datetime import datetime, timezone
from decimal import Decimal
from collections import Counter, defaultdict

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import boto3
from app.services.vector_service import VectorService

REGION = os.getenv("AWS_REGION", "us-east-2")
dynamodb = boto3.resource("dynamodb", region_name=REGION)
videos_table = dynamodb.Table(os.getenv("DYNAMODB_TABLE", "youtube-videos"))
clusters_table = dynamodb.Table("narrative-clusters")


def extract_week(source_key):
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


def dec(val):
    """Convert numbers to Decimal for DynamoDB."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def read_qdrant():
    vs = VectorService()
    all_chunks = []
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
            all_chunks.append(point.payload)
        if offset is None:
            break

    # Deduplicate to video level
    videos = {}
    for p in all_chunks:
        vid = p.get("transcript_index", "")
        if vid and vid not in videos:
            videos[vid] = p

    print(f"Read {len(all_chunks)} chunks → {len(videos)} unique videos from Qdrant")
    return videos


def get_dynamo_keys():
    """Build lookup: videoId → {PartitionKey, SortKey}"""
    keys = {}
    scan_kwargs = {"ProjectionExpression": "PartitionKey, SortKey"}
    while True:
        resp = videos_table.scan(**scan_kwargs)
        for item in resp["Items"]:
            keys[item["SortKey"]] = {
                "PartitionKey": item["PartitionKey"],
                "SortKey": item["SortKey"],
            }
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    print(f"Found {len(keys)} items in DynamoDB")
    return keys


def migrate_videos(videos, dynamo_keys):
    updated = 0
    skipped = 0
    not_found = 0

    for video_id, p in videos.items():
        key = dynamo_keys.get(video_id)
        if not key:
            not_found += 1
            continue

        source_key = p.get("source_key") or ""
        week = extract_week(source_key)
        cluster_id = p.get("cluster_id")
        topics = p.get("topics") or []
        category = p.get("category") or ""
        sentiment = p.get("sentiment") or ""
        key_claims = p.get("key_claims") or []
        is_breaking = bool(p.get("is_breaking", False))
        cluster_label = p.get("cluster_label") or ""
        cluster_confidence = p.get("cluster_confidence") or 0

        # Build update parts manually — only include non-empty values
        set_parts = []
        names = {}
        values = {}

        if topics:
            set_parts.append("#topics = :topics")
            names["#topics"] = "topics"
            values[":topics"] = topics

        if category:
            set_parts.append("#category = :category")
            names["#category"] = "category"
            values[":category"] = category

        if sentiment:
            set_parts.append("#sentiment = :sentiment")
            names["#sentiment"] = "sentiment"
            values[":sentiment"] = sentiment

        if key_claims:
            set_parts.append("#claims = :claims")
            names["#claims"] = "key_claims"
            values[":claims"] = key_claims

        # Always write these
        set_parts.append("#breaking = :breaking")
        names["#breaking"] = "is_breaking"
        values[":breaking"] = is_breaking

        if source_key:
            set_parts.append("#src = :src")
            names["#src"] = "source_key"
            values[":src"] = source_key

        set_parts.append("#week = :week")
        names["#week"] = "week"
        values[":week"] = week

        if cluster_id is not None and cluster_id != -1:
            set_parts.append("#cid = :cid")
            names["#cid"] = "cluster_id"
            values[":cid"] = dec(cluster_id)

            set_parts.append("#clabel = :clabel")
            names["#clabel"] = "cluster_label"
            values[":clabel"] = cluster_label

            set_parts.append("#cconf = :cconf")
            names["#cconf"] = "cluster_confidence"
            values[":cconf"] = dec(cluster_confidence)

        set_parts.append("#ts = :ts")
        names["#ts"] = "indexed_at"
        values[":ts"] = datetime.now(timezone.utc).isoformat()

        if not set_parts:
            skipped += 1
            continue

        try:
            videos_table.update_item(
                Key=key,
                UpdateExpression="SET " + ", ".join(set_parts),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )
            updated += 1
        except Exception as e:
            print(f"  FAILED {video_id}: {e}")
            skipped += 1

    print(f"Videos: updated={updated} skipped={skipped} not_found={not_found}")
    return updated


def migrate_clusters(videos):
    cluster_videos = defaultdict(list)
    for vid, p in videos.items():
        cid = p.get("cluster_id")
        if cid is not None and cid != -1:
            cluster_videos[int(cid)].append(p)

    print(f"Found {len(cluster_videos)} clusters")

    for cid, vids in cluster_videos.items():
        channels = set()
        sentiments = Counter()
        categories = Counter()
        topics_counter = Counter()
        breaking = 0
        views = 0
        likes = 0
        comments = 0

        for v in vids:
            channels.add(v.get("channel", ""))
            sentiments[v.get("sentiment", "neutral")] += 1
            categories[v.get("category", "Other")] += 1
            for t in v.get("topics", []):
                topics_counter[t] += 1
            if v.get("is_breaking"):
                breaking += 1
            views += v.get("view_count", 0)
            likes += v.get("like_count", 0)
            comments += v.get("comment_count", 0)

        # Get classified claims — handle None
        raw_claims = vids[0].get("classified_claims")
        if raw_claims and isinstance(raw_claims, dict):
            claims = raw_claims
        else:
            claims = {"consensus": [], "debated": [], "unique": []}

        # Convert any floats in claims to Decimal
        claims_clean = _clean_for_dynamo(claims)

        item = {
            "cluster_id": dec(cid),
            "cluster_label": vids[0].get("cluster_label", f"Cluster {cid}"),
            "video_count": dec(len(vids)),
            "channel_count": dec(len(channels)),
            "channels": sorted(channels),
            "top_topics": [t[0] for t in topics_counter.most_common(5)],
            "dominant_category": categories.most_common(1)[0][0] if categories else "Other",
            "dominant_sentiment": sentiments.most_common(1)[0][0] if sentiments else "neutral",
            "breaking_count": dec(breaking),
            "total_views": dec(views),
            "total_likes": dec(likes),
            "total_comments": dec(comments),
            "classified_claims": claims_clean,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        clusters_table.put_item(Item=item)
        print(f"  Cluster {cid}: {item['cluster_label']} ({len(vids)} videos)")

    print(f"Wrote {len(cluster_videos)} clusters")


def _clean_for_dynamo(obj):
    """Recursively convert floats to Decimal and remove None values."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return Decimal(str(round(obj, 4)))
    if isinstance(obj, int):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _clean_for_dynamo(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_clean_for_dynamo(i) for i in obj if i is not None]
    return obj


def verify():
    # Videos
    all_items = []
    scan_kwargs = {}
    while True:
        resp = videos_table.scan(
            **scan_kwargs,
            ProjectionExpression="SortKey, topics, sentiment, cluster_id, week"
        )
        all_items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    has_topics = sum(1 for i in all_items if i.get("topics"))
    has_sentiment = sum(1 for i in all_items if i.get("sentiment"))
    has_cluster = sum(1 for i in all_items if i.get("cluster_id") is not None)
    has_week = sum(1 for i in all_items if i.get("week"))

    print(f"\n=== youtube-videos ({len(all_items)} items) ===")
    print(f"  Has topics:     {has_topics}")
    print(f"  Has sentiment:  {has_sentiment}")
    print(f"  Has cluster_id: {has_cluster}")
    print(f"  Has week:       {has_week}")
    print(f"  No intel:       {len(all_items) - has_topics}")

    # Sample one migrated item
    for item in all_items:
        if item.get("topics"):
            resp = videos_table.get_item(
                Key={"PartitionKey": "NBCNews", "SortKey": item["SortKey"]}
            )
            if "Item" in resp:
                i = resp["Item"]
                print(f"\n  Sample: {i['SortKey']}")
                print(f"    topics: {i.get('topics', [])[:2]}...")
                print(f"    sentiment: {i.get('sentiment')}")
                print(f"    cluster_id: {i.get('cluster_id')}")
                print(f"    week: {i.get('week')}")
            break

    # Clusters
    resp = clusters_table.scan()
    print(f"\n=== narrative-clusters ({len(resp['Items'])} items) ===")
    for item in sorted(resp["Items"], key=lambda x: int(x["cluster_id"])):
        cid = item["cluster_id"]
        label = item.get("cluster_label", "?")
        vc = item.get("video_count", 0)
        claims = item.get("classified_claims") or {}
        c = len(claims.get("consensus", []))
        d = len(claims.get("debated", []))
        u = len(claims.get("unique", []))
        print(f"  Cluster {cid}: {label} ({vc} videos, claims: {c}c/{d}d/{u}u)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify()
        sys.exit(0)

    print("=== Qdrant → DynamoDB Migration v2 ===\n")

    videos = read_qdrant()
    dynamo_keys = get_dynamo_keys()

    print("\nMigrating videos...")
    migrate_videos(videos, dynamo_keys)

    print("\nMigrating clusters...")
    migrate_clusters(videos)

    print("\nVerifying...")
    verify()

    print("\n=== Done ===")