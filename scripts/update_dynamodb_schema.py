"""
update_dynamodb_schema.py - DynamoDB Schema Migration

Step 1: Add GSIs to youtube-videos table (cluster-index, week-index, sentiment-index)
Step 2: Create narrative-clusters table (partition key: cluster_id)
Step 3: Verify both tables are ready

Does NOT pre-populate empty fields — the pipeline writes real values during processing.

Requirements:
  pip install boto3 python-dotenv

Run:
  python scripts/update_dynamodb_schema.py
  python scripts/update_dynamodb_schema.py --skip-gsis    # skip GSI creation if already done
  python scripts/update_dynamodb_schema.py --verify-only  # just check table status
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────

REGION = os.getenv("AWS_REGION", "us-east-2")
YOUTUBE_VIDEOS_TABLE = os.getenv("DYNAMODB_TABLE", "youtube-videos")
NARRATIVE_CLUSTERS_TABLE = "narrative-clusters"


def get_clients():
    client = boto3.client("dynamodb", region_name=REGION)
    resource = boto3.resource("dynamodb", region_name=REGION)
    return client, resource


# ── Step 1: Add GSIs to youtube-videos ────────────────────────────────────────


def add_gsis_to_youtube_videos(client):
    """
    Add three GSIs to youtube-videos table.
    DynamoDB only allows one GSI creation at a time in a single update,
    so we add them sequentially and wait for each to become ACTIVE.
    """
    gsis = [
        {
            "name": "cluster-index",
            "key_schema": [
                {"AttributeName": "cluster_id", "KeyType": "HASH"},
                {"AttributeName": "SortKey", "KeyType": "RANGE"},
            ],
            "attribute_definitions": [
                {"AttributeName": "cluster_id", "AttributeType": "N"},
                {"AttributeName": "SortKey", "AttributeType": "S"},
            ],
        },
        {
            "name": "week-index",
            "key_schema": [
                {"AttributeName": "week", "KeyType": "HASH"},
                {"AttributeName": "SortKey", "KeyType": "RANGE"},
            ],
            "attribute_definitions": [
                {"AttributeName": "week", "AttributeType": "S"},
                {"AttributeName": "SortKey", "AttributeType": "S"},
            ],
        },
        {
            "name": "sentiment-index",
            "key_schema": [
                {"AttributeName": "sentiment", "KeyType": "HASH"},
                {"AttributeName": "publishedAt", "KeyType": "RANGE"},
            ],
            "attribute_definitions": [
                {"AttributeName": "sentiment", "AttributeType": "S"},
                {"AttributeName": "publishedAt", "AttributeType": "S"},
            ],
        },
    ]

    # Check which GSIs already exist
    try:
        desc = client.describe_table(TableName=YOUTUBE_VIDEOS_TABLE)
        existing_gsis = {
            gsi["IndexName"] for gsi in desc["Table"].get("GlobalSecondaryIndexes", [])
        }
    except ClientError as e:
        print(f"ERROR: Cannot describe table {YOUTUBE_VIDEOS_TABLE}: {e}")
        sys.exit(1)

    for gsi in gsis:
        gsi_name = gsi["name"]

        if gsi_name in existing_gsis:
            print(f"  GSI '{gsi_name}' already exists — skipping")
            continue

        print(f"  Creating GSI '{gsi_name}' on {YOUTUBE_VIDEOS_TABLE}...")

        try:
            client.update_table(
                TableName=YOUTUBE_VIDEOS_TABLE,
                AttributeDefinitions=gsi["attribute_definitions"],
                GlobalSecondaryIndexUpdates=[
                    {
                        "Create": {
                            "IndexName": gsi_name,
                            "KeySchema": gsi["key_schema"],
                            "Projection": {"ProjectionType": "ALL"},
                        }
                    }
                ],
            )
        except ClientError as e:
            if "already exists" in str(e).lower():
                print(f"  GSI '{gsi_name}' already exists")
                continue
            raise

        # Wait for GSI to become ACTIVE
        print(f"  Waiting for GSI '{gsi_name}' to become ACTIVE...")
        _wait_for_gsi(client, YOUTUBE_VIDEOS_TABLE, gsi_name)
        print(f"  GSI '{gsi_name}' is ACTIVE")


def _wait_for_gsi(client, table_name, gsi_name, timeout=300):
    """Poll table until the specified GSI is ACTIVE."""
    elapsed = 0
    while elapsed < timeout:
        desc = client.describe_table(TableName=table_name)
        for gsi in desc["Table"].get("GlobalSecondaryIndexes", []):
            if gsi["IndexName"] == gsi_name:
                status = gsi["IndexStatus"]
                if status == "ACTIVE":
                    return
                print(f"    {gsi_name}: {status} ({elapsed}s)")
        time.sleep(10)
        elapsed += 10

    raise TimeoutError(f"GSI '{gsi_name}' did not become ACTIVE within {timeout}s")


# ── Step 2: Create narrative-clusters table ───────────────────────────────────


def create_narrative_clusters_table(client):
    """
    Create the narrative-clusters table.
    - Partition key: cluster_id (Number)
    - No sort key (one item per cluster)
    - No GSIs (only ~12 items, scan or get_item is sufficient)
    - PAY_PER_REQUEST billing (free tier friendly, no capacity planning)
    """
    print(f"\nCreating table '{NARRATIVE_CLUSTERS_TABLE}'...")

    try:
        client.create_table(
            TableName=NARRATIVE_CLUSTERS_TABLE,
            KeySchema=[
                {"AttributeName": "cluster_id", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "cluster_id", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"  Table creation initiated. Waiting for ACTIVE status...")
        _wait_for_table(client, NARRATIVE_CLUSTERS_TABLE)
        print(f"  Table '{NARRATIVE_CLUSTERS_TABLE}' is ACTIVE")

    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"  Table '{NARRATIVE_CLUSTERS_TABLE}' already exists")
        else:
            raise


# --- VALIDATION FUNCTION ---
def validate_cluster_id_foreign_keys(dynamodb):
    """
    Validates that all non-empty cluster_id values in youtube-videos exist as primary keys in narrative-clusters.
    Prints any orphaned cluster_id values.
    """
    print("Validating cluster_id foreign key relationship...")
    videos_table = dynamodb.Table(YOUTUBE_VIDEOS_TABLE)
    clusters_table = dynamodb.Table(NARRATIVE_CLUSTERS_TABLE)
    # Gather all cluster_ids in narrative-clusters
    cluster_ids = set()
    scan_kwargs = {}
    done = False
    start_key = None
    while not done:
        if start_key:
            scan_kwargs["ExclusiveStartKey"] = start_key
        response = clusters_table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            if "cluster_id" in item:
                cluster_ids.add(item["cluster_id"])
        start_key = response.get("LastEvaluatedKey", None)
        done = start_key is None
    # Check all youtube-videos for cluster_id
    orphaned = set()
    scan_kwargs = {}
    done = False
    start_key = None
    while not done:
        if start_key:
            scan_kwargs["ExclusiveStartKey"] = start_key
        response = videos_table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            cid = item.get("cluster_id", "")
            if cid and cid not in cluster_ids:
                orphaned.add(cid)
        start_key = response.get("LastEvaluatedKey", None)
        done = start_key is None
    if orphaned:
        print(
            f"Orphaned cluster_id values in youtube-videos (no matching narrative-clusters): {orphaned}"
        )
    else:
        raise Exception("No cluster_id values found in youtube-videos to validate.")


def main():
    dynamodb = boto3.resource("dynamodb", region_name=DYNAMODB_REGION)
    client = boto3.client("dynamodb", region_name=DYNAMODB_REGION)
    add_new_attributes_to_youtube_videos(dynamodb)
    # create_narrative_clusters_table(client)
    # insert_sample_narrative_cluster(dynamodb)
    validate_cluster_id_foreign_keys(dynamodb)
    print("Done.")


def insert_sample_narrative_cluster(dynamodb):
    """
    Inserts a sample item into the narrative-clusters table with all required fields and nested map structure.
    """
    table = dynamodb.Table(NARRATIVE_CLUSTERS_TABLE)
    sample_item = {
        "cluster_id": "sample-cluster-001",
        "cluster_label": "Sample Cluster Label",
        "video_count": Decimal("10"),
        "channel_count": Decimal("3"),
        "channels": ["channelA", "channelB", "channelC"],
        "dominant_sentiment": "positive",
        "total_views": Decimal("12345"),
        "total_likes": Decimal("678"),
        "total_comments": Decimal("90"),
        "classified_claims": {
            "consensus": ["claim1", "claim2"],
            "debated": ["claim3"],
            "unique": ["claim4", "claim5"],
            "avg_clickbait_rating": Decimal("2.5"),
            "thumbnail_tone_breakdown": {
                "neutral": Decimal("5"),
                "positive": Decimal("3"),
                "negative": Decimal("2"),
            },
        },
        # Example GSI fields (optional, for demonstration)
        "SortKey": "2024-01-01T00:00:00Z",
        "week": "2024-W01",
        "sentiment": "positive",
        "publishedAt": "2024-01-01T00:00:00Z",
    }
    try:
        table.put_item(Item=sample_item)
        print("Sample narrative-clusters item inserted for schema demonstration.")
    except Exception as e:
        print(f"Failed to insert sample narrative-clusters item: {e}")


if __name__ == "__main__":
    main()
