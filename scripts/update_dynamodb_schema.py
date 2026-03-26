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
            gsi["IndexName"]
            for gsi in desc["Table"].get("GlobalSecondaryIndexes", [])
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


def _wait_for_table(client, table_name, timeout=120):
    """Poll until table is ACTIVE."""
    elapsed = 0
    while elapsed < timeout:
        desc = client.describe_table(TableName=table_name)
        status = desc["Table"]["TableStatus"]
        if status == "ACTIVE":
            return
        print(f"    {table_name}: {status} ({elapsed}s)")
        time.sleep(5)
        elapsed += 5

    raise TimeoutError(f"Table '{table_name}' did not become ACTIVE within {timeout}s")


# ── Step 3: Verify ────────────────────────────────────────────────────────────

def verify_tables(client):
    """Print status of both tables and all GSIs."""
    print("\n=== Verification ===\n")

    # youtube-videos
    try:
        desc = client.describe_table(TableName=YOUTUBE_VIDEOS_TABLE)
        table = desc["Table"]
        print(f"Table: {YOUTUBE_VIDEOS_TABLE}")
        print(f"  Status: {table['TableStatus']}")
        print(f"  Items: {table.get('ItemCount', 'unknown')}")
        print(f"  Key: {_format_key_schema(table['KeySchema'])}")
        print(f"  Billing: {table.get('BillingModeSummary', {}).get('BillingMode', 'unknown')}")

        gsis = table.get("GlobalSecondaryIndexes", [])
        if gsis:
            print(f"  GSIs ({len(gsis)}):")
            for gsi in gsis:
                print(f"    - {gsi['IndexName']}: {_format_key_schema(gsi['KeySchema'])} [{gsi['IndexStatus']}]")
        else:
            print("  GSIs: none")

    except ClientError as e:
        print(f"  ERROR: {e}")

    print()

    # narrative-clusters
    try:
        desc = client.describe_table(TableName=NARRATIVE_CLUSTERS_TABLE)
        table = desc["Table"]
        print(f"Table: {NARRATIVE_CLUSTERS_TABLE}")
        print(f"  Status: {table['TableStatus']}")
        print(f"  Items: {table.get('ItemCount', 'unknown')}")
        print(f"  Key: {_format_key_schema(table['KeySchema'])}")
        print(f"  Billing: {table.get('BillingModeSummary', {}).get('BillingMode', 'unknown')}")
        print(f"  GSIs: none (by design — only ~12 items)")

    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            print(f"Table: {NARRATIVE_CLUSTERS_TABLE}")
            print(f"  Status: DOES NOT EXIST")
        else:
            print(f"  ERROR: {e}")


def _format_key_schema(key_schema):
    parts = []
    for key in key_schema:
        key_type = "PK" if key["KeyType"] == "HASH" else "SK"
        parts.append(f"{key_type}={key['AttributeName']}")
    return ", ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DynamoDB schema migration")
    parser.add_argument(
        "--skip-gsis",
        action="store_true",
        help="Skip GSI creation on youtube-videos",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify table status, don't create anything",
    )
    args = parser.parse_args()

    client, resource = get_clients()

    if args.verify_only:
        verify_tables(client)
        return

    print("=== DynamoDB Schema Migration ===\n")

    # Step 1: GSIs on youtube-videos
    if args.skip_gsis:
        print("Step 1: Skipping GSI creation (--skip-gsis)\n")
    else:
        print(f"Step 1: Adding GSIs to {YOUTUBE_VIDEOS_TABLE}")
        add_gsis_to_youtube_videos(client)

    # Step 2: Create narrative-clusters
    print(f"\nStep 2: Creating {NARRATIVE_CLUSTERS_TABLE}")
    create_narrative_clusters_table(client)

    # Step 3: Verify
    verify_tables(client)

    print("\n=== Migration Complete ===")
    print("\nNext steps:")
    print("  1. Update pipeline_service.py to write intelligence fields to DynamoDB")
    print("  2. Update clustering_service.py to write cluster_id to DynamoDB + narrative-clusters")
    print("  3. Update claim_analysis_service.py to write to narrative-clusters")
    print("  4. Update trend_service.py to read from DynamoDB instead of Qdrant")


if __name__ == "__main__":
    main()