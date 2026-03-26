"""
update_dynamodb_schema.py - DynamoDB Schema Update Script

This script performs the following tasks:
1. Adds new attributes to existing items in the 'youtube-videos' table (backward compatible).
2. Creates the 'narrative-clusters' table with the specified schema and GSIs.

Requirements:
- boto3 (install with `pip install boto3`)
- AWS credentials configured (via environment or ~/.aws/credentials)

Run: python update_dynamodb_schema.py
"""

# Load .env for AWS credentials
from dotenv import load_dotenv

load_dotenv()

import boto3
from botocore.exceptions import ClientError
from decimal import Decimal

# --- CONFIG ---
DYNAMODB_REGION = "us-east-2"  # Update if needed
YOUTUBE_VIDEOS_TABLE = "youtube-videos"
NARRATIVE_CLUSTERS_TABLE = "narrative-clusters"

# --- NEW ATTRIBUTES FOR youtube-videos ---
NEW_ATTRIBUTES = {
    "topics": [],
    "sentiment": "",
    "category": "",
    "key_claims": [],
    "is_breaking": False,
    "cluster_id": "",
    "cluster_label": "",
    "thumbnail_sentiment": "",
    "clickbait_rating": Decimal("0"),
    "thumbnail_text": [],
}


def add_new_attributes_to_youtube_videos(dynamodb):
    print(f"Adding new attributes to all items in '{YOUTUBE_VIDEOS_TABLE}'...")
    table = dynamodb.Table(YOUTUBE_VIDEOS_TABLE)
    scan_kwargs = {}
    done = False
    start_key = None
    updated = 0
    while not done:
        if start_key:
            scan_kwargs["ExclusiveStartKey"] = start_key
        response = table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            update_needed = False
            update_expr = []
            expr_attr_vals = {}
            for attr, default in NEW_ATTRIBUTES.items():
                if attr not in item:
                    update_needed = True
                    update_expr.append(f"#attr_{attr} = :val_{attr}")
                    expr_attr_vals[f":val_{attr}"] = default
            if update_needed:
                update_expression = "SET " + ", ".join(update_expr)
                expr_attr_names = {
                    f"#attr_{attr}": attr for attr in NEW_ATTRIBUTES if attr not in item
                }
                # Try both key formats for robustness
                key = {"channel": item.get("channel"), "videoId": item.get("videoId")}
                if not key["channel"] or not key["videoId"]:
                    key = {
                        "PartitionKey": item.get("PartitionKey"),
                        "SortKey": item.get("SortKey"),
                    }
                table.update_item(
                    Key=key,
                    UpdateExpression=update_expression,
                    ExpressionAttributeNames=expr_attr_names,
                    ExpressionAttributeValues=expr_attr_vals,
                )
                updated += 1
        start_key = response.get("LastEvaluatedKey", None)
        done = start_key is None
    print(f"Updated {updated} items in '{YOUTUBE_VIDEOS_TABLE}'.")


def create_narrative_clusters_table(dynamodb):
    print(
        f"Creating table '{NARRATIVE_CLUSTERS_TABLE}' with GSIs and documenting all required attributes..."
    )
    # DynamoDB only enforces schema for key/index attributes, but we document all required fields here:
    # Required attributes (not enforced by DynamoDB):
    #   cluster_label (string)
    #   video_count (number)
    #   channel_count (number)
    #   channels (list of strings)
    #   dominant_sentiment (string)
    #   total_views (number)
    #   total_likes (number)
    #   total_comments (number)
    #   classified_claims (map):
    #       consensus (list of strings)
    #       debated (list of strings)
    #       unique (list of strings)
    #       avg_clickbait_rating (number)
    #       thumbnail_tone_breakdown (map of tone → count)
    # GSIs:
    #   1. cluster-index: PartitionKey=cluster_id, SortKey=SortKey
    #   2. week-index: PartitionKey=week, SortKey=SortKey
    #   3. sentiment-index: PartitionKey=sentiment, SortKey=publishedAt
    try:
        dynamodb.create_table(
            TableName=NARRATIVE_CLUSTERS_TABLE,
            KeySchema=[
                {"AttributeName": "cluster_id", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "cluster_id", "AttributeType": "S"},
                {"AttributeName": "SortKey", "AttributeType": "S"},
                {"AttributeName": "week", "AttributeType": "S"},
                {"AttributeName": "sentiment", "AttributeType": "S"},
                {"AttributeName": "publishedAt", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "cluster-index",
                    "KeySchema": [
                        {"AttributeName": "cluster_id", "KeyType": "HASH"},
                        {"AttributeName": "SortKey", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
                {
                    "IndexName": "week-index",
                    "KeySchema": [
                        {"AttributeName": "week", "KeyType": "HASH"},
                        {"AttributeName": "SortKey", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
                {
                    "IndexName": "sentiment-index",
                    "KeySchema": [
                        {"AttributeName": "sentiment", "KeyType": "HASH"},
                        {"AttributeName": "publishedAt", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            BillingMode="PROVISIONED",
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )
        print(f"Table '{NARRATIVE_CLUSTERS_TABLE}' creation initiated.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"Table '{NARRATIVE_CLUSTERS_TABLE}' already exists.")
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
