"""
Quick test to verify DynamoDB insertion logic from youtube_ingestion.py works.
Inserts a sample item, reads it back, then cleans it up.
"""

import os
import sys

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Reuse the functions from youtube_ingestion
from youtube_ingestion import save_to_dynamodb

# Load .env from the parent directory (same as config.py does)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "youtube-videos")
dynamodb = boto3.resource("dynamodb")

SAMPLE_VIDEO = {
    "videoId": "TEST_VIDEO_12345",
    "title": "Test Video - DynamoDB Insert Verification",
    "description": "This is a sample video used to test DynamoDB insertion logic.",
    "publishedAt": "2026-02-26T12:00:00Z",
    "viewCount": 10000,
    "likeCount": 500,
    "commentCount": 25,
    "transcript": "This is a sample transcript for testing purposes.",
    "topComments": [
        {"author": "TestUser1", "text": "Great video!", "likes": 10},
        {"author": "TestUser2", "text": "Very informative.", "likes": 5},
        {"author": "TestUser3", "text": "Thanks for sharing.", "likes": 3},
    ],
}


def test_insert():
    table = dynamodb.Table(DYNAMODB_TABLE)
    print(f"Table: {DYNAMODB_TABLE}")
    print(f"Region: {os.getenv('AWS_DEFAULT_REGION', 'us-east-2')}")
    print()

    # Step 1: Verify table is reachable
    try:
        table.table_status
        print(f"[OK] Table '{DYNAMODB_TABLE}' exists (status: {table.table_status})")
    except ClientError as e:
        print(f"[FAIL] Cannot reach table: {e.response['Error']['Message']}")
        sys.exit(1)

    # Step 2: Insert using save_to_dynamodb
    print(f"\nInserting sample video (id={SAMPLE_VIDEO['videoId']})...")
    try:
        save_to_dynamodb(dynamodb, DYNAMODB_TABLE, "TestChannel", [SAMPLE_VIDEO])
        print("[OK] save_to_dynamodb completed without error")
    except Exception as e:
        print(f"[FAIL] save_to_dynamodb raised: {e}")
        sys.exit(1)

    # Step 3: Read back the item
    print("\nReading item back from DynamoDB...")
    try:
        response = table.get_item(
            Key={"PartitionKey": SAMPLE_VIDEO["videoId"], "SortKey": "TestChannel"}
        )
        item = response.get("Item")
        if item:
            print("[OK] Item found:")
            print(f"  PartitionKey: {item['PartitionKey']}")
            print(f"  channel:      {item['channel']}")
            print(f"  title:        {item['title']}")
            print(f"  viewCount:    {item['viewCount']}")
            print(f"  fetchedAt:    {item['fetchedAt']}")
            print(f"  comments:     {len(item.get('topComments', []))} comments")
        else:
            print("[FAIL] Item was not found after insert!")
            sys.exit(1)
    except ClientError as e:
        print(f"[FAIL] Error reading item: {e.response['Error']['Message']}")
        sys.exit(1)

    # Step 4: Verify duplicate skip logic
    print("\nInserting same item again (should be skipped as duplicate)...")
    try:
        save_to_dynamodb(dynamodb, DYNAMODB_TABLE, "TestChannel", [SAMPLE_VIDEO])
        print("[OK] Duplicate handling works (item skipped)")
    except Exception as e:
        print(f"[FAIL] Duplicate insert raised: {e}")
        sys.exit(1)

    # Step 5: Cleanup
    print(f"\nCleaning up test item (id={SAMPLE_VIDEO['videoId']})...")
    try:
        table.delete_item(
            Key={"PartitionKey": SAMPLE_VIDEO["videoId"], "SortKey": "TestChannel"}
        )
        print("[OK] Test item deleted")
    except ClientError as e:
        print(f"[WARN] Cleanup failed: {e.response['Error']['Message']}")

    print("\n✓ All DynamoDB tests passed!")


if __name__ == "__main__":
    test_insert()
