import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import boto3
import json
from app.core.config import settings

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

def get_dynamo_items():
    dynamo = boto3.resource("dynamodb")
    table = dynamo.Table(settings.dynamodb_table)
    items = []
    response = table.scan(ProjectionExpression="PartitionKey, SortKey")
    items.extend(response["Items"])
    while "LastEvaluatedKey" in response:
        response = table.scan(
            ProjectionExpression="PartitionKey, SortKey",
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        items.extend(response["Items"])
    return items

def delete_missing_from_s3():
    print("Scanning S3...")
    s3_ids = get_s3_video_ids()
    print(f"  S3 videos: {len(s3_ids)}")

    print("Scanning DynamoDB...")
    dynamo_items = get_dynamo_items()
    print(f"  DynamoDB items: {len(dynamo_items)}")

    # Find items present in DynamoDB but missing from S3
    missing = [(item["PartitionKey"], item["SortKey"]) for item in dynamo_items if item["SortKey"] not in s3_ids]
    print(f"\nMissing in S3 (present in DynamoDB): {len(missing)}")
    for channel, video_id in missing:
        print(f"  {video_id} (channel: {channel})")

    if not missing:
        print("Nothing to delete.")
        return

    # Delete from DynamoDB
    dynamo = boto3.resource("dynamodb")
    table = dynamo.Table(settings.dynamodb_table)
    for channel, video_id in missing:
        resp = table.delete_item(Key={"PartitionKey": channel, "SortKey": video_id})
        code = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
        print(f"Deleted {video_id} (channel: {channel}): {code}")

if __name__ == "__main__":
    delete_missing_from_s3()