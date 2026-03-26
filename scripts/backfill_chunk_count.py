# scripts/backfill_chunk_count.py
import boto3
from app.core.config import settings
from app.services.vector_service import VectorService
from decimal import Decimal

dynamo = boto3.resource("dynamodb", region_name=settings.aws_region)
table = dynamo.Table(settings.dynamodb_table)
vs = VectorService()

# Build video_id → chunk count from Qdrant
print("Counting chunks per video in Qdrant...")
chunk_counts = {}
offset = None
while True:
    results, offset = vs.client.scroll(
        collection_name=vs.collection_name,
        limit=100,
        offset=offset,
        with_payload=["transcript_index"],
    )
    for r in results:
        vid = r.payload.get("transcript_index", "")
        if vid:
            chunk_counts[vid] = chunk_counts.get(vid, 0) + 1
    if offset is None:
        break

print(f"Found chunk counts for {len(chunk_counts)} videos in Qdrant")

# Scan DynamoDB for videos missing chunk_count
print("Scanning DynamoDB for missing chunk_count...")
scan_kwargs = {"ProjectionExpression": "PartitionKey, SortKey, chunk_count, topics"}
to_fix = []
while True:
    resp = table.scan(**scan_kwargs)
    for item in resp["Items"]:
        if item.get("topics") and not item.get("chunk_count"):
            to_fix.append(
                {
                    "PartitionKey": item["PartitionKey"],
                    "SortKey": item["SortKey"],
                }
            )
    if "LastEvaluatedKey" not in resp:
        break
    scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

print(f"Videos to backfill: {len(to_fix)}")

fixed = 0
skipped = 0
for item in to_fix:
    vid = item["SortKey"]
    count = chunk_counts.get(vid)
    if not count:
        print(f"  SKIP {vid} — not in Qdrant")
        skipped += 1
        continue

    table.update_item(
        Key={"PartitionKey": item["PartitionKey"], "SortKey": vid},
        UpdateExpression="SET #cc = :cc",
        ExpressionAttributeNames={"#cc": "chunk_count"},
        ExpressionAttributeValues={":cc": Decimal(str(count))},
    )
    print(f"  FIXED {vid} → chunk_count={count}")
    fixed += 1

print(f"\nDone. Fixed: {fixed} Skipped: {skipped}")
