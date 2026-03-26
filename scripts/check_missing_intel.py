# scripts/check_missing_intel.py
import boto3, os
from app.core.config import settings

dynamo = boto3.resource("dynamodb", region_name=settings.aws_region)
table = dynamo.Table(settings.dynamodb_table)

no_chunks = []
no_topics = []
no_cluster = []

scan_kwargs = {
    "ProjectionExpression": "PartitionKey, SortKey, chunk_count, topics, cluster_id"
}
while True:
    resp = table.scan(**scan_kwargs)
    for item in resp["Items"]:
        vid = item["SortKey"]
        if not item.get("chunk_count"):
            no_chunks.append(vid)
        if not item.get("topics"):
            no_topics.append(vid)
        if item.get("cluster_id") is None:
            no_cluster.append(vid)
    if "LastEvaluatedKey" not in resp:
        break
    scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

print(f"Missing chunk_count: {len(no_chunks)}")
print(f"Missing topics/intel: {len(no_topics)}")
print(f"No cluster_id: {len(no_cluster)}")
print(f"\nNo intel video IDs:")
for v in no_topics:
    print(f"  {v}")