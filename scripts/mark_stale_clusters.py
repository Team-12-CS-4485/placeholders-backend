"""
mark_stale_clusters.py — One-time script to mark clusters missing the current
week as declining. Future runs will handle this automatically via the freshness
check in run_clustering().
"""
import boto3
from collections import defaultdict
from decimal import Decimal
from datetime import datetime, timezone
from app.core.config import settings

dynamo = boto3.resource("dynamodb", region_name=settings.aws_region)
videos_table = dynamo.Table(settings.dynamodb_table)
clusters_table = dynamo.Table("narrative-clusters")


def scan_all(table, **extra):
    items, kwargs = [], dict(extra)
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


# Find which weeks each cluster has videos in
print("Scanning videos...")
videos = scan_all(
    videos_table,
    ProjectionExpression="SortKey, cluster_id, #wk",
    ExpressionAttributeNames={"#wk": "week"},
)

weeks_per_cluster = defaultdict(set)
for v in videos:
    cid = v.get("cluster_id")
    if cid is not None:
        weeks_per_cluster[int(cid)].add(v.get("week", "unknown"))

# Find the current (latest) week
all_weeks = set()
for ws in weeks_per_cluster.values():
    all_weeks |= ws
week_nums = []
for w in all_weeks:
    if w.startswith("week") and w[4:].isdigit():
        week_nums.append((int(w[4:]), w))
current_week = max(week_nums, key=lambda x: x[0])[1] if week_nums else None
print(f"Current week: {current_week}\n")

# Load clusters
clusters = scan_all(clusters_table)
now = datetime.now(timezone.utc).isoformat()

stale = []
for c in sorted(clusters, key=lambda x: int(x.get("cluster_id", 0))):
    cid = int(c.get("cluster_id", 0))
    label = c.get("cluster_label", "?")
    status = str(c.get("status", "active"))
    weeks = sorted(weeks_per_cluster.get(cid, set()))

    has_current = current_week in weeks
    marker = "OK" if has_current else "STALE"

    print(f"  {cid:3}: {label:<45} status={status:<12} weeks={weeks}  → {marker}")

    if not has_current and status == "active":
        stale.append((cid, label))

if not stale:
    print("\nNo stale clusters to mark.")
else:
    print(f"\n{len(stale)} cluster(s) to mark as declining:")
    for cid, label in stale:
        print(f"  → {cid}: {label}")

    confirm = input("\nMark these as declining? [y/N] ")
    if confirm.strip().lower() == "y":
        for cid, label in stale:
            clusters_table.update_item(
                Key={"cluster_id": Decimal(str(cid))},
                UpdateExpression="SET #status = :s, #updated = :u, #declined = :d",
                ExpressionAttributeNames={
                    "#status": "status",
                    "#updated": "updated_at",
                    "#declined": "declined_at",
                },
                ExpressionAttributeValues={
                    ":s": "declining",
                    ":u": now,
                    ":d": now,
                },
            )
            print(f"  MARKED declining: {cid} ({label})")
        print("Done.")
    else:
        print("Aborted.")
