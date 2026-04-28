"""
remap_cluster_ids.py - Remap cluster IDs to sequential integers

Reassigns cluster_id values across all three tables (narrative-clusters,
youtube-videos, articles) so active/declining clusters are numbered 0, 1, 2...
sorted by created_at ascending.

Inactive clusters are left untouched — their IDs stay retired so the
gap-filling logic in cluster_labeling_service can reuse them naturally.

Usage:
    python -m scripts.remap_cluster_ids               # dry run (default)
    python -m scripts.remap_cluster_ids --apply       # write changes
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import boto3
from decimal import Decimal

from app.core.config import settings

REGION = os.getenv("AWS_REGION", "us-east-2")
CLUSTERS_TABLE = "narrative-clusters"
VIDEOS_TABLE = settings.dynamodb_table
ARTICLES_TABLE = "articles"


def dec(val):
    return Decimal(str(val))


# ── Scan helpers ─────────────────────────────────────────────────────────────


def scan_all(table, **kwargs):
    items = []
    scan_kwargs = dict(kwargs)
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


# ── Phase 1: Build remapping ─────────────────────────────────────────────────


def build_id_map(dynamodb):
    clusters_table = dynamodb.Table(CLUSTERS_TABLE)
    items = scan_all(
        clusters_table,
        ProjectionExpression="cluster_id, cluster_label, #st, created_at",
        ExpressionAttributeNames={"#st": "status"},
    )

    # Only remap active and declining — leave inactive alone
    remappable = [
        i for i in items if i.get("status", "active") in ("active", "declining", "new")
    ]

    # Sort by created_at ascending so oldest story gets lowest ID
    remappable.sort(key=lambda i: i.get("created_at", ""))

    id_map = {}  # {old_id (int): new_id (int)}
    for new_id, item in enumerate(remappable):
        old_id = int(item["cluster_id"])
        if old_id != new_id:
            id_map[old_id] = new_id

    return id_map, remappable


# ── Phase 2: Remap narrative-clusters ────────────────────────────────────────


def remap_clusters_table(dynamodb, id_map, dry_run):
    if not id_map:
        print("  narrative-clusters: nothing to remap")
        return

    clusters_table = dynamodb.Table(CLUSTERS_TABLE)

    # Read full items for clusters that need remapping
    items_to_remap = []
    for old_id in id_map:
        resp = clusters_table.get_item(Key={"cluster_id": dec(old_id)})
        item = resp.get("Item")
        if item:
            items_to_remap.append(item)

    print(f"  narrative-clusters: {len(items_to_remap)} items to remap")

    if dry_run:
        for item in items_to_remap:
            old_id = int(item["cluster_id"])
            new_id = id_map[old_id]
            print(f"    {old_id} → {new_id}  [{item.get('cluster_label', '?')}]")
        return

    for item in items_to_remap:
        old_id = int(item["cluster_id"])
        new_id = id_map[old_id]

        new_item = dict(item)
        new_item["cluster_id"] = dec(new_id)

        # Delete old, put new (PK change requires delete + put)
        clusters_table.delete_item(Key={"cluster_id": dec(old_id)})
        clusters_table.put_item(Item=new_item)
        print(
            f"    remapped cluster {old_id} → {new_id}  [{item.get('cluster_label', '?')}]"
        )


# ── Phase 3: Remap youtube-videos ────────────────────────────────────────────


def remap_videos_table(dynamodb, id_map, dry_run):
    if not id_map:
        print("  youtube-videos: nothing to remap")
        return

    videos_table = dynamodb.Table(VIDEOS_TABLE)
    old_ids_dec = {dec(old_id) for old_id in id_map}

    # Scan only videos that have a cluster_id we need to remap
    items = scan_all(
        videos_table,
        ProjectionExpression="PartitionKey, SortKey, cluster_id",
        FilterExpression=boto3.dynamodb.conditions.Attr("cluster_id").exists(),
    )

    affected = [i for i in items if i.get("cluster_id") in old_ids_dec]
    print(f"  youtube-videos: {len(affected)} items to remap")

    if dry_run:
        return

    for item in affected:
        old_id = int(item["cluster_id"])
        new_id = id_map[old_id]
        videos_table.update_item(
            Key={"PartitionKey": item["PartitionKey"], "SortKey": item["SortKey"]},
            UpdateExpression="SET cluster_id = :new_id",
            ExpressionAttributeValues={":new_id": dec(new_id)},
        )

    print(f"    updated {len(affected)} videos")


# ── Phase 4: Remap articles ───────────────────────────────────────────────────


def remap_articles_table(dynamodb, id_map, dry_run):
    if not id_map:
        print("  articles: nothing to remap")
        return

    articles_table = dynamodb.Table(ARTICLES_TABLE)
    old_ids_dec = {dec(old_id) for old_id in id_map}

    items = scan_all(
        articles_table,
        ProjectionExpression="article_id, cluster_id",
    )

    affected = [i for i in items if i.get("cluster_id") in old_ids_dec]
    print(f"  articles: {len(affected)} items to remap")

    if dry_run:
        return

    for item in affected:
        old_id = int(item["cluster_id"])
        new_id = id_map[old_id]
        articles_table.update_item(
            Key={"article_id": item["article_id"]},
            UpdateExpression="SET cluster_id = :new_id",
            ExpressionAttributeValues={":new_id": dec(new_id)},
        )

    print(f"    updated {len(affected)} articles")


# ── Phase 5: Purge inactive clusters and their articles ───────────────────────


def purge_inactive_clusters(dynamodb, dry_run):
    clusters_table = dynamodb.Table(CLUSTERS_TABLE)
    articles_table = dynamodb.Table(ARTICLES_TABLE)

    all_clusters = scan_all(
        clusters_table,
        ProjectionExpression="cluster_id, cluster_label, #st",
        ExpressionAttributeNames={"#st": "status"},
    )

    inactive = [i for i in all_clusters if i.get("status") == "inactive"]
    print(f"  narrative-clusters: {len(inactive)} inactive clusters to delete")

    if not inactive:
        print("  articles: nothing to purge")
        return

    inactive_ids_dec = {i["cluster_id"] for i in inactive}

    all_articles = scan_all(
        articles_table,
        ProjectionExpression="article_id, cluster_id",
    )
    stale_articles = [
        a for a in all_articles if a.get("cluster_id") in inactive_ids_dec
    ]
    print(f"  articles: {len(stale_articles)} stale articles to delete")

    if dry_run:
        for i in sorted(inactive, key=lambda x: int(x["cluster_id"])):
            print(
                f"    delete cluster {int(i['cluster_id'])}  [{i.get('cluster_label', '?')}]"
            )
        return

    for i in inactive:
        clusters_table.delete_item(Key={"cluster_id": i["cluster_id"]})
        print(
            f"    deleted cluster {int(i['cluster_id'])}  [{i.get('cluster_label', '?')}]"
        )

    for a in stale_articles:
        articles_table.delete_item(Key={"article_id": a["article_id"]})

    print(f"    deleted {len(stale_articles)} stale articles")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Remap cluster IDs to sequential integers"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write changes to DynamoDB (default: dry run)",
    )
    args = parser.parse_args()
    dry_run = not args.apply

    if dry_run:
        print("DRY RUN — pass --apply to write changes\n")
    else:
        print("APPLYING changes to DynamoDB\n")

    dynamodb = boto3.resource("dynamodb", region_name=REGION)

    # Phase 1
    print("Building ID remapping...")
    id_map, remappable = build_id_map(dynamodb)

    if not id_map:
        print("All cluster IDs are already sequential. Nothing to do.")
    else:
        print(f"  {len(remappable)} active/declining clusters total")
        print(f"  {len(id_map)} need remapping\n")

        # Phase 2
        print("Phase 2: narrative-clusters")
        remap_clusters_table(dynamodb, id_map, dry_run)

        # Phase 3
        print("\nPhase 3: youtube-videos")
        remap_videos_table(dynamodb, id_map, dry_run)

        # Phase 4
        print("\nPhase 4: articles")
        remap_articles_table(dynamodb, id_map, dry_run)

    # Phase 5 — always runs regardless of whether remapping was needed
    print("\nPhase 5: purge inactive clusters")
    purge_inactive_clusters(dynamodb, dry_run)

    print("\nDone.")
    if dry_run:
        print("Run with --apply to commit these changes.")


if __name__ == "__main__":
    main()
