"""
delete_clusters.py - Delete one or more clusters and all their associated data

Removes:
  1. narrative-clusters record
  2. All cluster-weeks rows for the cluster
  3. All articles for the cluster

Does NOT touch youtube-videos (videos retain their cluster_id label but the
cluster itself is gone from all list/detail endpoints).

Usage:
    python scripts/delete_clusters.py --cluster-ids 1 4
    python scripts/delete_clusters.py --cluster-ids 1 4 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

import boto3
from boto3.dynamodb.conditions import Attr, Key

from app.core.config import settings

REGION = settings.aws_region


def delete_cluster(dynamodb, cluster_id: int, dry_run: bool) -> None:
    clusters_table = dynamodb.Table("narrative-clusters")
    cluster_weeks_table = dynamodb.Table("cluster-weeks")
    articles_table = dynamodb.Table("articles")

    cid_dec = Decimal(str(cluster_id))

    # ── 1. narrative-clusters ────────────────────────────────────────────────
    resp = clusters_table.get_item(Key={"cluster_id": cid_dec})
    cluster_item = resp.get("Item")
    if cluster_item:
        label = cluster_item.get("cluster_label", f"Cluster {cluster_id}")
        print(f"  [narrative-clusters] found: {label}")
        if not dry_run:
            clusters_table.delete_item(Key={"cluster_id": cid_dec})
            print(f"  [narrative-clusters] deleted cluster_id={cluster_id}")
    else:
        print(f"  [narrative-clusters] NOT FOUND — skipping")

    # ── 2. cluster-weeks ─────────────────────────────────────────────────────
    cw_items = []
    scan_kwargs: dict = {
        "KeyConditionExpression": Key("cluster_id").eq(cid_dec)
    }
    while True:
        r = cluster_weeks_table.query(**scan_kwargs)
        cw_items.extend(r.get("Items", []))
        if "LastEvaluatedKey" not in r:
            break
        scan_kwargs["ExclusiveStartKey"] = r["LastEvaluatedKey"]

    print(f"  [cluster-weeks] found {len(cw_items)} rows")
    if not dry_run:
        for item in cw_items:
            cluster_weeks_table.delete_item(
                Key={"cluster_id": item["cluster_id"], "week": item["week"]}
            )
        if cw_items:
            print(f"  [cluster-weeks] deleted {len(cw_items)} rows")

    # ── 3. articles ──────────────────────────────────────────────────────────
    art_items = []
    art_scan: dict = {
        "FilterExpression": Attr("cluster_id").eq(cid_dec),
        "ProjectionExpression": "article_id",
    }
    while True:
        r = articles_table.scan(**art_scan)
        art_items.extend(r.get("Items", []))
        if "LastEvaluatedKey" not in r:
            break
        art_scan["ExclusiveStartKey"] = r["LastEvaluatedKey"]

    print(f"  [articles] found {len(art_items)} articles")
    if not dry_run:
        for item in art_items:
            articles_table.delete_item(Key={"article_id": item["article_id"]})
        if art_items:
            print(f"  [articles] deleted {len(art_items)} articles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete clusters and all associated data")
    parser.add_argument("--cluster-ids", nargs="+", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] Deleting clusters: {args.cluster_ids}\n")

    for cid in args.cluster_ids:
        print(f"=== Cluster {cid} ===")
        delete_cluster(dynamodb, cid, args.dry_run)
        print()

    if args.dry_run:
        print("Dry run complete — no writes made.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
