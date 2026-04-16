"""
purge_cluster_data.py - Wipe all cluster data from DynamoDB

Clears the four tables that hold cluster state so a fresh all-time
clustering run can write clean data.

What gets deleted / stripped:
  narrative-clusters  — all rows deleted
  cluster-weeks       — all rows deleted
  articles            — all rows deleted
  youtube-videos      — cluster_id, cluster_label, cluster_confidence
                        stripped from every video that has them

Usage:
    python -m scripts.purge_cluster_data            # dry-run by default
    python -m scripts.purge_cluster_data --confirm  # actually writes
"""

import argparse
import logging
import boto3
from boto3.dynamodb.conditions import Attr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

REGION = "us-east-2"


def _scan_all(table, **kwargs):
    items = []
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def purge_narrative_clusters(table, dry_run):
    items = _scan_all(table, ProjectionExpression="cluster_id")
    logger.info(f"narrative-clusters: {len(items)} rows to delete")
    if not dry_run:
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={"cluster_id": item["cluster_id"]})
        logger.info("narrative-clusters: done")
    return len(items)


def purge_cluster_weeks(table, dry_run):
    items = _scan_all(
        table,
        ProjectionExpression="cluster_id, #wk",
        ExpressionAttributeNames={"#wk": "week"},
    )
    logger.info(f"cluster-weeks: {len(items)} rows to delete")
    if not dry_run:
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(
                    Key={"cluster_id": item["cluster_id"], "week": item["week"]}
                )
        logger.info("cluster-weeks: done")
    return len(items)


def purge_articles(table, dry_run):
    items = _scan_all(table, ProjectionExpression="article_id")
    logger.info(f"articles: {len(items)} rows to delete")
    if not dry_run:
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={"article_id": item["article_id"]})
        logger.info("articles: done")
    return len(items)


def strip_video_cluster_fields(table, dry_run):
    items = _scan_all(
        table,
        FilterExpression=Attr("cluster_id").exists(),
        ProjectionExpression="PartitionKey, SortKey",
    )
    logger.info(f"youtube-videos: {len(items)} videos with cluster fields to strip")
    if not dry_run:
        for item in items:
            table.update_item(
                Key={"PartitionKey": item["PartitionKey"], "SortKey": item["SortKey"]},
                UpdateExpression="REMOVE cluster_id, cluster_label, cluster_confidence",
            )
        logger.info("youtube-videos: done")
    return len(items)


def main():
    parser = argparse.ArgumentParser(description="Purge all cluster data from DynamoDB")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually execute the purge. Without this flag, runs as dry-run.",
    )
    args = parser.parse_args()
    dry_run = not args.confirm

    if dry_run:
        print("\n*** DRY RUN — pass --confirm to execute ***\n")
    else:
        print("\n*** LIVE RUN — deleting data ***\n")

    ddb = boto3.resource("dynamodb", region_name=REGION)

    counts = {}
    counts["narrative-clusters"] = purge_narrative_clusters(
        ddb.Table("narrative-clusters"), dry_run
    )
    counts["cluster-weeks"] = purge_cluster_weeks(ddb.Table("cluster-weeks"), dry_run)
    counts["articles"] = purge_articles(ddb.Table("articles"), dry_run)
    counts["youtube-videos (stripped)"] = strip_video_cluster_fields(
        ddb.Table("youtube-videos"), dry_run
    )

    print("\n--- Summary ---")
    for k, v in counts.items():
        action = "would affect" if dry_run else "affected"
        print(f"  {k}: {v} rows {action}")

    if dry_run:
        print("\nRe-run with --confirm to execute.")
    else:
        print("\nPurge complete. Run clustering next:")
        print("  python -m scripts.run_clustering --dry-run --verbose")
        print("  python -m scripts.run_clustering  (live write)")


if __name__ == "__main__":
    main()
