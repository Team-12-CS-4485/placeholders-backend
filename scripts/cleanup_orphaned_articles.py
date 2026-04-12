"""
cleanup_orphaned_articles.py

Finds articles in the articles table whose (cluster_id, week_number) no longer
has any video data in narrative-clusters.week_data. These are orphaned — they
were written under an older clustering run and no longer reflect current cluster
assignments.

Usage:
    python -m scripts.cleanup_orphaned_articles           # dry run (default)
    python -m scripts.cleanup_orphaned_articles --apply   # delete orphans
"""

from __future__ import annotations

import argparse
import os
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Attr
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-2")


def main(apply: bool) -> None:
    dynamo = boto3.resource("dynamodb", region_name=REGION)
    clusters_table = dynamo.Table("narrative-clusters")
    articles_table = dynamo.Table("articles")

    # ── Build valid (cluster_id, week_number) pairs from current week_data ───
    print("Reading narrative-clusters week_data…")
    valid_pairs: set[tuple[int, int]] = set()

    scan_kwargs: dict = {}
    while True:
        resp = clusters_table.scan(**scan_kwargs)
        for item in resp["Items"]:
            cid = int(item.get("cluster_id", -1))
            for wd in item.get("week_data", []):
                if int(wd.get("video_count", 0)) > 0:
                    w = wd.get("week", "")
                    if w.startswith("week") and w[4:].isdigit():
                        valid_pairs.add((cid, int(w[4:])))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    print(f"Valid (cluster, week) pairs in current clustering: {len(valid_pairs)}")

    # ── Scan articles table ──────────────────────────────────────────────────
    print("Scanning articles table…")
    orphans: list[dict] = []

    scan_kwargs = {
        "ProjectionExpression": "article_id, cluster_id, week_number, #wk, title",
        "ExpressionAttributeNames": {"#wk": "week"},
    }
    while True:
        resp = articles_table.scan(**scan_kwargs)
        for item in resp["Items"]:
            cid = int(item.get("cluster_id", -1))
            wn = int(item.get("week_number", -1))
            if (cid, wn) not in valid_pairs:
                orphans.append(item)
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    if not orphans:
        print("No orphaned articles found.")
        return

    print(f"\nFound {len(orphans)} orphaned article(s):\n")
    for a in sorted(
        orphans,
        key=lambda x: (int(x.get("cluster_id", 0)), int(x.get("week_number", 0))),
    ):
        print(
            f"  article_id={a['article_id']}  "
            f"cluster={int(a.get('cluster_id', 0))}  "
            f"week_number={int(a.get('week_number', 0))}  "
            f"title={str(a.get('title', ''))[:60]}"
        )

    if not apply:
        print("\n[DRY RUN] No deletions performed. Re-run with --apply to delete.")
        return

    print(f"\nDeleting {len(orphans)} orphaned article(s)…")
    deleted = 0
    for a in orphans:
        try:
            articles_table.delete_item(Key={"article_id": a["article_id"]})
            print(
                f"  DELETED {a['article_id']}  cluster={int(a.get('cluster_id', 0))}  week={int(a.get('week_number', 0))}"
            )
            deleted += 1
        except Exception as exc:
            print(f"  ERROR {a['article_id']}: {exc}")

    print(f"\nDone. Deleted {deleted}/{len(orphans)} orphaned articles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete orphaned articles (default is dry run)",
    )
    args = parser.parse_args()
    main(apply=args.apply)
