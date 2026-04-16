"""
backfill_article_titles.py - Overwrite article titles with cluster-weeks narrative_headline

For every article in the articles table, fetches the narrative_headline from
cluster-weeks for that (cluster_id, week) and writes it as the article title.

Usage:
    python scripts/backfill_article_titles.py
    python scripts/backfill_article_titles.py --dry-run
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
from boto3.dynamodb.conditions import Key

REGION = "us-east-2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ddb = boto3.resource("dynamodb", region_name=REGION)
    articles_table = ddb.Table("articles")
    cluster_weeks_table = ddb.Table("cluster-weeks")

    # Scan all articles
    articles = []
    kwargs: dict = {}
    while True:
        r = articles_table.scan(**kwargs)
        articles.extend(r["Items"])
        if "LastEvaluatedKey" not in r:
            break
        kwargs["ExclusiveStartKey"] = r["LastEvaluatedKey"]

    print(f"Found {len(articles)} articles")

    # Build headline map: {(cluster_id, week): headline}
    headline_map: dict[tuple[int, str], str] = {}
    unique_cluster_ids = {int(a["cluster_id"]) for a in articles}
    for cid in unique_cluster_ids:
        resp = cluster_weeks_table.query(
            KeyConditionExpression=Key("cluster_id").eq(Decimal(str(cid))),
            ProjectionExpression="#wk, narrative_headline",
            ExpressionAttributeNames={"#wk": "week"},
        )
        for row in resp.get("Items", []):
            headline = row.get("narrative_headline", "").strip()
            if headline:
                headline_map[(cid, row["week"])] = headline

    print(f"Loaded headlines for {len(unique_cluster_ids)} clusters\n")

    updated = skipped = missing = 0
    for article in articles:
        cid = int(article["cluster_id"])
        week = article.get("week", f"week{article.get('week_number', '')}")
        headline = headline_map.get((cid, week))

        if not headline:
            print(f"  [MISSING] cluster={cid} week={week} article={article['article_id']}")
            missing += 1
            continue

        current_title = article.get("title", "")
        if current_title == headline:
            skipped += 1
            continue

        print(f"  cluster={cid} {week}: {headline[:70]}")
        if not args.dry_run:
            articles_table.update_item(
                Key={"article_id": article["article_id"]},
                UpdateExpression="SET title = :t",
                ExpressionAttributeValues={":t": headline},
            )
        updated += 1

    print(f"\nUpdated: {updated}  Skipped (already correct): {skipped}  Missing headline: {missing}")
    if args.dry_run:
        print("Dry run — no writes made.")


if __name__ == "__main__":
    main()
