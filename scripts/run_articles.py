"""
run_articles.py - Generate Gemini news articles for all cluster × week combinations

Reads narrative-clusters (including classified_claims written by run_claim_analysis),
calls Gemini to generate headline/overview/body for each cluster × week, and writes
results to the articles table.

Idempotent by default — skips (cluster, week) pairs that already have an article.
Use --force to regenerate and overwrite existing articles.

Usage:
    python -m scripts.run_articles                      # all clusters, all weeks
    python -m scripts.run_articles --week week3         # one week only
    python -m scripts.run_articles --cluster-id 7       # one cluster only
    python -m scripts.run_articles --force              # overwrite existing articles
    python -m scripts.run_articles --dry-run            # preview job count, no Gemini
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.services.article_service import ArticleService


def main():
    parser = argparse.ArgumentParser(
        description="Generate articles for all cluster × week combinations"
    )
    parser.add_argument(
        "--week", default=None, help="Limit to a specific week (e.g. week3 or 3)"
    )
    parser.add_argument(
        "--cluster-id", type=int, default=None, help="Limit to a single cluster"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing articles"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview jobs without calling Gemini"
    )
    args = parser.parse_args()

    service = ArticleService()
    result = service.run_article_generation(
        week=args.week,
        force=args.force,
        cluster_id=args.cluster_id,
        dry_run=args.dry_run,
    )

    if result["dry_run"]:
        print(f"\n*** DRY RUN ***")
        print(f"Would generate: {result['articles_would_generate']}")
        print(f"Would skip:     {result['articles_would_skip']}")
        print(f"Weeks:          {result['weeks_processed']}")
        if result.get("per_cluster"):
            print("\n--- Per Cluster ---")
            for cid, info in sorted(
                result["per_cluster"].items(), key=lambda x: int(x[0])
            ):
                label = info.get("cluster_label", "?")
                weeks = info.get("weeks", {})
                to_gen = [
                    w for w, d in weeks.items() if d["status"] == "would_generate"
                ]
                to_skip = [w for w, d in weeks.items() if d["status"] == "would_skip"]
                print(f"  [{cid}] {label}")
                if to_gen:
                    print(f"    generate: {to_gen}")
                if to_skip:
                    print(f"    skip:     {to_skip}")
    else:
        print(f"\nGenerated: {result['articles_generated']}")
        print(f"Skipped:   {result['articles_skipped']}")
        print(f"Failed:    {result['articles_failed']}")
        print(f"Weeks:     {result['weeks_processed']}")
        if result["articles_failed"] > 0:
            print("\n--- Failures ---")
            for key, info in result["per_cluster"].items():
                if info.get("status") == "failed":
                    print(f"  {key}: {info.get('error', '?')}")


if __name__ == "__main__":
    main()
