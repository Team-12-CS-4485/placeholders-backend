"""
pipeline_worker.py - Full Pipeline Worker

Runs the complete analysis pipeline end-to-end in one command:
1. S3 transcript analysis → Gemini intelligence (incl. public sentiment) → DynamoDB + Qdrant
2. Sync missing videos (DynamoDB videos not yet in Qdrant)
3. Narrative clustering (UMAP + HDBSCAN → stable matching → DynamoDB)
4. Claim analysis (consensus/debated/unique → DynamoDB)
5. Article generation (Gemini news articles → DynamoDB articles table)

Usage:
    python -m app.workers.embedding_worker
    python -m app.workers.embedding_worker --skip-clustering    # ingest only
    python -m app.workers.embedding_worker --clustering-only    # skip ingest, just recluster
    python -m app.workers.embedding_worker --articles-only        # generate articles only
    python -m app.workers.embedding_worker --week week2           # articles for a specific week
"""

import argparse
import sys

from app.core.config import settings
from app.services.pipeline_service import PipelineService


def run_transcript_analysis_job(prefix=None, limit=None, output_file=None):
    """Step 1: Run the S3 → Gemini → DynamoDB + Qdrant pipeline."""
    pipeline = PipelineService()
    result = pipeline.run_s3_transcript_analysis(prefix=prefix, limit=limit)
    target_file = output_file or settings.analysis_output_file

    with open(target_file, "w", encoding="utf-8") as file:
        file.write(f"Prefix: {result['prefix']}\n")
        file.write(f"Objects Processed: {result['objects_processed']}\n")
        file.write(f"Videos Found: {result['videos_found']}\n")
        file.write(f"Videos Indexed: {result['videos_indexed']}\n")
        file.write(f"Total Chunks Stored: {result['total_chunks_stored']}\n\n")

        for obj in result["results"]:
            file.write(f"Object: {obj['key']}\n")
            file.write(f"Status: {obj['status']}\n")
            if obj.get("error"):
                file.write(f"Error: {obj['error']}\n")

            for video_result in obj["transcript_results"]:
                file.write(f"  Video ID: {video_result['video_id']}\n")
                file.write(f"  Chunks: {video_result['chunk_count']}\n")
                file.write(f"  Points Stored: {video_result['chunks_stored']}\n")
                if video_result.get("error"):
                    file.write(f"  Error: {video_result['error']}\n")
                elif video_result.get("intelligence"):
                    intel = video_result["intelligence"]
                    file.write(f"  Category: {intel.get('category')}\n")
                    file.write(f"  Sentiment: {intel.get('sentiment')}\n")
                    file.write(f"  Public Sentiment: {intel.get('public_sentiment')}\n")
                    file.write(f"  Topics: {', '.join(intel.get('topics', []))}\n")
                    file.write(f"  Is Breaking: {intel.get('is_breaking')}\n")
                    file.write(f"  Key Claims:\n")
                    for claim in intel.get("key_claims", []):
                        file.write(f"    - {claim}\n")
                file.write("\n")

            file.write("-" * 80 + "\n\n")

    return result


def run_sync():
    """Step 2: Sync any videos in DynamoDB but not yet in Qdrant."""
    print("\n" + "=" * 60)
    print("Step 2: Syncing missing videos")
    print("=" * 60)

    from scripts.sync_missing import main as sync_main

    sync_main()


def run_clustering():
    """Step 3: Run narrative clustering → stable matching → DynamoDB."""
    print("\n" + "=" * 60)
    print("Step 3: Running narrative clustering")
    print("=" * 60)

    from app.services.clustering_service import ClusteringService

    service = ClusteringService()
    summary = service.run_clustering(
        min_cluster_size=7,
        min_samples=2,
    )

    print(f"\n  Videos clustered:   {summary.get('total_videos', 0)}")
    print(f"  Clusters found:     {summary.get('cluster_count', 0)}")
    print(f"  Matched (existing): {summary.get('matched_clusters', 0)}")
    print(f"  New clusters:       {summary.get('new_clusters', 0)}")
    print(f"  Declined:           {summary.get('declined_clusters', 0)}")
    print(f"  Noise videos:       {summary.get('noise_videos', 0)}")

    if summary.get("clusters"):
        print("\n  Clusters:")
        for cid, info in summary["clusters"].items():
            new_tag = " [NEW]" if info.get("is_new") else ""
            print(
                f"    {cid}: {info['label']} "
                f"({info['video_count']} videos, {info['channel_count']} channels){new_tag}"
            )

    return summary


def run_claim_analysis():
    """Step 4: Classify claims → DynamoDB."""
    print("\n" + "=" * 60)
    print("Step 4: Running claim analysis")
    print("=" * 60)

    from app.services.claim_analysis_service import ClaimAnalysisService

    service = ClaimAnalysisService()
    summary = service.run_claim_analysis(max_per_type=3)

    print(f"\n  Clusters processed: {summary.get('clusters_processed', 0)}")
    print(f"  Claims written:     {summary.get('total_written', 0)}")

    if summary.get("per_cluster"):
        print("\n  Per cluster:")
        for cid, info in sorted(summary["per_cluster"].items()):
            print(
                f"    Cluster {cid}: "
                f"{info.get('selected_c', 0)}c / "
                f"{info.get('selected_d', 0)}d / "
                f"{info.get('selected_u', 0)}u"
            )

    return summary


def run_article_generation(
    week: str = None, force: bool = False, cluster_id: int = None
):
    """Step 5: Generate Gemini news articles → DynamoDB articles table."""
    print("\n" + "=" * 60)
    print("Step 5: Generating news articles")
    if week:
        print(f"         (week: {week})")
    print("=" * 60)

    from app.services.article_service import ArticleService

    service = ArticleService()
    summary = service.run_article_generation(
        week=week,
        force=force,
        cluster_id=cluster_id,
    )

    print(f"\n  Articles generated: {summary.get('articles_generated', 0)}")
    print(f"  Articles skipped:   {summary.get('articles_skipped', 0)}")
    print(f"  Articles failed:    {summary.get('articles_failed', 0)}")
    print(f"  Weeks processed:    {', '.join(summary.get('weeks_processed', []))}")

    per_cluster = summary.get("per_cluster", {})
    if per_cluster:
        print("\n  Per cluster:")
        for key, info in sorted(per_cluster.items()):
            status = info.get("status", "?")
            wk = info.get("week", "?")
            if status == "generated":
                aid = info.get("article_id", "")
                headline = info.get("headline", "")[:60]
                print(f"    {key} [{wk}]: ✓ {aid} — {headline}...")
            elif status == "skipped":
                print(f"    {key} [{wk}]: — skipped")
            else:
                err = info.get("error", "")[:80]
                print(f"    {key} [{wk}]: ✗ {err}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline worker")
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Only run ingestion + sync, skip clustering, claims, and articles",
    )
    parser.add_argument(
        "--clustering-only",
        action="store_true",
        help="Skip ingestion, only run clustering + claim analysis + articles",
    )
    parser.add_argument(
        "--articles-only",
        action="store_true",
        help="Skip everything, only run article generation",
    )
    parser.add_argument(
        "--week",
        type=str,
        default=None,
        help="Limit article generation to a specific week (e.g. 'week1' or '1')",
    )
    parser.add_argument(
        "--force-articles",
        action="store_true",
        help="Overwrite existing articles when generating",
    )
    parser.add_argument(
        "--cluster-id",
        type=int,
        default=None,
        help="Limit article generation to a single cluster ID",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="S3 prefix to ingest (e.g. youtube-data/week5/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max S3 objects to process",
    )
    args = parser.parse_args()

    # ── Articles-only mode ────────────────────────────────────────────────────
    if args.articles_only:
        print("=" * 60)
        print("Pipeline Worker — Articles Only")
        print("=" * 60)
        run_article_generation(
            week=args.week,
            force=args.force_articles,
            cluster_id=args.cluster_id,
        )
        print("\n=== Done ===")
        sys.exit(0)

    # ── Clustering-only mode ──────────────────────────────────────────────────
    if args.clustering_only:
        print("=" * 60)
        print("Pipeline Worker — Clustering Only")
        print("=" * 60)
        run_clustering()
        run_claim_analysis()
        run_article_generation(
            week=args.week,
            force=args.force_articles,
            cluster_id=args.cluster_id,
        )
        print("\n=== Done ===")
        sys.exit(0)

    # ── Full pipeline ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Pipeline Worker — Full Run")
    print("=" * 60)

    # Step 1: Ingest
    print("\nStep 1: Running transcript analysis pipeline")
    print("-" * 40)
    run_transcript_analysis_job(prefix=args.prefix, limit=args.limit)

    # Step 2: Sync missing
    run_sync()

    if args.skip_clustering:
        print("\n=== Done (clustering + articles skipped) ===")
        sys.exit(0)

    # Step 3: Clustering
    run_clustering()

    # Step 4: Claim analysis
    run_claim_analysis()

    # Step 5: Article generation
    run_article_generation(
        week=args.week,
        force=args.force_articles,
        cluster_id=args.cluster_id,
    )

    print("\n" + "=" * 60)
    print("=== Pipeline Complete ===")
    print("=" * 60)
