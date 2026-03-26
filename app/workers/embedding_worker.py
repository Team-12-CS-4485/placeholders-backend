"""
embedding_worker.py - Full Pipeline Worker

Runs the complete analysis pipeline end-to-end in one command:
1. S3 transcript analysis → Gemini intelligence → DynamoDB + Qdrant
2. Sync missing videos (DynamoDB videos not yet in Qdrant)
3. Narrative clustering (UMAP + HDBSCAN → DynamoDB)
4. Claim analysis (consensus/debated/unique → DynamoDB)

Usage:
    python -m app.workers.embedding_worker
    python -m app.workers.embedding_worker --skip-clustering    # ingest only
    python -m app.workers.embedding_worker --clustering-only    # skip ingest, just recluster
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

    from app.services.sync_missing import main as sync_main
    sync_main()


def run_clustering():
    """Step 3: Run narrative clustering → DynamoDB."""
    print("\n" + "=" * 60)
    print("Step 3: Running narrative clustering")
    print("=" * 60)

    from app.services.clustering_service import ClusteringService

    service = ClusteringService()
    summary = service.run_clustering(
        min_cluster_size=7,
        min_samples=2,
    )

    print(f"\n  Videos clustered: {summary.get('total_videos', 0)}")
    print(f"  Clusters found:   {summary.get('cluster_count', 0)}")
    print(f"  Noise videos:     {summary.get('noise_videos', 0)}")
    print(f"  DynamoDB updated: {summary.get('videos_updated', 0)}")
    print(f"  Clusters written: {summary.get('clusters_written', 0)}")

    if summary.get("clusters"):
        print("\n  Clusters:")
        for cid, info in summary["clusters"].items():
            print(
                f"    {cid}: {info['label']} "
                f"({info['video_count']} videos, {info['channel_count']} channels)"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline worker")
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Only run ingestion + sync, skip clustering and claim analysis",
    )
    parser.add_argument(
        "--clustering-only",
        action="store_true",
        help="Skip ingestion, only run clustering + claim analysis",
    )
    args = parser.parse_args()

    if args.clustering_only:
        # Skip ingestion, just recluster
        print("=" * 60)
        print("Pipeline Worker — Clustering Only")
        print("=" * 60)
        run_clustering()
        run_claim_analysis()
        print("\n=== Done ===")
        sys.exit(0)

    # Full pipeline
    print("=" * 60)
    print("Pipeline Worker — Full Run")
    print("=" * 60)

    # Step 1: Ingest
    print("\nStep 1: Running transcript analysis pipeline")
    print("-" * 40)
    run_transcript_analysis_job()

    # Step 2: Sync missing
    run_sync()

    if args.skip_clustering:
        print("\n=== Done (clustering skipped) ===")
        sys.exit(0)

    # Step 3: Clustering
    run_clustering()

    # Step 4: Claim analysis
    run_claim_analysis()

    print("\n" + "=" * 60)
    print("=== Pipeline Complete ===")
    print("=" * 60)