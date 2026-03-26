"""
run_clustering.py - Run narrative clustering on existing Qdrant data

Pulls all vectors, clusters with HDBSCAN, labels from existing topics,
patches cluster_id + cluster_label back to every chunk. Zero Gemini calls.

Usage:
    python -m scripts.run_clustering
    python -m scripts.run_clustering --min-cluster-size 4 --min-samples 3
"""

import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from app.services.clustering_service import ClusteringService


def main():
    parser = argparse.ArgumentParser(description="Run narrative clustering")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="Minimum videos to form a cluster (default: 3)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="HDBSCAN density parameter (default: 2)",
    )
    parser.add_argument(
        "--umap-components",
        type=int,
        default=15,
        help="UMAP target dimensions — 768 dims compressed to this (default: 15)",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP neighborhood size — higher = more global structure (default: 15)",
    )
    args = parser.parse_args()

    print("=== Narrative Clustering ===\n")

    service = ClusteringService()
    summary = service.run_clustering(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
    )

    print(f"\nTotal videos:     {summary['total_videos']}")
    print(f"Clusters found:   {summary['cluster_count']}")
    print(f"Noise (outliers): {summary['noise_videos']}")

    if summary["clusters"]:
        print("\n--- Clusters ---\n")
        for cid, info in summary["clusters"].items():
            print(f"  Cluster {cid}: {info['label']!r}")
            print(
                f"    Videos: {info['video_count']}  |  Channels: {info['channel_count']}"
            )
            print(f"    Channels: {', '.join(info['channels'])}")
            print(f"    Top topics: {', '.join(info['top_topics'][:3])}")
            print(f"    Category: {info['dominant_category']}")
            print(f"    Sentiment: {info['dominant_sentiment']}")
            print(f"    Breaking: {info['breaking_count']}")
            print(f"    Total views: {info['total_views']:,}")
            print()

    # Write summary to file
    output_path = "cluster_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
