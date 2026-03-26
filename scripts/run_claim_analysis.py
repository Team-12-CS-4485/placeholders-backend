"""
run_claim_analysis.py - Classify claims across narrative clusters

Embeds all claims locally, groups by semantic similarity, classifies
into consensus/controversial/unique, patches results back to Qdrant.
Zero Gemini calls.

Usage:
    python -m scripts.run_claim_analysis
    python -m scripts.run_claim_analysis --max-per-type 2
"""

import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from app.services.claim_analysis_service import ClaimAnalysisService


def main():
    parser = argparse.ArgumentParser(description="Run claim classification")
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=3,
        help="Max claims to keep per type per cluster (default: 3)",
    )
    args = parser.parse_args()

    print("=== Claim Analysis ===\n")

    service = ClaimAnalysisService()
    summary = service.run_claim_analysis(max_per_type=args.max_per_type)

    print(f"\nClusters processed: {summary['clusters_processed']}")
    print(f"Total chunks patched: {summary['total_patched']}")

    if summary.get("per_cluster"):
        print("\n--- Per Cluster ---\n")
        for cid, info in sorted(summary["per_cluster"].items()):
            print(f"  Cluster {cid}:")
            print(f"    Raw claims: {info['total_claims']}")
            print(f"    Similarity groups: {info['groups_formed']}")
            print(
                f"    Consensus found/selected: {info['consensus_found']}/{info['consensus_selected']}"
            )
            print(
                f"    Debated found/selected: {info['debated_found']}/{info['debated_selected']}"
            )
            print(
                f"    Unique found/selected: {info['unique_found']}/{info['unique_selected']}"
            )
            print()

    # Write summary
    output_path = "claim_analysis_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
