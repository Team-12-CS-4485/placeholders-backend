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

    if summary.get("per_cluster"):
        print("\n--- Per Cluster ---\n")
        for cid, info in sorted(summary["per_cluster"].items()):
            print(f"  Cluster {cid}:")
            print(f"    Raw claims: {info['total_claims']}")
            print(
                f"    Selected: {info.get('selected_c',0)}c / {info.get('selected_d',0)}d / {info.get('selected_u',0)}u"
            )
            print()

    # Write summary
    output_path = "claim_analysis_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
