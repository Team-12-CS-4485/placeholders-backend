"""
run_claim_analysis.py - Classify claims across narrative clusters

Embeds all claims locally, groups by semantic similarity, classifies
into consensus/debated/unique with risk scores, writes to DynamoDB.
Zero Gemini calls.

Usage:
    python -m scripts.run_claim_analysis
    python -m scripts.run_claim_analysis --dry-run
    python -m scripts.run_claim_analysis --max-per-type 5
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from app.services.claim_analysis_service import ClaimAnalysisService


class _DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def main():
    parser = argparse.ArgumentParser(description="Run claim classification")
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=3,
        help="Max claims to keep per type per cluster (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis but do not write to DynamoDB",
    )
    args = parser.parse_args()

    print("=== Claim Analysis ===\n")

    service = ClaimAnalysisService()
    result = service.run_claim_analysis(
        max_per_type=args.max_per_type, dry_run=args.dry_run
    )

    print(f"\nClusters processed: {result['clusters_processed']}")
    print(f"Total written:      {result['total_written']}")
    print(f"Dry run:            {result['dry_run']}")

    if result.get("per_cluster"):
        print("\n--- Per Cluster ---\n")
        for cid, info in sorted(result["per_cluster"].items()):
            print(f"  Cluster {cid}:")
            print(f"    Raw claims: {info['total_claims']}  Groups: {info['groups']}")
            print(
                f"    All:      {info['consensus']}c / {info['debated']}d / {info['unique']}u"
            )
            print(
                f"    Selected: {info.get('selected_c',0)}c / {info.get('selected_d',0)}d / {info.get('selected_u',0)}u"
            )

    # Write full results to file
    cluster_results = result.get("cluster_results", {})
    if cluster_results:
        output = {}
        for cid, claims_data in sorted(cluster_results.items()):
            output[str(cid)] = claims_data

        output_path = "claim_analysis_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, cls=_DecimalEncoder)
        print(f"\nFull results written to {output_path}")

        # Also write a readable text report
        report_path = "claim_analysis_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Claim Analysis Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(
                f"Clusters: {result['clusters_processed']}  Written: {result['total_written']}\n"
            )
            f.write("=" * 80 + "\n\n")

            for cid, claims_data in sorted(cluster_results.items()):
                total = sum(
                    len(claims_data.get(t, []))
                    for t in ["consensus", "debated", "unique"]
                )
                f.write(f"=== Cluster {cid} ({total} selected claims) ===\n\n")

                for claim_type in ["consensus", "debated", "unique"]:
                    claims = claims_data.get(claim_type, [])
                    if not claims:
                        continue
                    f.write(f"  [{claim_type.upper()}]\n")
                    for c in claims:
                        f.write(
                            f"    risk={c.get('risk_score', '?')}  channel={c.get('channel', '?')}\n"
                        )
                        f.write(f"    claim: {c.get('claim', '')}\n")
                        if c.get("sources"):
                            f.write(
                                f"    sources: {', '.join(c['sources'])} ({c.get('source_count', 0)})\n"
                            )
                        if c.get("framing_divergence"):
                            f.write(
                                f"    framing_divergence: {c['framing_divergence']}\n"
                            )
                        if c.get("perspectives"):
                            for p in c["perspectives"]:
                                f.write(
                                    f"      - {p['channel']} ({p['sentiment']}): {p.get('video_title', '')}\n"
                                )
                        if c.get("video_title"):
                            f.write(f"    video: {c['video_title']}\n")
                        f.write("\n")
                f.write("-" * 80 + "\n\n")

        print(f"Readable report written to {report_path}")
    else:
        print("\nNo cluster results to write (dry run or empty).")


if __name__ == "__main__":
    main()
