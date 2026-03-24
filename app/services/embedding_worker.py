"""
embedding_worker.py - Batch Transcript Analysis Worker

CLI entrypoint for running the transcript analysis pipeline as a batch job.
Executes the full pipeline (S3 -> chunk -> Gemini intelligence + thumbnail -> embed -> Qdrant)
and writes per-video results to a text output file.

Usage: python3 -m app.workers.embedding_worker
"""

from app.core.config import settings
from app.services.pipeline_service import PipelineService


def run_transcript_analysis_job(prefix=None, limit=None, output_file=None):
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
                else:
                    # Transcript intelligence
                    intel = video_result.get("intelligence") or {}
                    if intel:
                        file.write(f"  Category: {intel.get('category')}\n")
                        file.write(f"  Sentiment: {intel.get('sentiment')}\n")
                        file.write(f"  Topics: {', '.join(intel.get('topics', []))}\n")
                        file.write(f"  Is Breaking: {intel.get('is_breaking')}\n")
                        file.write(f"  Key Claims:\n")
                        for claim in intel.get("key_claims", []):
                            file.write(f"    - {claim}\n")

                    # Thumbnail intelligence
                    thumb = video_result.get("thumbnail") or {}
                    if thumb.get("thumbnail_tone"):
                        file.write(f"  Thumbnail Tone: {thumb.get('thumbnail_tone')}\n")
                        file.write(f"  Thumbnail Clickbait Score: {thumb.get('thumbnail_clickbait_score')}\n")
                        file.write(f"  Thumbnail Brand Consistent: {thumb.get('thumbnail_brand_consistent')}\n")
                        file.write(f"  Thumbnail Visual: {thumb.get('thumbnail_visual')}\n")
                        file.write(f"  Thumbnail Insight: {thumb.get('thumbnail_insight')}\n")

                file.write("\n")

            file.write("-" * 80 + "\n\n")

    return result


if __name__ == "__main__":
    run_transcript_analysis_job()

    # Auto-sync any missing videos after pipeline completes
    print("\n=== Running post-pipeline sync ===")
    from scripts.sync_missing import main as sync_main
    sync_main()
