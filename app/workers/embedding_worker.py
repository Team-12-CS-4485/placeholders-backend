from app.core.config import settings
from app.services.pipeline_service import PipelineService


def run_transcript_analysis_job(prefix=None, limit=None, output_file=None):
    pipeline = PipelineService()
    result = pipeline.run_s3_transcript_analysis(prefix=prefix, limit=limit)
    target_file = output_file or settings.analysis_output_file

    with open(target_file, "w", encoding="utf-8") as file:
        file.write(f"Prefix: {result['prefix']}\n")
        file.write(f"Objects Processed: {result['objects_processed']}\n")
        file.write(f"Transcripts Found: {result['transcripts_found']}\n")
        file.write(f"Transcripts Analyzed: {result['transcripts_analyzed']}\n\n")

        for obj in result["results"]:
            file.write(f"Object: {obj['key']}\n")
            file.write(f"Status: {obj['status']}\n")
            if obj.get("error"):
                file.write(f"Error: {obj['error']}\n")
            for transcript_result in obj["transcript_results"]:
                file.write(f"Transcript Index: {transcript_result['transcript_index']}\n")
                file.write(f"Chunk Count: {transcript_result['chunk_count']}\n")
                if transcript_result.get("error"):
                    file.write(f"Analysis Error: {transcript_result['error']}\n\n")
                else:
                    file.write(f"Summary:\n{transcript_result['final_summary']}\n\n")
            file.write("-" * 80 + "\n\n")

    return result


if __name__ == "__main__":
    run_transcript_analysis_job()
