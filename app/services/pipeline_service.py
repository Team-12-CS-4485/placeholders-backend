"""
pipeline_service.py - Core Pipeline Orchestrator

Coordinates the end-to-end transcript analysis workflow:
1. Loads transcript JSON objects from S3 via StorageService
2. Splits transcripts into semantic chunks via ChunkingService
3. Embeds and stores chunks locally via FAISSService
4. Runs Gemini-powered chunk analysis and final summary generation via EmbeddingService
5. Returns comprehensive results with chunk maps, analysis maps, and per-object status

Also provides semantic search over indexed chunks using FAISS.
"""

from app.core.config import settings
from app.core.logging import get_logger
from app.services.storage_service import StorageService
from app.services.embedding_service import EmbeddingService
from app.services.chunking_service import semantic_chunk
from app.services.faiss_service import FAISSService


class PipelineService:
    def __init__(self, storage_service=None, embedding_service=None, faiss_service=None):
        self.storage_service = storage_service or StorageService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.faiss_service = faiss_service or FAISSService()
        self.logger = get_logger(__name__)

    def run_s3_transcript_analysis(self, prefix=None, limit=None):
        use_prefix = prefix if prefix is not None else settings.s3_prefix
        use_limit = limit if limit is not None else settings.s3_object_limit
        self.logger.info(f"PIPELINE_START prefix={use_prefix} limit={use_limit}")

        try:
            source_objects = self.storage_service.load_transcripts_from_prefix(
                prefix=use_prefix,
                limit=use_limit,
            )

            object_results = []
            total_transcripts = 0
            analyzed_transcripts = 0
            total_chunks_stored = 0
            chunk_map = {}
            analysis_map = {}

            for source in source_objects:
                source_key = source.get("key", "")
                object_result = {
                    "key": source_key,
                    "status": "success",
                    "error": source.get("error"),
                    "transcript_results": [],
                }

                transcripts = source.get("transcripts", [])
                total_transcripts += len(transcripts)

                if source.get("error"):
                    object_result["status"] = "failed"
                    object_results.append(object_result)
                    continue

                for idx, transcript in enumerate(transcripts, start=1):
                    transcript_key = f"{source_key}::transcript_{idx}"

                    # Step 1: Semantic chunking
                    chunks = semantic_chunk(transcript)
                    chunk_map[transcript_key] = chunks

                    try:
                        # Embed locally (nomic) — no API call, no quota
                        vectors = self.embedding_service.embed_chunks(chunks)

                        # Upsert to Qdrant
                        points_indexed = self.vector_service.upsert_transcript_chunks(
                            transcript_key=transcript_key,
                            source_key=source_key,
                            transcript_index=idx,
                        )

                        # Gemini analysis skipped — no quota burned
                        analysis_map[transcript_key] = {
                            "status": "success",
                            "chunk_count": len(chunks),
                            "chunk_analyses": [],
                            "final_summary": "",
                            "qdrant_points_indexed": points_indexed,
                            "error": None,
                        }
                        object_result["transcript_results"].append({
                            "transcript_key": transcript_key,
                            "transcript_index": idx,
                            "chunk_count": len(chunks),
                            "final_summary": "",
                            "qdrant_points_indexed": points_indexed,
                            "error": None,
                        })
                        analyzed_transcripts += 1
                        total_points_indexed += points_indexed
                        print(
                            f"UPSERT_SUCCESS key={transcript_key} "
                            f"chunks={len(chunks)} points_indexed={points_indexed}"
                        )

                    except Exception as exc:
                        analysis_map[transcript_key] = {
                            "status": "failed",
                            "chunk_count": len(chunks),
                            "chunk_analyses": [],
                            "final_summary": "",
                            "chunks_stored": 0,
                            "error": str(exc),
                        }
                        object_result["status"] = "partial_failed"
                        object_result["transcript_results"].append({
                            "transcript_key": transcript_key,
                            "transcript_index": idx,
                            "error": str(exc),
                            "chunk_count": len(chunks),
                            "final_summary": "",
                            "chunks_stored": 0,
                        })
                        print(
                            f"UPSERT_FAILURE key={transcript_key} "
                            f"chunks={len(chunks)} error={exc}"
                        )

                object_results.append(object_result)

            response = {
                "prefix": use_prefix,
                "object_limit": use_limit,
                "objects_processed": len(source_objects),
                "transcripts_found": total_transcripts,
                "transcripts_analyzed": analyzed_transcripts,
                "total_chunks_stored": total_chunks_stored,
                "chunk_map": chunk_map,
                "analysis_map": analysis_map,
                "results": object_results,
            }

            print(
                "PIPELINE_SUMMARY "
                f"objects={response['objects_processed']} "
                f"transcripts_found={response['transcripts_found']} "
                f"transcripts_indexed={response['transcripts_analyzed']} "
                f"qdrant_points_indexed={response['qdrant_points_indexed']}"
            )

            if analyzed_transcripts > 0:
                self.logger.info(
                    "PIPELINE_SUCCESS "
                    f"objects={response['objects_processed']} "
                    f"transcripts={response['transcripts_analyzed']} "
                    f"points={total_points_indexed}"
                )
            else:
                self.logger.error(
                    "PIPELINE_FAILURE "
                    f"objects={response['objects_processed']} "
                    f"transcripts={response['transcripts_analyzed']}"
                )
            return response

        except Exception as exc:
            self.logger.error(f"PIPELINE_FAILURE error={exc}")
            raise

    def search_similar_chunks(self, query: str, limit: int = 5):
        self.logger.info(f"QDRANT_SEARCH_START limit={limit}")

        query_vector = self.embedding_service.embed_query(query)

        hits = self.vector_service.search_similar_chunks(
            query_vector=query_vector,
            limit=limit,
        )
        self.logger.info(f"QDRANT_SEARCH_SUCCESS hits={len(hits)}")
        return {
            "query": query,
            "limit": limit,
            "hits": hits,
        }