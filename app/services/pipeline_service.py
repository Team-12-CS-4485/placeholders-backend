from app.core.config import settings
from app.core.logging import get_logger
from app.services.storage_service import StorageService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


class PipelineService:
    def __init__(self, storage_service=None, embedding_service=None, vector_service=None):
        self.storage_service = storage_service or StorageService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_service = vector_service or VectorService()
        self.logger = get_logger(__name__)

    def run_s3_transcript_analysis(self, prefix=None, limit=None):
        use_prefix = prefix if prefix is not None else settings.s3_prefix
        use_limit = limit if limit is not None else settings.s3_object_limit
        self.logger.info(f"PIPELINE_START prefix={use_prefix} limit={use_limit}")

        try:
            source_objects = self.storage_service.load_transcripts_from_prefix(
                prefix=use_prefix,
                limit=use_limit
            )

            object_results = []
            total_transcripts = 0
            analyzed_transcripts = 0
            total_points_indexed = 0
            chunk_map = {}
            analysis_map = {}

            for source in source_objects:
                source_key = source.get("key", "")
                object_result = {
                    "key": source_key,
                    "status": "success",
                    "error": source.get("error"),
                    "transcript_results": []
                }

                transcripts = source.get("transcripts", [])
                total_transcripts += len(transcripts)

                if source.get("error"):
                    object_result["status"] = "failed"
                    object_results.append(object_result)
                    continue

                for idx, transcript in enumerate(transcripts, start=1):
                    transcript_key = f"{source_key}::transcript_{idx}"
                    chunks = self.embedding_service.chunk_text(transcript)
                    chunk_map[transcript_key] = chunks
                    try:
                        vectors = self.embedding_service.embed_chunks(chunks)
                        points_indexed = self.vector_service.upsert_transcript_chunks(
                            transcript_key=transcript_key,
                            source_key=source_key,
                            transcript_index=idx,
                            chunks=chunks,
                            vectors=vectors,
                        )
                        chunk_analyses = self.embedding_service.analyze_chunks(chunks)
                        final_summary = self.embedding_service.summarize_analyses(chunk_analyses)
                        analysis_map[transcript_key] = {
                            "status": "success",
                            "chunk_count": len(chunks),
                            "chunk_analyses": chunk_analyses,
                            "final_summary": final_summary,
                            "qdrant_points_indexed": points_indexed,
                            "error": None,
                        }
                        object_result["transcript_results"].append({
                            "transcript_key": transcript_key,
                            "transcript_index": idx,
                            "chunk_count": len(chunks),
                            "final_summary": final_summary,
                            "qdrant_points_indexed": points_indexed,
                            "error": None,
                        })
                        analyzed_transcripts += 1
                        total_points_indexed += points_indexed
                        print(
                            f"ANALYSIS_SUCCESS key={transcript_key} "
                            f"chunks={len(chunks)} points_indexed={points_indexed}"
                        )
                    except Exception as exc:
                        analysis_map[transcript_key] = {
                            "status": "failed",
                            "chunk_count": len(chunks),
                            "chunk_analyses": [],
                            "final_summary": "",
                            "qdrant_points_indexed": 0,
                            "error": str(exc),
                        }
                        object_result["status"] = "partial_failed"
                        object_result["transcript_results"].append({
                            "transcript_key": transcript_key,
                            "transcript_index": idx,
                            "error": str(exc),
                            "chunk_count": len(chunks),
                            "final_summary": "",
                            "qdrant_points_indexed": 0,
                        })
                        print(
                            f"ANALYSIS_FAILURE key={transcript_key} "
                            f"chunks={len(chunks)} error={exc}"
                        )

                object_results.append(object_result)

            response = {
                "prefix": use_prefix,
                "object_limit": use_limit,
                "objects_processed": len(source_objects),
                "transcripts_found": total_transcripts,
                "transcripts_analyzed": analyzed_transcripts,
                "qdrant_collection": settings.qdrant_collection,
                "qdrant_points_indexed": total_points_indexed,
                "chunk_map": chunk_map,
                "analysis_map": analysis_map,
                "results": object_results
            }

            print(
                "PIPELINE_CONSOLE_SUMMARY "
                f"objects={response['objects_processed']} "
                f"transcripts_found={response['transcripts_found']} "
                f"transcripts_analyzed={response['transcripts_analyzed']} "
                f"qdrant_points_indexed={response['qdrant_points_indexed']}"
            )

            if analyzed_transcripts > 0:
                self.logger.info(
                    "PIPELINE_SUCCESS "
                    f"objects={response['objects_processed']} transcripts={response['transcripts_analyzed']}"
                )
            else:
                self.logger.error(
                    "PIPELINE_FAILURE "
                    f"objects={response['objects_processed']} transcripts={response['transcripts_analyzed']}"
                )
            return response

        except Exception as exc:
            self.logger.error(f"PIPELINE_FAILURE error={exc}")
            raise

    def search_similar_chunks(self, query, limit=5):
        self.logger.info(f"QDRANT_SEARCH_START limit={limit}")
        query_vector = self.embedding_service.embed_text(query)
        hits = self.vector_service.search_similar_chunks(query_vector=query_vector, limit=limit)
        self.logger.info(f"QDRANT_SEARCH_SUCCESS hits={len(hits)}")
        return {
            "collection": settings.qdrant_collection,
            "query": query,
            "limit": limit,
            "hits": hits,
        }
