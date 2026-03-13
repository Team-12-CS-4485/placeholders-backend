"""
pipeline_service.py - Core Pipeline Orchestrator

Coordinates the end-to-end transcript analysis workflow:
1. Loads video objects from S3 via StorageService (transcript + metadata)
2. DynamoDB join for fresh viewCount/likeCount via StorageService
3. Semantic chunking via EmbeddingService
4. One Gemini call per video → topics, category, sentiment, key_claims, is_breaking
5. Embeds chunks locally via EmbeddingService (nomic, free)
6. Upserts to Qdrant with full enriched payload via VectorService

Final Qdrant payload per chunk:
{
  "transcript_key": "youtube-data/week1/cnbc.json::videoId",
  "source_key": "youtube-data/week1/cnbc.json",
  "chunk_index": 1,
  "text": "...",
  "channel": "CNBC",
  "title": "...",
  "published_at": "2026-03-04T19:57:29Z",
  "view_count": 145000,
  "like_count": 3200,
  "comment_count": 400,
  "topics": ["AI Policy", "Pentagon"],
  "category": "Technology",
  "sentiment": "negative",
  "key_claims": ["DoD may invoke Defense Production Act"],
  "is_breaking": true
}
"""

import time

from app.core.config import settings
from app.core.logging import get_logger
from app.services.storage_service import StorageService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


class PipelineService:
    def __init__(self, storage_service=None, embedding_service=None, vector_service=None):
        self.storage_service = storage_service or StorageService()
        self.embedding_service = embedding_service or EmbeddingService(
            api_keys=settings.genai_api_keys
        )
        self.vector_service = vector_service or VectorService()
        self.logger = get_logger(__name__)

    def run_s3_transcript_analysis(self, prefix=None, limit=None):
        use_prefix = prefix if prefix is not None else settings.s3_prefix
        use_limit = limit if limit is not None else settings.s3_object_limit
        self.logger.info(f"PIPELINE_START prefix={use_prefix} limit={use_limit}")

        try:
            source_objects = self.storage_service.load_videos_from_prefix(
                prefix=use_prefix,
                limit=use_limit,
            )

            object_results = []
            total_videos = 0
            analyzed_videos = 0
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

                if source.get("error"):
                    object_result["status"] = "failed"
                    object_results.append(object_result)
                    continue

                videos = source.get("videos", [])
                total_videos += len(videos)

                for video in videos:
                    video_id      = video.get("videoId", "")
                    title         = video.get("title", "")
                    channel       = video.get("channel", "")
                    published_at  = video.get("published_at", "")
                    view_count    = video.get("view_count", 0)
                    like_count    = video.get("like_count", 0)
                    comment_count = video.get("comment_count", 0)
                    transcript    = video.get("transcript", "")

                    # Unique key per video (not per source file)
                    transcript_key = f"{source_key}::{video_id}"
                    
                    # Check if already indexed — skip to save Gemini quota
                    existing = self.vector_service.client.scroll(
                        collection_name=self.vector_service.collection_name,
                        scroll_filter=models.Filter(
                            must=[models.FieldCondition(
                                key="transcript_index",
                                match=models.MatchValue(value=video_id)
                            )]
                        ),
                        limit=1,
                    )[0]
                    if existing:
                        self.logger.info(f"VIDEO_SKIP_ALREADY_INDEXED videoId={video_id} channel={channel}")
                        analyzed_videos += 1
                        total_chunks_stored += len(existing)
                        continue
                    

                    # Step 1: Chunk
                    chunks = self.embedding_service.chunk_text(transcript)
                    chunk_map[transcript_key] = chunks

                    try:
                        # Step 2: Gemini intelligence — one call per video
                        self.logger.info(
                            f"GEMINI_INTELLIGENCE_START videoId={video_id} chunks={len(chunks)} key=#{self.embedding_service.current_key_index + 1}"
                            )
                        intelligence = self.embedding_service.extract_video_intelligence(
                            chunks=chunks,
                            title=title,
                        )
                        self.logger.info(
                            f"GEMINI_INTELLIGENCE_DONE videoId={video_id} "
                            f"category={intelligence['category']} "
                            f"sentiment={intelligence['sentiment']} "
                            f"topics={intelligence['topics']}"
                        )
                        time.sleep(10)  # brief pause between videos to avoid rate limits

                        # Step 3: Embed chunks locally
                        vectors = self.embedding_service.embed_chunks(chunks)

                        # Step 4: Build enriched payload metadata (same for every chunk in this video)
                        video_metadata = {
                            "channel":       channel,
                            "title":         title,
                            "published_at":  published_at,
                            "view_count":    view_count,
                            "like_count":    like_count,
                            "comment_count": comment_count,
                            **intelligence,  # topics, category, sentiment, key_claims, is_breaking
                        }

                        # Step 5: Upsert to Qdrant with full payload
                        points_indexed = self.vector_service.upsert_transcript_chunks(
                            transcript_key=transcript_key,
                            source_key=source_key,
                            transcript_index=video_id,
                            chunks=chunks,
                            vectors=vectors,
                            extra_metadata=video_metadata,
                        )

                        analysis_map[transcript_key] = {
                            "status": "success",
                            "chunk_count": len(chunks),
                            "chunks_stored": points_indexed,
                            "intelligence": intelligence,
                            "error": None,
                        }
                        object_result["transcript_results"].append({
                            "transcript_key": transcript_key,
                            "video_id":       video_id,
                            "chunk_count":    len(chunks),
                            "chunks_stored":  points_indexed,
                            "intelligence":   intelligence,
                            "error":          None,
                        })
                        analyzed_videos += 1
                        total_chunks_stored += points_indexed
                        self.logger.info(
                            f"VIDEO_INDEXED videoId={video_id} channel={channel} "
                            f"chunks={len(chunks)} points={points_indexed}"
                        )

                    except Exception as exc:
                        analysis_map[transcript_key] = {
                            "status": "failed",
                            "chunk_count": len(chunks),
                            "chunks_stored": 0,
                            "intelligence": None,
                            "error": str(exc),
                        }
                        object_result["status"] = "partial_failed"
                        object_result["transcript_results"].append({
                            "transcript_key": transcript_key,
                            "video_id":       video_id,
                            "chunk_count":    len(chunks),
                            "chunks_stored":  0,
                            "intelligence":   None,
                            "error":          str(exc),
                        })
                        self.logger.error(
                            f"VIDEO_INDEX_FAILURE videoId={video_id} error={exc}"
                        )

                object_results.append(object_result)

            response = {
                "prefix":             use_prefix,
                "object_limit":       use_limit,
                "objects_processed":  len(source_objects),
                "videos_found":       total_videos,
                "videos_indexed":     analyzed_videos,
                "total_chunks_stored": total_chunks_stored,
                "chunk_map":          chunk_map,
                "analysis_map":       analysis_map,
                "results":            object_results,
            }

            if analyzed_videos > 0:
                self.logger.info(
                    f"PIPELINE_SUCCESS objects={len(source_objects)} "
                    f"videos={analyzed_videos} chunks={total_chunks_stored}"
                )
            else:
                self.logger.error(
                    f"PIPELINE_FAILURE objects={len(source_objects)} videos={analyzed_videos}"
                )

            return response

        except Exception as exc:
            self.logger.error(f"PIPELINE_FAILURE error={exc}")
            raise

    def search_similar_chunks(self, query: str, limit: int = 5):
        self.logger.info(f"SEARCH_START query='{query[:50]}' limit={limit}")
        query_vector = self.embedding_service.embed_query(query)
        hits = self.vector_service.search_similar_chunks(
            query_vector=query_vector,
            limit=limit,
        )
        self.logger.info(f"SEARCH_SUCCESS hits={len(hits)}")
        return {
            "query": query,
            "limit": limit,
            "hits": hits,
        }