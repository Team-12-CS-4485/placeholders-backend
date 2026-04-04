"""
pipeline_service.py - Core Pipeline Orchestrator

Coordinates the end-to-end transcript analysis workflow:
1. Loads video objects from S3 via StorageService (transcript + metadata)
2. DynamoDB join for fresh viewCount/likeCount via StorageService
3. Semantic chunking via EmbeddingService
4. ONE combined Gemini call per video (multimodal: transcript + thumbnail image)
   → topics, category, sentiment, key_claims, is_breaking, thumbnail fields
5. Writes intelligence + thumbnail to DynamoDB (youtube-videos table)
6. Embeds chunks locally via EmbeddingService (nomic, free)
7. Upserts to Qdrant with search-only payload (vector, text, transcript_index)

DynamoDB youtube-videos item gets:
  topics, category, sentiment, key_claims, is_breaking, source_key, week, chunk_count
  thumbnail_visual, thumbnail_tone, thumbnail_clickbait_score,
  thumbnail_brand_consistent, thumbnail_insight

Qdrant transcript_chunks point gets:
  vector, text, chunk_index, word_count, transcript_index, source_key
"""

import re
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from qdrant_client.http import models

from app.core.config import settings
from app.core.logging import get_logger
from app.services.storage_service import StorageService
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeminiService
from app.services.vector_service import VectorService


def _extract_week(source_key: str) -> str:
    """Extract week from source_key path."""
    if not source_key:
        return "unknown"
    match = re.search(r"(week\d+)", source_key, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", source_key)
    if date_match:
        try:
            dt = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            wk = dt.isocalendar()[1] - 10 + 1
            if wk >= 1:
                return f"week{wk}"
        except ValueError:
            pass
    return "unknown"


def _dec(val):
    """Convert numbers to Decimal for DynamoDB."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def _thumbnail_is_empty(thumb: dict) -> bool:
    """
    Returns True if analyze_thumbnail() returned all-default values,
    indicating a Gemini 503 error or unavailable image.

    A real result always has a non-empty tone AND score >= 1.
    """
    return (
        not thumb.get("thumbnail_visual")
        and not thumb.get("thumbnail_tone")
        and thumb.get("thumbnail_clickbait_score", 0) == 0
    )


class PipelineService:
    def __init__(
        self,
        storage_service=None,
        embedding_service=None,
        gemini_service=None,
        vector_service=None,
    ):
        self.storage_service = storage_service or StorageService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.gemini_service = gemini_service or GeminiService(
            api_keys=settings.genai_api_keys
        )
        self.vector_service = vector_service or VectorService()
        self.logger = get_logger(__name__)

        self._dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
        self._videos_table = self._dynamodb.Table(settings.dynamodb_table)

    # ── DynamoDB: intelligence write ──────────────────────────────────────────

    def _write_intelligence_to_dynamodb(
        self,
        channel: str,
        video_id: str,
        source_key: str,
        intelligence: dict,
        chunk_count: int,
    ) -> None:
        """
        Write Gemini intelligence fields to the existing DynamoDB video item.
        Only writes fields that have real values — never writes None or empty strings
        to GSI key fields (week, sentiment, cluster_id).
        """
        week = _extract_week(source_key)

        set_parts = []
        names = {}
        values = {}

        topics = intelligence.get("topics") or []
        if topics:
            set_parts.append("#topics = :topics")
            names["#topics"] = "topics"
            values[":topics"] = topics

        category = intelligence.get("category") or ""
        if category:
            set_parts.append("#category = :category")
            names["#category"] = "category"
            values[":category"] = category

        sentiment = intelligence.get("sentiment") or ""
        if sentiment:
            set_parts.append("#sentiment = :sentiment")
            names["#sentiment"] = "sentiment"
            values[":sentiment"] = sentiment

        key_claims = intelligence.get("key_claims") or []
        if key_claims:
            set_parts.append("#claims = :claims")
            names["#claims"] = "key_claims"
            values[":claims"] = key_claims

        set_parts.append("#breaking = :breaking")
        names["#breaking"] = "is_breaking"
        values[":breaking"] = bool(intelligence.get("is_breaking", False))

        public_sentiment = intelligence.get("public_sentiment") or ""
        if public_sentiment:
            set_parts.append("#psent = :psent")
            names["#psent"] = "public_sentiment"
            values[":psent"] = public_sentiment

        public_sentiment_score = intelligence.get("public_sentiment_score")
        if public_sentiment_score is not None:
            set_parts.append("#pscore = :pscore")
            names["#pscore"] = "public_sentiment_score"
            values[":pscore"] = _dec(public_sentiment_score)

        if source_key:
            set_parts.append("#src = :src")
            names["#src"] = "source_key"
            values[":src"] = source_key

        set_parts.append("#week = :week")
        names["#week"] = "week"
        values[":week"] = week

        set_parts.append("#chunks = :chunks")
        names["#chunks"] = "chunk_count"
        values[":chunks"] = _dec(chunk_count)

        set_parts.append("#ts = :ts")
        names["#ts"] = "indexed_at"
        values[":ts"] = datetime.now(timezone.utc).isoformat()

        if not set_parts:
            return

        try:
            self._videos_table.update_item(
                Key={"PartitionKey": channel, "SortKey": video_id},
                UpdateExpression="SET " + ", ".join(set_parts),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )
            self.logger.info(
                f"DYNAMO_INTEL_WRITTEN videoId={video_id} "
                f"fields={list(names.values())}"
            )
        except Exception as exc:
            self.logger.error(f"DYNAMO_INTEL_FAILED videoId={video_id} error={exc}")

    # ── DynamoDB: thumbnail write ─────────────────────────────────────────────

    def _write_thumbnail_to_dynamodb(
        self,
        channel: str,
        video_id: str,
        thumbnail: dict,
    ) -> None:
        """
        Write Gemini vision thumbnail fields to the existing DynamoDB video item.

        Fields written:
          thumbnail_visual           str
          thumbnail_tone             str
          thumbnail_clickbait_score  int (Decimal)
          thumbnail_brand_consistent bool
          thumbnail_insight          str

        Skips entirely if thumbnail result is all defaults (503 / no image).
        """
        if _thumbnail_is_empty(thumbnail):
            self.logger.info(f"THUMBNAIL_SKIP_EMPTY videoId={video_id}")
            return

        set_parts = []
        names = {}
        values = {}

        if thumbnail.get("thumbnail_visual"):
            set_parts.append("#tvis = :tvis")
            names["#tvis"] = "thumbnail_visual"
            values[":tvis"] = str(thumbnail["thumbnail_visual"])

        if thumbnail.get("thumbnail_tone"):
            set_parts.append("#ttone = :ttone")
            names["#ttone"] = "thumbnail_tone"
            values[":ttone"] = str(thumbnail["thumbnail_tone"])

        # Always write score and brand_consistent (safe defaults are fine to store)
        set_parts.append("#tscore = :tscore")
        names["#tscore"] = "thumbnail_clickbait_score"
        values[":tscore"] = _dec(thumbnail.get("thumbnail_clickbait_score", 0))

        set_parts.append("#tbrand = :tbrand")
        names["#tbrand"] = "thumbnail_brand_consistent"
        values[":tbrand"] = bool(thumbnail.get("thumbnail_brand_consistent", False))

        if thumbnail.get("thumbnail_insight"):
            set_parts.append("#tins = :tins")
            names["#tins"] = "thumbnail_insight"
            values[":tins"] = str(thumbnail["thumbnail_insight"])

        if not set_parts:
            return

        try:
            self._videos_table.update_item(
                Key={"PartitionKey": channel, "SortKey": video_id},
                UpdateExpression="SET " + ", ".join(set_parts),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )
            self.logger.info(
                f"DYNAMO_THUMBNAIL_WRITTEN videoId={video_id} "
                f"tone={thumbnail.get('thumbnail_tone')} "
                f"score={thumbnail.get('thumbnail_clickbait_score')}"
            )
        except Exception as exc:
            self.logger.error(f"DYNAMO_THUMBNAIL_FAILED videoId={video_id} error={exc}")

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run_s3_transcript_analysis(self, prefix=None, limit=None, dry_run=False):
        use_prefix = prefix if prefix is not None else settings.s3_prefix
        use_limit = limit if limit is not None else settings.s3_object_limit
        self.logger.info(
            f"PIPELINE_START prefix={use_prefix} limit={use_limit} dry_run={dry_run}"
        )

        if dry_run:
            source_objects = self.storage_service.load_videos_from_prefix(
                prefix=use_prefix,
                limit=use_limit,
            )
            total_videos = sum(len(s.get("videos", [])) for s in source_objects)
            self.logger.info(
                f"PIPELINE_DRY_RUN objects={len(source_objects)} videos={total_videos}"
            )
            return {
                "prefix": use_prefix,
                "object_limit": use_limit,
                "objects_processed": len(source_objects),
                "videos_found": total_videos,
                "videos_indexed": 0,
                "total_chunks_stored": 0,
                "results": [],
                "dry_run": True,
            }

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
                    video_id = video.get("videoId", "")
                    title = video.get("title", "")
                    channel = video.get("channel", "")
                    transcript = video.get("transcript", "")
                    top_comments = video.get("top_comments", [])

                    transcript_key = f"{source_key}::{video_id}"

                    # Check if already indexed in Qdrant — skip to save Gemini quota
                    try:
                        existing = self.vector_service.client.scroll(
                            collection_name=self.vector_service.collection_name,
                            scroll_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="transcript_index",
                                        match=models.MatchValue(value=video_id),
                                    )
                                ]
                            ),
                            limit=1,
                        )[0]
                        if existing:
                            self.logger.info(
                                f"VIDEO_SKIP_ALREADY_INDEXED videoId={video_id} channel={channel}"
                            )
                            analyzed_videos += 1
                            total_chunks_stored += len(existing)
                            continue
                    except Exception:
                        pass

                    # Step 1: Chunk
                    chunks = self.embedding_service.chunk_text(transcript)
                    chunk_map[transcript_key] = chunks

                    try:
                        # Step 2: Combined Gemini intelligence + thumbnail — ONE call
                        self.logger.info(
                            f"GEMINI_COMBINED_START videoId={video_id} "
                            f"chunks={len(chunks)} "
                            f"key=#{self.gemini_service.current_key_index + 1}"
                        )
                        intelligence, thumbnail = (
                            self.gemini_service.extract_full_video_intelligence(
                                chunks=chunks,
                                title=title,
                                top_comments=top_comments,
                                video_id=video_id,
                            )
                        )

                        # Guard: Gemini 503 returns empty topics — skip this video
                        if not intelligence.get("topics"):
                            self.logger.warning(
                                f"GEMINI_INTELLIGENCE_EMPTY videoId={video_id} "
                                f"— skipping (likely 503)"
                            )
                            object_result["transcript_results"].append(
                                {
                                    "transcript_key": transcript_key,
                                    "video_id": video_id,
                                    "chunk_count": len(chunks),
                                    "chunks_stored": 0,
                                    "intelligence": None,
                                    "thumbnail": None,
                                    "error": "Gemini returned empty intelligence (503)",
                                }
                            )
                            continue

                        self.logger.info(
                            f"GEMINI_COMBINED_DONE videoId={video_id} "
                            f"category={intelligence['category']} "
                            f"sentiment={intelligence['sentiment']} "
                            f"topics={intelligence['topics']} "
                            f"thumbnail_tone={thumbnail.get('thumbnail_tone')}"
                        )

                        # Step 3: Write intelligence to DynamoDB
                        self._write_intelligence_to_dynamodb(
                            channel=channel,
                            video_id=video_id,
                            source_key=source_key,
                            intelligence=intelligence,
                            chunk_count=len(chunks),
                        )

                        # Step 4: Write thumbnail to DynamoDB (if non-empty)
                        if _thumbnail_is_empty(thumbnail):
                            self.logger.warning(
                                f"THUMBNAIL_EMPTY videoId={video_id} "
                                f"— skipping thumbnail write (no image)"
                            )
                        else:
                            self._write_thumbnail_to_dynamodb(
                                channel=channel,
                                video_id=video_id,
                                thumbnail=thumbnail,
                            )

                        # Step 6: Embed chunks locally
                        vectors = self.embedding_service.embed_chunks(chunks)

                        # Step 7: Upsert to Qdrant — SEARCH FIELDS ONLY
                        points_indexed = self.vector_service.upsert_transcript_chunks(
                            transcript_key=transcript_key,
                            source_key=source_key,
                            transcript_index=video_id,
                            chunks=chunks,
                            vectors=vectors,
                            extra_metadata=None,
                        )

                        analysis_map[transcript_key] = {
                            "status": "success",
                            "chunk_count": len(chunks),
                            "chunks_stored": points_indexed,
                            "intelligence": intelligence,
                            "thumbnail": thumbnail,
                            "error": None,
                        }
                        object_result["transcript_results"].append(
                            {
                                "transcript_key": transcript_key,
                                "video_id": video_id,
                                "chunk_count": len(chunks),
                                "chunks_stored": points_indexed,
                                "intelligence": intelligence,
                                "thumbnail": thumbnail,
                                "error": None,
                            }
                        )
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
                            "thumbnail": None,
                            "error": str(exc),
                        }
                        object_result["status"] = "partial_failed"
                        object_result["transcript_results"].append(
                            {
                                "transcript_key": transcript_key,
                                "video_id": video_id,
                                "chunk_count": len(chunks),
                                "chunks_stored": 0,
                                "intelligence": None,
                                "thumbnail": None,
                                "error": str(exc),
                            }
                        )
                        self.logger.error(
                            f"VIDEO_FAILED videoId={video_id} error={exc}"
                        )

                object_results.append(object_result)

            result = {
                "prefix": use_prefix,
                "object_limit": use_limit,
                "objects_processed": len(source_objects),
                "videos_found": total_videos,
                "videos_indexed": analyzed_videos,
                "total_chunks_stored": total_chunks_stored,
                "results": object_results,
            }
            self.logger.info(
                f"PIPELINE_COMPLETE videos_found={total_videos} "
                f"videos_indexed={analyzed_videos} "
                f"chunks_stored={total_chunks_stored}"
            )
            return result

        except Exception as exc:
            self.logger.error(f"PIPELINE_FATAL error={exc}")
            raise
