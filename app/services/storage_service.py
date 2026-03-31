import base64

"""
storage_service.py - S3 Storage Service

Handles all AWS S3 interactions for the pipeline:
- Lists object keys under a given S3 prefix with pagination
- Fetches and parses JSON objects from S3
- Extracts transcripts alongside full video metadata (videoId, title, channel, etc.)
- Cleans transcript text by stripping VTT caption metadata headers
- DynamoDB join for fresh view/like/comment counts
"""

import json
import re
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    def __init__(self, s3_client=None, dynamodb_resource=None, bucket=None):
        self.s3_client = s3_client or boto3.client("s3")
        self.dynamodb = dynamodb_resource or boto3.resource("dynamodb")
        self.bucket = bucket or settings.s3_bucket
        self.dynamodb = dynamodb_resource or boto3.resource(
            "dynamodb", region_name=settings.aws_region
        )

    def list_videos(self, limit: int = 20, cursor: str = None):
        list_kwargs = {
            "Bucket": self.bucket,
            "Prefix": settings.s3_prefix,
            "MaxKeys": limit,
        }
        if cursor:
            list_kwargs["ContinuationToken"] = base64.b64decode(
                cursor.encode()
            ).decode()

        response = self.s3_client.list_objects_v2(**list_kwargs)

        items = []
        for obj in response.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            try:
                payload = self.get_json_object(key)
            except (ClientError, json.JSONDecodeError, UnicodeDecodeError):
                continue

            channel = payload.get("channel", "")
            for video in payload.get("videos", []):
                items.append(
                    {
                        "video_id": video.get("videoId", ""),
                        "channel": channel,
                        "title": video.get("title", ""),
                        "description": video.get("description", ""),
                        "published_at": video.get("publishedAt", ""),
                        "view_count": int(video.get("viewCount", 0)),
                        "like_count": int(video.get("likeCount", 0)),
                        "comment_count": int(video.get("commentCount", 0)),
                    }
                )

        next_cursor = None
        if response.get("IsTruncated"):
            token = response.get("NextContinuationToken", "")
            next_cursor = base64.b64encode(token.encode()).decode()

        return {
            "items": items,
            "total_returned": len(items),
            "next_cursor": next_cursor,
        }

    def get_video_by_id(self, video_id: str) -> Optional[dict]:
        """
        Fetch a single video detail by videoId.

        Looks up the video in DynamoDB first (gets all intelligence fields + source_key),
        then fetches only the matching S3 file for transcript, description, and top_comments.
        Falls back to a full S3 scan if source_key is missing from DynamoDB.
        """
        from app.services.dynamo_service import DynamoService

        dynamo = DynamoService(dynamodb_resource=self.dynamodb)
        dynamo_item = dynamo.get_video_by_id(video_id)

        # Try to fetch only the known S3 file; fall back to full scan if unavailable
        source_key = dynamo_item.get("source_key") if dynamo_item else None

        s3_video = self._find_video_in_s3(video_id, source_key)
        if s3_video is None and source_key:
            # source_key may be stale; retry with full scan
            s3_video = self._find_video_in_s3(video_id, source_key=None)

        if s3_video is None and dynamo_item is None:
            return None

        top_comments = []
        for comment in (s3_video or {}).get("topComments", []):
            if not isinstance(comment, dict):
                continue
            top_comments.append(
                {
                    "author": comment.get("author", ""),
                    "text": comment.get("text", ""),
                    "likes": int(comment.get("likes", 0)),
                }
            )

        # Base fields — prefer DynamoDB values (fresher stats), fall back to S3
        base = dynamo.map_video_item(dynamo_item) if dynamo_item else {}
        s3 = s3_video or {}

        return {
            "video_id": video_id,
            "channel": base.get("channel") or s3.get("channel", ""),
            "title": base.get("title") or s3.get("title", ""),
            "published_at": base.get("published_at") or s3.get("publishedAt", ""),
            "view_count": base.get("view_count") or int(s3.get("viewCount", 0)),
            "like_count": base.get("like_count") or int(s3.get("likeCount", 0)),
            "comment_count": base.get("comment_count")
            or int(s3.get("commentCount", 0)),
            "week": base.get("week"),
            "topics": base.get("topics"),
            "category": base.get("category"),
            "sentiment": base.get("sentiment"),
            "key_claims": base.get("key_claims"),
            "is_breaking": base.get("is_breaking"),
            "thumbnail_tone": base.get("thumbnail_tone"),
            "thumbnail_clickbait_score": base.get("thumbnail_clickbait_score"),
            "thumbnail_insight": base.get("thumbnail_insight"),
            "thumbnail_brand_consistent": base.get("thumbnail_brand_consistent"),
            # S3-only fields
            "description": s3.get("description", ""),
            "transcript": self._clean_transcript(s3.get("transcript", "")),
            "top_comments": top_comments,
        }

    def _find_video_in_s3(
        self, video_id: str, source_key: Optional[str]
    ) -> Optional[dict]:
        """
        Find a video dict from S3.
        If source_key is provided, fetch only that file.
        Otherwise fall back to scanning all files under the prefix.
        """
        if source_key:
            try:
                payload = self.get_json_object(source_key)
                for video in payload.get("videos", []):
                    if video.get("videoId", "") == video_id:
                        video["channel"] = payload.get("channel", "")
                        return video
            except (ClientError, json.JSONDecodeError, UnicodeDecodeError):
                pass
            return None

        # Full scan fallback
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=settings.s3_prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if not key or key.endswith("/"):
                    continue
                try:
                    payload = self.get_json_object(key)
                except (ClientError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                for video in payload.get("videos", []):
                    if video.get("videoId", "") == video_id:
                        video["channel"] = payload.get("channel", "")
                        return video
        return None

    def list_object_keys(self, prefix=None, limit=None):
        use_prefix = prefix if prefix is not None else settings.s3_prefix
        use_limit = limit if limit is not None else settings.s3_object_limit
        keys = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=use_prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if key and not key.endswith("/"):
                    keys.append(key)
                    if len(keys) >= use_limit:
                        return keys
        return keys

    def get_json_object(self, key):
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)

    def _clean_transcript(self, text):
        cleaned = text.strip()
        cleaned = re.sub(
            r"^\s*Kind:\s*captions\s+Language:\s*[a-zA-Z-]+\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def get_video_metadata(self, channel: str, video_id: str) -> dict:
        """
        DynamoDB join — fetches fresh viewCount, likeCount, commentCount
        for a given channel + videoId. Returns empty dict if not found.
        """
        try:
            table = self.dynamodb.Table(settings.dynamodb_table)
            response = table.get_item(
                Key={
                    "PartitionKey": channel,
                    "SortKey": video_id,
                }
            )
            item = response.get("Item")
            if not item:
                logger.warning(f"DYNAMO_MISS channel={channel} videoId={video_id}")
                return {}
            return {
                "view_count": int(item.get("viewCount", 0)),
                "like_count": int(item.get("likeCount", 0)),
                "comment_count": int(item.get("commentCount", 0)),
            }
        except ClientError as e:
            logger.error(f"DYNAMO_ERROR channel={channel} videoId={video_id} error={e}")
            return {}

    def extract_videos(self, payload: dict, source_key: str) -> list[dict]:
        """
        Extracts a list of video objects from an S3 payload.
        Each returned dict contains transcript + full video metadata.
        Skips videos with no transcript.
        """
        videos = []
        channel = payload.get("channel", "")

        raw_videos = payload.get("videos", [])
        if not isinstance(raw_videos, list):
            return videos

        for video in raw_videos:
            if not isinstance(video, dict):
                continue

            transcript_raw = video.get("transcript", "")
            if not isinstance(transcript_raw, str) or not transcript_raw.strip():
                continue

            transcript = self._clean_transcript(transcript_raw)
            if not transcript:
                continue

            video_id = video.get("videoId", "")

            # DynamoDB join for fresh counts
            fresh_stats = {}
            if channel and video_id:
                fresh_stats = self.get_video_metadata(channel, video_id)

            # Extract top comments (up to 5)
            top_comments = []
            for comment in video.get("topComments", [])[:5]:
                if isinstance(comment, dict):
                    top_comments.append(comment.get("text", ""))

            videos.append(
                {
                    "videoId": video_id,
                    "title": video.get("title", ""),
                    "channel": channel,
                    "published_at": video.get("publishedAt", ""),
                    "view_count": fresh_stats.get(
                        "view_count", int(video.get("viewCount", 0))
                    ),
                    "like_count": fresh_stats.get(
                        "like_count", int(video.get("likeCount", 0))
                    ),
                    "comment_count": fresh_stats.get(
                        "comment_count", int(video.get("commentCount", 0))
                    ),
                    "transcript": transcript,
                    "top_comments": top_comments,
                }
            )

        return videos

    def load_videos_from_prefix(self, prefix=None, limit=None) -> list[dict]:
        """
        Main entry point for the pipeline.
        Returns list of source objects, each with a list of enriched video dicts.
        """
        results = []
        keys = self.list_object_keys(prefix=prefix, limit=limit)

        for key in keys:
            try:
                payload = self.get_json_object(key)
                videos = self.extract_videos(payload, source_key=key)
                results.append(
                    {
                        "key": key,
                        "videos": videos,
                        "video_count": len(videos),
                    }
                )
                logger.info(f"S3_LOAD key={key} videos={len(videos)}")
            except (ClientError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                results.append(
                    {
                        "key": key,
                        "error": str(exc),
                        "videos": [],
                        "video_count": 0,
                    }
                )
                logger.error(f"S3_LOAD_ERROR key={key} error={exc}")

        return results

    # ── Optional / legacy ─────────────────────────────────────────────────────

    def extract_transcripts(self, payload):
        """Legacy method — returns plain transcript strings. Use extract_videos() instead."""
        transcripts = []
        if isinstance(payload, dict):
            direct = payload.get("transcript")
            if isinstance(direct, str) and direct.strip():
                transcripts.append(self._clean_transcript(direct))
            for video in payload.get("videos", []):
                if isinstance(video, dict):
                    t = video.get("transcript")
                    if isinstance(t, str) and t.strip():
                        transcripts.append(self._clean_transcript(t))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    t = item.get("transcript")
                    if isinstance(t, str) and t.strip():
                        transcripts.append(self._clean_transcript(t))
        seen, unique = set(), []
        for t in transcripts:
            if t and t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def load_transcripts_from_prefix(self, prefix=None, limit=None):
        """Legacy method — use load_videos_from_prefix() instead."""
        results = []
        keys = self.list_object_keys(prefix=prefix, limit=limit)
        for key in keys:
            try:
                payload = self.get_json_object(key)
                transcripts = self.extract_transcripts(payload)
                results.append(
                    {
                        "key": key,
                        "transcripts": transcripts,
                        "transcript_count": len(transcripts),
                    }
                )
            except (ClientError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                results.append(
                    {
                        "key": key,
                        "error": str(exc),
                        "transcripts": [],
                        "transcript_count": 0,
                    }
                )
        return results
