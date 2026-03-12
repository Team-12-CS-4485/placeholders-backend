import base64
"""
storage_service.py - S3 Storage Service

Handles all AWS S3 interactions for the pipeline:
- Lists object keys under a given S3 prefix with pagination
- Fetches and parses JSON objects from S3
- Extracts and deduplicates transcripts from flexible payload structures
  (supports both dict and list formats, nested under "videos" key or top-level)
- Cleans transcript text by stripping VTT caption metadata headers
"""

import json
import re
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings


class StorageService:
    def __init__(self, s3_client=None, bucket=None, dynamodb_resource=None):
        self.s3_client = s3_client or boto3.client("s3")
        self.bucket = bucket or settings.s3_bucket
        self.dynamodb = dynamodb_resource or boto3.resource("dynamodb", region_name=settings.aws_region)

    def list_videos(self, limit: int = 20, cursor: str = None):
        list_kwargs = {
            "Bucket": self.bucket,
            "Prefix": settings.s3_prefix,
            "MaxKeys": limit,
        }
        if cursor:
            list_kwargs["ContinuationToken"] = base64.b64decode(cursor.encode()).decode()

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
                items.append({
                    "video_id": video.get("videoId", ""),
                    "channel": channel,
                    "title": video.get("title", ""),
                    "description": video.get("description", ""),
                    "published_at": video.get("publishedAt", ""),
                    "view_count": int(video.get("viewCount", 0)),
                    "like_count": int(video.get("likeCount", 0)),
                    "comment_count": int(video.get("commentCount", 0)),
                })

        next_cursor = None
        if response.get("IsTruncated"):
            token = response.get("NextContinuationToken", "")
            next_cursor = base64.b64encode(token.encode()).decode()

        return {"items": items, "total_returned": len(items), "next_cursor": next_cursor}

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

    def extract_transcripts(self, payload):
        transcripts = []

        if isinstance(payload, dict):
            direct_value = payload.get("transcript")
            if isinstance(direct_value, str) and direct_value.strip():
                transcripts.append(self._clean_transcript(direct_value))

            videos = payload.get("videos")
            if isinstance(videos, list):
                for video in videos:
                    if not isinstance(video, dict):
                        continue
                    transcript = video.get("transcript")
                    if isinstance(transcript, str) and transcript.strip():
                        transcripts.append(self._clean_transcript(transcript))

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    transcript = item.get("transcript")
                    if isinstance(transcript, str) and transcript.strip():
                        transcripts.append(self._clean_transcript(transcript))

        unique_transcripts = []
        seen = set()
        for transcript in transcripts:
            if not transcript:
                continue
            if transcript not in seen:
                seen.add(transcript)
                unique_transcripts.append(transcript)

        return unique_transcripts

    def load_transcripts_from_prefix(self, prefix=None, limit=None):
        results = []
        keys = self.list_object_keys(prefix=prefix, limit=limit)

        for key in keys:
            try:
                payload = self.get_json_object(key)
                transcripts = self.extract_transcripts(payload)
                results.append({
                    "key": key,
                    "transcripts": transcripts,
                    "transcript_count": len(transcripts)
                })
            except (ClientError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                results.append({
                    "key": key,
                    "error": str(exc),
                    "transcripts": [],
                    "transcript_count": 0
                })

        return results
