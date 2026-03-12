import base64
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
        table = self.dynamodb.Table(settings.dynamodb_table)
        scan_kwargs = {
            "Limit": limit,
            "ProjectionExpression": "SortKey, channel, title, description, publishedAt, viewCount, likeCount, commentCount",
        }
        if cursor:
            scan_kwargs["ExclusiveStartKey"] = json.loads(
                base64.b64decode(cursor.encode()).decode()
            )

        response = table.scan(**scan_kwargs)

        items = [
            {
                "video_id": item.get("SortKey", ""),
                "channel": item.get("channel", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "published_at": item.get("publishedAt", ""),
                "view_count": int(item.get("viewCount", 0)),
                "like_count": int(item.get("likeCount", 0)),
                "comment_count": int(item.get("commentCount", 0)),
            }
            for item in response.get("Items", [])
        ]

        next_cursor = None
        last_key = response.get("LastEvaluatedKey")
        if last_key:
            next_cursor = base64.b64encode(json.dumps(last_key).encode()).decode()

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
