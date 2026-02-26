import json
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings


class StorageService:
    def __init__(self, s3_client=None, bucket=None):
        self.s3_client = s3_client or boto3.client("s3")
        self.bucket = bucket or settings.s3_bucket

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

    def extract_transcripts(self, payload):
        transcripts = []

        if isinstance(payload, dict):
            direct_value = payload.get("transcript")
            if isinstance(direct_value, str) and direct_value.strip():
                transcripts.append(direct_value.strip())

            videos = payload.get("videos")
            if isinstance(videos, list):
                for video in videos:
                    if not isinstance(video, dict):
                        continue
                    transcript = video.get("transcript")
                    if isinstance(transcript, str) and transcript.strip():
                        transcripts.append(transcript.strip())

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    transcript = item.get("transcript")
                    if isinstance(transcript, str) and transcript.strip():
                        transcripts.append(transcript.strip())

        unique_transcripts = []
        seen = set()
        for transcript in transcripts:
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
