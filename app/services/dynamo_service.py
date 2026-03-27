"""
dynamo_service.py - DynamoDB Read Service

Single abstraction for all API read paths. Replaces S3 (videos listing)
and Qdrant (trend aggregation) as the primary data source for read APIs.

Tables accessed:
  youtube-videos       PK=PartitionKey (channel)  SK=SortKey (videoId)
  narrative-clusters   PK=cluster_id

All DynamoDB numbers are returned as Python Decimal by boto3.
The _deserialize() helper normalises these to int/float and sets to lists
before returning data to callers.
"""

from __future__ import annotations

import base64
import json
import logging
from decimal import Decimal
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


class DynamoService:
    def __init__(self, dynamodb_resource=None):
        self._dynamodb = dynamodb_resource or boto3.resource(
            "dynamodb", region_name=settings.aws_region
        )
        self._videos_table = self._dynamodb.Table(settings.dynamodb_table)
        self._clusters_table = self._dynamodb.Table("narrative-clusters")

    # ── Decimal / type normalisation ─────────────────────────────────────────

    def _deserialize(self, obj):
        """Recursively convert Decimal → int/float and set → list."""
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        if isinstance(obj, dict):
            return {k: self._deserialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._deserialize(i) for i in obj]
        if isinstance(obj, set):
            return [self._deserialize(i) for i in sorted(obj)]
        return obj

    # ── Videos ───────────────────────────────────────────────────────────────

    def scan_videos(
        self,
        limit: int = 20,
        cursor: str = None,
        week: str = None,
    ) -> dict:
        """
        Paginated scan of youtube-videos table.

        cursor  — base64-encoded JSON of DynamoDB LastEvaluatedKey dict.
        week    — optional filter (e.g. "week1") on the `week` attribute.

        Returns:
            {items: [VideoItem...], total_returned: int, next_cursor: str|None}
        """
        # Normalise ?week=2 → "week2"
        if week and week.isdigit():
            week = f"week{week}"

        scan_kwargs: dict = {"Limit": limit}

        if cursor:
            try:
                raw = base64.b64decode(cursor.encode()).decode()
                scan_kwargs["ExclusiveStartKey"] = json.loads(raw)
            except Exception:
                logger.warning(f"DYNAMO_SCAN_BAD_CURSOR cursor={cursor!r}")

        if week:
            scan_kwargs["FilterExpression"] = Attr("week").eq(week)

        try:
            response = self._videos_table.scan(**scan_kwargs)
        except ClientError as exc:
            logger.error(f"DYNAMO_SCAN_VIDEOS_ERROR error={exc}")
            return {"items": [], "total_returned": 0, "next_cursor": None}

        items = []
        for raw in response.get("Items", []):
            item = self._deserialize(raw)
            items.append(
                {
                    "video_id": item.get("SortKey", ""),
                    "channel": item.get("channel") or item.get("PartitionKey", ""),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "published_at": item.get("publishedAt", ""),
                    "view_count": int(item.get("viewCount") or 0),
                    "like_count": int(item.get("likeCount") or 0),
                    "comment_count": int(item.get("commentCount") or 0),
                }
            )

        next_cursor = None
        last_key = response.get("LastEvaluatedKey")
        if last_key:
            clean_key = self._deserialize(last_key)
            next_cursor = base64.b64encode(json.dumps(clean_key).encode()).decode()

        logger.info(f"DYNAMO_SCAN_VIDEOS returned={len(items)} week={week}")
        return {
            "items": items,
            "total_returned": len(items),
            "next_cursor": next_cursor,
        }

    # ── Clusters ─────────────────────────────────────────────────────────────

    def get_all_clusters(self) -> list[dict]:
        """
        Full scan of narrative-clusters (small table, no pagination needed).
        Returns list of deserialised cluster dicts.
        """
        items = []
        scan_kwargs: dict = {}

        while True:
            try:
                response = self._clusters_table.scan(**scan_kwargs)
            except ClientError as exc:
                logger.error(f"DYNAMO_SCAN_CLUSTERS_ERROR error={exc}")
                break

            for raw in response.get("Items", []):
                items.append(self._deserialize(raw))

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key

        logger.info(f"DYNAMO_SCAN_CLUSTERS returned={len(items)}")
        return items

    def get_cluster(self, cluster_id: int) -> dict:
        """
        Single get_item on narrative-clusters by cluster_id PK.
        Raises KeyError if not found.
        """
        try:
            response = self._clusters_table.get_item(
                Key={"cluster_id": Decimal(str(cluster_id))}
            )
        except ClientError as exc:
            logger.error(
                f"DYNAMO_GET_CLUSTER_ERROR cluster_id={cluster_id} error={exc}"
            )
            raise KeyError(cluster_id)

        item = response.get("Item")
        if not item:
            logger.warning(f"DYNAMO_CLUSTER_MISS cluster_id={cluster_id}")
            raise KeyError(cluster_id)

        return self._deserialize(item)
