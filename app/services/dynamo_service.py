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
        cluster_id: int = None,
    ) -> dict:
        """
        Paginated scan of youtube-videos table.

        When filters (week, cluster_id) are active, DynamoDB's Limit applies
        before FilterExpression — so a Limit=20 scan may return 0 results even
        when matching items exist. To fix this, filtered scans exhaust all
        DynamoDB pages first, then paginate in Python. The cursor for filtered
        scans is a base64-encoded integer offset.

        Unfiltered scans use native DynamoDB key-based pagination (cursor is a
        base64-encoded LastEvaluatedKey JSON).

        Returns:
            {items: [VideoItem...], total_returned: int, next_cursor: str|None}
        """
        # Normalise ?week=2 → "week2"
        if week and week.isdigit():
            week = f"week{week}"

        # Build filter expression
        filter_expr = None
        if week:
            filter_expr = Attr("week").eq(week)
        if cluster_id is not None:
            cluster_filter = Attr("cluster_id").eq(cluster_id)
            filter_expr = (
                filter_expr & cluster_filter if filter_expr else cluster_filter
            )

        if filter_expr is not None:
            return self._scan_videos_filtered(
                limit, cursor, filter_expr, week, cluster_id
            )
        return self._scan_videos_unfiltered(limit, cursor)

    def _scan_videos_filtered(
        self,
        limit: int,
        cursor: str,
        filter_expr,
        week,
        cluster_id,
    ) -> dict:
        """
        Filtered scan: exhaust all DynamoDB pages, then slice in Python.
        Cursor is a base64-encoded integer offset into the full result set.
        """
        offset = 0
        if cursor:
            try:
                offset = int(base64.b64decode(cursor.encode()).decode())
            except Exception:
                logger.warning(f"DYNAMO_SCAN_BAD_CURSOR cursor={cursor!r}")

        all_items = []
        scan_kwargs: dict = {"FilterExpression": filter_expr}

        try:
            while True:
                response = self._videos_table.scan(**scan_kwargs)
                for raw in response.get("Items", []):
                    all_items.append(self.map_video_item(self._deserialize(raw)))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                scan_kwargs["ExclusiveStartKey"] = last_key
        except ClientError as exc:
            logger.error(f"DYNAMO_SCAN_VIDEOS_ERROR error={exc}")
            return {"items": [], "total_returned": 0, "next_cursor": None}

        page = all_items[offset : offset + limit]
        next_offset = offset + limit
        next_cursor = (
            base64.b64encode(str(next_offset).encode()).decode()
            if next_offset < len(all_items)
            else None
        )

        logger.info(
            f"DYNAMO_SCAN_VIDEOS_FILTERED total={len(all_items)} "
            f"offset={offset} returned={len(page)} week={week} cluster_id={cluster_id}"
        )
        return {"items": page, "total_returned": len(page), "next_cursor": next_cursor}

    def _scan_videos_unfiltered(self, limit: int, cursor: str) -> dict:
        """
        Unfiltered scan: native DynamoDB key-based pagination.
        Cursor is a base64-encoded LastEvaluatedKey JSON dict.
        """
        scan_kwargs: dict = {"Limit": limit}

        if cursor:
            try:
                raw = base64.b64decode(cursor.encode()).decode()
                scan_kwargs["ExclusiveStartKey"] = json.loads(raw)
            except Exception:
                logger.warning(f"DYNAMO_SCAN_BAD_CURSOR cursor={cursor!r}")

        try:
            response = self._videos_table.scan(**scan_kwargs)
        except ClientError as exc:
            logger.error(f"DYNAMO_SCAN_VIDEOS_ERROR error={exc}")
            return {"items": [], "total_returned": 0, "next_cursor": None}

        items = [
            self.map_video_item(self._deserialize(raw))
            for raw in response.get("Items", [])
        ]

        next_cursor = None
        last_key = response.get("LastEvaluatedKey")
        if last_key:
            clean_key = self._deserialize(last_key)
            next_cursor = base64.b64encode(json.dumps(clean_key).encode()).decode()

        logger.info(f"DYNAMO_SCAN_VIDEOS returned={len(items)}")
        return {
            "items": items,
            "total_returned": len(items),
            "next_cursor": next_cursor,
        }

    def map_video_item(self, item: dict) -> dict:
        """Map a deserialized youtube-videos DynamoDB item to an API-ready dict."""
        video_id = item.get("SortKey", "")
        thumbnail_url = (
            f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            if video_id
            else None
        )
        return {
            "video_id": video_id,
            "channel": item.get("channel") or item.get("PartitionKey", ""),
            "title": item.get("title", ""),
            "published_at": item.get("publishedAt", ""),
            "view_count": int(item.get("viewCount") or 0),
            "like_count": int(item.get("likeCount") or 0),
            "comment_count": int(item.get("commentCount") or 0),
            "week": item.get("week"),
            "topics": item.get("topics") or None,
            "category": item.get("category") or None,
            "sentiment": item.get("sentiment") or None,
            "key_claims": item.get("key_claims") or None,
            "is_breaking": item.get("is_breaking"),
            "cluster_id": item.get("cluster_id"),
            "cluster_label": item.get("cluster_label") or None,
            "thumbnail_url": thumbnail_url,
            "thumbnail_tone": item.get("thumbnail_tone") or None,
            "thumbnail_clickbait_score": item.get("thumbnail_clickbait_score"),
            "thumbnail_insight": item.get("thumbnail_insight") or None,
            "thumbnail_brand_consistent": item.get("thumbnail_brand_consistent"),
        }

    def get_video_by_id(self, video_id: str) -> Optional[dict]:
        """
        Scan youtube-videos for a single video by SortKey (videoId).
        Returns the full deserialized item dict, or None if not found.
        """
        scan_kwargs: dict = {"FilterExpression": Attr("SortKey").eq(video_id)}
        try:
            while True:
                response = self._videos_table.scan(**scan_kwargs)
                items = response.get("Items", [])
                if items:
                    return self._deserialize(items[0])
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                scan_kwargs["ExclusiveStartKey"] = last_key
        except ClientError as exc:
            logger.error(f"DYNAMO_GET_VIDEO_ERROR video_id={video_id} error={exc}")
            return None

        return None

    # ── Clusters ─────────────────────────────────────────────────────────────

    def get_all_clusters(self, include_inactive: bool = False) -> list[dict]:
        """
        Full scan of narrative-clusters (small table, no pagination needed).
        Returns list of deserialised cluster dicts.
        Filters out inactive clusters by default.
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
                item = self._deserialize(raw)
                if not include_inactive and item.get("status") == "inactive":
                    continue
                items.append(item)

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
