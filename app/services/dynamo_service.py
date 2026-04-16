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
from datetime import date
from decimal import Decimal
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Attr, Key
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
        self._cluster_weeks_table = self._dynamodb.Table("cluster-weeks")

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

    def get_videos_by_channel(
        self,
        channel_name: str,
        limit: int = 20,
        cursor: str = None,
    ) -> dict:
        """
        Query youtube-videos by PartitionKey (channel name).
        Uses native DynamoDB key-based pagination.
        """
        query_kwargs: dict = {
            "KeyConditionExpression": Key("PartitionKey").eq(channel_name),
            "Limit": limit,
        }

        if cursor:
            try:
                raw = base64.b64decode(cursor.encode()).decode()
                query_kwargs["ExclusiveStartKey"] = json.loads(raw)
            except Exception:
                logger.warning(f"DYNAMO_QUERY_BAD_CURSOR cursor={cursor!r}")

        try:
            response = self._videos_table.query(**query_kwargs)
        except ClientError as exc:
            logger.error(
                f"DYNAMO_QUERY_CHANNEL_ERROR channel={channel_name} error={exc}"
            )
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

        logger.info(
            f"DYNAMO_QUERY_CHANNEL channel={channel_name} returned={len(items)}"
        )
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

        logger.debug(f"DYNAMO_SCAN_CLUSTERS returned={len(items)}")
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

    # ── Video titles by cluster ─────────────────────────────────────────────

    def get_video_titles_for_cluster(
        self, cluster_id: int, week: str | None = None, limit: int = 6
    ) -> list[str]:
        """
        Pull video titles from youtube-videos via the cluster-index GSI.
        If week is provided, filters to only that week's videos.
        Returns empty list gracefully if the GSI doesn't exist.
        """
        try:
            kwargs: dict = {
                "IndexName": "cluster-index",
                "KeyConditionExpression": Key("cluster_id").eq(
                    Decimal(str(cluster_id))
                ),
                "ProjectionExpression": "#t, #wk",
                "ExpressionAttributeNames": {"#t": "title", "#wk": "week"},
            }
            if week:
                kwargs["FilterExpression"] = Attr("week").eq(week)
            else:
                kwargs["Limit"] = limit

            items = []
            while True:
                resp = self._videos_table.query(**kwargs)
                items.extend(resp.get("Items", []))
                if not week or "LastEvaluatedKey" not in resp:
                    break
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

            titles = [item["title"] for item in items if item.get("title")]
            return titles[:limit]
        except Exception as exc:
            logger.warning(
                f"DYNAMO_VIDEO_TITLES_WARN cluster={cluster_id} " f"error={exc}"
            )
            return []

    def get_clusters_for_week(self, week: str) -> list[dict]:
        """
        Scan cluster-weeks for all items matching a given week.
        Returns deserialized items (cluster_id, narrative_headline, etc.).
        """
        try:
            items = []
            scan_kwargs: dict = {"FilterExpression": Attr("week").eq(week)}
            while True:
                resp = self._cluster_weeks_table.scan(**scan_kwargs)
                for raw in resp.get("Items", []):
                    items.append(self._deserialize(raw))
                if "LastEvaluatedKey" not in resp:
                    break
                scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            logger.debug(f"CLUSTER_WEEKS_SCAN week={week} found={len(items)}")
            return items
        except Exception as exc:
            logger.warning(f"CLUSTER_WEEKS_SCAN_WARN week={week} error={exc}")
            return []

    def get_video_data_for_cluster(
        self, cluster_id: int, week: str, limit: int = 10
    ) -> list[dict]:
        """
        Pull title + key_claims for videos in a given cluster+week via cluster-index GSI.
        Returns up to `limit` results as [{"title": ..., "key_claims": [...]}].
        """
        try:
            kwargs: dict = {
                "IndexName": "cluster-index",
                "KeyConditionExpression": Key("cluster_id").eq(
                    Decimal(str(cluster_id))
                ),
                "FilterExpression": Attr("week").eq(week),
                "ProjectionExpression": "#t, key_claims, #wk",
                "ExpressionAttributeNames": {"#t": "title", "#wk": "week"},
            }
            items = []
            while True:
                resp = self._videos_table.query(**kwargs)
                items.extend(resp.get("Items", []))
                if "LastEvaluatedKey" not in resp:
                    break
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

            results = []
            for item in items[:limit]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "key_claims": list(item.get("key_claims", [])),
                    }
                )
            return results
        except Exception as exc:
            logger.warning(
                f"DYNAMO_VIDEO_DATA_WARN cluster={cluster_id} week={week} error={exc}"
            )
            return []

    def save_cluster_week_snapshot(
        self, cluster_id: int, week: str, data: dict
    ) -> None:
        """
        Write a cluster+week snapshot to cluster-weeks using update_item.

        narrative_headline, narrative_summary, and created_at are only set if
        they do not already exist (if_not_exists) so that manual edits and
        prior runs are never clobbered. All other fields are always overwritten.
        """
        # Fields that must never be overwritten once set
        protected = {"narrative_headline", "narrative_summary", "created_at"}

        update_parts = []
        expr_names: dict = {}
        expr_values: dict = {}

        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, float):
                v = Decimal(str(round(v, 4)))
            elif isinstance(v, int) and not isinstance(v, bool):
                v = Decimal(str(v))

            safe_name = f"#f_{k}"
            val_key = f":v_{k}"
            expr_names[safe_name] = k
            expr_values[val_key] = v

            if k in protected:
                update_parts.append(
                    f"{safe_name} = if_not_exists({safe_name}, {val_key})"
                )
            else:
                update_parts.append(f"{safe_name} = {val_key}")

        if not update_parts:
            return

        try:
            self._cluster_weeks_table.update_item(
                Key={
                    "cluster_id": Decimal(str(cluster_id)),
                    "week": week,
                },
                UpdateExpression="SET " + ", ".join(update_parts),
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
            )
        except Exception as exc:
            logger.warning(
                f"CLUSTER_WEEK_SAVE_WARN cluster={cluster_id} week={week} error={exc}"
            )

    def get_cluster_week_row(self, cluster_id: int, week: str) -> dict | None:
        """Fetch a single cluster-weeks row by (cluster_id, week)."""
        try:
            resp = self._cluster_weeks_table.get_item(
                Key={"cluster_id": Decimal(str(cluster_id)), "week": week}
            )
            return resp.get("Item")
        except Exception as exc:
            logger.warning(
                f"CLUSTER_WEEK_ROW_WARN cluster={cluster_id} week={week} error={exc}"
            )
            return None

    def get_all_cluster_week_headlines(self, cluster_id: int) -> dict[str, str]:
        """
        Query cluster-weeks for all weeks of a given cluster.
        Returns {week_name: narrative_headline} for every week that has a headline.
        Used to enrich week_data in detail responses.
        """
        try:
            kwargs: dict = {
                "KeyConditionExpression": Key("cluster_id").eq(
                    Decimal(str(cluster_id))
                ),
                "ProjectionExpression": "#wk, narrative_headline",
                "ExpressionAttributeNames": {"#wk": "week"},
            }
            items = []
            while True:
                resp = self._cluster_weeks_table.query(**kwargs)
                items.extend(resp.get("Items", []))
                if "LastEvaluatedKey" not in resp:
                    break
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            return {
                i["week"]: i.get("narrative_headline", "")
                for i in items
                if i.get("narrative_headline")
            }
        except Exception as exc:
            logger.warning(
                f"CLUSTER_WEEK_ALL_HEADLINES_WARN cluster={cluster_id} error={exc}"
            )
            return {}

    def get_cluster_week_headlines(
        self, cluster_id: int, last_n: int = 4
    ) -> list[dict]:
        """
        Query cluster-weeks for all weeks of a given cluster_id.
        Returns the last `last_n` entries sorted oldest → newest as
        [{"week": "week3", "headline": "..."}].
        """
        try:
            kwargs: dict = {
                "KeyConditionExpression": Key("cluster_id").eq(
                    Decimal(str(cluster_id))
                ),
                "ProjectionExpression": "#wk, narrative_headline",
                "ExpressionAttributeNames": {"#wk": "week"},
            }
            items = []
            while True:
                resp = self._cluster_weeks_table.query(**kwargs)
                items.extend(resp.get("Items", []))
                if "LastEvaluatedKey" not in resp:
                    break
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

            # Sort by week number
            def _week_num(item):
                wk = item.get("week", "")
                return (
                    int(wk[4:]) if wk.startswith("week") and wk[4:].isdigit() else 9999
                )

            items.sort(key=_week_num)
            return [
                {"week": i["week"], "headline": i.get("narrative_headline", "")}
                for i in items
                if i.get("narrative_headline")
            ][-last_n:]
        except Exception as exc:
            logger.warning(
                f"CLUSTER_WEEK_HEADLINES_WARN cluster={cluster_id} error={exc}"
            )
            return []

    # ── Articles ────────────────────────────────────────────────────────────

    @staticmethod
    def _week_start_date(week_number: int) -> str:
        """
        Return the ISO date of the Monday that starts a project week.
        Project week N = ISO week (N + 9) of 2026.
        e.g. week1 → 2026-03-02, week6 → 2026-04-06
        Returns "" for invalid week numbers.
        """
        try:
            iso_week = week_number + 9
            return date.fromisocalendar(2026, iso_week, 1).isoformat()
        except Exception:
            return ""

    def _articles_table(self):
        return self._dynamodb.Table("articles")

    def article_exists(self, cluster_id: int, week_number: int) -> bool:
        """Check if an article already exists for cluster + week."""
        scan_kwargs: dict = {
            "FilterExpression": (
                Attr("cluster_id").eq(cluster_id) & Attr("week_number").eq(week_number)
            ),
            "ProjectionExpression": "article_id",
        }
        try:
            while True:
                resp = self._articles_table().scan(**scan_kwargs)
                if resp.get("Items"):
                    return True
                if "LastEvaluatedKey" not in resp:
                    break
                scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            return False
        except Exception as exc:
            logger.warning(f"ARTICLE_EXISTS_CHECK_WARN error={exc}")
            return False

    def save_article(self, item: dict) -> None:
        """Put an article item into the articles table."""
        self._articles_table().put_item(Item=item)

    def delete_articles(self, cluster_id: int, week_number: int) -> None:
        """Remove existing articles for a cluster + week."""
        try:
            resp = self._articles_table().scan(
                FilterExpression=(
                    Attr("cluster_id").eq(Decimal(str(cluster_id)))
                    & Attr("week_number").eq(Decimal(str(week_number)))
                ),
                ProjectionExpression="article_id",
            )
            for item in resp.get("Items", []):
                self._articles_table().delete_item(
                    Key={"article_id": item["article_id"]}
                )
        except Exception as exc:
            logger.warning(f"ARTICLE_DELETE_WARN cluster={cluster_id} error={exc}")

    def get_articles(
        self,
        cluster_id: Optional[int] = None,
        week_number: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List articles, optionally filtered. Returns metadata (no body)."""
        filter_parts = []
        if cluster_id is not None:
            filter_parts.append(Attr("cluster_id").eq(cluster_id))
        if week_number is not None:
            filter_parts.append(Attr("week_number").eq(week_number))

        scan_kwargs: dict = {
            "ProjectionExpression": (
                "article_id, cluster_id, cluster_label, "
                "week_number, #wk, title, overview, created_at"
            ),
            "ExpressionAttributeNames": {"#wk": "week"},
        }

        if filter_parts:
            expr = filter_parts[0]
            for part in filter_parts[1:]:
                expr = expr & part
            scan_kwargs["FilterExpression"] = expr

        items: list[dict] = []
        while True:
            try:
                resp = self._articles_table().scan(**scan_kwargs)
            except ClientError as exc:
                logger.error(f"ARTICLE_LIST_ERROR error={exc}")
                break

            for raw in resp.get("Items", []):
                item = self._deserialize(raw)
                wn = int(item.get("week_number", 0))
                items.append(
                    {
                        "article_id": item.get("article_id", ""),
                        "cluster_id": int(item.get("cluster_id", 0)),
                        "cluster_label": item.get("cluster_label", ""),
                        "week_number": wn,
                        "week_start_date": self._week_start_date(wn),
                        "title": item.get("title", ""),
                        "overview": item.get("overview", ""),
                        "created_at": item.get("created_at", ""),
                    }
                )
                if len(items) >= limit:
                    break

            if len(items) >= limit or "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        items.sort(key=lambda x: (-x["week_number"], x["cluster_id"]))
        return items[:limit]

    def get_article_by_id(self, article_id: str) -> Optional[dict]:
        """Fetch a single article (including body) by article_id."""
        try:
            resp = self._articles_table().get_item(Key={"article_id": article_id})
        except ClientError as exc:
            logger.error(f"ARTICLE_GET_ERROR id={article_id} error={exc}")
            return None

        item = resp.get("Item")
        if not item:
            return None

        item = self._deserialize(item)
        wn = int(item.get("week_number", 0))
        return {
            "article_id": item.get("article_id", ""),
            "cluster_id": int(item.get("cluster_id", 0)),
            "cluster_label": item.get("cluster_label", ""),
            "week_number": wn,
            "week_start_date": self._week_start_date(wn),
            "title": item.get("title", ""),
            "overview": item.get("overview", ""),
            "body": item.get("body", ""),
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
        }
