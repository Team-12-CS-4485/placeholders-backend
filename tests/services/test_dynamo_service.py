"""
Unit tests for DynamoService helper methods.
Tests pure/local logic — no real DynamoDB calls.
"""

from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.services.dynamo_service import DynamoService


@pytest.fixture
def svc():
    mock_resource = MagicMock()
    return DynamoService(dynamodb_resource=mock_resource)


# ── _deserialize ─────────────────────────────────────────────────────────────


def test_deserialize_decimal_int(svc):
    assert svc._deserialize(Decimal("5")) == 5
    assert isinstance(svc._deserialize(Decimal("5")), int)


def test_deserialize_decimal_float(svc):
    result = svc._deserialize(Decimal("3.14"))
    assert abs(result - 3.14) < 0.001
    assert isinstance(result, float)


def test_deserialize_nested(svc):
    raw = {"count": Decimal("10"), "score": Decimal("0.5"), "tags": ["a", "b"]}
    result = svc._deserialize(raw)
    assert result == {"count": 10, "score": 0.5, "tags": ["a", "b"]}


def test_deserialize_set_to_list(svc):
    result = svc._deserialize({Decimal("1"), Decimal("2"), Decimal("3")})
    assert sorted(result) == [1, 2, 3]


def test_deserialize_passthrough_str(svc):
    assert svc._deserialize("hello") == "hello"


def test_deserialize_passthrough_bool(svc):
    assert svc._deserialize(True) is True
    assert svc._deserialize(False) is False


# ── map_video_item ────────────────────────────────────────────────────────────


def test_map_video_item_basic_fields(svc):
    raw = {
        "SortKey": "abc123",
        "PartitionKey": "CNBC",
        "channel": "CNBC",
        "title": "Market Update",
        "publishedAt": "2026-03-10T12:00:00+00:00",
        "viewCount": 142000,
        "likeCount": 3200,
        "commentCount": 410,
        "week": "week1",
    }
    item = svc.map_video_item(raw)
    assert item["video_id"] == "abc123"
    assert item["channel"] == "CNBC"
    assert item["title"] == "Market Update"
    assert item["view_count"] == 142000
    assert item["like_count"] == 3200
    assert item["comment_count"] == 410
    assert item["week"] == "week1"


def test_map_video_item_thumbnail_url_derived(svc):
    raw = {"SortKey": "abc123", "viewCount": 0, "likeCount": 0, "commentCount": 0}
    item = svc.map_video_item(raw)
    assert (
        item["thumbnail_url"] == "https://img.youtube.com/vi/abc123/maxresdefault.jpg"
    )


def test_map_video_item_thumbnail_url_none_when_no_video_id(svc):
    raw = {"SortKey": "", "viewCount": 0, "likeCount": 0, "commentCount": 0}
    item = svc.map_video_item(raw)
    assert item["thumbnail_url"] is None


def test_map_video_item_intelligence_fields(svc):
    raw = {
        "SortKey": "abc123",
        "viewCount": 0,
        "likeCount": 0,
        "commentCount": 0,
        "topics": ["inflation", "fed"],
        "category": "Wall Street & Markets",
        "sentiment": "negative",
        "key_claims": ["Fed holds rates"],
        "is_breaking": True,
        "cluster_id": 4,
        "cluster_label": "Federal Reserve Policy",
    }
    item = svc.map_video_item(raw)
    assert item["topics"] == ["inflation", "fed"]
    assert item["category"] == "Wall Street & Markets"
    assert item["sentiment"] == "negative"
    assert item["key_claims"] == ["Fed holds rates"]
    assert item["is_breaking"] is True
    assert item["cluster_id"] == 4
    assert item["cluster_label"] == "Federal Reserve Policy"


def test_map_video_item_thumbnail_analysis_fields(svc):
    raw = {
        "SortKey": "abc123",
        "viewCount": 0,
        "likeCount": 0,
        "commentCount": 0,
        "thumbnail_tone": "urgency",
        "thumbnail_clickbait_score": 3,
        "thumbnail_insight": "Bold overlay",
        "thumbnail_brand_consistent": True,
    }
    item = svc.map_video_item(raw)
    assert item["thumbnail_tone"] == "urgency"
    assert item["thumbnail_clickbait_score"] == 3
    assert item["thumbnail_insight"] == "Bold overlay"
    assert item["thumbnail_brand_consistent"] is True


def test_map_video_item_missing_fields_default_to_none(svc):
    raw = {"SortKey": "abc123", "viewCount": 0, "likeCount": 0, "commentCount": 0}
    item = svc.map_video_item(raw)
    assert item["topics"] is None
    assert item["category"] is None
    assert item["sentiment"] is None
    assert item["cluster_id"] is None
    assert item["thumbnail_tone"] is None


def test_map_video_item_channel_fallback_to_partition_key(svc):
    raw = {
        "SortKey": "vid1",
        "PartitionKey": "BBC",
        "viewCount": 0,
        "likeCount": 0,
        "commentCount": 0,
    }
    item = svc.map_video_item(raw)
    assert item["channel"] == "BBC"


# ── _week_start_date ──────────────────────────────────────────────────────────


def test_week_start_date_week1():
    # week1 = ISO week 10 of 2026 = Monday 2026-03-02
    result = DynamoService._week_start_date(1)
    assert result == "2026-03-02"


def test_week_start_date_week6():
    # week6 = ISO week 15 of 2026 = Monday 2026-04-06
    result = DynamoService._week_start_date(6)
    assert result == "2026-04-06"


def test_week_start_date_invalid():
    result = DynamoService._week_start_date(9999)
    assert result == ""
