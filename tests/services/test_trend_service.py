"""
Unit tests for TrendService pure helper methods.
No DynamoDB or network calls — all inputs passed directly.
"""

from collections import Counter
from unittest.mock import patch, MagicMock

import pytest

with patch("app.services.pipeline_service.PipelineService"), patch(
    "app.services.storage_service.StorageService"
):
    from app.services.trend_service import TrendService


@pytest.fixture
def svc():
    mock_dynamo = MagicMock()
    return TrendService(dynamo_service=mock_dynamo)


# ── _compute_heat_score ───────────────────────────────────────────────────────


def test_heat_score_zero(svc):
    assert svc._compute_heat_score(0, 0, 0) == 0.0


def test_heat_score_channels_dominate(svc):
    score = svc._compute_heat_score(channel_count=10, breaking_count=0, total_views=0)
    assert score == 30.0


def test_heat_score_breaking_weight(svc):
    score = svc._compute_heat_score(channel_count=0, breaking_count=5, total_views=0)
    assert score == 10.0


def test_heat_score_views_weight(svc):
    score = svc._compute_heat_score(
        channel_count=0, breaking_count=0, total_views=1_000_000
    )
    assert score == 10.0


def test_heat_score_combined(svc):
    score = svc._compute_heat_score(
        channel_count=7, breaking_count=3, total_views=500_000
    )
    # 7*3 + 3*2 + 500000/100000 = 21 + 6 + 5 = 32
    assert score == 32.0


# ── _compute_sentiment_label ─────────────────────────────────────────────────


def test_sentiment_label_empty(svc):
    assert svc._compute_sentiment_label(Counter()) == "Neutral"


def test_sentiment_label_mostly_negative(svc):
    label = svc._compute_sentiment_label(Counter({"negative": 8, "neutral": 2}))
    assert label == "Negative"


def test_sentiment_label_mostly_positive(svc):
    label = svc._compute_sentiment_label(Counter({"positive": 6, "neutral": 4}))
    assert label == "Positive"


def test_sentiment_label_neutral_dominant(svc):
    label = svc._compute_sentiment_label(Counter({"neutral": 7, "negative": 3}))
    assert label == "Neutral"


def test_sentiment_label_polarized(svc):
    label = svc._compute_sentiment_label(
        Counter({"negative": 5, "positive": 4, "neutral": 1})
    )
    assert label == "Polarized"


# ── _classify_trend ───────────────────────────────────────────────────────────


def test_classify_trend_no_data(svc):
    trend_type, badge = svc._classify_trend([], 3, 100000, Counter())
    assert trend_type == "holding"


def test_classify_trend_single_week_emerging(svc):
    week_data = [{"week": "week1", "video_count": 5}]
    trend_type, badge = svc._classify_trend(week_data, 3, 100000, Counter())
    assert trend_type == "emerging"
    assert badge == "New"


def test_classify_trend_fading(svc):
    week_data = [
        {"week": "week1", "video_count": 10},
        {"week": "week2", "video_count": 0},
    ]
    trend_type, badge = svc._classify_trend(week_data, 3, 100000, Counter())
    assert trend_type == "fading"
    assert badge == "Fading"


def test_classify_trend_surging(svc):
    week_data = [
        {"week": "week1", "video_count": 5},
        {"week": "week2", "video_count": 10},
    ]
    trend_type, badge = svc._classify_trend(week_data, 3, 100000, Counter())
    assert trend_type == "surging"


def test_classify_trend_dominant_high_channel_count(svc):
    week_data = [
        {"week": "week1", "video_count": 10},
        {"week": "week2", "video_count": 11},  # +10% — not surging threshold
    ]
    trend_type, badge = svc._classify_trend(week_data, 7, 1000000, Counter())
    assert trend_type == "dominant"
    assert badge == "High Impact"


def test_classify_trend_new_cluster_from_zero(svc):
    week_data = [
        {"week": "week1", "video_count": 0},
        {"week": "week2", "video_count": 5},
    ]
    trend_type, badge = svc._classify_trend(week_data, 3, 50000, Counter())
    assert trend_type == "emerging"
    assert badge == "New"


# ── _compute_recent_sentiment_label ──────────────────────────────────────────


def test_recent_sentiment_label_no_weeks(svc):
    label = svc._compute_recent_sentiment_label([], "Negative")
    assert label == "Negative"


def test_recent_sentiment_label_reversal(svc):
    week_data = [
        {
            "week": "week1",
            "sentiment_breakdown": {"positive": 8, "negative": 2},
        },
        {
            "week": "week2",
            "sentiment_breakdown": {"negative": 8, "positive": 2},
        },
    ]
    label = svc._compute_recent_sentiment_label(week_data, "Negative")
    assert "Reversal" in label
    assert "Negative" in label


def test_recent_sentiment_label_shift(svc):
    week_data = [
        {
            "week": "week1",
            "sentiment_breakdown": {"positive": 5, "negative": 5},  # Polarized
        },
        {
            "week": "week2",
            "sentiment_breakdown": {"negative": 9, "neutral": 1},
        },
    ]
    label = svc._compute_recent_sentiment_label(week_data, "Negative")
    assert "Shift" in label or "Negative" in label
