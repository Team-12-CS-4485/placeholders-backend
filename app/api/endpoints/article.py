"""
article.py - Pydantic schemas for Article API responses
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── List view (no body) ───────────────────────────────────────────────────────


class ArticleListItem(BaseModel):
    article_id: str
    cluster_id: int
    cluster_label: str
    week_number: int
    title: str
    overview: str
    created_at: str


# ── Full detail (includes body) ───────────────────────────────────────────────


class ArticleDetail(ArticleListItem):
    body: str
    updated_at: Optional[str] = None


# ── List response ─────────────────────────────────────────────────────────────


class ArticleListResponse(BaseModel):
    articles: list[ArticleListItem]
    total: int


# ── Generation request / response ─────────────────────────────────────────────


class ArticleGenerationRequest(BaseModel):
    week: Optional[str] = Field(
        default=None,
        description="Specific week to generate articles for (e.g. 'week1' or '1'). "
        "If omitted, all weeks are processed.",
    )
    force: bool = Field(
        default=False,
        description="Overwrite existing articles for the given cluster + week.",
    )
    cluster_id: Optional[int] = Field(
        default=None,
        description="Only generate an article for this specific cluster.",
    )


class ArticleGenerationSummary(BaseModel):
    articles_generated: int
    articles_skipped: int
    articles_failed: int
    weeks_processed: list[str]
