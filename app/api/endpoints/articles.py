"""
articles.py - Article API Endpoints

POST /api/articles/generate              → trigger article generation
GET  /api/articles                        → list all articles (no body)
GET  /api/articles/{article_id}          → full article detail (with body)
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.article import (
    ArticleGenerationRequest,
    ArticleGenerationSummary,
    ArticleDetail,
    ArticleListItem,
    ArticleListResponse,
)
from app.services.article_service import ArticleService

router = APIRouter(prefix="/api/articles", tags=["articles"])


@router.post("/generate", response_model=ArticleGenerationSummary)
def generate_articles(request: ArticleGenerationRequest):
    """
    Trigger article generation for all clusters × weeks.

    - `week`       – limit to a specific week (e.g. "week1" or "1")
    - `cluster_id` – limit to a single cluster
    - `force`      – overwrite articles that already exist
    """
    try:
        service = ArticleService()
        result = service.run_article_generation(
            week=request.week,
            force=request.force,
            cluster_id=request.cluster_id,
            dry_run=request.dry_run,
        )
        return {
            "articles_generated": result["articles_generated"],
            "articles_skipped": result["articles_skipped"],
            "articles_failed": result["articles_failed"],
            "weeks_processed": result["weeks_processed"],
            "dry_run": result["dry_run"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Article generation failed: {exc}"
        ) from exc


@router.get("", response_model=ArticleListResponse)
def list_articles(
    cluster_id: Optional[int] = Query(
        default=None,
        description="Filter to a specific cluster ID",
    ),
    week: Optional[int] = Query(
        default=None,
        description="Filter to a specific week number (e.g. 1, 2, 3)",
    ),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    List published articles.  No body included — use the detail endpoint for full content.
    """
    try:
        service = ArticleService()
        articles = service.get_articles(
            cluster_id=cluster_id,
            week_number=week,
            limit=limit,
        )
        return {"articles": articles, "total": len(articles)}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list articles: {exc}"
        ) from exc


@router.get("/{article_id}", response_model=ArticleDetail)
def get_article(article_id: str):
    """Fetch a single article by ID, including the full body text."""
    try:
        service = ArticleService()
        article = service.get_article_by_id(article_id)
        if not article:
            raise HTTPException(
                status_code=404, detail=f"Article '{article_id}' not found"
            )
        return article
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch article: {exc}"
        ) from exc
