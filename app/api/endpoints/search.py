"""
search.py - Search API Endpoint

GET /api/search?q= : Semantic search over indexed videos. Returns VideoItem-shaped
                     results ranked by relevance, with a matching excerpt per result.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.schemas.search import SearchResponse
from app.services.dynamo_service import DynamoService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.core.config import settings

router = APIRouter(prefix="/api/search", tags=["search"])
logger = logging.getLogger(__name__)


@router.get("", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Max results to return"),
):
    """
    Semantic search over indexed transcript chunks.

    Embeds the query, finds the closest matching chunks in Qdrant, deduplicates
    by video, then fetches full video metadata from DynamoDB. Returns VideoItem-
    shaped results with a relevance score and matching excerpt.
    """
    try:
        query_vector = EmbeddingService(api_keys=settings.genai_api_keys).embed_query(q)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

    try:
        # Fetch more chunks than needed so deduplication still yields `limit` videos
        raw_hits = VectorService().search_similar_chunks(
            query_vector=query_vector,
            limit=limit * 4,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Vector search failed: {exc}"
        ) from exc

    # Deduplicate by video_id — keep the highest-scoring chunk per video
    seen: dict[str, dict] = {}
    for hit in raw_hits:
        video_id = hit.get("transcript_index") or hit.get("source_key", "")
        if not video_id:
            continue
        if video_id not in seen or hit["score"] > seen[video_id]["score"]:
            seen[video_id] = hit

    top_hits = sorted(seen.values(), key=lambda h: h["score"], reverse=True)[:limit]

    # Fetch full video metadata from DynamoDB
    dynamo = DynamoService()
    results = []
    for hit in top_hits:
        video_id = hit.get("transcript_index") or hit.get("source_key", "")
        item = dynamo.get_video_by_id(video_id)
        if not item:
            logger.warning(f"SEARCH_DYNAMO_MISS video_id={video_id}")
            continue
        mapped = dynamo.map_video_item(item)
        results.append(
            {
                **mapped,
                "score": round(hit["score"], 4),
                "excerpt": hit.get("text", ""),
            }
        )

    return {
        "query": q,
        "limit": limit,
        "total": len(results),
        "results": results,
    }
