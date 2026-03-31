"""
pipeline.py - Pipeline API Endpoints

POST /api/pipeline/run    : Runs the full pipeline — ingest → cluster → claim analysis → articles
POST /api/pipeline/search : Semantic search over indexed transcript chunks
"""

from fastapi import APIRouter, HTTPException

from app.schemas.pipeline import (
    PipelineRunRequest,
    FullPipelineResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)
from app.services.pipeline_service import PipelineService
from app.services.clustering_service import ClusteringService
from app.services.claim_analysis_service import ClaimAnalysisService
from app.services.article_service import ArticleService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.core.config import settings

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run", response_model=FullPipelineResponse)
def run_full_pipeline(request: PipelineRunRequest):
    """
    Runs the full pipeline in sequence:
    1. Ingest    — S3 → chunk → Gemini → embed → Qdrant
    2. Cluster   — UMAP/HDBSCAN → patches cluster_id/label back to Qdrant
    3. Claims    — classifies consensus/debated/unique → patches back to Qdrant
    4. Articles  — generates Gemini news articles → DynamoDB articles table
    """
    try:
        # Step 1: Ingest
        ingestion_result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix,
            limit=request.limit,
        )

        # Step 2: Cluster
        clustering_result = ClusteringService().run_clustering()

        # Step 3: Claim analysis
        claim_result = ClaimAnalysisService().run_claim_analysis()

        # Step 4: Article generation (non-fatal — pipeline succeeds even if this fails)
        try:
            article_result = ArticleService().run_article_generation()
        except Exception as article_exc:
            # Log but don't fail the whole pipeline
            import logging
            logging.getLogger(__name__).error(
                f"ARTICLE_GENERATION_FAILED (non-fatal) error={article_exc}"
            )
            article_result = {
                "articles_generated": 0,
                "articles_skipped": 0,
                "articles_failed": 0,
                "weeks_processed": [],
            }

        return {
            "ingestion": {
                "objects_processed": ingestion_result["objects_processed"],
                "videos_found": ingestion_result["videos_found"],
                "videos_indexed": ingestion_result["videos_indexed"],
                "total_chunks_stored": ingestion_result["total_chunks_stored"],
            },
            "clustering": {
                "total_videos": clustering_result.get("total_videos", 0),
                "cluster_count": clustering_result.get("cluster_count", 0),
                "noise_videos": clustering_result.get("noise_videos", 0),
                "total_chunks_patched": clustering_result.get(
                    "total_chunks_patched", 0
                ),
            },
            "claim_analysis": {
                "clusters_processed": claim_result.get("clusters_processed", 0),
                "total_patched": claim_result.get("total_patched", 0),
            },
            "articles": {
                "articles_generated": article_result.get("articles_generated", 0),
                "articles_skipped": article_result.get("articles_skipped", 0),
                "articles_failed": article_result.get("articles_failed", 0),
                "weeks_processed": article_result.get("weeks_processed", []),
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@router.post("/search", response_model=VectorSearchResponse)
def search_similar_chunks(request: VectorSearchRequest):
    try:
        limit = request.limit or 5
        query_vector = EmbeddingService(api_keys=settings.genai_api_keys).embed_query(
            request.query
        )
        hits = VectorService().search_similar_chunks(
            query_vector=query_vector, limit=limit
        )
        return {"query": request.query, "limit": limit, "hits": hits}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
