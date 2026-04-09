"""
pipeline.py - Pipeline API Endpoints

POST /api/pipeline/run    : Starts the full pipeline in background, returns immediately
POST /api/pipeline/ingest : Ingest only — starts in background, returns immediately
POST /api/pipeline/cluster: Cluster only — UMAP/HDBSCAN → DynamoDB (synchronous)
POST /api/pipeline/claims : Claim analysis only — classify → DynamoDB (synchronous)
POST /api/pipeline/search : Semantic search over indexed transcript chunks

All write endpoints accept dry_run=true to preview without writing.
/ingest and /run with dry_run=true run synchronously (no Gemini calls, fast).
"""

import threading

from fastapi import APIRouter, HTTPException, Request

from app.core.cache import invalidate_all
from app.core.logging import get_logger
from app.core.rate_limit import limiter

logger = get_logger(__name__)

from app.schemas.pipeline import (
    PipelineRunRequest,
    DryRunRequest,
    FullPipelineResponse,
    IngestionSummary,
    ClusteringSummary,
    ClaimAnalysisSummary,
    ArticlesSummary,
    VectorSearchRequest,
    VectorSearchResponse,
    JobSubmitResponse,
    JobStatusResponse,
)
from app.services.pipeline_service import PipelineService
from app.services.clustering_service import ClusteringService
from app.services.claim_analysis_service import ClaimAnalysisService
from app.services.article_service import ArticleService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services import job_service
from app.core.config import settings

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


# ── Background workers ────────────────────────────────────────────────────────


def _run_ingest_background(request: PipelineRunRequest, job_id: str) -> None:
    try:
        result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix,
            limit=request.limit,
            job_id=job_id,
        )
        job_service.complete_job(job_id, result)
    except Exception as exc:
        logger.error(f"INGEST_BACKGROUND_FATAL error={exc}")
        job_service.fail_job(job_id, str(exc))


def _run_full_pipeline_background(request: PipelineRunRequest, job_id: str) -> None:
    try:
        ingest_result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix,
            limit=request.limit,
            job_id=job_id,
        )
        ClusteringService().run_clustering()
        ClaimAnalysisService().run_claim_analysis()
        article_result = {}
        try:
            article_result = ArticleService().run_article_generation()
        except Exception as exc:
            logger.error(f"ARTICLE_GENERATION_FAILED (non-fatal) error={exc}")
        invalidate_all()
        logger.info("PIPELINE_BACKGROUND_COMPLETE cache invalidated")
        job_service.complete_job(
            job_id,
            {
                "ingestion": ingest_result,
                "articles": article_result,
            },
        )
    except Exception as exc:
        logger.error(f"PIPELINE_BACKGROUND_FATAL error={exc}")
        job_service.fail_job(job_id, str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/run", response_model=JobSubmitResponse)
@limiter.limit("2/minute")
def run_full_pipeline(http_request: Request, request: PipelineRunRequest):
    """
    Starts the full pipeline in the background:
    1. Ingest — S3 → chunk → Gemini (10 parallel workers) → embed → Qdrant
    2. Cluster — UMAP/HDBSCAN → DynamoDB
    3. Claim analysis — classifies consensus/debated/unique → DynamoDB
    4. Article generation — Gemini articles (10 parallel workers) → DynamoDB

    Returns immediately with a job_id. Poll GET /api/pipeline/jobs/{job_id} for status.
    dry_run=true runs synchronously (no Gemini calls).
    """
    if request.dry_run:
        try:
            PipelineService().run_s3_transcript_analysis(
                prefix=request.prefix,
                limit=request.limit,
                dry_run=True,
            )
            ClusteringService().run_clustering(dry_run=True)
            ClaimAnalysisService().run_claim_analysis(dry_run=True)
            ArticleService().run_article_generation(dry_run=True)
            return {"job_id": None, "status": "complete"}
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Dry run failed: {exc}"
            ) from exc

    job_id = job_service.create_job()
    threading.Thread(
        target=_run_full_pipeline_background,
        args=(request, job_id),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running"}


@router.post("/ingest", response_model=JobSubmitResponse)
@limiter.limit("5/minute")
def run_ingest(http_request: Request, request: PipelineRunRequest):
    """
    Ingest only — S3 → chunk → Gemini (10 parallel workers) → DynamoDB → Qdrant.
    Returns immediately with a job_id. Poll GET /api/pipeline/jobs/{job_id} for status.
    dry_run=true runs synchronously (no Gemini calls, fast).
    """
    if request.dry_run:
        try:
            result = PipelineService().run_s3_transcript_analysis(
                prefix=request.prefix,
                limit=request.limit,
                dry_run=True,
            )
            return {"job_id": None, "status": "complete"}
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Dry run failed: {exc}"
            ) from exc

    job_id = job_service.create_job()
    threading.Thread(
        target=_run_ingest_background,
        args=(request, job_id),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Poll this after submitting a pipeline run to check progress and results."""
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/cluster", response_model=ClusteringSummary)
def run_cluster(request: DryRunRequest):
    """Cluster only — reads vectors from Qdrant + metadata from DynamoDB, runs UMAP/HDBSCAN."""
    try:
        result = ClusteringService().run_clustering(dry_run=request.dry_run)
        return {
            "total_videos": result.get("total_videos", 0),
            "cluster_count": result.get("cluster_count", 0),
            "noise_videos": result.get("noise_videos", 0),
            "total_chunks_patched": result.get("videos_updated", 0),
            "dry_run": result.get("dry_run", False),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Clustering failed: {exc}"
        ) from exc


@router.post("/claims", response_model=ClaimAnalysisSummary)
def run_claims(request: DryRunRequest):
    """Claim analysis only — reads clustered videos from DynamoDB, classifies claims."""
    try:
        result = ClaimAnalysisService().run_claim_analysis(dry_run=request.dry_run)
        return {
            "clusters_processed": result.get("clusters_processed", 0),
            "total_patched": result.get("total_written", 0),
            "dry_run": result.get("dry_run", False),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Claim analysis failed: {exc}"
        ) from exc


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
