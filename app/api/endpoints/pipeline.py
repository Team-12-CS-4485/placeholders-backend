"""
pipeline.py - Pipeline API Endpoints

POST /api/pipeline/run    : Starts the full pipeline in background, returns job_id
POST /api/pipeline/ingest : Ingest only — starts in background, returns job_id
POST /api/pipeline/cluster: Cluster only — UMAP/HDBSCAN → DynamoDB (synchronous)
POST /api/pipeline/claims : Claim analysis only — classify → DynamoDB (synchronous)

GET  /api/pipeline/jobs/{job_id} : Poll status of a background pipeline job

POST /api/pipeline/search : Semantic search over indexed transcript chunks

All write endpoints accept dry_run=true to preview without writing.
/ingest and /run with dry_run=true run synchronously (no Gemini calls, fast).
"""

import threading

from fastapi import APIRouter, HTTPException

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
        job_service.fail_job(job_id, str(exc))


def _run_full_pipeline_background(request: PipelineRunRequest, job_id: str) -> None:
    try:
        ingest_result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix,
            limit=request.limit,
            job_id=job_id,
        )
        clustering_result = ClusteringService().run_clustering()
        claim_result = ClaimAnalysisService().run_claim_analysis()

        try:
            article_result = ArticleService().run_article_generation()
        except Exception as article_exc:
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

        job_service.complete_job(
            job_id,
            {
                "ingestion": {
                    "objects_processed": ingest_result["objects_processed"],
                    "videos_found": ingest_result["videos_found"],
                    "videos_indexed": ingest_result["videos_indexed"],
                    "total_chunks_stored": ingest_result["total_chunks_stored"],
                    "failed_videos": ingest_result.get("failed_videos", []),
                },
                "clustering": {
                    "total_videos": clustering_result.get("total_videos", 0),
                    "cluster_count": clustering_result.get("cluster_count", 0),
                    "noise_videos": clustering_result.get("noise_videos", 0),
                    "videos_updated": clustering_result.get("videos_updated", 0),
                },
                "claim_analysis": {
                    "clusters_processed": claim_result.get("clusters_processed", 0),
                    "total_written": claim_result.get("total_written", 0),
                },
                "articles": {
                    "articles_generated": article_result.get("articles_generated", 0),
                    "articles_skipped": article_result.get("articles_skipped", 0),
                    "articles_failed": article_result.get("articles_failed", 0),
                    "weeks_processed": article_result.get("weeks_processed", []),
                },
            },
        )
    except Exception as exc:
        job_service.fail_job(job_id, str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/run", response_model=JobSubmitResponse)
def run_full_pipeline(request: PipelineRunRequest):
    """
    Starts the full pipeline in the background:
    1. Ingest — S3 → chunk → Gemini (10 parallel workers) → embed → Qdrant
    2. Cluster — UMAP/HDBSCAN → DynamoDB
    3. Claim analysis — classifies consensus/debated/unique → DynamoDB
    4. Article generation — Gemini articles (10 parallel workers) → DynamoDB

    Returns job_id immediately. Poll GET /api/pipeline/jobs/{job_id} for status.
    dry_run=true runs synchronously and returns immediately (no Gemini calls).
    """
    if request.dry_run:
        try:
            ingestion_result = PipelineService().run_s3_transcript_analysis(
                prefix=request.prefix,
                limit=request.limit,
                dry_run=True,
            )
            clustering_result = ClusteringService().run_clustering(dry_run=True)
            claim_result = ClaimAnalysisService().run_claim_analysis(dry_run=True)
            article_result = ArticleService().run_article_generation(dry_run=True)
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
def run_ingest(request: PipelineRunRequest):
    """
    Ingest only — S3 → chunk → Gemini (10 parallel workers) → DynamoDB → Qdrant.
    Returns job_id immediately. Poll GET /api/pipeline/jobs/{job_id} for status.
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
    """Poll the status of a background pipeline job."""
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
