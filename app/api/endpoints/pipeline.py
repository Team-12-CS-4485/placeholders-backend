"""
pipeline.py - Pipeline API Endpoints

POST /api/pipeline/run    : Runs the full pipeline — ingest → cluster → claim analysis
POST /api/pipeline/ingest : Ingest only — S3 → Gemini → DynamoDB → Qdrant
POST /api/pipeline/cluster: Cluster only — UMAP/HDBSCAN → DynamoDB
POST /api/pipeline/claims : Claim analysis only — classify → DynamoDB
POST /api/pipeline/search : Semantic search over indexed transcript chunks

All write endpoints accept dry_run=true to preview without writing.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.pipeline import (
    PipelineRunRequest,
    DryRunRequest,
    FullPipelineResponse,
    IngestionSummary,
    ClusteringSummary,
    ClaimAnalysisSummary,
    VectorSearchRequest,
    VectorSearchResponse,
)
from app.services.pipeline_service import PipelineService
from app.services.clustering_service import ClusteringService
from app.services.claim_analysis_service import ClaimAnalysisService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.core.config import settings

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run", response_model=FullPipelineResponse)
def run_full_pipeline(request: PipelineRunRequest):
    """
    Runs the full pipeline in sequence:
    1. Ingest — S3 → chunk → Gemini → embed → Qdrant
    2. Cluster — UMAP/HDBSCAN → DynamoDB
    3. Claim analysis — classifies consensus/debated/unique → DynamoDB

    Pass dry_run=true to preview counts without writing anything.
    """
    try:
        ingestion_result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix,
            limit=request.limit,
            dry_run=request.dry_run,
        )
        clustering_result = ClusteringService().run_clustering(
            dry_run=request.dry_run,
        )
        claim_result = ClaimAnalysisService().run_claim_analysis(
            dry_run=request.dry_run,
        )

        return {
            "ingestion": {
                "objects_processed": ingestion_result["objects_processed"],
                "videos_found": ingestion_result["videos_found"],
                "videos_indexed": ingestion_result["videos_indexed"],
                "total_chunks_stored": ingestion_result["total_chunks_stored"],
                "dry_run": ingestion_result.get("dry_run", False),
            },
            "clustering": {
                "total_videos": clustering_result.get("total_videos", 0),
                "cluster_count": clustering_result.get("cluster_count", 0),
                "noise_videos": clustering_result.get("noise_videos", 0),
                "total_chunks_patched": clustering_result.get("videos_updated", 0),
                "dry_run": clustering_result.get("dry_run", False),
            },
            "claim_analysis": {
                "clusters_processed": claim_result.get("clusters_processed", 0),
                "total_patched": claim_result.get("total_written", 0),
                "dry_run": claim_result.get("dry_run", False),
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@router.post("/ingest", response_model=IngestionSummary)
def run_ingest(request: PipelineRunRequest):
    """Ingest only — S3 → chunk → Gemini → DynamoDB → Qdrant."""
    try:
        result = PipelineService().run_s3_transcript_analysis(
            prefix=request.prefix,
            limit=request.limit,
            dry_run=request.dry_run,
        )
        return {
            "objects_processed": result["objects_processed"],
            "videos_found": result["videos_found"],
            "videos_indexed": result["videos_indexed"],
            "total_chunks_stored": result["total_chunks_stored"],
            "dry_run": result.get("dry_run", False),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


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
        raise HTTPException(status_code=500, detail=f"Clustering failed: {exc}") from exc


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
        raise HTTPException(status_code=500, detail=f"Claim analysis failed: {exc}") from exc


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
