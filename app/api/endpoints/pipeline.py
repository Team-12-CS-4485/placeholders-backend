from fastapi import APIRouter, HTTPException
from app.schemas.pipeline import (
    PipelineRunRequest,
    PipelineRunResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)
from app.services.pipeline_service import PipelineService


router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/s3-transcript-analysis", response_model=PipelineRunResponse)
def run_s3_transcript_analysis(request: PipelineRunRequest):
    try:
        service = PipelineService()
        return service.run_s3_transcript_analysis(
            prefix=request.prefix,
            limit=request.limit
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@router.post("/search", response_model=VectorSearchResponse)
def search_similar_chunks(request: VectorSearchRequest):
    try:
        service = PipelineService()
        return service.search_similar_chunks(
            query=request.query,
            limit=request.limit or 5,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
