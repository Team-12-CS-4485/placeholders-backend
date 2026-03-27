"""
pipeline.py - Pydantic Request/Response Schemas
"""

from typing import Optional
from pydantic import BaseModel, Field

# ── Requests ──────────────────────────────────────────────────────────────────


class PipelineRunRequest(BaseModel):
    prefix: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=50)


class VectorSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: Optional[int] = Field(default=5, ge=1, le=50)


# ── Pipeline response models ──────────────────────────────────────────────────


class VideoIntelligence(BaseModel):
    topics: list[str]
    category: str
    sentiment: str
    key_claims: list[str]
    is_breaking: bool


class ThumbnailIntelligence(BaseModel):
    thumbnail_visual: str
    thumbnail_tone: str
    thumbnail_clickbait_score: int
    thumbnail_brand_consistent: bool
    thumbnail_insight: str


class VideoAnalysisResult(BaseModel):
    transcript_key: str
    video_id: str
    chunk_count: int
    chunks_stored: int
    intelligence: Optional[VideoIntelligence] = None
    thumbnail: Optional[ThumbnailIntelligence] = None
    error: Optional[str] = None


class ObjectPipelineResult(BaseModel):
    key: str
    status: str
    error: Optional[str] = None
    transcript_results: list[VideoAnalysisResult]


class AnalysisMapEntry(BaseModel):
    status: str
    chunk_count: int
    chunks_stored: int
    intelligence: Optional[VideoIntelligence] = None
    thumbnail: Optional[ThumbnailIntelligence] = None
    error: Optional[str] = None


class PipelineRunResponse(BaseModel):
    prefix: str
    object_limit: int
    objects_processed: int
    videos_found: int
    videos_indexed: int
    total_chunks_stored: int
    chunk_map: dict[str, list[str]]
    analysis_map: dict[str, AnalysisMapEntry]
    results: list[ObjectPipelineResult]


# ── Full pipeline response ────────────────────────────────────────────────────


class IngestionSummary(BaseModel):
    objects_processed: int
    videos_found: int
    videos_indexed: int
    total_chunks_stored: int


class ClusteringSummary(BaseModel):
    total_videos: int
    cluster_count: int
    noise_videos: int
    total_chunks_patched: int


class ClaimAnalysisSummary(BaseModel):
    clusters_processed: int
    total_patched: int


class FullPipelineResponse(BaseModel):
    ingestion: IngestionSummary
    clustering: ClusteringSummary
    claim_analysis: ClaimAnalysisSummary


# ── Search response models ────────────────────────────────────────────────────


class SearchHit(BaseModel):
    id: str
    score: float
    transcript_key: str
    source_key: str
    channel: str
    title: str
    published_at: str
    view_count: int
    category: str
    sentiment: str
    topics: list[str]
    key_claims: list[str]
    is_breaking: bool
    chunk_index: int
    text: str
    thumbnail_visual: str
    thumbnail_tone: str
    thumbnail_clickbait_score: int
    thumbnail_brand_consistent: bool
    thumbnail_insight: str


class VectorSearchResponse(BaseModel):
    query: str
    limit: int
    hits: list[SearchHit]
