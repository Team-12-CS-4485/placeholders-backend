"""
pipeline.py - Pydantic Request/Response Schemas
"""

from typing import Optional
from pydantic import BaseModel, Field

# ── Requests ──────────────────────────────────────────────────────────────────


class PipelineRunRequest(BaseModel):
    prefix: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=50)
    dry_run: bool = False


class DryRunRequest(BaseModel):
    dry_run: bool = False


class VectorSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: Optional[int] = Field(default=5, ge=1, le=50)


# ── Pipeline response models ──────────────────────────────────────────────────


# ── Full pipeline response ────────────────────────────────────────────────────


class IngestionSummary(BaseModel):
    objects_processed: int
    videos_found: int
    videos_indexed: int
    total_chunks_stored: int
    dry_run: bool = False


class ClusteringSummary(BaseModel):
    total_videos: int
    cluster_count: int
    noise_videos: int
    total_chunks_patched: int
    dry_run: bool = False


class ClaimAnalysisSummary(BaseModel):
    clusters_processed: int
    total_patched: int
    dry_run: bool = False


class ArticlesSummary(BaseModel):
    articles_generated: int
    articles_skipped: int
    articles_failed: int
    weeks_processed: list[str]
    dry_run: bool = False


class FullPipelineResponse(BaseModel):
    ingestion: IngestionSummary
    clustering: ClusteringSummary
    claim_analysis: ClaimAnalysisSummary
    articles: ArticlesSummary


# ── Async job models ──────────────────────────────────────────────────────────


class JobSubmitResponse(BaseModel):
    job_id: Optional[str]
    status: str  # "running" | "complete"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "running" | "complete" | "failed"
    progress: int
    total: int
    errors: list[str]
    failed_videos: list[str]
    started_at: str
    completed_at: Optional[str]
    result: Optional[dict]


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
