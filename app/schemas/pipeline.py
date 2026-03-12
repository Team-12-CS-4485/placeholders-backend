"""
pipeline.py - Pydantic Request/Response Schemas

Defines all API request and response models for the pipeline endpoints:
- PipelineRunRequest/Response : S3 transcript analysis pipeline execution
- VectorSearchRequest/Response : Semantic search over indexed transcript chunks (FAISS)
- Supporting models: TranscriptAnalysisResult, ObjectPipelineResult, AnalysisMapEntry, SearchHit
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    prefix: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=50)


class TranscriptAnalysisResult(BaseModel):
    transcript_key: str
    transcript_index: int
    chunk_count: int
    final_summary: str
    chunks_stored: int = 0
    error: Optional[str] = None


class ObjectPipelineResult(BaseModel):
    key: str
    status: str
    error: Optional[str] = None
    transcript_results: list[TranscriptAnalysisResult]


class AnalysisMapEntry(BaseModel):
    status: str
    chunk_count: int
    chunk_analyses: list[str]
    final_summary: str
    chunks_stored: int = 0
    error: Optional[str] = None


class PipelineRunResponse(BaseModel):
    prefix: str
    object_limit: int
    objects_processed: int
    transcripts_found: int
    transcripts_analyzed: int
    total_chunks_stored: int
    chunk_map: dict[str, list[str]]
    analysis_map: dict[str, AnalysisMapEntry]
    results: list[ObjectPipelineResult]


class VectorSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: Optional[int] = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    source_key: str = ""
    transcript_index: int = 0
    chunk_index: int = 0
    text: str = ""
    word_count: int = 0


class VectorSearchResponse(BaseModel):
    query: str
    limit: int
    hits: list[SearchHit]
