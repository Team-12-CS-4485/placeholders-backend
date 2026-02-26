from typing import Optional
from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    prefix: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=50)


class TranscriptAnalysisResult(BaseModel):
    transcript_key: str
    transcript_index: int
    chunk_count: int
    final_summary: str
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
    error: Optional[str] = None


class PipelineRunResponse(BaseModel):
    prefix: str
    object_limit: int
    objects_processed: int
    transcripts_found: int
    transcripts_analyzed: int
    chunk_map: dict[str, list[str]]
    analysis_map: dict[str, AnalysisMapEntry]
    results: list[ObjectPipelineResult]
