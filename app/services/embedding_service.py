from __future__ import annotations

import logging
from typing import Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.services.chunking_service import SemanticChunker, get_default_chunker

logger = logging.getLogger(__name__)


class EmbeddingService:
    
    def __init__(
        self,
        client=None,
        chunker: Optional[SemanticChunker] = None,
    ):
        self.model_id = settings.gemini_model_id
        self.client = client or genai.Client(api_key=settings.genai_api_key)

        # One model for both chunking and embedding — loaded once, shared
        self._chunker = chunker or get_default_chunker(
            model_name=settings.embedding_model_id
        )

        if self._chunker.available:
            logger.info(
                f"EMBEDDING_LOCAL model={settings.embedding_model_id} dims=768 cost=free"
            )
        else:
            logger.warning(
                "Local embedding model not available — install sentence-transformers. "
                "Chunking will fall back to character mode and embed_chunks will raise."
            )

    # ── Chunking ─────────────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> list[str]:

        return self._chunker.chunk(text)

    # ── Gemini analysis ───────────────────────────────────────────────────────

    def _get_text(self, response) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(response)

    def analyze_chunks(self, chunks: list[str]) -> list[str]:

        analyses = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            prompt = (
                f"Analyze transcript chunk {idx}/{total}. "
                "Return key points, notable claims, and a concise summary:\n\n"
            )
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt + chunk],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=settings.gemini_thinking_level
                    )
                ),
            )
            analyses.append(self._get_text(response))
        return analyses

    def summarize_analyses(self, analyses: list[str]) -> str:
        """Merge per-chunk analyses into one final transcript report."""
        if not analyses:
            return ""
        joined = "\n\n".join(
            f"Chunk {idx + 1} Analysis:\n{a}" for idx, a in enumerate(analyses)
        )
        prompt = (
            "Combine the chunk analyses into one final transcript report with: "
            "overall summary, key themes, key claims, and potential follow-up questions.\n\n"
        )
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt + joined],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=settings.gemini_thinking_level
                )
            ),
        )
        return self._get_text(response)

    def analyze_transcript(self, transcript: str) -> dict:
        chunks = self.chunk_text(transcript)
        analyses = self.analyze_chunks(chunks)
        final_summary = self.summarize_analyses(analyses)
        return {
            "chunk_count": len(chunks),
            "chunk_analyses": analyses,
            "final_summary": final_summary,
        }

    # ── Embedding (local, free) ───────────────────────────────────────────────

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
     
        vectors = self._chunker.embed([text], is_query=is_query)
        return vectors[0]

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
       
        if not chunks:
            return []
        logger.info(f"EMBED_LOCAL chunks={len(chunks)}")
        return self._chunker.embed(chunks, is_query=False)

    def embed_query(self, query: str) -> list[float]:
      
        return self.embed_text(query, is_query=True)

    # ── Utility ──────────────────────────────────────────────────────────────

    @property
    def chunker(self) -> SemanticChunker:
        return self._chunker