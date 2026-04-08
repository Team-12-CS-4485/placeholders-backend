"""
embedding_service.py - Local Embedding & Chunking Service

Handles local-only operations (zero API calls):
- Semantic chunking via sentence-transformers model
- Local chunk embedding (nomic-ai/nomic-embed-text-v1.5, 768 dims, free)

Gemini-powered analysis has been moved to gemini_service.py.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import settings
from app.services.chunking_service import SemanticChunker, get_default_chunker

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(
        self,
        chunker: Optional[SemanticChunker] = None,
        api_keys=None,
    ):
        # api_keys accepted for backward compatibility but not used here
        self._chunker = chunker or get_default_chunker(
            model_name=settings.embedding_model_id
        )

    # ── Chunking ─────────────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> list[str]:
        return self._chunker.chunk(text)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        return self._chunker.embed([text], is_query=is_query)[0]

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        logger.debug(f"EMBED_LOCAL chunks={len(chunks)}")
        return self._chunker.embed(chunks, is_query=False)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query, is_query=True)

    @property
    def chunker(self) -> SemanticChunker:
        return self._chunker
