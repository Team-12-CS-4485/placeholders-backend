"""
embedding_service.py - Gemini Analysis Service

Handles Gemini API interactions for transcript analysis:
- Per-chunk analysis using Gemini with extended thinking (configurable level)
- Final summary generation by combining chunk analyses into a comprehensive report

Chunking is now handled by chunking_service.py (semantic chunking).
Embedding + storage is now handled by faiss_service.py (sentence-transformers + FAISS).
"""

from google import genai
from google.genai import types
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, client=None):
        self.model_id = settings.gemini_model_id
        self.client = client or genai.Client(api_key=settings.genai_api_key)

    def _get_text(self, response):
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(response)

    def analyze_chunks(self, chunks):
        """Analyze each chunk with Gemini, returning key points, claims, and summary."""
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
                )
            )
            analyses.append(self._get_text(response))
            logger.info(f"GEMINI_ANALYZE chunk={idx}/{total}")

        return analyses

    def summarize_analyses(self, analyses):
        """Combine chunk analyses into one final transcript report."""
        if not analyses:
            return ""

        joined = "\n\n".join(
            f"Chunk {idx + 1} Analysis:\n{analysis}"
            for idx, analysis in enumerate(analyses)
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
            )
        )
        logger.info("GEMINI_SUMMARIZE complete")
        return self._get_text(response)
