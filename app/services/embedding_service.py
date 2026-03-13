"""
embedding_service.py - Embedding & Gemini Analysis Service

Handles:
- Semantic chunking via local sentence-transformers model
- Local chunk embedding (nomic-ai/nomic-embed-text-v1.5, 768 dims, free)
- Gemini-powered video intelligence extraction (topics, sentiment, claims) — one call per video
- Gemini chunk analysis and summarization
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.services.chunking_service import SemanticChunker, get_default_chunker

logger = logging.getLogger(__name__)

CATEGORY_OPTIONS = [
    # US Politics
    "US Elections & Voting",
    "Congress & Legislation",
    "White House & Executive",
    "Supreme Court & Justice System",
    "Immigration & Border",
    # International
    "Middle East Conflict",
    "Russia-Ukraine War",
    "China & Taiwan",
    "Europe & NATO",
    "Global Diplomacy",
    # Economy & Business
    "Wall Street & Markets",
    "Federal Reserve & Interest Rates",
    "Trade & Tariffs",
    "Corporate News",
    "Jobs & Labor",
    # Technology
    "AI & Machine Learning",
    "Big Tech & Regulation",
    "Cybersecurity",
    "Space & Science",
    # Society & Crime
    "Gun Violence & Mass Shootings",
    "Crime & Law Enforcement",
    "Civil Rights & Social Justice",
    "Education",
    # Health
    "Public Health & Disease",
    "Healthcare Policy",
    "Mental Health",
    # Environment
    "Climate & Energy Policy",
    "Natural Disasters",
    # Other
    "Entertainment & Celebrity",
    "Sports",
    "Other",
]

CATEGORY_OPTIONS_STR = ", ".join(CATEGORY_OPTIONS)


class EmbeddingService:

    def __init__(self, client=None, chunker: Optional[SemanticChunker] = None):
        self.model_id = settings.gemini_model_id
        self.client = client or genai.Client(api_key=settings.genai_api_key)

        self._chunker = chunker or get_default_chunker(
            model_name=settings.embedding_model_id
        )

        if self._chunker.available:
            logger.info(f"EMBEDDING_LOCAL model={settings.embedding_model_id} dims=768 cost=free")
        else:
            logger.warning(
                "Local embedding model not available — install sentence-transformers. "
                "Chunking will fall back to character mode and embed_chunks will raise."
            )

    # ── Chunking ─────────────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> list[str]:
        return self._chunker.chunk(text)

    # ── Gemini helpers ────────────────────────────────────────────────────────

    def _get_text(self, response) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(response)

    def _gemini(self, prompt: str, max_retries: int = 6) -> str:
        wait = 15
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level=settings.gemini_thinking_level
                        )
                    ),
                )
                return self._get_text(response)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                is_rate_limit = (
                    "429" in exc_str
                    or "RESOURCE_EXHAUSTED" in exc_str
                    or "rate" in exc_str.lower()
                    or "quota" in exc_str.lower()
                    or "too many" in exc_str.lower()
                    or getattr(exc, "status_code", None) == 429
                    or getattr(exc, "code", None) == 429
                )
                if is_rate_limit:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"GEMINI_RATE_LIMIT attempt={attempt+1}/{max_retries} "
                            f"waiting={wait}s error={exc_str[:100]}"
                        )
                        time.sleep(wait)
                        wait = min(wait * 2, 300)
                    else:
                        raise RuntimeError(
                            f"Gemini rate limit exceeded after {max_retries} retries"
                        ) from exc
                else:
                    raise
        raise last_exc

    # ── Video intelligence — ONE call per video ───────────────────────────────

    def extract_video_intelligence(self, chunks: list[str], title: str = "") -> dict:
        """
        Single Gemini call per video. Concatenates all chunks and extracts:
        - topics (list of 3-5 short topic strings)
        - category (one of the fixed category options)
        - sentiment (positive / negative / neutral)
        - key_claims (list of up to 5 one-sentence claims)
        - is_breaking (bool — urgent/developing story language detected)

        Returns a dict with those keys. Falls back to safe defaults on parse error.
        """
        full_text = "\n\n".join(chunks)
        title_line = f'Video title: "{title}"\n\n' if title else ""

        prompt = (
            f"{title_line}"
            f"Below is a news video transcript split into chunks.\n\n"
            f"{full_text}\n\n"
            "Return ONLY a valid JSON object with exactly these keys:\n"
            "{\n"
            '  "topics": ["topic1", "topic2"],\n'
            '  "category": "<category>",\n'
            '  "sentiment": "<positive|negative|neutral>",\n'
            '  "key_claims": ["claim1", "claim2"],\n'
            '  "is_breaking": <true|false>\n'
            "}\n\n"
            "Rules:\n"
            "- topics: 3 to 5 short phrases, specific to this video\n"
            f"- category: must be exactly one of these options: {CATEGORY_OPTIONS_STR}\n"
            "- sentiment: overall tone of the coverage\n"
            "- key_claims: up to 5 specific factual claims made in the video, one sentence each\n"
            "- is_breaking: true if the transcript uses urgent/developing/breaking language\n"
            "Return raw JSON only, no markdown, no explanation."
        )

        raw = ""
        try:
            raw = self._gemini(prompt)
            # Strip markdown fences if Gemini adds them anyway
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean)

            return {
                "topics":     data.get("topics", [])[:5],
                "category":   data.get("category", "Other") if data.get("category") in CATEGORY_OPTIONS else "Other",
                "sentiment":  data.get("sentiment", "neutral") if data.get("sentiment") in ("positive", "negative", "neutral") else "neutral",
                "key_claims": data.get("key_claims", [])[:5],
                "is_breaking": bool(data.get("is_breaking", False)),
            }

        except (json.JSONDecodeError, Exception) as exc:
            logger.error(f"GEMINI_INTELLIGENCE_PARSE_ERROR error={exc} raw={raw[:200]}")
            return {
                "topics": [],
                "category": "Other",
                "sentiment": "neutral",
                "key_claims": [],
                "is_breaking": False,
            }

    # ── Chunk-level analysis (kept for optional deep analysis) ────────────────

    def analyze_chunks(self, chunks: list[str]) -> list[str]:
        analyses = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            prompt = (
                f"Analyze transcript chunk {idx}/{total}. "
                "Return key points, notable claims, and a concise summary:\n\n"
                + chunk
            )
            analyses.append(self._gemini(prompt))
        return analyses

    def summarize_analyses(self, analyses: list[str]) -> str:
        if not analyses:
            return ""
        joined = "\n\n".join(
            f"Chunk {idx + 1} Analysis:\n{a}" for idx, a in enumerate(analyses)
        )
        prompt = (
            "Combine the chunk analyses into one final transcript report with: "
            "overall summary, key themes, key claims, and potential follow-up questions.\n\n"
            + joined
        )
        logger.info("GEMINI_SUMMARIZE complete")
        return self._gemini(prompt)

    def analyze_transcript(self, transcript: str) -> dict:
        chunks = self.chunk_text(transcript)
        analyses = self.analyze_chunks(chunks)
        final_summary = self.summarize_analyses(analyses)
        return {
            "chunk_count": len(chunks),
            "chunk_analyses": analyses,
            "final_summary": final_summary,
        }

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        return self._chunker.embed([text], is_query=is_query)[0]

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        logger.info(f"EMBED_LOCAL chunks={len(chunks)}")
        return self._chunker.embed(chunks, is_query=False)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query, is_query=True)

    @property
    def chunker(self) -> SemanticChunker:
        return self._chunker