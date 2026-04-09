"""
gemini_service.py - Gemini API Client & Video Analysis

Handles all Google Gemini interactions:
- API client management with key rotation (up to 30 keys)
- Retry logic for rate limits (429) and server errors (503)
- Video intelligence extraction (topics, sentiment, claims)
- Thumbnail analysis (visual tone, clickbait score)
- Combined intelligence + thumbnail in one call
- Chunk-level analysis and summarization
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Optional, Union

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

THUMBNAIL_TONE_OPTIONS = (
    "urgency",
    "fear",
    "anger",
    "sadness",
    "curiosity",
    "hope",
    "neutral",
    "celebratory",
    "alarming",
)


class KeysExhaustedError(Exception):
    """Raised when all keys in a worker's pool are rate-limited."""

    pass


class GeminiService:

    def __init__(
        self,
        client=None,
        chunker: Optional[SemanticChunker] = None,
        api_keys=None,
        fail_on_exhaustion: bool = False,
    ):
        self.model_id = settings.gemini_model_id

        self.api_keys = api_keys or [settings.genai_api_key]
        self.current_key_index = 0
        self.client = client or genai.Client(
            api_key=self.api_keys[0], http_options={"timeout": 60_000}
        )
        self.fail_on_exhaustion = fail_on_exhaustion

        self._chunker = chunker or get_default_chunker(
            model_name=settings.embedding_model_id
        )

    # ── Chunking (kept for analyze_transcript compatibility) ────────────────

    def chunk_text(self, text: str) -> list[str]:
        return self._chunker.chunk(text)

    # ── Gemini helpers ────────────────────────────────────────────────────────

    def _get_text(self, response) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(response)

    def _rotate_key(self) -> bool:
        """Rotate to the next API key. Returns True if a new key is available."""
        next_index = self.current_key_index + 1
        if next_index >= len(self.api_keys):
            logger.error("API_KEY_EXHAUSTED all keys have hit quota")
            return False
        self.current_key_index = next_index
        new_key = self.api_keys[self.current_key_index]
        self.client = genai.Client(api_key=new_key, http_options={"timeout": 60_000})
        logger.warning(
            f"API_KEY_ROTATED key_index={self.current_key_index}/{len(self.api_keys)-1}"
        )
        return True

    @staticmethod
    def _is_retryable(exc) -> tuple[bool, str]:
        """Check if a Gemini exception is retryable. Returns (retryable, type)."""
        exc_str = str(exc)
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

        if (
            "429" in exc_str
            or "RESOURCE_EXHAUSTED" in exc_str
            or "rate" in exc_str.lower()
            or "quota" in exc_str.lower()
            or "too many" in exc_str.lower()
            or status == 429
        ):
            return True, "rate_limit"

        if (
            "503" in exc_str
            or "500" in exc_str
            or "UNAVAILABLE" in exc_str
            or "INTERNAL" in exc_str
            or "overloaded" in exc_str.lower()
            or "disconnected" in exc_str.lower()
            or "without sending a response" in exc_str.lower()
            or status in (500, 503)
        ):
            return True, "server_error"

        return False, "unknown"

    def _gemini(self, prompt: Union[str, list], max_retries: int = 6) -> str:
        contents = prompt if isinstance(prompt, list) else [prompt]
        last_exc = None
        for attempt in range(max_retries * len(self.api_keys)):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level=settings.gemini_thinking_level,
                        ),
                        max_output_tokens=8192,
                    ),
                )
                time.sleep(0.5)  # pace RPM across keys
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    logger.debug(
                        f"GEMINI_TOKENS in={usage.prompt_token_count}"
                        f" out={usage.candidates_token_count}"
                        f" total={usage.total_token_count}"
                    )
                return self._get_text(response)
            except Exception as exc:
                last_exc = exc
                retryable, err_type = self._is_retryable(exc)

                if not retryable:
                    raise

                if err_type == "rate_limit":
                    if self._rotate_key():
                        logger.warning(
                            f"GEMINI_KEY_ROTATED attempt={attempt+1} retrying immediately"
                        )
                        continue
                    if self.fail_on_exhaustion:
                        raise KeysExhaustedError(
                            f"All {len(self.api_keys)} keys exhausted for this worker"
                        )
                    logger.warning(
                        "ALL_KEYS_EXHAUSTED resetting to key 0 and waiting 60s"
                    )
                    self.current_key_index = 0
                    self.client = genai.Client(
                        api_key=self.api_keys[0], http_options={"timeout": 60_000}
                    )
                    time.sleep(60)
                    continue

                if err_type == "server_error":
                    wait = min(5 * (attempt + 1), 30)
                    logger.warning(
                        f"GEMINI_503 attempt={attempt+1} waiting {wait}s before retry"
                    )
                    time.sleep(wait)
                    continue
        raise last_exc

    def _gemini_vision(self, prompt: Union[str, list], max_retries: int = 6) -> str:
        contents = prompt if isinstance(prompt, list) else [prompt]
        last_exc = None
        for attempt in range(max_retries * len(self.api_keys)):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    ),
                )
                time.sleep(0.5)  # pace RPM across keys
                return self._get_text(response)
            except Exception as exc:
                last_exc = exc
                retryable, err_type = self._is_retryable(exc)

                if not retryable:
                    raise

                if err_type == "rate_limit":
                    if self._rotate_key():
                        logger.warning(
                            f"GEMINI_KEY_ROTATED (vision) attempt={attempt+1} "
                            f"retrying immediately"
                        )
                        continue
                    if self.fail_on_exhaustion:
                        raise KeysExhaustedError(
                            f"All {len(self.api_keys)} keys exhausted for this worker"
                        )
                    logger.warning(
                        "ALL_KEYS_EXHAUSTED (vision) resetting to key 0 and waiting 60s"
                    )
                    self.current_key_index = 0
                    self.client = genai.Client(
                        api_key=self.api_keys[0], http_options={"timeout": 60_000}
                    )
                    time.sleep(60)
                    continue

                if err_type == "server_error":
                    wait = min(5 * (attempt + 1), 30)
                    logger.warning(
                        f"GEMINI_VISION_503 attempt={attempt+1} waiting {wait}s "
                        f"before retry"
                    )
                    time.sleep(wait)
                    continue
        raise last_exc

    # ── Thumbnail helpers ─────────────────────────────────────────────────────

    def _fetch_thumbnail(self, video_id: str) -> tuple:
        """
        Try maxresdefault then hqdefault from YouTube's image CDN.
        Returns (bytes, mime_type) on success, (None, '') if unavailable.
        Rejects YouTube's placeholder grey tile (< 5 KB).
        """
        for url in [
            f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        ]:
            try:
                with urllib.request.urlopen(url, timeout=8) as resp:
                    data = resp.read()
                if len(data) > 5000:
                    return data, "image/jpeg"
            except Exception:
                continue
        return None, ""

    def analyze_thumbnail(self, video_id: str, title: str = "") -> dict:
        """
        Fetches the YouTube thumbnail for video_id and extracts structured
        metadata via a single Gemini vision call (thinking_level=low).

        Returns a dict with these keys — safe defaults on any failure:
          thumbnail_visual           str   description of image contents (≤20 words)
          thumbnail_tone             str   one of the THUMBNAIL_TONE_OPTIONS
          thumbnail_clickbait_score  int   1 (editorial) – 5 (pure clickbait); 0 = unknown
          thumbnail_brand_consistent bool  looks like professional broadcast news
          thumbnail_insight          str   one-sentence engagement strategy
        """
        _default = {
            "thumbnail_visual": "",
            "thumbnail_tone": "",
            "thumbnail_clickbait_score": 0,
            "thumbnail_brand_consistent": False,
            "thumbnail_insight": "",
        }

        img_bytes, mime = self._fetch_thumbnail(video_id)
        if not img_bytes:
            logger.warning(f"THUMBNAIL_FETCH_MISS videoId={video_id}")
            return _default

        tone_options = ", ".join(THUMBNAIL_TONE_OPTIONS)
        title_line = f'Video title: "{title}"\n\n' if title else ""

        prompt = (
            f"{title_line}"
            "Analyze this YouTube news thumbnail. "
            "Return ONLY a valid JSON object with exactly these keys:\n"
            "{\n"
            '  "thumbnail_visual": "<key visual elements and text overlays, '
            '≤20 words>",\n'
            '  "thumbnail_tone": "<tone>",\n'
            '  "thumbnail_clickbait_score": <1-5>,\n'
            '  "thumbnail_brand_consistent": <true|false>,\n'
            '  "thumbnail_insight": "<one sentence on engagement strategy>"\n'
            "}\n\n"
            "Rules:\n"
            f"- thumbnail_tone: exactly one of: {tone_options}\n"
            "- thumbnail_clickbait_score: 1=purely editorial, 5=pure clickbait\n"
            "- thumbnail_brand_consistent: true if it looks like "
            "professional broadcast news\n"
            "- thumbnail_visual: objective description, under 20 words\n"
            "- thumbnail_insight: one sentence, no more\n"
            "Return raw JSON only, no markdown fences, no explanation."
        )

        raw = ""
        try:
            contents = [
                types.Part.from_bytes(data=img_bytes, mime_type=mime),
                types.Part.from_text(text=prompt),
            ]
            raw = self._gemini_vision(contents)
            clean = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            data = json.loads(clean)

            score = data.get("thumbnail_clickbait_score", 0)
            try:
                score = int(score)
            except (ValueError, TypeError):
                score = 0

            tone = str(data.get("thumbnail_tone", "")).lower().strip()
            if tone not in THUMBNAIL_TONE_OPTIONS:
                tone = "neutral"

            return {
                "thumbnail_visual": str(data.get("thumbnail_visual", "")),
                "thumbnail_tone": tone,
                "thumbnail_clickbait_score": score if 1 <= score <= 5 else 0,
                "thumbnail_brand_consistent": bool(
                    data.get("thumbnail_brand_consistent", False)
                ),
                "thumbnail_insight": str(data.get("thumbnail_insight", "")),
            }
        except (json.JSONDecodeError, Exception) as exc:
            logger.error(
                f"THUMBNAIL_PARSE_ERROR videoId={video_id} "
                f"error={exc} raw={raw[:200]}"
            )
            return _default

    # ── Combined intelligence + thumbnail — ONE call per video ────────────────

    def extract_full_video_intelligence(
        self,
        chunks: list[str],
        title: str = "",
        top_comments: list[str] = None,
        video_id: str = "",
    ) -> tuple[dict, dict]:
        """
        Single Gemini call per video that extracts BOTH video intelligence
        AND thumbnail analysis. Uses a multimodal prompt when a thumbnail
        is available, otherwise falls back to text-only.

        Returns (intelligence_dict, thumbnail_dict).
        """
        full_text = "\n\n".join(chunks)
        title_line = f'Video title: "{title}"\n\n' if title else ""

        comments_block = ""
        if top_comments:
            comments_text = "\n".join(f"- {c}" for c in top_comments if c.strip())
            if comments_text:
                comments_block = f"\n\nTop viewer comments:\n{comments_text}\n"

        # Try to fetch thumbnail
        img_bytes, mime = None, ""
        if video_id:
            img_bytes, mime = self._fetch_thumbnail(video_id)

        has_thumbnail = img_bytes is not None
        tone_options = ", ".join(THUMBNAIL_TONE_OPTIONS)

        # Build JSON schema section
        json_schema = (
            '  "topics": ["topic1", "topic2"],\n'
            '  "category": "<category>",\n'
            '  "sentiment": "<positive|negative|neutral>",\n'
            '  "key_claims": ["claim1", "claim2"],\n'
            '  "is_breaking": <true|false>,\n'
            '  "public_sentiment": "<positive|negative|neutral|mixed>",\n'
            '  "public_sentiment_score": <float between -1.0 and 1.0>'
        )
        if has_thumbnail:
            json_schema += ",\n"
            json_schema += (
                '  "thumbnail_visual": '
                '"<key visual elements, ≤20 words>",\n'
                '  "thumbnail_tone": "<tone>",\n'
                '  "thumbnail_clickbait_score": <1-5>,\n'
                '  "thumbnail_brand_consistent": <true|false>,\n'
                '  "thumbnail_insight": '
                '"<one sentence on engagement strategy>"'
            )

        # Build rules section
        rules = (
            "Rules:\n"
            "- topics: 3 to 5 broad reusable tags like 'Iran Conflict', "
            "'Oil Markets', 'Drone Warfare'. NOT headline-specific phrases. "
            "Should apply across multiple videos on the same story.\n"
            f"- category: must be exactly one of these options: "
            f"{CATEGORY_OPTIONS_STR}\n"
            "- sentiment: overall tone of the coverage "
            "(the creator/video perspective)\n"
            "- key_claims: up to 5 specific factual claims made in the "
            "video, one sentence each\n"
            "- is_breaking: true if the transcript uses "
            "urgent/developing/breaking language\n"
            "- public_sentiment: audience reaction based on the viewer "
            "comments. positive = supportive/agreeing, "
            "negative = critical/angry, neutral = informational, "
            "mixed = divided opinions. If no comments provided, "
            "default to neutral.\n"
            "- public_sentiment_score: -1.0 (very negative) to "
            "1.0 (very positive), 0.0 = neutral. "
            "Based on viewer comments. If no comments, use 0.0.\n"
        )
        if has_thumbnail:
            rules += (
                f"- thumbnail_tone: exactly one of: {tone_options}\n"
                "- thumbnail_clickbait_score: 1=purely editorial, "
                "5=pure clickbait\n"
                "- thumbnail_brand_consistent: true if it looks like "
                "professional broadcast news\n"
                "- thumbnail_visual: objective description of the attached "
                "thumbnail, under 20 words\n"
                "- thumbnail_insight: one sentence, no more\n"
            )

        prompt_text = (
            f"{title_line}"
            f"Below is a news video transcript.\n\n"
            f"{full_text}"
            f"{comments_block}\n\n"
            f"Return ONLY a valid JSON object with these keys:\n"
            f"{{\n{json_schema}\n}}\n\n"
            f"{rules}"
            "Return raw JSON only, no markdown fences, no explanation."
        )

        # Build contents (multimodal if thumbnail available)
        if has_thumbnail:
            contents = [
                types.Part.from_bytes(data=img_bytes, mime_type=mime),
                types.Part.from_text(text=prompt_text),
            ]
        else:
            contents = [prompt_text]

        _thumb_default = {
            "thumbnail_visual": "",
            "thumbnail_tone": "",
            "thumbnail_clickbait_score": 0,
            "thumbnail_brand_consistent": False,
            "thumbnail_insight": "",
        }
        _intel_default = {
            "topics": [],
            "category": "Other",
            "sentiment": "neutral",
            "key_claims": [],
            "is_breaking": False,
            "public_sentiment": "neutral",
            "public_sentiment_score": 0.0,
        }

        raw = ""
        try:
            raw = (
                self._gemini_vision(contents)
                if has_thumbnail
                else self._gemini(contents)
            )
            clean = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            data = json.loads(clean)

            # ── Parse intelligence fields ──
            public_sentiment = data.get("public_sentiment", "neutral")
            if public_sentiment not in ("positive", "negative", "neutral", "mixed"):
                public_sentiment = "neutral"

            public_sentiment_score = data.get("public_sentiment_score", 0.0)
            try:
                public_sentiment_score = max(
                    -1.0, min(1.0, float(public_sentiment_score))
                )
            except (ValueError, TypeError):
                public_sentiment_score = 0.0

            intelligence = {
                "topics": data.get("topics", [])[:5],
                "category": (
                    data.get("category", "Other")
                    if data.get("category") in CATEGORY_OPTIONS
                    else "Other"
                ),
                "sentiment": (
                    data.get("sentiment", "neutral")
                    if data.get("sentiment") in ("positive", "negative", "neutral")
                    else "neutral"
                ),
                "key_claims": data.get("key_claims", [])[:5],
                "is_breaking": bool(data.get("is_breaking", False)),
                "public_sentiment": public_sentiment,
                "public_sentiment_score": public_sentiment_score,
            }

            # ── Parse thumbnail fields ──
            if has_thumbnail:
                score = data.get("thumbnail_clickbait_score", 0)
                try:
                    score = int(score)
                except (ValueError, TypeError):
                    score = 0

                tone = str(data.get("thumbnail_tone", "")).lower().strip()
                if tone not in THUMBNAIL_TONE_OPTIONS:
                    tone = "neutral"

                thumbnail = {
                    "thumbnail_visual": str(data.get("thumbnail_visual", "")),
                    "thumbnail_tone": tone,
                    "thumbnail_clickbait_score": (score if 1 <= score <= 5 else 0),
                    "thumbnail_brand_consistent": bool(
                        data.get("thumbnail_brand_consistent", False)
                    ),
                    "thumbnail_insight": str(data.get("thumbnail_insight", "")),
                }
            else:
                thumbnail = _thumb_default

            return intelligence, thumbnail

        except (json.JSONDecodeError, Exception) as exc:
            logger.error(
                f"GEMINI_COMBINED_PARSE_ERROR videoId={video_id} "
                f"error={exc} raw={raw[:200]}"
            )
            return _intel_default, _thumb_default

    # ── Chunk-level analysis (optional deep analysis) ────────────────────────

    def analyze_chunks(self, chunks: list[str]) -> list[str]:
        analyses = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            prompt = (
                f"Analyze transcript chunk {idx}/{total}. "
                "Return key points, notable claims, and a concise "
                "summary:\n\n" + chunk
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
            "Combine the chunk analyses into one final transcript report "
            "with: overall summary, key themes, key claims, and potential "
            "follow-up questions.\n\n" + joined
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
