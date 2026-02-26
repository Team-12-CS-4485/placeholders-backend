from google import genai
from google.genai import types
from app.core.config import settings


class EmbeddingService:
    def __init__(self, client=None):
        self.model_id = settings.gemini_model_id
        self.client = client or genai.Client(api_key=settings.genai_api_key)

    def chunk_text(self, text, chunk_size=None, overlap=None):
        use_chunk_size = chunk_size if chunk_size is not None else settings.chunk_size_chars
        use_overlap = overlap if overlap is not None else settings.chunk_overlap_chars

        if not text or not text.strip():
            return []
        if use_chunk_size <= 0:
            return [text.strip()]
        if use_overlap >= use_chunk_size:
            use_overlap = max(0, use_chunk_size // 10)

        chunks = []
        start = 0
        clean_text = text.strip()

        while start < len(clean_text):
            end = start + use_chunk_size
            chunks.append(clean_text[start:end])
            if end >= len(clean_text):
                break
            start = end - use_overlap

        return chunks

    def _get_text(self, response):
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(response)

    def analyze_chunks(self, chunks):
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

        return analyses

    def summarize_analyses(self, analyses):
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
        return self._get_text(response)

    def analyze_transcript(self, transcript):
        chunks = self.chunk_text(transcript)
        analyses = self.analyze_chunks(chunks)
        final_summary = self.summarize_analyses(analyses)
        return {
            "chunk_count": len(chunks),
            "chunk_analyses": analyses,
            "final_summary": final_summary
        }
