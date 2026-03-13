"""
config.py - Centralized Configuration

Loads all application settings from environment variables with sensible defaults.
Covers S3 storage, Gemini API (model ID, thinking level, embedding model),
text chunking parameters, Qdrant vector DB connection, and output file paths.

See config/.env.example for the full list of supported environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.s3_bucket = os.getenv("S3_BUCKET", "")
        self.s3_prefix = os.getenv("S3_PREFIX", "youtube-data/")
        self.s3_object_limit = int(os.getenv("S3_OBJECT_LIMIT", "3"))
        self.genai_api_key = os.getenv("GENAI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        self.genai_api_keys = [self.genai_api_key]
        for i in range(2, 20):
            key = os.getenv(f"GENAI_API_KEY_{i}") or os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                self.genai_api_keys.append(key)
        self.gemini_model_id = os.getenv("GEMINI_MODEL_ID", "gemini-3-flash-preview")
        self.gemini_thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "medium")
        self.embedding_model_id = os.getenv(
            "EMBEDDING_MODEL_ID", "nomic-ai/nomic-embed-text-v1.5"
        )
        self.semantic_similarity_threshold = float(
            os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.48")
        )
        self.semantic_min_words = int(os.getenv("SEMANTIC_MIN_WORDS", "40"))
        self.semantic_max_words = int(os.getenv("SEMANTIC_MAX_WORDS", "350"))
        self.chunk_size_chars = int(os.getenv("CHUNK_SIZE_CHARS", "6000"))
        self.chunk_overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "400"))
        self.analysis_output_file = os.getenv("ANALYSIS_OUTPUT_FILE", "transcript_analysis.txt")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "transcript_chunks")
        self.dynamodb_table = os.getenv("DYNAMODB_TABLE", "youtube-videos")



settings = Settings()