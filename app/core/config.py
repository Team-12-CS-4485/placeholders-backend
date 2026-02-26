import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.s3_bucket = os.getenv("S3_BUCKET", "")
        self.s3_prefix = os.getenv("S3_PREFIX", "youtube-data/")
        self.s3_object_limit = int(os.getenv("S3_OBJECT_LIMIT", "3"))
        self.genai_api_key = os.getenv("GENAI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        self.gemini_model_id = os.getenv("GEMINI_MODEL_ID", "gemini-3-flash-preview")
        self.chunk_size_chars = int(os.getenv("CHUNK_SIZE_CHARS", "6000"))
        self.chunk_overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "400"))
        self.gemini_thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "medium")
        self.analysis_output_file = os.getenv("ANALYSIS_OUTPUT_FILE", "transcript_analysis.txt")


settings = Settings()
