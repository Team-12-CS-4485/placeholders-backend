"""
Configuration module for YouTube API and project settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# YouTube API Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
AWS_REGION = os.getenv('AWS_REGION', 'us-east-2')
S3_BUCKET = os.getenv('S3_BUCKET', 'youtube-ingestion-data-us-east-2')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'youtube-videos')

# Data Collection Parameters
MAX_VIDEOS_PER_CHANNEL = int(os.getenv('MAX_VIDEOS_PER_CHANNEL', 5))
MAX_VIDEO_DURATION_MINUTES = int(os.getenv('MAX_VIDEO_DURATION_MINUTES', 30))
MIN_VIEW_COUNT = int(os.getenv('MIN_VIEW_COUNT', 5000))
TIME_WINDOW_DAYS = int(os.getenv('TIME_WINDOW_DAYS', 180))
COMMENTS_PER_VIDEO = int(os.getenv('COMMENTS_PER_VIDEO', 3))
PROXY_USERNAME = os.getenv('PROXY_USERNAME', '')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD', '')
TRANSCRIPT_FALLBACK_URL_TEMPLATE = os.getenv(
    'TRANSCRIPT_FALLBACK_URL_TEMPLATE',
    'https://youtubetranscript.com/?server_vid2={video_id}'
)
TRANSCRIPT_FALLBACK_API_KEY = os.getenv('TRANSCRIPT_FALLBACK_API_KEY', '')
TRANSCRIPT_FALLBACK_AUTH_HEADER = os.getenv('TRANSCRIPT_FALLBACK_AUTH_HEADER', 'X-API-Key')
TRANSCRIPT_REQUEST_TIMEOUT_SECONDS = int(os.getenv('TRANSCRIPT_REQUEST_TIMEOUT_SECONDS', 12))
DYNAMODB_MAX_ITEM_BYTES = int(os.getenv('DYNAMODB_MAX_ITEM_BYTES', 380000))
MAX_TRANSCRIPT_CHARS = int(os.getenv('MAX_TRANSCRIPT_CHARS', 120000))

NEWS_CHANNELS = {
    "CNN": "UCupvZG-5ko_eiXAupbDfxWw",
    "CNBC": "UCvJJ_dzjViJCoLf5uKUTwoA",
    "BBCNews": "UC16niRr50-MSBwiO3YDb3RA",
    "FoxNews": "UCXIJgqnII2ZOINSWNOGFThA",
    "ABCNews": "UCBi2mrWuNuyYy4gbM6fU18Q",
    "NBCNews": "UCeY0bbntWzzVIaj2z3QigXg",
    "PBSNewsHour": "UC6ZFN9Tx6xh-skXCuRHCDpQ",
    'CBSNews': 'UC8p1vwvWtl6T73JiExfWs1g',
    'NewYorkTimes': 'UCqnbDFdCpuN8CMEg0VuEBqA',
    'WashingtonPost': 'UCHd62-u_v4DvJ8TCFtpi4GA',
}


# Validation
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env file. Please set it up.")
