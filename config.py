"""
Configuration module for YouTube API and project settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# YouTube API Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MAX_VIDEOS_PER_CHANNEL = int(os.getenv("MAX_VIDEOS_PER_CHANNEL",5))
MIN_VIEW_COUNT = int(os.getenv("MIN_VIEW_COUNT", 5000))

NEWS_CHANNELS = {
    "CNN": "@CNN",
    "CNBC": "@CNBC",
    "BBCNews": "@BBCNews",
    "FoxNews": "@FoxNews",
    "ABCNews": "@ABCNews",
    "NBCNews": "@NBCNews",
    "PBSNewsHour": "@PBSNewsHour",
}

# Validation
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env file. Please set it up.")
