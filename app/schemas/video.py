from typing import List, Optional

from pydantic import BaseModel


class VideoItem(BaseModel):
    video_id: str
    channel: str
    title: str
    published_at: str
    view_count: int
    like_count: int
    comment_count: int
    week: Optional[str] = None
    # AI intelligence fields (written by pipeline, may be absent on older records)
    topics: Optional[List[str]] = None
    category: Optional[str] = None
    sentiment: Optional[str] = None
    key_claims: Optional[List[str]] = None
    is_breaking: Optional[bool] = None
    # Cluster assignment (written by clustering pipeline)
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    # Thumbnail analysis fields
    thumbnail_tone: Optional[str] = None
    thumbnail_clickbait_score: Optional[int] = None
    thumbnail_insight: Optional[str] = None
    thumbnail_brand_consistent: Optional[bool] = None


class VideoTopComment(BaseModel):
    author: str
    text: str
    likes: int


class VideoDetailItem(VideoItem):
    description: str
    transcript: str
    top_comments: List[VideoTopComment]


class VideoListResponse(BaseModel):
    items: List[VideoItem]
    total_returned: int
    next_cursor: Optional[str] = None
