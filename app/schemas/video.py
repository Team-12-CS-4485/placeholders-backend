from typing import List, Optional

from pydantic import BaseModel


class VideoItem(BaseModel):
    video_id: str
    channel: str
    title: str
    description: str
    published_at: str
    view_count: int
    like_count: int
    comment_count: int

class VideoTopComment(BaseModel):
    author: str
    text: str
    likes: int


class VideoDetailItem(VideoItem):
    transcript: str
    top_comments: List[VideoTopComment]


class VideoListResponse(BaseModel):
    items: List[VideoItem]
    total_returned: int
    next_cursor: Optional[str] = None
