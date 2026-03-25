from qdrant_client import QdrantClient
from app.core.config import settings

qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

# Total chunks
info = qdrant.get_collection(settings.qdrant_collection)
print(f"Total chunks: {info.points_count}")

# Unique videos
video_ids = set()
offset = None
while True:
    results, offset = qdrant.scroll(
        collection_name=settings.qdrant_collection,
        limit=100,
        offset=offset,
        with_payload=["transcript_index", "channel"],
    )
    for r in results:
        vid = r.payload.get("transcript_index")
        if vid:
            video_ids.add(vid)
    if offset is None:
        break

print(f"Unique videos indexed: {len(video_ids)}")
