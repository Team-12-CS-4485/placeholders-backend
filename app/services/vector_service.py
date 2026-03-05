import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings


class VectorService:
    def __init__(self, client=None):
        self.collection_name = settings.qdrant_collection
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    def ensure_collection(self, vector_size):
        collections = self.client.get_collections().collections
        collection_names = {collection.name for collection in collections}
        if self.collection_name in collection_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def collection_exists(self):
        collections = self.client.get_collections().collections
        collection_names = {collection.name for collection in collections}
        return self.collection_name in collection_names

    def upsert_transcript_chunks(self, transcript_key, source_key, transcript_index, chunks, vectors):
        if not chunks or not vectors:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors length mismatch")

        self.ensure_collection(len(vectors[0]))
        points = []
        for chunk_index, (chunk_text, vector) in enumerate(zip(chunks, vectors), start=1):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{transcript_key}:{chunk_index}"))
            payload = {
                "transcript_key": transcript_key,
                "source_key": source_key,
                "transcript_index": transcript_index,
                "chunk_index": chunk_index,
                "text": chunk_text,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search_similar_chunks(self, query_vector, limit=5):
        if not self.collection_exists():
            return []

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

        hits = []
        for result in results:
            payload = result.payload or {}
            hits.append(
                {
                    "id": str(result.id),
                    "score": float(result.score),
                    "transcript_key": payload.get("transcript_key", ""),
                    "source_key": payload.get("source_key", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                }
            )
        return hits
