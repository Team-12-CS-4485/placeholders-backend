from __future__ import annotations

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self, client: Optional[QdrantClient] = None):
        self.collection_name = settings.qdrant_collection
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    # ── Collection management ────────────────────────────────────────────────

    def _get_stored_vector_size(self) -> Optional[int]:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.config.params.vectors.size
        except Exception:
            return None

    def ensure_collection(self, vector_size: int) -> None:
        stored_size = self._get_stored_vector_size()

        if stored_size is not None:
            if stored_size != vector_size:
                logger.warning(
                    f"QDRANT_DIMENSION_MISMATCH collection={self.collection_name} "
                    f"stored={stored_size} model_produces={vector_size}. "
                    "Set a new QDRANT_COLLECTION name in .env to use the new model, "
                    "or delete the existing collection."
                )
            return  # collection already exists

        logger.info(
            f"QDRANT_COLLECTION_CREATE name={self.collection_name} "
            f"vector_size={vector_size} distance=COSINE"
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            ),
        )

        # Keyword indexes — exact match + filtering
        keyword_fields = (
            "transcript_key",
            "source_key",
            "transcript_index",
            "channel",
            "category",
            "sentiment",
            "topics",  # array — Qdrant indexes each element
            "thumbnail_tone",  # urgency/fear/anger/etc.
        )
        for field in keyword_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"PAYLOAD_INDEX_CREATED field={field} type=keyword")
            except Exception as exc:
                logger.warning(f"PAYLOAD_INDEX_SKIP field={field} reason={exc}")

        # Bool indexes
        bool_fields = (
            "is_breaking",
            "thumbnail_brand_consistent",
        )
        for field in bool_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                logger.info(f"PAYLOAD_INDEX_CREATED field={field} type=bool")
            except Exception as exc:
                logger.warning(f"PAYLOAD_INDEX_SKIP field={field} reason={exc}")

        # Integer indexes — range queries (e.g. view_count > 100000, clickbait_score >= 4)
        integer_fields = (
            "view_count",
            "like_count",
            "comment_count",
            "thumbnail_clickbait_score",
        )
        for field in integer_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
                logger.info(f"PAYLOAD_INDEX_CREATED field={field} type=integer")
            except Exception as exc:
                logger.warning(f"PAYLOAD_INDEX_SKIP field={field} reason={exc}")

        logger.info(f"QDRANT_COLLECTION_READY name={self.collection_name}")

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)

    def delete_collection(self) -> None:
        """Drop the collection — use when switching embedding models."""
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)
            logger.info(f"QDRANT_COLLECTION_DELETED name={self.collection_name}")

    def add_thumbnail_indexes(self) -> None:
        """
        Backfill payload indexes for thumbnail fields on an existing collection.
        Safe to call multiple times — logs a warning if an index already exists
        but does not raise. Run this once after upgrading from a pre-thumbnail
        collection rather than recreating the entire collection.
        """
        if not self.collection_exists():
            logger.warning("ADD_THUMBNAIL_INDEXES skipped — collection does not exist")
            return

        for field in ("thumbnail_tone",):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"THUMBNAIL_INDEX_CREATED field={field} type=keyword")
            except Exception as exc:
                logger.warning(f"THUMBNAIL_INDEX_SKIP field={field} reason={exc}")

        for field in ("thumbnail_brand_consistent",):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                logger.info(f"THUMBNAIL_INDEX_CREATED field={field} type=bool")
            except Exception as exc:
                logger.warning(f"THUMBNAIL_INDEX_SKIP field={field} reason={exc}")

        for field in ("thumbnail_clickbait_score",):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
                logger.info(f"THUMBNAIL_INDEX_CREATED field={field} type=integer")
            except Exception as exc:
                logger.warning(f"THUMBNAIL_INDEX_SKIP field={field} reason={exc}")

    # ── Upsert ───────────────────────────────────────────────────────────────

    def upsert_transcript_chunks(
        self,
        transcript_key: str,
        source_key: str,
        transcript_index,
        chunks: list[str],
        vectors: list[list[float]],
        extra_metadata: dict = None,
    ) -> int:

        if not chunks or not vectors:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have the same length "
                f"for transcript_key={transcript_key}"
            )

        self.ensure_collection(vector_size=len(vectors[0]))

        base_payload = {
            "transcript_key": transcript_key,
            "source_key": source_key,
            "transcript_index": str(transcript_index),
        }
        if extra_metadata:
            base_payload.update(extra_metadata)

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{transcript_key}:{chunk_idx}")),
                vector=vector,
                payload={
                    **base_payload,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                },
            )
            for chunk_idx, (chunk_text, vector) in enumerate(
                zip(chunks, vectors), start=1
            )
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(
            f"QDRANT_UPSERT_OK transcript_key={transcript_key} "
            f"points={len(points)} vector_size={len(vectors[0])}"
        )
        return len(points)

    # ── Search ───────────────────────────────────────────────────────────────

    def search_similar_chunks(
        self,
        query_vector: list[float],
        limit: int = 5,
        source_key_filter: Optional[str] = None,
    ) -> list[dict]:

        if not self.collection_exists():
            logger.warning("QDRANT_SEARCH_SKIP — collection does not exist")
            return []

        query_filter = None
        if source_key_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_key",
                        match=models.MatchValue(value=source_key_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

        return [
            {
                "id": str(r.id),
                "score": float(r.score),
                "transcript_key": r.payload.get("transcript_key", ""),
                "source_key": r.payload.get("source_key", ""),
                "channel": r.payload.get("channel", ""),
                "title": r.payload.get("title", ""),
                "published_at": r.payload.get("published_at", ""),
                "view_count": r.payload.get("view_count", 0),
                "category": r.payload.get("category", ""),
                "sentiment": r.payload.get("sentiment", ""),
                "topics": r.payload.get("topics", []),
                "key_claims": r.payload.get("key_claims", []),
                "is_breaking": r.payload.get("is_breaking", False),
                "chunk_index": r.payload.get("chunk_index", 0),
                "text": r.payload.get("text", ""),
                # thumbnail fields
                "thumbnail_visual": r.payload.get("thumbnail_visual", ""),
                "thumbnail_tone": r.payload.get("thumbnail_tone", ""),
                "thumbnail_clickbait_score": r.payload.get(
                    "thumbnail_clickbait_score", 0
                ),
                "thumbnail_brand_consistent": r.payload.get(
                    "thumbnail_brand_consistent", False
                ),
                "thumbnail_insight": r.payload.get("thumbnail_insight", ""),
            }
            for r in results
        ]
