"""
faiss_service.py - FAISS Vector Storage Service with Hybrid Search

Handles local vector storage using FAISS (Facebook AI Similarity Search).
Embeds text chunks using sentence-transformers (all-mpnet-base-v2, 768 dims),
normalizes vectors for cosine similarity, and stores them in a local FAISS index.

Search uses a hybrid approach:
- Semantic search via FAISS (vector cosine similarity)
- Keyword search via BM25 (term frequency scoring)
- Final score = weighted combination of both (default 0.6 semantic + 0.4 keyword)

Index files:
- data/faiss_index.bin : The FAISS vector index
- data/chunk_metadata.pkl : Chunk text and metadata
"""

import math
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from app.core.logging import get_logger

logger = get_logger(__name__)

# Must use same model as chunking_service.py
model = SentenceTransformer("all-mpnet-base-v2")

EMBEDDING_DIMENSION = 768  # all-mpnet-base-v2 outputs 768 dims
INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/chunk_metadata.pkl"


def _tokenize(text):
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _bm25_score(query_tokens, doc_tokens, avg_dl, doc_count, doc_freqs, k1=1.5, b=0.75):
    """Compute BM25 score for a single document against a query."""
    score = 0.0
    dl = len(doc_tokens)
    doc_token_counts = Counter(doc_tokens)

    for token in query_tokens:
        if token not in doc_freqs:
            continue
        tf = doc_token_counts.get(token, 0)
        df = doc_freqs[token]
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        score += idf * tf_norm

    return score


class FAISSService:
    def __init__(self, index_path=None, metadata_path=None):
        self.index_path = index_path or INDEX_PATH
        self.metadata_path = metadata_path or METADATA_PATH

    def load_or_create_index(self):
        """Loads existing FAISS index from disk, or creates a fresh one."""
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            logger.info(f"FAISS_LOAD existing index with {index.ntotal} vectors")
        else:
            index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            logger.info("FAISS_CREATE new index")
        return index

    def load_metadata(self):
        """Loads chunk metadata from disk. Returns empty list if nothing saved yet."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                return pickle.load(f)
        return []

    def save(self, index, metadata):
        """Persists the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path) or "data", exist_ok=True)
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"FAISS_SAVE vectors={index.ntotal}")

    def embed_and_store(self, chunks, source_key="", transcript_index=0, extra_metadata=None):
        """
        Embeds chunks and stores them in the local FAISS index.

        Args:
            chunks (List[str]): Text chunks to embed and store
            source_key (str): S3 key or identifier for the source object
            transcript_index (int): Index of the transcript within the source
            extra_metadata (dict): Additional metadata to store with each chunk

        Returns:
            int: Number of chunks stored
        """
        if not chunks:
            return 0

        index = self.load_or_create_index()
        all_metadata = self.load_metadata()

        # Embed all chunks at once
        embeddings = model.encode(chunks, show_progress_bar=False)

        # Normalize vectors (required for cosine similarity with IndexFlatIP)
        faiss.normalize_L2(embeddings)

        # Add vectors to FAISS index
        index.add(embeddings)

        # Store metadata alongside — FAISS only stores vectors, not text
        for i, chunk in enumerate(chunks):
            meta = {
                "source_key": source_key,
                "transcript_index": transcript_index,
                "chunk_index": i,
                "text": chunk,
                "word_count": len(chunk.split()),
            }
            if extra_metadata:
                meta.update(extra_metadata)
            all_metadata.append(meta)

        # Save everything to disk
        self.save(index, all_metadata)

        logger.info(f"FAISS_STORE chunks={len(chunks)} source={source_key}")
        return len(chunks)

    def _keyword_search(self, query_text, all_metadata, top_k=20):
        """
        BM25 keyword search over stored chunk text.
        Returns indices and scores sorted by relevance.
        """
        if not all_metadata:
            return [], []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return [], []

        # Tokenize all documents
        doc_tokens_list = [_tokenize(m.get("text", "")) for m in all_metadata]
        doc_count = len(doc_tokens_list)
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(doc_count, 1)

        # Compute document frequencies for query terms
        doc_freqs = {}
        for token in set(query_tokens):
            doc_freqs[token] = sum(1 for dt in doc_tokens_list if token in dt)

        # Score each document
        scored = []
        for i, doc_tokens in enumerate(doc_tokens_list):
            score = _bm25_score(query_tokens, doc_tokens, avg_dl, doc_count, doc_freqs)
            if score > 0:
                scored.append((i, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        indices = [s[0] for s in scored]
        scores = [s[1] for s in scored]
        return indices, scores

    def search_similar_chunks(self, query_text, top_k=5, semantic_weight=0.6):
        """
        Hybrid search combining FAISS semantic similarity and BM25 keyword matching.

        Args:
            query_text (str): Search query
            top_k (int): Number of results to return
            semantic_weight (float): Weight for semantic score (0-1). Keyword weight = 1 - semantic_weight.

        Returns:
            List of dicts with chunk text, combined score, and metadata
        """
        index = self.load_or_create_index()
        all_metadata = self.load_metadata()

        if index.ntotal == 0:
            logger.info("FAISS_SEARCH index is empty")
            return []

        keyword_weight = 1.0 - semantic_weight

        # --- Semantic search via FAISS ---
        # Fetch more candidates than needed so we can merge with keyword results
        fetch_k = min(top_k * 3, index.ntotal)
        query_embedding = model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        sem_scores, sem_indices = index.search(query_embedding, fetch_k)

        # Build a map of index -> normalized semantic score (already 0-1 from cosine)
        semantic_map = {}
        for score, idx in zip(sem_scores[0], sem_indices[0]):
            if idx == -1:
                continue
            semantic_map[idx] = max(0.0, float(score))  # cosine similarity 0-1

        # --- Keyword search via BM25 ---
        kw_indices, kw_scores = self._keyword_search(query_text, all_metadata, top_k=fetch_k)

        # Normalize BM25 scores to 0-1 range
        keyword_map = {}
        if kw_scores:
            max_kw = max(kw_scores)
            for idx, score in zip(kw_indices, kw_scores):
                keyword_map[idx] = score / max_kw if max_kw > 0 else 0.0

        # --- Combine scores ---
        all_candidates = set(semantic_map.keys()) | set(keyword_map.keys())
        combined = []
        for idx in all_candidates:
            sem = semantic_map.get(idx, 0.0)
            kw = keyword_map.get(idx, 0.0)
            final_score = (semantic_weight * sem) + (keyword_weight * kw)
            combined.append((idx, final_score, sem, kw))

        # Sort by combined score, take top_k
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:top_k]

        results = []
        for idx, final_score, sem_score, kw_score in combined:
            meta = all_metadata[idx]
            results.append({
                "score": round(final_score, 4),
                "semantic_score": round(sem_score, 4),
                "keyword_score": round(kw_score, 4),
                **meta,
            })

        logger.info(
            f"HYBRID_SEARCH query='{query_text[:50]}' "
            f"hits={len(results)} weights=semantic:{semantic_weight}/keyword:{keyword_weight}"
        )
        return results
