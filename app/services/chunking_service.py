from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

NOMIC_PASSAGE_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


# ── Sentence splitter ────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:

    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    merged: list[str] = []
    for fragment in raw:
        if merged and len(fragment.split()) < 4:
            merged[-1] = merged[-1] + " " + fragment
        else:
            merged.append(fragment)
    return [s.strip() for s in merged if s.strip()]


# ── SemanticChunker ──────────────────────────────────────────────────────────


class SemanticChunker:

    _model = None
    _model_name_loaded: Optional[str] = None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        similarity_threshold: float = 0.48,
        min_words_per_chunk: int = 40,
        max_words_per_chunk: int = 350,
        use_nomic_prefix: bool = True,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_words_per_chunk = min_words_per_chunk
        self.max_words_per_chunk = max_words_per_chunk
        self.use_nomic_prefix = use_nomic_prefix
        self._available: Optional[bool] = None  # deferred — loaded on first use

    def _try_load_model(self) -> bool:
        if (
            SemanticChunker._model is not None
            and SemanticChunker._model_name_loaded == self.model_name
        ):
            return True
        try:
            from sentence_transformers import SentenceTransformer

            SemanticChunker._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
            )
            SemanticChunker._model_name_loaded = self.model_name
            logger.info(f"CHUNKER_READY model={self.model_name}")
            return True
        except ImportError:
            logger.error(
                "CHUNKER_LOAD_FAILED sentence-transformers not installed — "
                "fix: pip install sentence-transformers"
            )
            return False
        except Exception as exc:
            logger.error(f"CHUNKER_LOAD_FAILED model={self.model_name} error={exc}")
            return False

    @property
    def model(self):
        return SemanticChunker._model

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._try_load_model()
        return self._available

    def chunk(self, text: str) -> list[str]:

        if not text or not text.strip():
            return []
        if self.available:
            try:
                return self._semantic_chunk(text)
            except Exception as exc:
                logger.warning(
                    f"SEMANTIC_CHUNK_ERROR error={exc} — falling back to char chunking"
                )
        return self._character_chunk(text)

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:

        if not self.available:
            raise RuntimeError(
                "Local model not loaded. "
                "Ensure sentence-transformers is installed: pip install sentence-transformers"
            )
        if not texts:
            return []

        if self.use_nomic_prefix:
            prefix = NOMIC_QUERY_PREFIX if is_query else NOMIC_PASSAGE_PREFIX
            prefixed = [prefix + t for t in texts]
        else:
            prefixed = texts

        vectors = self.model.encode(
            prefixed,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
        )
        return [v.tolist() for v in vectors]

    def _semantic_chunk(self, text: str) -> list[str]:
        import numpy as np  # type: ignore

        sentences = _split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        prefixed = (
            [NOMIC_PASSAGE_PREFIX + s for s in sentences]
            if self.use_nomic_prefix
            else sentences
        )
        embeddings = self.model.encode(
            prefixed,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=64,
        )

        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            current_words = len(" ".join(current).split())

            if current_words >= self.max_words_per_chunk:
                chunks.append(" ".join(current))
                current = [sentences[i]]
                continue

            start_idx = i - len(current)
            centroid = np.mean(embeddings[start_idx:i], axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
            similarity = float(np.dot(centroid_norm, embeddings[i]))

            if (
                similarity < self.similarity_threshold
                and current_words >= self.min_words_per_chunk
            ):
                chunks.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            chunks.append(" ".join(current))

        logger.debug(
            f"SEMANTIC_CHUNK_DONE sentences={len(sentences)} chunks={len(chunks)}"
        )
        return chunks

    def _character_chunk(
        self, text: str, chunk_size: int = 1500, overlap: int = 150
    ) -> list[str]:

        clean = text.strip()
        chunks, start = [], 0
        while start < len(clean):
            end = start + chunk_size
            chunks.append(clean[start:end])
            if end >= len(clean):
                break
            start = end - overlap
        logger.warning(
            f"CHAR_CHUNK_FALLBACK chunks={len(chunks)} — semantic model unavailable"
        )
        return chunks


# ── Module-level singleton ───────────────────────────────────────────────────

_default_chunker: Optional[SemanticChunker] = None


def get_default_chunker(model_name: str = DEFAULT_MODEL) -> SemanticChunker:
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = SemanticChunker(model_name=model_name)
    return _default_chunker


def semantic_chunk(text: str) -> list[str]:
    return get_default_chunker().chunk(text)
