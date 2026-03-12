# Pipeline Stages

The pipeline is triggered via `POST /api/pipeline/s3-transcript-analysis` and runs synchronously end-to-end.

---

## Overview

```
S3 (raw JSON) → Extract Transcripts → Chunk → Embed → Qdrant Index → Gemini Analysis
```

---

## Stage 1: Load from S3

**Service:** `StorageService.load_transcripts_from_prefix`

Scans S3 under a given prefix (default: `youtube-data/`) up to a configured object limit. Each object is a JSON file with one of two shapes:

- Channel blob: `{ "channel": "...", "videos": [ { "transcript": "...", ... } ] }`
- Single video: `{ "transcript": "..." }`

Transcripts are extracted and deduplicated per object. Objects that fail to load (missing, malformed JSON, decode error) are marked with `status: "failed"` and skipped in downstream stages.

**Config:** `S3_PREFIX`, `S3_OBJECT_LIMIT`, `S3_BUCKET`

---

## Stage 2: Chunk

**Service:** `EmbeddingService.chunk_text`

Each transcript is split into overlapping character-level chunks. Overlap ensures context is preserved across chunk boundaries.

**Config:** `CHUNK_SIZE_CHARS` (default: 6000), `CHUNK_OVERLAP_CHARS` (default: 400)

---

## Stage 3: Embed

**Service:** `EmbeddingService.embed_chunks`

Each chunk is embedded using Google's `text-embedding-004` model via the `google-genai` SDK. Returns a list of float vectors, one per chunk.

**Config:** `EMBEDDING_MODEL_ID`, `GENAI_API_KEY` / `GEMINI_API_KEY`

---

## Stage 4: Index into Qdrant

**Service:** `VectorService.upsert_transcript_chunks`

Upserts vector points into Qdrant. Each point stores:
- The embedding vector
- Payload: `transcript_key`, `source_key`, `transcript_index`, `chunk_index`, chunk text

**Config:** `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`

---

## Stage 5: Analyze with Gemini

**Service:** `EmbeddingService.analyze_chunks` → `EmbeddingService.summarize_analyses`

Each chunk is analyzed individually by Gemini, then a final summary is produced across all chunk analyses. Thinking depth is configurable.

**Config:** `GEMINI_MODEL_ID`, `GEMINI_THINKING_LEVEL` (low / medium / high)

---

## Status Values

| Status           | Meaning                                                    |
|------------------|------------------------------------------------------------|
| `success`        | All transcripts in the object processed without error      |
| `partial_failed` | At least one transcript failed, others succeeded           |
| `failed`         | Object could not be loaded at all (S3/JSON error)          |

---

## Running as a Worker

The pipeline can also be run outside the API as a standalone job:

```bash
python -m app.workers.embedding_worker
```

This runs the full pipeline and writes a summary to `ANALYSIS_OUTPUT_FILE` (default: `transcript_analysis.txt`).
