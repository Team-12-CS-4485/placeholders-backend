# placeholders-backend

Backend services for the project.

## Current structure

```text
.
├── app/
│   ├── api/endpoints/
│   ├── core/
│   ├── schemas/
│   ├── services/
│   └── workers/
├── config/
├── data/
├── data_collection/
├── docs/
├── scripts/
└── tests/
```

## Where code goes

- `data_collection/`
  - Current/legacy batch ingestion pipeline.
  - `youtube_ingestion.py` is the existing YouTube -> filter -> S3/DynamoDB job.
- `app/main.py`
  - FastAPI entrypoint (app bootstrap and route registration).
- `app/api/endpoints/`
  - HTTP route handlers for frontend APIs (`videos`, `trends`, `pipeline`).
- `app/schemas/`
  - Pydantic request/response models used by endpoints.
- `app/services/`
  - Business logic layer.
  - `ingestion_service.py`: ingestion/filter orchestration.
  - `storage_service.py`: S3/DynamoDB read/write logic.
  - `embedding_service.py`: transcript chunking + Gemini analysis.
  - `pipeline_service.py`: runs multi-step pipeline flow.
- `app/workers/`
  - Background/CLI job entrypoints.
  - `embedding_worker.py` runs transcript analysis from S3 objects.
- `app/core/`
  - Shared app setup (configuration, logging).
- `config/.env.example`
  - Environment variable template.
- `docs/`
  - Architecture, API contract, and pipeline stage notes.
- `tests/`
  - Test buckets by layer (`api`, `services`, `workers`, `integration`).

## Current transcript analysis test flow

- API trigger: `POST /api/pipeline/s3-transcript-analysis`
- Source: first 3 objects under S3 prefix `youtube-data/`.
- Input field used: `transcript` values from each JSON object.
- For each transcript, a key is created in this format:
  - `{s3_object_key}::transcript_{index}`
- Chunk map produced:
  - `chunk_map[transcript_key] = [chunk1, chunk2, ...]`
- Gemini analysis:
  - Each chunk is sent to `gemini-3-flash-preview`.
  - Chunk analyses are merged into one final summary per transcript.
- Analysis map produced:
  - `analysis_map[transcript_key] = {status, chunk_count, chunk_analyses, final_summary, error}`
- Console output:
  - Prints per-transcript success/failure and final pipeline summary.
- Runner: `app/workers/embedding_worker.py`.

## Environment variables for test flow

- `S3_BUCKET`
- `S3_PREFIX` (default: `youtube-data/`)
- `S3_OBJECT_LIMIT` (default: `3`)
- `GENAI_API_KEY`
- `GEMINI_MODEL_ID` (default: `gemini-3-flash-preview`)
- `GEMINI_THINKING_LEVEL` (default: `medium`)
- `CHUNK_SIZE_CHARS` (default: `6000`)
- `CHUNK_OVERLAP_CHARS` (default: `400`)
- `ANALYSIS_OUTPUT_FILE` (default: `transcript_analysis.txt`)

## How to run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set env vars (or copy from `config/.env.example`):
```bash
export S3_BUCKET="your-bucket-name"
export GENAI_API_KEY="your-google-genai-key"
export S3_PREFIX="youtube-data/"
export S3_OBJECT_LIMIT="3"
```

3. Run API server:
```bash
uvicorn app.main:app --reload
```

4. Trigger pipeline via API:
```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/s3-transcript-analysis \
  -H "Content-Type: application/json" \
  -d '{"prefix":"youtube-data/","limit":3}'
```

5. Optional: run pipeline directly (without API):
```bash
python3 -m app.workers.embedding_worker
```
