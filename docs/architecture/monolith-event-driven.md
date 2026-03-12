# Architecture: Monolith vs Event-Driven

## Current State: Synchronous Monolith

All pipeline stages (S3 load → chunk → embed → index → analyze) run synchronously within a single API request. The HTTP call to `POST /api/pipeline/s3-transcript-analysis` blocks until everything is complete.

```
HTTP Request
    └── PipelineService
            ├── StorageService   (S3 fetch)
            ├── EmbeddingService (chunk + embed + analyze via Gemini)
            └── VectorService    (Qdrant upsert)
HTTP Response
```

**Tradeoffs:**
- Simple to run and debug — one process, one log stream
- Request timeouts are a real risk as the number of S3 objects grows
- Gemini API calls and embedding are slow; a single object with multiple transcripts can take 30+ seconds
- No retry logic if a stage fails partway through

---

## Target State: Event-Driven

The intended direction is to decouple ingestion from processing. YouTube ingestion writes to S3/DynamoDB, which triggers downstream processing asynchronously.

```
YouTube Ingestion (data_collection/)
    └── writes to S3 + DynamoDB

S3 Event / Queue
    └── Embedding Worker (app/workers/embedding_worker.py)
            ├── chunk + embed
            ├── Qdrant upsert
            └── Gemini analysis
```

**Benefits:**
- Ingestion and processing scale independently
- Worker can retry failed objects without affecting the API
- API stays responsive — pipeline status would be polled rather than waited on

---

## Current Gap

`embedding_worker.py` already exists as a standalone job but is not yet wired to any event source (e.g., SQS, S3 event notification, or a scheduled trigger). It currently has to be run manually or called via the API.

The next step to move toward event-driven would be:
1. Configure an S3 event notification or SQS queue to trigger on new object uploads
2. Have the worker consume from that queue instead of scanning S3 on demand
3. Add a status tracking mechanism (e.g., a DynamoDB table) so the API can report processing state per object
