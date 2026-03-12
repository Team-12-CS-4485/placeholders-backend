# Frontend API Endpoints

Base URL: `http://localhost:8000` (local)

---

## Health

### `GET /health`
Returns server status.

**Response**
```json
{ "status": "ok" }
```

---

## Videos

### `GET /api/videos`
Returns a paginated list of ingested videos. Does not include transcripts or comments.

**Query Parameters**

| Parameter | Type   | Default | Max | Description                          |
|-----------|--------|---------|-----|--------------------------------------|
| `limit`   | int    | 20      | 100 | Number of videos to return           |
| `cursor`  | string | —       | —   | Pagination token from previous response |

**Response**
```json
{
  "items": [
    {
      "video_id": "abc123",
      "channel": "CNBC",
      "title": "Market Update",
      "description": "...",
      "published_at": "2025-01-15T12:00:00+00:00",
      "view_count": 142000,
      "like_count": 3200,
      "comment_count": 410
    }
  ],
  "total_returned": 20,
  "next_cursor": "eyJQYXJ0aXRpb25LZXkiOiAiQ05CQyJ9"
}
```

**Pagination**

Pass `next_cursor` from the response as the `cursor` param on the next request. When `next_cursor` is `null`, you have reached the last page.

```
GET /api/videos?limit=20
GET /api/videos?limit=20&cursor=<next_cursor>
```

---

## Pipeline

### `POST /api/pipeline/s3-transcript-analysis`
Triggers the full pipeline: loads transcripts from S3, chunks and embeds them, indexes into Qdrant.

**Request Body**
```json
{
  "prefix": "youtube-data/",
  "limit": 3
}
```
Both fields are optional and fall back to env defaults (`S3_PREFIX`, `S3_OBJECT_LIMIT`).

**Response**
```json
{
  "prefix": "youtube-data/",
  "object_limit": 3,
  "objects_processed": 3,
  "transcripts_found": 6,
  "transcripts_analyzed": 6,
  "qdrant_collection": "transcript_chunks",
  "qdrant_points_indexed": 48,
  "results": [...]
}
```

Each item in `results` has a `status` of `"success"`, `"failed"`, or `"partial_failed"`.

---

### `POST /api/pipeline/search`
Semantic search over indexed transcript chunks using a text query.

**Request Body**
```json
{
  "query": "federal reserve interest rate decision",
  "limit": 5
}
```
`limit` is optional, defaults to 5.

**Response**
```json
{
  "collection": "transcript_chunks",
  "query": "federal reserve interest rate decision",
  "limit": 5,
  "hits": [
    {
      "score": 0.91,
      "transcript_key": "youtube-data/week1/cnbc.json::transcript_1",
      "source_key": "youtube-data/week1/cnbc.json",
      "chunk_index": 2,
      "text": "..."
    }
  ]
}
```
