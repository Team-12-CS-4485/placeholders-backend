# Local Development and Testing Guide

---

## 1. Environment Setup

The project uses two dependency files:

- **`requirements.txt`** — Core dependencies to run the application (FastAPI, AWS clients, AI libraries, Qdrant client)
- **`requirements-dev.txt`** — Development and testing tools (pytest, black, flake8)

### Installation

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install   # sets up black to run on every git commit
```

Or using the Makefile:

```bash
make install
```

---

## 2. Environment Variables

Copy the example env file and fill in your values:

```bash
cp config/.env.example .env
```

Minimum required to run the API locally:

```bash
AWS_REGION=us-east-2
S3_BUCKET=your-bucket
DYNAMODB_TABLE=youtube-videos
GENAI_API_KEY=your-gemini-key
QDRANT_URL=http://localhost:6333
```

The API will start without Qdrant running, but search and pipeline endpoints will fail. All read endpoints (`/api/trends`, `/api/narratives`, `/api/videos`, etc.) only require DynamoDB access.

---

## 3. Running Locally

```bash
# Start Qdrant (only needed for pipeline + search)
docker compose -f docker-compose.qdrant.yml up -d

# Start the API server
uvicorn app.main:app --reload
# Docs at http://localhost:8000/docs
```

### Makefile Commands

| Command            | Description |
|--------------------|-------------|
| `make install`     | Install all dependencies |
| `make run`         | Start FastAPI dev server with auto-reload |
| `make test`        | Run pytest suite |
| `make lint`        | Run flake8 |
| `make format`      | Auto-format with black |
| `make docker-build`| Build Docker image locally |

---

## 4. Test Structure

```
tests/
├── api/          # Endpoint tests (e.g. test_health.py)
├── services/     # Unit tests for service logic
├── integration/  # Multi-service interaction tests
```

### Running Tests

```bash
# All tests
make test
# or
pytest tests/

# Specific file
pytest tests/api/test_health.py

# Verbose
pytest tests/ -v
```

---

## 5. CI/CD Pipeline (GitHub Actions)

Defined in `.github/workflows/ci.yml`. Triggers on every push and pull request to `main`/`master`.

**Steps:**
1. Install Python 3.11 + dependencies
2. Run `flake8` — fails on critical errors (syntax, undefined names)
3. Run `black` — auto-formats and commits back any changes (does not fail the build)
4. Run `pytest tests/`
5. Build Docker image (validates Dockerfile)
6. On merge to `main`: trigger Render deploy via `RENDER_DEPLOY_HOOK_URL` secret

> Render's native auto-deploy should be **disabled**. Deployment is triggered exclusively by CI after all checks pass.

### Pre-commit (local)

`pre-commit install` (run once after cloning) sets up black to auto-format on every `git commit`. This keeps code consistently formatted before it ever reaches CI.

---

## 6. Common Issues

**`ModuleNotFoundError: No module named 'app.schemas.article'`**
The schema file must be at `app/schemas/article.py`. If it exists elsewhere, move it.

**Test collection fails due to missing AWS credentials**
Tests that instantiate services touching DynamoDB/S3 will fail without valid credentials. Either mock the boto3 clients or set up a local DynamoDB (e.g. via `docker run amazon/dynamodb-local`).

**Qdrant connection refused**
Start Qdrant with `docker compose -f docker-compose.qdrant.yml up -d` before running pipeline or search tests.
