"""
main.py - FastAPI Application Entry Point

Sets up the Newsify backend API server with:
- CORS middleware (all origins allowed for development)
- HTTP request/response logging middleware
- Pipeline router for transcript analysis and search endpoints
- Health check endpoint at GET /health

Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.endpoints.trends import router as trends_router
from app.api.endpoints.pipeline import router as pipeline_router
from app.api.endpoints.videos import router as videos_router
from app.core.logging import get_logger

app = FastAPI(title="Placeholders Backend API")
logger = get_logger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router)
app.include_router(videos_router)
app.include_router(trends_router)


@app.middleware("http")
async def log_requests(request, call_next):
    try:
        response = await call_next(request)
        if response.status_code < 400:
            logger.info(
                f"API_SUCCESS method={request.method} path={request.url.path} status={response.status_code}"
            )
        else:
            logger.error(
                f"API_FAILURE method={request.method} path={request.url.path} status={response.status_code}"
            )
        return response
    except Exception as exc:
        logger.error(
            f"API_FAILURE method={request.method} path={request.url.path} error={exc}"
        )
        raise


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}
