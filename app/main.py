"""
main.py - FastAPI Application Entry Point

Sets up the Newsify backend API server with:
- CORS middleware (all origins allowed for development)
- HTTP request/response logging middleware (with request ID + elapsed time)
- Global exception handlers (consistent error response shape)
- Pipeline router for transcript analysis and search endpoints
- Trends router for cluster trend data
- Videos router for video listing and detail
- Articles router for generated news articles
- Health check endpoint at GET /health

Run with: uvicorn app.main:app --reload
"""

import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.endpoints.trends import router as trends_router, weeks_router
from app.api.endpoints.pipeline import router as pipeline_router
from app.api.endpoints.videos import router as videos_router
from app.api.endpoints.narratives import router as narratives_router
from app.api.endpoints.search import router as search_router
from app.api.endpoints.stats import router as stats_router
from app.api.endpoints.articles import router as articles_router
from app.core.logging import get_logger, request_id_var
from app.core.rate_limit import limiter

app = FastAPI(title="Placeholders Backend API")
logger = get_logger(__name__)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
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
app.include_router(weeks_router)
app.include_router(narratives_router)
app.include_router(search_router)
app.include_router(stats_router)
app.include_router(articles_router)


# ── Exception handlers ────────────────────────────────────────────────────────


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": exc.detail},
        headers={"X-Request-ID": request_id_var.get()},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": first.get("msg", "Invalid request parameters"),
        },
        headers={"X-Request-ID": request_id_var.get()},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"UNHANDLED_ERROR path={request.url.path} error={exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred"},
        headers={"X-Request-ID": request_id_var.get()},
    )


# ── HTTP logging middleware ───────────────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request_id_var.set(request_id)
    _start = time.time()
    try:
        response = await call_next(request)
        elapsed_ms = int((time.time() - _start) * 1000)
        response.headers["X-Request-ID"] = request_id
        if response.status_code < 400:
            logger.info(
                f"API_SUCCESS method={request.method} path={request.url.path} status={response.status_code} elapsed_ms={elapsed_ms}"
            )
        else:
            logger.error(
                f"API_FAILURE method={request.method} path={request.url.path} status={response.status_code} elapsed_ms={elapsed_ms}"
            )
        return response
    except Exception as exc:
        elapsed_ms = int((time.time() - _start) * 1000)
        logger.error(
            f"API_FAILURE method={request.method} path={request.url.path} error={exc} elapsed_ms={elapsed_ms}"
        )
        raise


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}
