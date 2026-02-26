from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints.pipeline import router as pipeline_router
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
        logger.error(f"API_FAILURE method={request.method} path={request.url.path} error={exc}")
        raise


@app.get("/health")
def health():
    return {"status": "ok"}
