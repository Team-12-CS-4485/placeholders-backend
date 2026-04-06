"""
job_service.py - In-memory async job tracker

Stores pipeline job state so endpoints can return immediately and
clients can poll GET /api/pipeline/jobs/{job_id} for status.

State is lost on server restart — acceptable for pipeline jobs,
which can simply be re-submitted.
"""

import threading
import uuid
from datetime import datetime, timezone

_store: dict = {}
_lock = threading.Lock()


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _store[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "total": 0,
            "errors": [],
            "failed_videos": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "result": None,
        }
    return job_id


def update_total(job_id: str, total: int) -> None:
    with _lock:
        job = _store.get(job_id)
        if job:
            job["total"] = total


def increment_progress(job_id: str, failed_video: str = None) -> None:
    with _lock:
        job = _store.get(job_id)
        if not job:
            return
        job["progress"] += 1
        if failed_video:
            job["failed_videos"].append(failed_video)


def complete_job(job_id: str, result: dict) -> None:
    with _lock:
        job = _store.get(job_id)
        if not job:
            return
        job["status"] = "complete"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["result"] = result


def fail_job(job_id: str, error: str) -> None:
    with _lock:
        job = _store.get(job_id)
        if not job:
            return
        job["status"] = "failed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["errors"].append(error)


def get_job(job_id: str) -> dict | None:
    return _store.get(job_id)
