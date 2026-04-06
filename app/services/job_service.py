"""
job_service.py - DynamoDB-backed async job tracker

Stores pipeline job state in the 'pipeline-jobs' DynamoDB table so
endpoints can return immediately and clients can poll
GET /api/pipeline/jobs/{job_id} for status.

Table schema:
  PK: job_id (S)
  Attributes: status, progress, total, errors, failed_videos,
              started_at, completed_at, result
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

TABLE_NAME = "pipeline-jobs"

_dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
_table = _dynamodb.Table(TABLE_NAME)


def _dec(val):
    """Convert numbers to Decimal for DynamoDB."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    return val


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job() -> str:
    job_id = str(uuid.uuid4())
    item = {
        "job_id": job_id,
        "status": "running",
        "progress": _dec(0),
        "total": _dec(0),
        "errors": [],
        "failed_videos": [],
        "started_at": _now(),
        "completed_at": None,
        "result": None,
    }
    # DynamoDB doesn't allow None values in top-level attributes on put_item
    # Remove None values
    clean_item = {k: v for k, v in item.items() if v is not None}
    _table.put_item(Item=clean_item)
    logger.info(f"JOB_CREATED job_id={job_id}")
    return job_id


def update_total(job_id: str, total: int) -> None:
    try:
        _table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #total = :total",
            ExpressionAttributeNames={"#total": "total"},
            ExpressionAttributeValues={":total": _dec(total)},
        )
    except Exception as exc:
        logger.error(f"JOB_UPDATE_TOTAL_FAILED job_id={job_id} error={exc}")


def increment_progress(job_id: str, failed_video: str = None) -> None:
    try:
        if failed_video:
            _table.update_item(
                Key={"job_id": job_id},
                UpdateExpression=(
                    "SET #progress = #progress + :one, "
                    "#fv = list_append(if_not_exists(#fv, :empty), :vid)"
                ),
                ExpressionAttributeNames={
                    "#progress": "progress",
                    "#fv": "failed_videos",
                },
                ExpressionAttributeValues={
                    ":one": _dec(1),
                    ":vid": [failed_video],
                    ":empty": [],
                },
            )
        else:
            _table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #progress = #progress + :one",
                ExpressionAttributeNames={"#progress": "progress"},
                ExpressionAttributeValues={":one": _dec(1)},
            )
    except Exception as exc:
        logger.error(f"JOB_INCREMENT_FAILED job_id={job_id} error={exc}")


def complete_job(job_id: str, result: dict) -> None:
    try:
        # DynamoDB can't store floats — recursively convert
        clean_result = _clean_for_dynamo(result) if result else {}
        _table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET #status = :status, #completed = :completed, #result = :result"
            ),
            ExpressionAttributeNames={
                "#status": "status",
                "#completed": "completed_at",
                "#result": "result",
            },
            ExpressionAttributeValues={
                ":status": "complete",
                ":completed": _now(),
                ":result": clean_result,
            },
        )
        logger.info(f"JOB_COMPLETED job_id={job_id}")
    except Exception as exc:
        logger.error(f"JOB_COMPLETE_FAILED job_id={job_id} error={exc}")


def fail_job(job_id: str, error: str) -> None:
    try:
        _table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET #status = :status, #completed = :completed, "
                "#errors = list_append(if_not_exists(#errors, :empty), :err)"
            ),
            ExpressionAttributeNames={
                "#status": "status",
                "#completed": "completed_at",
                "#errors": "errors",
            },
            ExpressionAttributeValues={
                ":status": "failed",
                ":completed": _now(),
                ":err": [error],
                ":empty": [],
            },
        )
        logger.info(f"JOB_FAILED job_id={job_id} error={error}")
    except Exception as exc:
        logger.error(f"JOB_FAIL_WRITE_FAILED job_id={job_id} error={exc}")


def get_job(job_id: str) -> dict | None:
    try:
        resp = _table.get_item(Key={"job_id": job_id})
        item = resp.get("Item")
        if not item:
            return None
        # Convert Decimals back to int/float for JSON serialization
        return _deserialize(item)
    except Exception as exc:
        logger.error(f"JOB_GET_FAILED job_id={job_id} error={exc}")
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _deserialize(obj):
    """boto3 Decimal → int/float, set → list for JSON serialization."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deserialize(i) for i in obj]
    if isinstance(obj, set):
        return sorted(_deserialize(i) for i in obj)
    return obj


def _clean_for_dynamo(obj):
    """Recursively convert floats to Decimal and remove None values for DynamoDB."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return Decimal(str(round(obj, 4)))
    if isinstance(obj, int):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _clean_for_dynamo(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_clean_for_dynamo(i) for i in obj if i is not None]
    return obj
