"""
pipeline.py - Consolidated YouTube Ingestion Pipeline

Single weekly run across all NEWS_CHANNELS with cookie-authenticated yt-dlp,
resumable archive tracking, and 429 retry/skip logging.

Usage:
    python pipeline.py
"""

import json
import logging
import os
import re
import sys
import time as _time
import tempfile
from datetime import datetime, timedelta, timezone

import boto3
import yt_dlp
from botocore.exceptions import ClientError

from config import (
    YOUTUBE_API_KEY,
    S3_BUCKET,
    DYNAMODB_TABLE,
    NEWS_CHANNELS,
    MAX_VIDEOS_PER_CHANNEL,
    MAX_VIDEO_DURATION_MINUTES,
    MIN_VIEW_COUNT,
    TIME_WINDOW_DAYS,
    COMMENTS_PER_VIDEO,
    MAX_TRANSCRIPT_CHARS,
)
from youtube_ingestion import (
    build_client,
    get_uploads_playlist,
    get_latest_videos,
    get_video_statistics,
    get_top_comments,
    save_to_dynamodb,
    get_existing_video_ids,
    is_within_duration_limit,
)
from rate_limit import QuotaExhaustedError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARCHIVE_LOCAL = "/tmp/archive.txt"
_SKIPPED_LOG_S3_KEY = "logs/skipped_videos.txt"
_TRANSCRIPT_RETRY_DELAYS = [60, 120, 300]  # seconds — mirrors rate_limit._BACKOFF_DELAYS

# ---------------------------------------------------------------------------
# AWS clients
# ---------------------------------------------------------------------------

s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Archive helpers (manual, because extract_info(download=False) never
# triggers yt-dlp's native archive writer)
# ---------------------------------------------------------------------------

def load_archive(archive_path: str = _ARCHIVE_LOCAL) -> set:
    """
    Return the set of video IDs already recorded in archive_path.
    Returns an empty set if the file does not exist yet.
    Format per line: 'youtube <video_id>'
    """
    ids: set = set()
    try:
        with open(archive_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("youtube "):
                    ids.add(line[len("youtube "):])
    except FileNotFoundError:
        pass
    return ids


def add_to_archive(video_id: str, archive_path: str = _ARCHIVE_LOCAL) -> None:
    """Append a successfully-processed video ID to archive_path."""
    with open(archive_path, "a", encoding="utf-8") as fh:
        fh.write(f"youtube {video_id}\n")


# ---------------------------------------------------------------------------
# Transcript fetching with inline 429 retry
# ---------------------------------------------------------------------------

def get_transcript(
    video_id: str,
    skipped_videos: list = None,
) -> str | None:
    """
    Fetch the English VTT transcript for video_id via yt-dlp.

    - Authenticates via Chrome cookies (cookiesfrombrowser).
    - Uses randomised human-like delays (sleep_interval 3-8s).
    - On HTTP 429: waits and retries up to 3 times (60 / 120 / 300s).
    - After all retries exhausted: appends video_id to skipped_videos, returns None.
    - Non-429 yt-dlp errors: logs warning, returns None (normal filter result).
    - Truncates transcript to MAX_TRANSCRIPT_CHARS.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    last_exc = None

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            "noplaylist": True,
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "sleep_interval": 3,
            "max_sleep_interval": 8,
            "sleep_interval_requests": 1,
            "ignoreconfig": True,
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        }

        for attempt, delay in enumerate([0] + _TRANSCRIPT_RETRY_DELAYS, start=1):
            if delay:
                logger.warning(
                    f"[{video_id}] yt-dlp 429 — waiting {delay}s before attempt {attempt}"
                )
                _time.sleep(delay)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                last_exc = None
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "too many requests" in msg:
                    logger.warning(
                        f"[{video_id}] yt-dlp 429 on attempt {attempt}/{len(_TRANSCRIPT_RETRY_DELAYS) + 1}"
                    )
                    last_exc = exc
                    continue
                # Non-429 error — no retry needed
                logger.warning(f"[{video_id}] yt-dlp error: {exc}")
                return None
        else:
            # All retries exhausted on 429
            logger.error(f"[{video_id}] Permanently rate-limited after all retries — skipping")
            if skipped_videos is not None:
                skipped_videos.append(video_id)
            return None

        # Find the subtitle file yt-dlp wrote to tmpdir (pick largest if multiple)
        vtt_files = [
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if f.endswith(".vtt")
        ]
        if not vtt_files:
            return None

        with open(max(vtt_files, key=os.path.getsize), "r", encoding="utf-8") as fh:
            vtt_content = fh.read()

        # Parse VTT into plain text (deduplicated)
        lines = []
        seen: set = set()
        for line in vtt_content.splitlines():
            line = line.strip()
            if not line or line.startswith("WEBVTT") or "-->" in line or re.match(r"^\d+$", line):
                continue
            line = re.sub(r"<[^>]+>", "", line)
            if line and line not in seen:
                seen.add(line)
                lines.append(line)

        transcript = " ".join(lines)
        if not transcript:
            return None

        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[:MAX_TRANSCRIPT_CHARS]

        return transcript


# ---------------------------------------------------------------------------
# S3 save (timestamp-based key, no week prefix)
# ---------------------------------------------------------------------------

def _save_channel_to_s3(channel_name: str, videos: list) -> None:
    """Save channel data to S3 under a timestamp-based key."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    key = f"youtube-data/{timestamp}/{channel_name.lower()}.json"
    payload = {
        "channel": channel_name,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "videos": videos,
    }
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, default=str),
        ContentType="application/json",
    )
    logger.info(f"Saved {len(videos)} videos to s3://{S3_BUCKET}/{key}")


# ---------------------------------------------------------------------------
# Skipped video log
# ---------------------------------------------------------------------------

def upload_skipped_log(video_ids: list) -> None:
    """
    Append rate-limited video IDs to s3://S3_BUCKET/logs/skipped_videos.txt.
    Downloads the existing file first (if present), appends new timestamped
    entries, then re-uploads.
    """
    if not video_ids:
        logger.info("No skipped videos to log.")
        return

    existing_lines = []
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=_SKIPPED_LOG_S3_KEY)
        existing_lines = obj["Body"].read().decode("utf-8").splitlines()
        logger.info(f"Loaded {len(existing_lines)} existing entries from {_SKIPPED_LOG_S3_KEY}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "NoSuchBucket"):
            logger.info("No existing skipped log — will create new one.")
        else:
            logger.warning(f"Could not read existing skipped log: {exc}")

    run_timestamp = datetime.now(timezone.utc).isoformat()
    new_lines = [f"{run_timestamp}  {vid}" for vid in video_ids]
    body = "\n".join(existing_lines + new_lines) + "\n"

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=_SKIPPED_LOG_S3_KEY,
            Body=body.encode("utf-8"),
            ContentType="text/plain",
        )
        logger.info(
            f"Uploaded {len(new_lines)} skipped video entries to "
            f"s3://{S3_BUCKET}/{_SKIPPED_LOG_S3_KEY}"
        )
    except ClientError as exc:
        logger.error(f"Failed to upload skipped log: {exc}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(partial: bool = False, dry_run: bool = False) -> None:
    """
    Full ingestion run across all NEWS_CHANNELS in a single pass.

    1. Download cookies from S3 (hard-fail if missing)
    2. Load /tmp/archive.txt for resume support
    3. For each channel: fetch → dedup → filter → save → archive
    4. Upload skipped video log to S3

    dry_run=True: runs all API calls and filters but skips all writes
    (DynamoDB, S3, archive). Useful for testing without consuming quota.
    """
    if dry_run:
        logger.info("=== DRY RUN — no data will be written ===")
    logger.info("=== YouTube Ingestion Pipeline starting ===")

    # Step 1 — archive
    processed_archive = load_archive(_ARCHIVE_LOCAL)
    logger.info(f"Archive loaded: {len(processed_archive)} previously processed video IDs")

    # Shared state
    youtube = build_client(YOUTUBE_API_KEY)
    skipped_videos: list = []
    results = []
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=TIME_WINDOW_DAYS)

    logger.info(
        f"Processing {len(NEWS_CHANNELS)} channels | "
        f"time window: last {TIME_WINDOW_DAYS} days"
    )

    # Step 3 — channel loop
    for idx, (channel_name, channel_id) in enumerate(NEWS_CHANNELS.items(), start=1):
        try:
            logger.info(f"[{idx}/{len(NEWS_CHANNELS)}] {channel_name}")

            uploads_playlist = get_uploads_playlist(youtube, channel_id)
            if not uploads_playlist:
                logger.warning(f"[{channel_name}] Channel not found — skipping")
                results.append({"channel": channel_name, "status": "error", "error": "not found"})
                continue

            raw_videos = get_latest_videos(youtube, uploads_playlist, MAX_VIDEOS_PER_CHANNEL)
            if not raw_videos:
                logger.warning(f"[{channel_name}] No videos returned")
                results.append({"channel": channel_name, "status": "error", "error": "no videos"})
                continue

            # Dedup: DynamoDB existing IDs + local archive
            existing_ids = get_existing_video_ids(dynamodb, DYNAMODB_TABLE, channel_name)
            all_seen = existing_ids | processed_archive
            before = len(raw_videos)
            raw_videos = [v for v in raw_videos if v["videoId"] not in all_seen]
            logger.info(
                f"[{channel_name}] {before} fetched, "
                f"{before - len(raw_videos)} already seen, {len(raw_videos)} to process"
            )

            if not raw_videos:
                logger.info(f"[{channel_name}] All videos already ingested")
                results.append({"channel": channel_name, "status": "success", "videoCount": 0})
                continue

            # Statistics batch call
            stats_map = get_video_statistics(youtube, [v["videoId"] for v in raw_videos])

            # Per-video filter (cost order: time → views → duration → comments → transcript)
            filtered: list = []
            for video in raw_videos:
                if len(filtered) >= MAX_VIDEOS_PER_CHANNEL:
                    break

                vid = video["videoId"]

                # Time window
                try:
                    pub = datetime.fromisoformat(video["publishedAt"].replace("Z", "+00:00"))
                    if pub < cutoff_date:
                        logger.info(f"[{channel_name}] {vid} — too old, skipping")
                        continue
                except Exception:
                    logger.warning(f"[{channel_name}] {vid} — unparseable publishedAt, skipping")
                    continue

                details = stats_map.get(vid, {})
                stats = details.get("statistics", {})
                duration = details.get("duration", "")

                # View count
                view_count = int(stats.get("viewCount", 0))
                if view_count < MIN_VIEW_COUNT:
                    logger.info(
                        f"[{channel_name}] {vid} — {view_count} views < {MIN_VIEW_COUNT}, skipping"
                    )
                    continue

                # Duration
                if not is_within_duration_limit(duration, MAX_VIDEO_DURATION_MINUTES):
                    logger.info(f"[{channel_name}] {vid} — exceeds duration limit, skipping")
                    continue

                # Comments
                top_comments = get_top_comments(youtube, vid, COMMENTS_PER_VIDEO)
                if len(top_comments) < COMMENTS_PER_VIDEO:
                    logger.info(
                        f"[{channel_name}] {vid} — only {len(top_comments)} comments, skipping"
                    )
                    continue

                # Transcript (most expensive — last)
                transcript = get_transcript(
                    vid,
                    skipped_videos=skipped_videos,
                )
                if not transcript:
                    logger.info(f"[{channel_name}] {vid} — no transcript, skipping")
                    continue

                video["transcript"] = transcript
                video["topComments"] = top_comments
                video["viewCount"] = view_count
                video["likeCount"] = int(stats.get("likeCount", 0))
                video["commentCount"] = int(stats.get("commentCount", 0))
                filtered.append(video)

            logger.info(
                f"[{channel_name}] {len(filtered)}/{MAX_VIDEOS_PER_CHANNEL} videos passed all filters"
            )

            if not filtered:
                results.append({
                    "channel": channel_name,
                    "status": "error",
                    "error": "no videos passed filters",
                })
                continue

            # Persist — DynamoDB first, then S3
            if dry_run:
                logger.info(f"[DRY-RUN] [{channel_name}] Would save {len(filtered)} videos to DynamoDB")
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
                logger.info(f"[DRY-RUN] [{channel_name}] Would save to s3://{S3_BUCKET}/youtube-data/{timestamp}/{channel_name.lower()}.json")
            else:
                try:
                    save_to_dynamodb(dynamodb, DYNAMODB_TABLE, channel_name, filtered)
                except Exception as exc:
                    logger.error(f"[{channel_name}] DynamoDB save failed: {exc}")
                    results.append({"channel": channel_name, "status": "error", "error": str(exc)})
                    continue

                try:
                    _save_channel_to_s3(channel_name, filtered)
                except Exception as exc:
                    logger.error(f"[{channel_name}] S3 save failed: {exc}")
                    results.append({"channel": channel_name, "status": "error", "error": str(exc)})
                    continue

                # Update archive
                for video in filtered:
                    add_to_archive(video["videoId"], _ARCHIVE_LOCAL)
                    processed_archive.add(video["videoId"])

            results.append({
                "channel": channel_name,
                "status": "success",
                "videoCount": len(filtered),
            })
            label = "would save" if dry_run else "saved"
            logger.info(f"[{channel_name}] Done — {len(filtered)} videos {label}")

        except QuotaExhaustedError:
            logger.error(f"[{channel_name}] YouTube API quota exhausted — stopping run")
            results.append({"channel": channel_name, "status": "error", "error": "quota exhausted"})
            break
        except Exception as exc:
            logger.error(f"[{channel_name}] Unexpected error: {exc}", exc_info=True)
            results.append({"channel": channel_name, "status": "error", "error": str(exc)})

        # Inter-channel sleep
        if idx < len(NEWS_CHANNELS):
            logger.info("[rate-limit] Sleeping 60s before next channel...")
            _time.sleep(60)

    # Step 4 — skipped log
    if dry_run:
        if skipped_videos:
            logger.info(f"[DRY-RUN] Would upload {len(skipped_videos)} skipped video IDs to S3")
    else:
        upload_skipped_log(skipped_videos)

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    logger.info(f"=== Pipeline complete: {successful} channels OK, {failed} errors ===")
    for r in results:
        if r["status"] == "success":
            logger.info(f"  {r['channel']}: {r.get('videoCount', 0)} videos")
        else:
            logger.error(f"  {r['channel']}: {r.get('error')}")
    if skipped_videos:
        logger.warning(
            f"  {len(skipped_videos)} videos skipped due to rate-limiting: {skipped_videos}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv
    run_pipeline(dry_run=dry_run)


if __name__ == "__main__":
    main()
