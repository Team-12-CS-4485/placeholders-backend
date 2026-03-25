"""
pipeline.py - Consolidated YouTube Ingestion Pipeline

Single weekly run across all NEWS_CHANNELS with cookie-authenticated yt-dlp,
resumable archive tracking, and 429 retry/skip logging.

Usage:
    python pipeline.py
"""

import json
import logging
import sys
import time as _time
from datetime import datetime, timedelta, timezone

import boto3
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
)
from youtube_ingestion import (
    build_client,
    get_uploads_playlist,
    get_latest_videos,
    get_video_statistics,
    get_top_comments,
    get_video_transcript,
    save_to_dynamodb,
    get_existing_video_ids,
    is_within_duration_limit,
    refresh_cookies,
    _COOKIE_PATH,
)
from rate_limit import QuotaExhaustedError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARCHIVE_LOCAL = "/tmp/archive.txt"
_SKIPPED_LOG_S3_KEY = "logs/skipped_videos.txt"

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

    1. Export fresh cookies from Chrome to /tmp/cookies.txt (hard-fail if Chrome unavailable)
    2. Load /tmp/archive.txt for resume support
    3. For each channel: fetch → dedup → filter → save → archive
    4. Upload skipped video log to S3

    dry_run=True: runs all API calls and filters but skips all writes
    (DynamoDB, S3, archive). Useful for testing without consuming quota.
    """
    if dry_run:
        logger.info("=== DRY RUN — no data will be written ===")
    logger.info("=== YouTube Ingestion Pipeline starting ===")

    # Step 1 — cookies
    refresh_cookies(_COOKIE_PATH)

    # Step 2 — archive
    processed_archive = load_archive(_ARCHIVE_LOCAL)
    logger.info(f"Archive loaded: {len(processed_archive)} previously processed video IDs")

    # Shared state
    youtube = build_client(YOUTUBE_API_KEY)
    skipped_videos: list = []
    results = []
    retry_channels: list = []  # channels that hit 429 during the main pass
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
                transcript = get_video_transcript(
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
            msg = str(exc)
            if "429" in msg or "Too Many Requests" in msg.lower():
                logger.warning(f"[{channel_name}] 429 rate-limited — will retry after all channels complete")
                retry_channels.append((channel_name, channel_id))
            else:
                logger.error(f"[{channel_name}] Unexpected error: {exc}", exc_info=True)
            results.append({"channel": channel_name, "status": "error", "error": msg})

        # Inter-channel sleep
        if idx < len(NEWS_CHANNELS):
            logger.info("[rate-limit] Sleeping 60s before next channel...")
            _time.sleep(60)

    # Step 4 — retry 429'd channels
    if retry_channels:
        logger.info(
            f"[retry-pass] {len(retry_channels)} channel(s) hit 429 — "
            f"waiting 5 minutes before retrying: {[c for c, _ in retry_channels]}"
        )
        _time.sleep(300)

        for channel_name, channel_id in retry_channels:
            # Remove the earlier failed result so the summary reflects the retry outcome
            results = [r for r in results if r["channel"] != channel_name]
            try:
                logger.info(f"[retry-pass] Retrying {channel_name}")

                uploads_playlist = get_uploads_playlist(youtube, channel_id)
                if not uploads_playlist:
                    logger.warning(f"[{channel_name}] Channel not found — skipping")
                    results.append({"channel": channel_name, "status": "error", "error": "not found"})
                    continue

                raw_videos = get_latest_videos(youtube, uploads_playlist, MAX_VIDEOS_PER_CHANNEL)
                if not raw_videos:
                    results.append({"channel": channel_name, "status": "error", "error": "no videos"})
                    continue

                existing_ids = get_existing_video_ids(dynamodb, DYNAMODB_TABLE, channel_name)
                all_seen = existing_ids | processed_archive
                raw_videos = [v for v in raw_videos if v["videoId"] not in all_seen]

                if not raw_videos:
                    logger.info(f"[{channel_name}] All videos already ingested")
                    results.append({"channel": channel_name, "status": "success", "videoCount": 0})
                    continue

                stats_map = get_video_statistics(youtube, [v["videoId"] for v in raw_videos])

                filtered: list = []
                for video in raw_videos:
                    if len(filtered) >= MAX_VIDEOS_PER_CHANNEL:
                        break

                    vid = video["videoId"]

                    try:
                        pub = datetime.fromisoformat(video["publishedAt"].replace("Z", "+00:00"))
                        if pub < cutoff_date:
                            continue
                    except Exception:
                        continue

                    details = stats_map.get(vid, {})
                    stats = details.get("statistics", {})
                    duration = details.get("duration", "")

                    view_count = int(stats.get("viewCount", 0))
                    if view_count < MIN_VIEW_COUNT:
                        continue

                    if not is_within_duration_limit(duration, MAX_VIDEO_DURATION_MINUTES):
                        continue

                    top_comments = get_top_comments(youtube, vid, COMMENTS_PER_VIDEO)
                    if len(top_comments) < COMMENTS_PER_VIDEO:
                        continue

                    transcript = get_video_transcript(vid, skipped_videos=skipped_videos)
                    if not transcript:
                        continue

                    video["transcript"] = transcript
                    video["topComments"] = top_comments
                    video["viewCount"] = view_count
                    video["likeCount"] = int(stats.get("likeCount", 0))
                    video["commentCount"] = int(stats.get("commentCount", 0))
                    filtered.append(video)

                logger.info(
                    f"[retry-pass] [{channel_name}] {len(filtered)}/{MAX_VIDEOS_PER_CHANNEL} videos passed all filters"
                )

                if not filtered:
                    results.append({"channel": channel_name, "status": "error", "error": "no videos passed filters"})
                    continue

                if dry_run:
                    logger.info(f"[DRY-RUN] [retry-pass] [{channel_name}] Would save {len(filtered)} videos to DynamoDB")
                else:
                    try:
                        save_to_dynamodb(dynamodb, DYNAMODB_TABLE, channel_name, filtered)
                    except Exception as exc:
                        logger.error(f"[retry-pass] [{channel_name}] DynamoDB save failed: {exc}")
                        results.append({"channel": channel_name, "status": "error", "error": str(exc)})
                        continue

                    try:
                        _save_channel_to_s3(channel_name, filtered)
                    except Exception as exc:
                        logger.error(f"[retry-pass] [{channel_name}] S3 save failed: {exc}")
                        results.append({"channel": channel_name, "status": "error", "error": str(exc)})
                        continue

                    for video in filtered:
                        add_to_archive(video["videoId"], _ARCHIVE_LOCAL)
                        processed_archive.add(video["videoId"])

                results.append({"channel": channel_name, "status": "success", "videoCount": len(filtered)})
                label = "would save" if dry_run else "saved"
                logger.info(f"[retry-pass] [{channel_name}] Done — {len(filtered)} videos {label}")

            except QuotaExhaustedError:
                logger.error(f"[retry-pass] [{channel_name}] Quota exhausted — stopping")
                results.append({"channel": channel_name, "status": "error", "error": "quota exhausted"})
                break
            except Exception as exc:
                logger.error(f"[retry-pass] [{channel_name}] Failed again: {exc}", exc_info=True)
                results.append({"channel": channel_name, "status": "error", "error": str(exc)})

    # Step 5 — skipped log
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
