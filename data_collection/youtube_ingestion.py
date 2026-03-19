"""
youtube_ingestion.py - YouTube Data Collection Pipeline

Collects videos from configured news channels via the YouTube Data API v3.
For each channel, fetches recent uploads and applies multi-stage filtering:
1. Time window (default 180 days)
2. Minimum view count (default 5,000)
3. Maximum duration (default 30 minutes)
4. Minimum comment count (default 3)
5. Transcript availability (extracted via yt-dlp)

Passing videos are enriched with transcripts and top comments, then saved to
both S3 (as JSON for the analysis pipeline) and DynamoDB (for metadata persistence).
"""

from googleapiclient.discovery import build
import yt_dlp
from config import NEWS_CHANNELS, CHANNEL_BATCHES, YOUTUBE_API_KEY, MAX_VIDEOS_PER_CHANNEL, MIN_VIEW_COUNT, TIME_WINDOW_DAYS, COMMENTS_PER_VIDEO, MAX_VIDEO_DURATION_MINUTES
from rate_limit import retry_api_call, retry_transcript, QuotaExhaustedError
import json
import re
import time as _time
from datetime import datetime, time, timedelta, timezone
import os
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal
import logging

# Setup loggingmax
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
logger.propagate = False

# AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE")


def build_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


@retry_api_call
def get_uploads_playlist(youtube, channel_id: str):
    
    response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    if not response.get("items"):
        return None

    #returns id of the channels playlist which has all the uploaded videos
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


@retry_api_call
def get_latest_videos(youtube, uploads_playlist_id: str, max_results=MAX_VIDEOS_PER_CHANNEL):
    #Since we have so many filter lets fetch more videos than we need and then filter down to the top 5 that meet all criteria. This way we have a better chance of getting 5 good videos per channel.
    max_results = max_results * 5  # Fetch more to account for filtering
    response = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=max_results
    ).execute()

    videos = []
    for item in response.get("items", []):
        videos.append({
            "videoId": item["contentDetails"]["videoId"],
            "title": item["snippet"]["title"],
            "description": item["snippet"]["description"],
            "publishedAt": item["snippet"]["publishedAt"],
        })

    return videos


@retry_api_call
def get_video_statistics(youtube, video_ids):
    response = youtube.videos().list(
        part="statistics,contentDetails",
        id=",".join(video_ids)   
    ).execute()

    stats_map = {}
    for item in response.get("items", []):
        stats_map[item["id"]] = {
            "statistics": item["statistics"],
            "duration": item.get("contentDetails", {}).get("duration", "")
        }

    return stats_map


def is_within_duration_limit(duration_str, max_minutes=MAX_VIDEO_DURATION_MINUTES):
    """Check if an ISO 8601 duration is at most max_minutes.

    Supports formats: PT#M#S, PT#H#M#S, P#DT#H#M#S
    """
    match = re.match(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return False
    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    total_minutes = days * 24 * 60 + hours * 60 + minutes + seconds / 60
    return total_minutes <= max_minutes


@retry_api_call
def get_top_comments(youtube, video_id, max_results):
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            order="relevance",
            textFormat="plainText"
        ).execute()

        comments = []
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": comment["authorDisplayName"],
                "text": comment["textDisplay"],
                "likes": comment["likeCount"]
            })

        return comments
    except QuotaExhaustedError:
        raise
    except Exception as e:
        logger.warning(f"Could not fetch comments for video {video_id}: {e}")
        return []



@retry_transcript
def get_video_transcript(video_id):
    """Fetch transcript using yt-dlp Python API without writing to disk."""
    import urllib.request
    url = f"https://www.youtube.com/watch?v={video_id}"
    from time import sleep
    sleep(3)  # Sleep to avoid hitting rate limits on yt-dlp or YouTube when processing multiple videos in a loop
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.warning(f"[{video_id}] yt-dlp error fetching transcript: {e}")
        return None

    # Get English subtitles (manual first, then auto)
    subtitles = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}

    tracks = subtitles.get("en") or auto_subs.get("en")

    if not tracks:
        return None

    # Get VTT subtitle URL
    vtt_url = None
    for track in tracks:
        if track.get("ext") == "vtt":
            vtt_url = track.get("url")
            break

    if not vtt_url:
        return None

    try:
        # Download subtitle content directly
        with urllib.request.urlopen(vtt_url) as response:
            vtt_content = response.read().decode("utf-8")
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            raise  # let @retry_transcript catch and retry
        logger.warning(f"[{video_id}] Error downloading VTT: {e}")
        return None

    lines = []
    seen = set()

    for line in vtt_content.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or re.match(r"^\d+$", line):
            continue

        line = re.sub(r"<[^>]+>", "", line)

        if line and line not in seen:
            seen.add(line)
            lines.append(line)

    transcript = " ".join(lines)
    return transcript if transcript else None


_S3_CONFIG_KEY = "youtube-data/run_config.json"


def get_current_week(s3_client, bucket: str) -> int:
    """Read current week number from S3 config. Returns 1 if not yet set."""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=_S3_CONFIG_KEY)
        config = json.loads(obj["Body"].read().decode("utf-8"))
        return int(config.get("current_week", 1))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "NoSuchBucket"):
            return 1
        raise


def save_current_week(s3_client, bucket: str, week: int):
    """Write the next week number back to S3 config."""
    s3_client.put_object(
        Bucket=bucket,
        Key=_S3_CONFIG_KEY,
        Body=json.dumps({"current_week": week}),
        ContentType="application/json",
    )
    logger.info(f"Week counter updated to {week} in S3 config")


def save_to_s3(channel_name, videos, week_num: int):
    """Save channel data to S3"""
    try:
        timestamp = datetime.now().isoformat()
        s3_key = f"youtube-data/week{week_num}/{channel_name.lower()}.json"
        
        payload = {
            "channel": channel_name,
            "fetched_at": timestamp,
            "videos": videos
        }
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(payload, ensure_ascii=False),
            ContentType='application/json'
        )
        
        logger.info(f"Saved {len(videos)} videos to S3: s3://{S3_BUCKET}/{s3_key}")
        return s3_key
        
    except ClientError as e:
        logger.error(f"Error saving to S3: {e}")
        raise


def convert_to_decimal(obj):
    """Convert numeric values to Decimal for DynamoDB compatibility."""
    if isinstance(obj, dict):
        return {k: convert_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_decimal(item) for item in obj]
    elif isinstance(obj, (int, float)):
        return Decimal(str(obj))
    else:
        return obj


def save_to_dynamodb(dynamodb, table_name, channel_name, videos):
    """Save videos to DynamoDB, persisting old data and skipping duplicates."""
    table = dynamodb.Table(table_name)

    # Deduplicate by videoId before writing
    seen_ids = set()
    unique_videos = []
    for video in videos:
        if video["videoId"] not in seen_ids:
            seen_ids.add(video["videoId"])
            unique_videos.append(video)

    # Write new items (put_item will skip if PartitionKey already exists)
    written = 0
    skipped = 0
    try:
        for video in unique_videos:
            item = {
                'PartitionKey': channel_name,
                'SortKey': video['videoId'],
                'channel': channel_name,
                'title': video.get('title', ''),
                'description': video.get('description', ''),
                'publishedAt': video.get('publishedAt', ''),
                'viewCount': video.get('viewCount', 0),
                'likeCount': video.get('likeCount', 0),
                'commentCount': video.get('commentCount', 0),
                'transcript': video.get('transcript', ''),
                'topComments': video.get('topComments', []),
                'fetchedAt': datetime.now(timezone.utc).isoformat(),
            }
            item = convert_to_decimal(item)

            try:
                # Only write if PartitionKey doesn't already exist (prevents overwriting old data)
                table.put_item(
                    Item=item,
                    ConditionExpression='attribute_not_exists(PartitionKey)'
                )
                written += 1
            except ClientError as e:
                if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                    logger.info(f"Video {video['videoId']} already exists in DynamoDB, skipping")
                    skipped += 1
                else:
                    raise

        logger.info(f"DynamoDB for {channel_name}: {written} new, {skipped} already existed")
    except Exception as e:
        logger.error(f"Error saving to DynamoDB: {e}")
        raise


def get_existing_video_ids(dynamodb, table_name, channel_name):
    """Query DynamoDB for all video IDs already stored for a channel."""
    table = dynamodb.Table(table_name)
    existing_ids = set()
    try:
        response = table.query(
            KeyConditionExpression=Key('PartitionKey').eq(channel_name),
            ProjectionExpression='SortKey'
        )
        for item in response.get('Items', []):
            existing_ids.add(item['SortKey'])
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression=Key('PartitionKey').eq(channel_name),
                ProjectionExpression='SortKey',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            for item in response.get('Items', []):
                existing_ids.add(item['SortKey'])
    except ClientError as e:
        logger.warning(f"Could not query existing videos for {channel_name}: {e}")
    return existing_ids


def main(partial: bool = False, batch: int = 0):
    """
    Process channels and save to S3 and DynamoDB.

    batch=0  → all channels (default / manual --run-now)
    batch=1/2/3 → only that batch group (scheduled cron runs)
    partial=True → skip advancing the week counter after the run
    """
    logger.info("YouTube Video Ingestion")

    # Select channel subset based on batch
    if batch and 1 <= batch <= len(CHANNEL_BATCHES):
        batch_names = set(CHANNEL_BATCHES[batch - 1])
        channels = {k: v for k, v in NEWS_CHANNELS.items() if k in batch_names}
        logger.info(f"BATCH {batch} — {list(channels)}")
    else:
        channels = NEWS_CHANNELS

    if partial:
        logger.info("PARTIAL RUN — week counter will NOT advance after this run")
    logger.info(f"Time Window: Last {TIME_WINDOW_DAYS} days")
    logger.info(f"S3 Bucket: {S3_BUCKET}")
    logger.info(f"DynamoDB Table: {DYNAMODB_TABLE}")
    logger.info(f"Processing {len(channels)} channels")

    week_num = get_current_week(s3_client, S3_BUCKET)
    logger.info(f"S3 week label: week{week_num}")

    youtube = build_client(YOUTUBE_API_KEY)
    results = []
    quota_exhausted = False

    for idx, (channel_name, channel_id) in enumerate(channels.items(), 1):
        try:
            logger.info(f"[{idx}/{len(channels)}] Processing {channel_name}...")
            
            uploads_playlist = get_uploads_playlist(youtube, channel_id)
            if not uploads_playlist:
                logger.warning(f"[{channel_name}] Channel not found")
                results.append({"channel": channel_name, "status": "error", "error": "Channel not found"})
                continue
            
            videos = get_latest_videos(youtube, uploads_playlist, MAX_VIDEOS_PER_CHANNEL)
            logger.info(f"[{channel_name}] Retrieved {len(videos)} videos")
            
            if not videos:
                logger.warning(f"[{channel_name}] No videos found")
                results.append({"channel": channel_name, "status": "error", "error": "No videos found"})
                continue
            
            # Check DynamoDB for existing video IDs to skip duplicates early
            existing_ids = get_existing_video_ids(dynamodb, DYNAMODB_TABLE, channel_name)
            videos = [v for v in videos if v["videoId"] not in existing_ids]
            if existing_ids:
                logger.info(f"[{channel_name}] Skipped {len(existing_ids)} already-ingested videos")
            
            if not videos:
                logger.info(f"[{channel_name}] All videos already ingested")
                results.append({"channel": channel_name, "status": "success", "videoCount": 0})
                continue
            
            # Get statistics and duration info
            video_ids = [v["videoId"] for v in videos]
            stats_map = get_video_statistics(youtube, video_ids)
            
            # Single pass: filter by time window, duration, transcript, comments, and enrich
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=TIME_WINDOW_DAYS)
            filtered_videos = []
            for video in videos:
                if len(filtered_videos) >= MAX_VIDEOS_PER_CHANNEL:
                    logger.info(f"[{channel_name}] Reached target of {MAX_VIDEOS_PER_CHANNEL} videos, moving to next channel")
                    break
                
                vid = video["videoId"]
                
                # Filter by time window
                try:
                    published_date = datetime.fromisoformat(video["publishedAt"].replace('Z', '+00:00'))
                    if published_date < cutoff_date:
                        logger.info(f"[{channel_name}] Skipping {vid} - published {(datetime.now(timezone.utc) - published_date).days} days ago")
                        continue
                except Exception as e:
                    logger.warning(f"[{channel_name}] Could not parse date for {vid}: {e}")
                    continue
                
                details = stats_map.get(vid, {})
                stats = details.get("statistics", {})
                duration = details.get("duration", "")
                
                # Filter by view count
                view_count = int(stats.get("viewCount", 0))
                if view_count < MIN_VIEW_COUNT:
                    logger.info(f"[{channel_name}] Skipping {vid} - only {view_count} views (min: {MIN_VIEW_COUNT})")
                    continue
                
                # Filter by duration
                if not is_within_duration_limit(duration):
                    logger.info(f"[{channel_name}] Skipping {vid} - exceeds {MAX_VIDEO_DURATION_MINUTES} min duration limit")
                    continue
                
                # Filter by minimum number of comments (before transcript to save tokens)
                top_comments = get_top_comments(youtube, vid, COMMENTS_PER_VIDEO)
                if len(top_comments) < COMMENTS_PER_VIDEO:
                    logger.info(f"[{channel_name}] Skipping {vid} - only {len(top_comments)} comments (need {COMMENTS_PER_VIDEO})")
                    continue

                # Filter by transcript availability (most expensive check — last)
                transcript = get_video_transcript(vid)
                if not transcript:
                    logger.info(f"[{channel_name}] Skipping {vid} - no transcript available")
                    continue
                
                # Video passed all filters — enrich and keep
                video["transcript"] = transcript
                video["topComments"] = top_comments
                video["viewCount"] = view_count
                video["likeCount"] = int(stats.get("likeCount", 0))
                video["commentCount"] = int(stats.get("commentCount", 0))
                filtered_videos.append(video)
            
            logger.info(f"[{channel_name}] Filtered to {len(filtered_videos)}/{MAX_VIDEOS_PER_CHANNEL} videos after all checks")
            
            if not filtered_videos:
                logger.warning(f"[{channel_name}] No videos met criteria")
                results.append({"channel": channel_name, "status": "error", "error": "No videos met filtering criteria"})
                continue
            
            # Save to DynamoDB first (source of truth for dedup)
            try:
                save_to_dynamodb(dynamodb, DYNAMODB_TABLE, channel_name, filtered_videos)
            except Exception as e:
                logger.error(f"[{channel_name}] Failed to save to DynamoDB: {e}")
                results.append({"channel": channel_name, "status": "error", "error": f"DynamoDB save failed: {e}"})
                continue
            
            # Save to S3 only after DynamoDB succeeds
            try:
                save_to_s3(channel_name, filtered_videos, week_num)
            except Exception as e:
                logger.error(f"[{channel_name}] Failed to save to S3: {e}")
                results.append({"channel": channel_name, "status": "error", "error": f"S3 save failed: {e}"})
                continue
            
            logger.info(f"[{channel_name}] Successfully processed {len(filtered_videos)} videos")
            results.append({"channel": channel_name, "videoCount": len(filtered_videos), "status": "success"})

            # Sleep between channels to reduce IP-level rate limiting
            if idx < len(channels):
                logger.info("[rate-limit] Sleeping 60s before next channel...")
                _time.sleep(60)

        except QuotaExhaustedError:
            logger.error(f"[{channel_name}] YouTube API quota exhausted — stopping run, will resume next cycle")
            results.append({"channel": channel_name, "status": "error", "error": "quota exhausted"})
            quota_exhausted = True
            break
        except Exception as e:
            logger.error(f"[{channel_name}] Error: {e}")
            results.append({"channel": channel_name, "status": "error", "error": str(e)})
    
    # Summary
    logger.info("Ingestion Summary:")
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Successful: {successful}/{len(channels)}")
    logger.info(f"Failed: {failed}/{len(channels)}")
    
    for result in results:
        if result.get("status") == "success":
            logger.info(f"  {result['channel']}: {result.get('videoCount', 0)} videos")
        else:
            logger.error(f"  {result['channel']}: {result.get('error', 'Unknown error')}")

    if quota_exhausted:
        processed_channels = {r["channel"] for r in results}
        skipped_channels = [ch for ch in channels if ch not in processed_channels]
        logger.warning("INCOMPLETE BATCH RUN — quota was exhausted before all channels were processed")
        logger.warning(f"  Channels processed : {[r['channel'] for r in results]}")
        logger.warning(f"  Channels not reached: {skipped_channels}")
        logger.warning(f"  week{week_num} in S3 is partial — rerun will write to week{week_num + 1}")

    # Advance week counter only on a full (non-partial) run
    if partial:
        logger.info(f"PARTIAL RUN — week counter stays at {week_num}, next run will still write to week{week_num}")
    else:
        save_current_week(s3_client, S3_BUCKET, week_num + 1)

    logger.info("YouTube ingestion completed")

if __name__ == "__main__":
    main()