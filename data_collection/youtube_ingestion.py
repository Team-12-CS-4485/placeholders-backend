from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from config import NEWS_CHANNELS, YOUTUBE_API_KEY, MAX_VIDEOS_PER_CHANNEL, MIN_VIEW_COUNT, TIME_WINDOW_DAYS, COMMENTS_PER_VIDEO, MAX_VIDEO_DURATION_MINUTES
import json
from datetime import datetime, timedelta, timezone
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

# AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE")


def build_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def get_uploads_playlist(youtube, channel_id: str):
    
    response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    if not response.get("items"):
        return None

    #returns id of the channels playlist which has all the uploaded videos
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


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
    import re
    match = re.match(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return False
    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    total_minutes = days * 24 * 60 + hours * 60 + minutes + seconds / 60
    return total_minutes <= max_minutes


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
    except Exception as e:
        logger.warning(f"Could not fetch comments for video {video_id}: {e}")
        return []


def get_video_transcript(video_id):
    try:
        # Fetch transcript from YouTube
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        
        if not transcript:
            return ""
        
        transcript_text = ""
        for entry in transcript:
            if isinstance(entry, dict):
                transcript_text += entry.get("text", "") + " "
            else:
                transcript_text += (entry.text if hasattr(entry, "text") else str(entry)) + " "
        
        return transcript_text.strip()
        
    except (TranscriptsDisabled, NoTranscriptFound):
        return ""
    except Exception as e:
        logger.warning(f"Error fetching transcript for {video_id}: {e}")
        return ""


def save_to_s3(channel_name, videos):
    """Save channel data to S3"""
    try:
        timestamp = datetime.now().isoformat()
        s3_key = f"youtube-data/week1/{channel_name.lower()}/{timestamp.replace(':', '-')}.json"
        
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
                'PartitionKey': video['videoId'],
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


def main():
    """Process all channels and save to S3 and DynamoDB"""
    logger.info("YouTube Video Ingestion")
    logger.info(f"Time Window: Last {TIME_WINDOW_DAYS} days")
    logger.info(f"S3 Bucket: {S3_BUCKET}")
    logger.info(f"DynamoDB Table: {DYNAMODB_TABLE}")
    logger.info(f"Processing {len(NEWS_CHANNELS)} channels")
    
    youtube = build_client(YOUTUBE_API_KEY)
    results = []
    
    for idx, (channel_name, channel_id) in enumerate(NEWS_CHANNELS.items(), 1):
        try:
            logger.info(f"[{idx}/{len(NEWS_CHANNELS)}] Processing {channel_name}...")
            
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
            
            # Save to S3
            try:
                save_to_s3(channel_name, filtered_videos)
            except Exception as e:
                logger.error(f"[{channel_name}] Failed to save to S3: {e}")
                results.append({"channel": channel_name, "status": "error", "error": f"S3 save failed: {e}"})
                continue
            
            #Save to DynamoDB (replacing old entries)
            try:
                save_to_dynamodb(dynamodb, DYNAMODB_TABLE, channel_name, filtered_videos)
            except Exception as e:
                logger.error(f"[{channel_name}] Failed to save to DynamoDB: {e}")
                results.append({"channel": channel_name, "status": "error", "error": f"DynamoDB save failed: {e}"})
                continue
            
            logger.info(f"[{channel_name}] Successfully processed {len(filtered_videos)} videos")
            results.append({"channel": channel_name, "videoCount": len(filtered_videos), "status": "success"})
            
        except Exception as e:
            logger.error(f"[{channel_name}] Error: {e}")
            results.append({"channel": channel_name, "status": "error", "error": str(e)})
    
    # Summary
    logger.info("Ingestion Summary:")
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Successful: {successful}/{len(NEWS_CHANNELS)}")
    logger.info(f"Failed: {failed}/{len(NEWS_CHANNELS)}")
    
    for result in results:
        if result.get("status") == "success":
            logger.info(f" {result['channel']}: {result.get('videoCount', 0)} videos")
        else:
            logger.error(f"{result['channel']}: {result.get('error', 'Unknown error')}")
    
    logger.info("YouTube ingestion completed")

if __name__ == "__main__":
    main()