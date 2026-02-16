from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from config import YOUTUBE_API_KEY, NEWS_CHANNELS, MAX_VIDEOS_PER_CHANNEL, MIN_VIEW_COUNT
import json
from datetime import datetime
import os
import boto3
from botocore.exceptions import ClientError
import logging
from s3_utils import save_to_s3

# Setup logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Format: add timestamp, level, and message for CloudWatch
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# AWS clients
dynamodb = boto3.resource('dynamodb')


def build_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def get_uploads_playlist(youtube, handle: str):
    # Remove @ if present, YouTube API uses username not handle
    username = handle.lstrip('@')
    
    response = youtube.channels().list(
        part="contentDetails",
        forUsername=username
    ).execute()

    if not response.get("items"):
        return None

    #returns id of the channels playlist which has all the uploaded videos
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_latest_videos(youtube, uploads_playlist_id: str, max_results=MAX_VIDEOS_PER_CHANNEL):
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
        part="statistics",
        id=",".join(video_ids)   
    ).execute()

    stats_map = {}
    for item in response.get("items", []):
        stats_map[item["id"]] = item["statistics"]

    return stats_map


def get_top_comments(youtube, video_id, max_results=3):
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


def get_video_transcript(video_id):
    try:
        # Use fetch() as instance method - instantiate YouTubeTranscriptApi first
        transcript = YouTubeTranscriptApi().fetch(video_id)
        
        if not transcript:
            logger.info(f"No transcript data for video {video_id}")
            return ""
        
        # Combine all transcript entries into a single string
        # Handle both dict and object formats from different versions
        transcript_text = ""
        for entry in transcript:
            if isinstance(entry, dict):
                transcript_text += entry.get("text", "") + " "
            else:
                # Try attribute access for object format (v1.2.4+)
                transcript_text += (entry.text if hasattr(entry, "text") else str(entry)) + " "
        
        transcript_text = transcript_text.strip()
        
        logger.info(f"Fetched transcript for video {video_id} ({len(transcript_text)} chars)")
        return transcript_text
        
    except (TranscriptsDisabled, NoTranscriptFound):
        # Expected - many videos don't have transcripts enabled
        logger.info(f"No transcript available for video {video_id} (subtitles disabled or not found)")
        return ""
    except Exception as e:
        logger.warning(f"Unexpected error fetching transcript for {video_id}: {type(e).__name__}: {e}")
        return ""


def write_to_dynamodb(channel_name, videos, table_name):
    """
    Write video data to DynamoDB for querying.
    """
    try:
        table = dynamodb.Table(table_name)
        timestamp = datetime.now().isoformat()
        
        with table.batch_writer() as batch:
            for video in videos:
                item = {
                    "PartitionKey": channel_name,
                    "SortKey": video["videoId"],
                    "channel": channel_name,
                    "videoId": video["videoId"],
                    "title": video["title"],
                    "description": video["description"],
                    "publishedAt": video["publishedAt"],
                    "viewCount": video.get("viewCount", 0),
                    "likeCount": video.get("likeCount", 0),
                    "commentCount": video.get("commentCount", 0),
                    "topComments": video.get("topComments", []),
                    "transcript": video.get("transcript", ""),
                    "fetchedAt": timestamp
                }
                batch.put_item(Item=item)
        
        logger.info(f"Wrote {len(videos)} videos to DynamoDB table {table_name}")
    except ClientError as e:
        logger.error(f"Error writing to DynamoDB: {e}")
        raise


def process_channel(youtube, channel_name, handle, s3_bucket, dynamodb_table):
    try:
        logger.info(f"[{channel_name}] Starting channel processing")
        
        uploads_playlist = get_uploads_playlist(youtube, handle)
        if not uploads_playlist:
            logger.warning(f"[{channel_name}] Channel not found with handle: {handle}")
            return None

        videos = get_latest_videos(youtube, uploads_playlist, MAX_VIDEOS_PER_CHANNEL)
        logger.info(f"[{channel_name}] Retrieved {len(videos)} videos from playlist")
        
        if not videos:
            logger.warning(f"[{channel_name}] No videos found")
            return None

        # Get video statistics
        video_ids = [v["videoId"] for v in videos]
        stats_map = get_video_statistics(youtube, video_ids)
        logger.info(f"[{channel_name}] Retrieved statistics for {len(video_ids)} videos")

        # Enrich videos with stats and comments
        for video in videos:
            stats = stats_map.get(video["videoId"], {})
            view_count = int(stats.get("viewCount", 0))
            
            # Filter by minimum view count
            if view_count >= MIN_VIEW_COUNT:
                video["viewCount"] = view_count
                video["likeCount"] = int(stats.get("likeCount", 0))
                video["commentCount"] = int(stats.get("commentCount", 0))
                
                # Try to get comments, but handle videos with disabled comments
                try:
                    video["topComments"] = get_top_comments(
                        youtube,
                        video["videoId"],
                        3
                    )
                except Exception as comment_err:
                    if "commentsDisabled" in str(comment_err):
                        logger.warning(f"[{channel_name}] Comments disabled for video {video['videoId']}")
                        video["topComments"] = []
                    else:
                        logger.warning(f"[{channel_name}] Could not fetch comments for video {video['videoId']}: {comment_err}")
                        video["topComments"] = []
                
                # Fetch transcript for the video
                video["transcript"] = get_video_transcript(video["videoId"])

        # Filter videos that meet minimum view count
        filtered_videos = [v for v in videos if v.get("viewCount", 0) >= MIN_VIEW_COUNT]
        logger.info(f"[{channel_name}] Filtered to {len(filtered_videos)} videos meeting minimum view count ({MIN_VIEW_COUNT})")
        
        if not filtered_videos:
            logger.warning(f"[{channel_name}] No videos met minimum view count ({MIN_VIEW_COUNT})")
            return None

        # Save to S3 if bucket provided
        if s3_bucket:
            try:
                save_to_s3(channel_name, filtered_videos, s3_bucket)
            except Exception as e:
                logger.error(f"[{channel_name}] Failed to save to S3: {e}", exc_info=True)
                raise
        
        # Save to DynamoDB (required)
        try:
            write_to_dynamodb(channel_name, filtered_videos, dynamodb_table)
        except Exception as e:
            logger.error(f"[{channel_name}] Failed to save to DynamoDB: {e}", exc_info=True)
            raise
        
        logger.info(f"[{channel_name}] Successfully processed {len(filtered_videos)} videos")
        return {
            "channel": channel_name,
            "videoCount": len(filtered_videos),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"[{channel_name}] Error processing channel: {e}", exc_info=True)
        return {
            "channel": channel_name,
            "status": "error",
            "error": str(e)
        }


def api_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "X-Content-Type-Options": "nosniff",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS, POST, GET",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        },
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    logger.info(f"Lambda invoked with event: {json.dumps(event)[:200]}...")
    
    try:
        # Handle API Gateway OPTIONS request (CORS preflight)
        if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
            logger.info("Responding to CORS preflight request")
            return api_response(200, {"message": "OK"})
        
        # Extract body if coming from API Gateway
        request_body = event
        if isinstance(event.get("body"), str):
            request_body = json.loads(event.get("body", "{}"))
        
        logger.info(f"Processing request with body: {json.dumps(request_body)[:200]}")
        
        # Get S3 bucket and DynamoDB table from environment or request (required)
        s3_bucket = request_body.get("s3_bucket") or os.getenv("S3_BUCKET")
        dynamodb_table = request_body.get("dynamodb_table") or os.getenv("DYNAMODB_TABLE")
        
        if not s3_bucket or not dynamodb_table:
            logger.error("Missing required configuration: S3_BUCKET and DYNAMODB_TABLE are required")
            return api_response(400, {
                "error": "Missing required configuration",
                "details": "S3_BUCKET and DYNAMODB_TABLE environment variables must be set",
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info(f"Configuration - S3: {s3_bucket}, DynamoDB: {dynamodb_table}")
        
        # Process all channels
        logger.info("=" * 60)
        logger.info(f"Starting YouTube ingestion for {len(NEWS_CHANNELS)} channels")
        logger.info("=" * 60)
        
        youtube = build_client(YOUTUBE_API_KEY)
        results = []
        
        for channel_name, handle in NEWS_CHANNELS.items():
            result = process_channel(
                youtube,
                channel_name,
                handle,
                s3_bucket=s3_bucket,
                dynamodb_table=dynamodb_table
            )
            if result:
                results.append(result)
        
        logger.info("=" * 60)
        logger.info(f"Completed processing. {len(results)}/{len(NEWS_CHANNELS)} channels successful")
        logger.info("=" * 60)
        
        # Prepare response
        response_body = {
            "message": "YouTube ingestion completed successfully",
            "channelsProcessed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully processed {len(results)} channels")
        return api_response(200, response_body)
    
    except ValueError as e:
        logger.error(f"JSON parsing error: {e}", exc_info=True)
        return api_response(400, {
            "error": "Invalid request body. Expected JSON.",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        return api_response(500, {
            "error": "Internal server error",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        })


