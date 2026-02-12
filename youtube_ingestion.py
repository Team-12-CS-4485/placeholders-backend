from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY, NEWS_CHANNELS, MAX_VIDEOS_PER_CHANNEL, MIN_VIEW_COUNT
import json
from datetime import datetime
import os


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def build_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def get_uploads_playlist(youtube, handle: str):
    response = youtube.channels().list(
        part="contentDetails",
        forHandle=handle
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


def save_to_json(channel_name, data):
    filename = os.path.join(
        DATA_DIR,
        f"{channel_name.lower()}_latest.json"
    )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "channel": channel_name,
            "fetched_at": datetime.now().isoformat(),
            "videos": data
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {filename}")


def main():
    youtube = build_client(YOUTUBE_API_KEY)

    channel_name = "CNN"
    handle = NEWS_CHANNELS[channel_name]

    print(f"Fetching latest videos from {handle}...")

    uploads_playlist = get_uploads_playlist(youtube, handle)
    if not uploads_playlist:
        print("Channel not found.")
        return

    videos = get_latest_videos(youtube, uploads_playlist, 5)

    video_ids = [v["videoId"] for v in videos]
    stats_map = get_video_statistics(youtube, video_ids)

    for video in videos:
        stats = stats_map.get(video["videoId"], {})
        video["viewCount"] = int(stats.get("viewCount", 0))
        video["likeCount"] = int(stats.get("likeCount", 0))
        video["commentCount"] = int(stats.get("commentCount", 0))

        video["topComments"] = get_top_comments(
                youtube,
                video["videoId"],
                3
        )

    save_to_json(channel_name, videos)



if __name__ == "__main__":
    main()
