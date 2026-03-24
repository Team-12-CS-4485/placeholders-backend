"""
News Trend Analyzer — DynamoDB Edition
Pulls video data from DynamoDB (PartitionKey=channel, SortKey=videoId),
analyzes thumbnails via Google Gemini Flash, and writes per-channel reports.

DynamoDB Table: youtube-videos
  PartitionKey : channel  (String)  — e.g. "NBCNews", "CNBC", "WashingtonPost"
  SortKey      : videoId  (String)  — YouTube video ID

Usage:
  python3 analyzer.py                         # analyze ALL channels
  python3 analyzer.py --channel NBCNews       # single channel
  python3 analyzer.py --batch-size 10         # thumbnails per Gemini call (default 10)

"""

import base64
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Load .env file if present (python-dotenv)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:

    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
        print(
            "Loaded .env (fallback parser — run `pip install python-dotenv` for full support)"
        )

import datetime
import hashlib
from pathlib import Path

# Configs

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
API_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get(
    "AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-east-1")
)
DYNAMO_TABLE = os.environ.get(
    "DYNAMODB_TABLE", os.environ.get("DYNAMO_TABLE", "youtube-videos")
)

REQUEST_DELAY = 5  # seconds between Gemini calls (rate limit buffer)
DEFAULT_BATCH = 10  # thumbnails per Gemini API call


# AWS SIGV4 — lightweight DynamoDB client


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.HMAC(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(key, date_stamp, region, service):
    k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)  # key already bytes here
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    return _sign(k_service, "aws4_request")


def dynamo_request(action: str, payload: dict) -> dict:
    """Sign and send a DynamoDB API request using AWS SigV4."""
    service = "dynamodb"
    host = f"dynamodb.{AWS_REGION}.amazonaws.com"
    endpoint = f"https://{host}/"

    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    body = json.dumps(payload).encode("utf-8")

    canonical_headers = (
        f"content-type:application/x-amz-json-1.0\n"
        f"host:{host}\n"
        f"x-amz-date:{amz_date}\n"
        f"x-amz-target:DynamoDB_20120810.{action}\n"
    )
    signed_headers = "content-type;host;x-amz-date;x-amz-target"
    payload_hash = hashlib.sha256(body).hexdigest()

    canonical_request = "\n".join(
        ["POST", "/", "", canonical_headers, signed_headers, payload_hash]
    )

    credential_scope = f"{date_stamp}/{AWS_REGION}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )

    signing_key = _get_signature_key(AWS_SECRET_KEY, date_stamp, AWS_REGION, service)
    signature = hmac.HMAC(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    auth_header = (
        f"AWS4-HMAC-SHA256 Credential={AWS_ACCESS_KEY}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/x-amz-json-1.0",
            "X-Amz-Date": amz_date,
            "X-Amz-Target": f"DynamoDB_20120810.{action}",
            "Authorization": auth_header,
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def dynamo_query_channel(channel: str) -> list[dict]:
    """
    Query all videos for a given channel using the PartitionKey.
    Handles pagination automatically (LastEvaluatedKey).
    Returns list of dicts: {videoId, title, channel, description, ...}
    """
    videos = []
    exclusive_start_key = None

    while True:
        payload = {
            "TableName": DYNAMO_TABLE,
            "KeyConditionExpression": "PartitionKey = :ch",
            "ExpressionAttributeValues": {":ch": {"S": channel}},
        }
        if exclusive_start_key:
            payload["ExclusiveStartKey"] = exclusive_start_key

        result = dynamo_request("Query", payload)

        for item in result.get("Items", []):
            videos.append(
                {
                    "videoId": item.get("SortKey", {}).get("S", ""),
                    "title": item.get("title", {}).get(
                        "S", item.get("SortKey", {}).get("S", "")
                    ),
                    "channel": item.get("channel", {}).get("S", channel),
                    "description": item.get("description", {}).get("S", ""),
                    "commentCount": item.get("commentCount", {}).get("N", "0"),
                }
            )

        exclusive_start_key = result.get("LastEvaluatedKey")
        if not exclusive_start_key:
            break

    return videos


def dynamo_list_channels() -> list[str]:
    """
    Scan the table to discover all unique PartitionKey (channel) values.
    Uses a ProjectionExpression to only pull the key — minimises RCUs.
    """
    channels = set()
    exclusive_start_key = None

    while True:
        payload = {
            "TableName": DYNAMO_TABLE,
            "ProjectionExpression": "PartitionKey",
        }
        if exclusive_start_key:
            payload["ExclusiveStartKey"] = exclusive_start_key

        result = dynamo_request("Scan", payload)

        for item in result.get("Items", []):
            ch = item.get("PartitionKey", {}).get("S")
            if ch:
                channels.add(ch)

        exclusive_start_key = result.get("LastEvaluatedKey")
        if not exclusive_start_key:
            break

    return sorted(channels)


# GEMINI HELPERS


def gemini_request(contents: list, system_instruction: str = None) -> str:
    """Send a request to Gemini with exponential backoff for 429 rate-limit errors."""
    payload = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    post_data = json.dumps(payload).encode()
    url = f"{API_BASE}?key={GEMINI_API_KEY}"

    max_retries = 6
    wait = 15

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=post_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
            return result["candidates"][0]["content"]["parts"][0]["text"]

        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429:
                if attempt < max_retries - 1:
                    print(
                        f"  [Rate limited] Waiting {wait}s… (retry {attempt+1}/{max_retries})"
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, 480)
                else:
                    raise RuntimeError(
                        f"Rate limit exceeded after {max_retries} retries.\n"
                        f"Try: increase REQUEST_DELAY (currently {REQUEST_DELAY}s)\n"
                        "Or upgrade your Gemini tier at aistudio.google.com"
                    )
            else:
                raise RuntimeError(f"Gemini API error {e.code}: {body[:400]}")


def fetch_thumbnail_bytes(video_id: str):
    """Download a YouTube thumbnail. Returns (bytes, mime) or (None, None)."""
    for url in [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    ]:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                data = r.read()
            if len(data) > 5000:
                return data, "image/jpeg"
        except Exception:
            continue
    return None, None


def analyze_thumbnail_batch(videos: list, batch_num: int, total_batches: int) -> str:
    """
    Send one batch of thumbnails to Gemini in a single API call.
    Returns the full analysis text for that batch.
    """
    parts = []

    for i, v in enumerate(videos, 1):
        vid = v["videoId"]
        title = v.get("title", vid)
        print(f"    Fetching thumbnail {i}/{len(videos)}: {title[:60]}")
        img_bytes, mime = fetch_thumbnail_bytes(vid)

        parts.append(
            {"text": f'\n--- Video {i}: "{title}" (https://youtu.be/{vid}) ---'}
        )
        if img_bytes:
            b64 = base64.b64encode(img_bytes).decode()
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        else:
            parts.append({"text": "[Thumbnail unavailable]"})

    parts.append(
        {
            "text": (
                "\n\nFor EACH video above (identified by its number and title), provide a thumbnail analysis "
                "across these 5 dimensions (keep each video's analysis under 150 words):\n"
                "1. Visual Elements — what is depicted, text overlays, graphics\n"
                "2. Emotional Tone — urgency, fear, curiosity, etc.\n"
                "3. Clickbait Rating — rate 1-5 and explain briefly\n"
                "4. Brand Consistency — does it look like professional broadcast news\n"
                "5. Key Insight — one sentence on its engagement strategy\n\n"
                "Format each video as:\n"
                "[Video N] <title>\n"
                "1. ...\n2. ...\n3. ...\n4. ...\n5. ..."
            )
        }
    )

    contents = [{"role": "user", "parts": parts}]

    print(
        f"    Sending batch {batch_num}/{total_batches} ({len(videos)} thumbnails) to Gemini…"
    )
    print(f"    Waiting {REQUEST_DELAY}s (rate limit buffer)…")
    time.sleep(REQUEST_DELAY)

    return gemini_request(
        contents,
        system_instruction=(
            "You are a concise, insightful media analyst specializing in "
            "visual communication and news media."
        ),
    )


# REPORT WRITER


def analyze_channel(channel: str, videos: list, out_dir: Path, batch_size: int):
    """Fetch thumbnails and run Gemini analysis for one channel, writing a report file."""
    safe_name = channel.replace("/", "_").replace(" ", "_")
    report_path = out_dir / f"{safe_name}_analysis.txt"

    total = len(videos)
    batches = [videos[i : i + batch_size] for i in range(0, total, batch_size)]
    n_batch = len(batches)

    print(f"\n{'='*60}")
    print(
        f"  Channel : {channel}  ({total} videos, {n_batch} batch(es) of ≤{batch_size})"
    )
    print(f"{'='*60}")

    with open(report_path, "w", encoding="utf-8") as txt:
        sep = "=" * 70
        txt.write(sep + "\n")
        txt.write("  NEWS ANALYSIS REPORT\n")
        txt.write(
            f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        txt.write(f"  Channel   : {channel}\n")
        txt.write(f"  Videos    : {total}\n")
        txt.write(sep + "\n\n")
        txt.write("Thumbnail Image Analysis\n")
        txt.write("-" * 40 + "\n\n")

        for idx, batch in enumerate(batches, 1):
            print(f"\n  Batch {idx}/{n_batch}:")
            analysis = analyze_thumbnail_batch(batch, idx, n_batch)
            txt.write(f"--- Batch {idx}/{n_batch} ---\n")
            txt.write(analysis + "\n\n")
            print(f"  Batch {idx} written.")

        txt.write(sep + "\n")
        txt.write(f"Report saved to: {report_path}\n")

    print(f"  Report → {report_path}")
    return report_path


# MAIN


def main():

    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not AWS_ACCESS_KEY:
        missing.append("AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_KEY:
        missing.append("AWS_SECRET_ACCESS_KEY")
    if missing:
        print("ERROR: Missing environment variable(s):", ", ".join(missing))
        sys.exit(1)

    args = sys.argv[1:]
    channel_filter = None
    batch_size = DEFAULT_BATCH
    out_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "reports"

    i = 0
    while i < len(args):
        if args[i] == "--channel" and i + 1 < len(args):
            channel_filter = args[i + 1]
            i += 2
        elif args[i] == "--batch-size" and i + 1 < len(args):
            batch_size = int(args[i + 1])
            i += 2
        elif args[i] == "--out-dir" and i + 1 < len(args):
            out_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    if channel_filter:
        channels = [channel_filter]
        print(f"Single-channel mode: {channel_filter}")
    else:
        print(f"Discovering channels in table '{DYNAMO_TABLE}'…")
        channels = dynamo_list_channels()
        print(f"Found {len(channels)} channel(s): {', '.join(channels)}")

    reports = []
    for ch in channels:
        print(f"\nQuerying DynamoDB for channel: {ch}")
        videos = dynamo_query_channel(ch)
        if not videos:
            print(f"  No videos found for {ch}, skipping.")
            continue
        rpt = analyze_channel(ch, videos, out_dir, batch_size)
        reports.append(rpt)

    print(f"\n{'='*60}")
    print(f"All done! {len(reports)} report(s) written:")
    for r in reports:
        print(f"  {r}")


if __name__ == "__main__":
    main()
