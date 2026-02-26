"""
News Trend Analyzer
Analyzes a YouTube news channel JSON dump using Google Gemini Flash 2.0.
- Finds trends and patterns across all videos/transcripts/comments
- Performs thumbnail image analysis for each video
- Writes output to news_analysis_report.txt in the SAME folder as the JSON
- Handles rate limits with exponential backoff (free tier: ~15 RPM)
"""

import json
import base64
import urllib.request
import urllib.error
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
API_BASE       = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


REQUEST_DELAY = 5

# ── HELPERS ──────────────────────────────────────────────────────────────────

def log(txt_file, message):
    """Print to console AND write to the txt report file."""
    print(message)
    txt_file.write(message + "\n")


def gemini_request(contents: list, system_instruction: str = None) -> str:
    """Send a request to Gemini with exponential backoff for 429 rate limit errors."""
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
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
            return result["candidates"][0]["content"]["parts"][0]["text"]

        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429:
                if attempt < max_retries - 1:
                    print(f"  [Rate limited] Waiting {wait}s... (retry {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    wait = min(wait * 2, 480)
                else:
                    raise RuntimeError(
                        f"Rate limit exceeded after {max_retries} retries.\n"
                        f"Try: increase REQUEST_DELAY in the script (currently {REQUEST_DELAY}s)\n"
                        "Or upgrade your Gemini tier at aistudio.google.com"
                    )
            else:
                raise RuntimeError(f"Gemini API error {e.code}: {body[:400]}")


def fetch_thumbnail_bytes(video_id: str):
    """Download YouTube thumbnail. Returns (bytes, mime) or (None, None)."""
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


def analyze_thumbnail(video_id: str, title: str) -> str:
    """Send thumbnail to Gemini Vision and return analysis."""
    img_bytes, mime = fetch_thumbnail_bytes(video_id)
    if img_bytes is None:
        return "  [Could not fetch thumbnail]"
    b64 = base64.b64encode(img_bytes).decode()
    contents = [{
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": mime, "data": b64}},
            {"text": (
                f'This is the YouTube thumbnail for the news video: "{title}"\n\n'
                "Analyze it across 5 dimensions (keep total response under 150 words):\n"
                "1. Visual Elements - what is depicted, text overlays, graphics\n"
                "2. Emotional Tone - urgency, fear, curiosity, etc.\n"
                "3. Clickbait Rating - rate 1-5 and explain briefly\n"
                "4. Brand Consistency - does it look like professional broadcast news\n"
                "5. Key Insight - one sentence on its engagement strategy"
            )},
        ],
    }]
    print(f"  Waiting {REQUEST_DELAY}s (rate limit buffer)...")
    time.sleep(REQUEST_DELAY)
    return gemini_request(contents)


def analyze_trends(videos: list, channel: str) -> str:
    """Ask Gemini to find trends/patterns across all videos."""
    summaries = []
    for v in videos:
        snippet  = (v.get("transcript") or "")[:800]
        comments = "; ".join(
            f'"{c["text"][:120]}" ({c["likes"]} likes)'
            for c in (v.get("topComments") or [])[:3]
        )
        summaries.append(
            f"[{v['publishedAt'][:10]}] {v['title']}\n"
            f"Views: {v.get('viewCount',0):,}  Likes: {v.get('likeCount',0):,}  "
            f"Comments: {v.get('commentCount',0):,}\n"
            f"Transcript: {snippet}\n"
            f"Top comments: {comments}\n"
        )

    prompt = (
        f"You are a media analyst. Below is data from {len(videos)} {channel} YouTube videos "
        f"published within a short window. Analyze and identify:\n\n"
        "1. Major Story Themes - dominant news topics\n"
        "2. Audience Sentiment Patterns - recurring emotions/opinions in comments\n"
        "3. Engagement Patterns - which stories got traction and why\n"
        "4. Narrative Framing - how does the channel frame these stories\n"
        "5. Trending Topics - hashtag-worthy trends from this batch\n"
        "6. Anomalies / Outliers - stories that break the expected pattern\n"
        "7. Executive Summary - 2-3 sentences on the overall news cycle\n\n"
        "---\n\n" + "\n\n".join(summaries)
    )
    print(f"  Waiting {REQUEST_DELAY}s (rate limit buffer)...")
    time.sleep(REQUEST_DELAY)
    return gemini_request(
        [{"role": "user", "parts": [{"text": prompt}]}],
        system_instruction="You are a concise, insightful media trend analyst."
    )


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main(json_path: str):
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nTo see a demo without an API key, run:")
        print("  python3 analyzer.py --demo <path/to/data.json>")
        sys.exit(1)

    json_path   = Path(json_path).resolve()
    report_path = json_path.parent / "news_analysis_report.txt"

    with open(json_path) as f:
        data = json.load(f)

    channel = data.get("channel", "Unknown")
    videos  = data.get("videos", [])
    fetched = data.get("fetched_at", "N/A")

    total_calls = 1 + len(videos)
    est_seconds = total_calls * REQUEST_DELAY
    print(f"  Note: {total_calls} API calls with {REQUEST_DELAY}s delay = ~{est_seconds}s minimum runtime")

    with open(report_path, "w", encoding="utf-8") as txt:
        sep = "=" * 70

        log(txt, sep)
        log(txt, "  NEWS ANALYSIS REPORT")
        log(txt, f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(txt, f"  Channel   : {channel}")
        log(txt, f"  Fetched   : {fetched}")
        log(txt, f"  Videos    : {len(videos)}")
        log(txt, sep)

        
        log(txt, "\nSTEP 1 — Trend & Pattern Analysis")
        log(txt, "-" * 40)
        print("  Calling Gemini for trend analysis...")
        trends = analyze_trends(videos, channel)
        log(txt, trends)

        
        log(txt, "\n" + sep)
        log(txt, "STEP 2 — Thumbnail Image Analysis")
        log(txt, "-" * 40)

        for i, v in enumerate(videos, 1):
            vid   = v["videoId"]
            title = v["title"]
            log(txt, f"\n[{i}/{len(videos)}] {title}")
            log(txt, f"  https://youtu.be/{vid}")
            print(f"  Analyzing thumbnail {i}/{len(videos)}...")
            analysis = analyze_thumbnail(vid, title)
            log(txt, analysis)
            log(txt, "")

        log(txt, sep)
        log(txt, f"Report saved to: {report_path}")

    print(f"\nDone! Report written to:\n  {report_path}")


# ── DEMO MODE ─────────────────────────────────────────────────────────────────

def demo(json_path: str):
    json_path   = Path(json_path).resolve()
    report_path = json_path.parent / "news_analysis_report.txt"

    with open(json_path) as f:
        data = json.load(f)

    channel = data.get("channel", "Unknown")
    fetched = data.get("fetched_at", "N/A")
    videos  = data.get("videos", [])

    demo_content = f"""======================================================================
  NEWS ANALYSIS REPORT  [DEMO MODE - no real API calls]
  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Channel   : {channel}
  Fetched   : {fetched}
  Videos    : {len(videos)}
======================================================================

STEP 1 — Trend & Pattern Analysis
----------------------------------------------------------------------
1. Major Story Themes
   - Missing Person / Crime: Nancy Guthrie disappearance dominates coverage.
   - School Shooting Accountability: Georgia father's trial for enabling the shooter.
   - Celebrity Obituary: Robert Duvall's death and legacy.
   - Geopolitics / Human Rights: Rubio addressing Navalny poisoning, US-Russia tensions.

2. Audience Sentiment Patterns
   - Skepticism of media motives ("distraction from real news").
   - Emotional investment in missing-person narratives.
   - Anger / outrage in gun control adjacent stories.
   - Nostalgia and mourning in celebrity death content.

3. Engagement Patterns
   - Missing person + celebrity host (Savannah Guthrie) drove highest view counts.
   - Trial content has moderate engagement; audiences follow real-time updates.
   - Light/viral content (UPS driver & turkeys) offers palette-cleansing engagement.

4. Narrative Framing
   - Victim-centered framing in crime stories; family anguish highlighted.
   - Accountability framing in trial coverage.
   - Legacy / tribute framing for Robert Duvall.

5. Trending Topics
   #NancyGuthrie #GeorgiaSchoolShooting #RobertDuvall #Navalny #ABCNews

6. Anomalies / Outliers
   - The UPS driver / turkey story is a clear tonal outlier — humor/viral
     insert amid otherwise heavy coverage.

7. Executive Summary
   This 24-hour news cycle blends investigative crime coverage, accountability
   journalism, and celebrity human interest with a single viral humor piece for
   engagement balance. Audience skepticism about media priorities is a recurring
   undercurrent across comment sections.

======================================================================
STEP 2 — Thumbnail Image Analysis
----------------------------------------------------------------------

[1/{len(videos)}] {videos[0]['title']}
  https://youtu.be/{videos[0]['videoId']}
  1. Visual Elements: Split-screen of anchor on-air and search scene. Bold text overlay.
  2. Emotional Tone: Urgency mixed with relief; designed to compel click-through.
  3. Clickbait Rating: 3/5 - uses a dramatic keyword but grounded in fact.
  4. Brand Consistency: Yes - professional color palette, broadcast news style.
  5. Key Insight: Leverages celebrity anchor connection to personalize the story.

[2/{len(videos)}] {videos[1]['title']}
  https://youtu.be/{videos[1]['videoId']}
  1. Visual Elements: Courtroom imagery and headline text overlay.
  2. Emotional Tone: Somber, gravity-conveying; evokes dread and outrage.
  3. Clickbait Rating: 2/5 - straightforward accountability journalism framing.
  4. Brand Consistency: Yes - matches broadcast investigative reporting aesthetic.
  5. Key Insight: Positions this as a landmark legal accountability moment.

  ... (run without --demo to analyze all {len(videos)} thumbnails with real AI) ...

======================================================================
Report saved to: {report_path}
"""

    print(demo_content)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(demo_content)
    print(f"Demo report written to:\n  {report_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args      = sys.argv[1:]
    demo_mode = "--demo" in args
    path_args = [a for a in args if not a.startswith("--")]

    if not path_args:
        print("Usage:")
        print("  python3 analyzer.py <path/to/data.json>")
        print("  python3 analyzer.py --demo <path/to/data.json>")
        sys.exit(1)

    json_file = path_args[0]

    if not Path(json_file).exists():
        print(f"ERROR: File not found: {json_file}")
        sys.exit(1)

    if demo_mode:
        demo(json_file)
    else:
        main(json_file)