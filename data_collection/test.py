import yt_dlp
import re

video_id = "C7cbJgrr-es"
url = f"https://www.youtube.com/watch?v={video_id}"

ydl_opts = {
    'skip_download': True,
    'quiet': True,
    'writesubtitles': True,
    'writeautomaticsub': True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)

# Get English subtitles (manual first, then auto)
subtitles = info.get("subtitles") or {}
auto_subs = info.get("automatic_captions") or {}

tracks = subtitles.get("en") or auto_subs.get("en")

if not tracks:
    print("No transcript found.")
else:
    # Get VTT subtitle URL
    vtt_url = None
    for track in tracks:
        if track.get("ext") == "vtt":
            vtt_url = track.get("url")
            break

    if not vtt_url:
        print("No VTT transcript found.")
    else:
        # Download subtitle content directly
        import urllib.request
        with urllib.request.urlopen(vtt_url) as response:
            vtt_content = response.read().decode("utf-8")

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
        print(transcript)