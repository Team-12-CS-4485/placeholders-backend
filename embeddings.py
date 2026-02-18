


import json
import re
import sys
from collections import Counter

COMMON_WORDS = {
    "the","and","is","in","to","of","a","that","it","on","for","with","as",
    "this","was","but","are","be","at","by","an","or","from","so","if",
    "we","you","they","he","she","his","her","their","them","its","it's",
    "will","has","have","had","been","were","which","about","into","more"
}

def clean_text(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)
    return [w for w in words if w not in COMMON_WORDS]

def get_top_words(text, n):
    words = clean_text(text)
    counts = Counter(words)
    return [word for word, _ in counts.most_common(n)]

def process_file(filename, n):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", [])
    results = []

    for video in videos:
        video_id = video.get("videoId")

        # Combine all useful text fields
        combined_text = ""

        combined_text += video.get("title", "") + " "
        combined_text += video.get("description", "") + " "
        combined_text += video.get("transcript", "") + " "

        # Include comments
        for comment in video.get("topComments", []):
            combined_text += comment.get("text", "") + " "

        top_words = get_top_words(combined_text, n)

        results.append({
            "video_id": video_id,
            "top_words": top_words
        })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_video_topics.py <json_file> <top_n>")
        sys.exit(1)

    filename = sys.argv[1]
    n = int(sys.argv[2])

    output = process_file(filename, n)

    print(json.dumps(output, indent=2))

