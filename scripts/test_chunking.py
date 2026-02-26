import os
import sys
import tiktoken

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_collection.chunking import semantic_chunk

TRANSCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/transcript.txt"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/chunks_output.txt"))


def main():
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript = f.read()

    chunks = semantic_chunk(transcript)
    enc = tiktoken.get_encoding("cl100k_base")

    total_tokens = len(enc.encode(transcript))
    print(f"Total tokens in transcript: {total_tokens}")
    print(f"Total chunks created: {len(chunks)}\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            tokens = enc.encode(chunk)
            word_count = len(chunk.split())

            print(f"Chunk {i}: {len(tokens)} tokens | {word_count} words")
            print(f"{chunk.strip()[:200]}")
            print()

            f.write("=" * 80 + "\n")
            f.write(f"Chunk {i}\n")
            f.write(f"Word Count: {word_count}\n")
            f.write("-" * 80 + "\n")
            f.write(chunk + "\n\n")

    all_tokens = sum(len(enc.encode(c)) for c in chunks)
    print(f"Total tokens across all chunks: {all_tokens}")
    print(f"Chunks saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()