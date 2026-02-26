import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_collection.embeddings import embed_and_store, query_similar_chunks

TRANSCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/transcript.txt"))

def main():
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Store the transcript
    print("Embedding and storing transcript...\n")
    num_chunks = embed_and_store(
        transcript,
        metadata={"channel": "CNN", "date": "2025-02-26"}
    )
    print(f"\nStored {num_chunks} chunks\n")

    # Run a few test queries and print results
    queries = [
        "DOJ redacting Epstein files",
        "lawmakers frustrated with Justice Department",
        "British royal family Epstein connection",
        "victims survivors intimidation",
    ]

    for query in queries:
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)
        results = query_similar_chunks(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  Result {i} | Score: {r['score']:.3f}")
            print(f"  {r['text'][:300].strip()}")
        print()

if __name__ == "__main__":
    main()