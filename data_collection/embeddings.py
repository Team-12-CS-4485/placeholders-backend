from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from data_collection.chunking import semantic_chunk

# Must use same model as chunking.py
model = SentenceTransformer('all-mpnet-base-v2')

EMBEDDING_DIMENSION = 768  # all-mpnet-base-v2 outputs 768 dims
INDEX_PATH = "data/faiss_index.bin"       # where the FAISS index is saved
METADATA_PATH = "data/chunk_metadata.pkl" # where chunk text + metadata is saved


def load_or_create_index():
    """
    Loads existing FAISS index from disk, or creates a fresh one.
    """
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        print(f"Loaded existing index with {index.ntotal} vectors")
    else:
        # IndexFlatIP = inner product (equivalent to cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        print("Created new FAISS index")
    return index


def load_metadata():
    """
    Loads chunk metadata (text, video_id, channel etc) from disk.
    Returns empty list if nothing saved yet.
    """
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return []


def save(index, metadata):
    """
    Persists the FAISS index and metadata to disk.
    """
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved index ({index.ntotal} vectors) and metadata to disk")


def embed_and_store(transcript, metadata={}):
    """
    Chunks a transcript, embeds each chunk, and stores in local FAISS index.

    Args:
        transcript (str): Full transcript text
        metadata (dict): Extra info e.g. { "video_id": "abc123", "channel": "CNN", "date": "2025-02-26" }

    Returns:
        int: Number of chunks stored
    """
    index = load_or_create_index()
    all_metadata = load_metadata()

    # Step 1: Chunk the transcript
    chunks = semantic_chunk(transcript)
    print(f"Created {len(chunks)} chunks")

    # Step 2: Embed all chunks at once
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Step 3: Normalize vectors (required for cosine similarity with IndexFlatIP)
    faiss.normalize_L2(embeddings)

    # Step 4: Add vectors to FAISS index
    index.add(embeddings)

    # Step 5: Store metadata alongside — FAISS only stores vectors, not text
    for i, chunk in enumerate(chunks):
        all_metadata.append({
            **metadata,
            "chunk_index": i,
            "text": chunk,
            "word_count": len(chunk.split())
        })

    # Step 6: Save everything to disk
    save(index, all_metadata)

    return len(chunks)


def query_similar_chunks(query_text, top_k=5):
    """
    Embeds a query and retrieves the most similar chunks from local FAISS index.

    Args:
        query_text (str): e.g. "what are networks saying about Trump?"
        top_k (int): Number of results to return

    Returns:
        List of dicts with chunk text, similarity score, and metadata
    """
    index = load_or_create_index()
    all_metadata = load_metadata()

    if index.ntotal == 0:
        print("Index is empty — run embed_and_store() first")
        return []

    # Embed and normalize the query — must match how chunks were embedded
    query_embedding = model.encode([query_text])
    faiss.normalize_L2(query_embedding)

    # Search — returns distances and indices of top_k matches
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 if not enough results
            continue
        results.append({
            "score": float(score),  # cosine similarity 0-1
            **all_metadata[idx]
        })

    return results


# Example usage
if __name__ == "__main__":
    sample_transcript = """
    There is new bipartisan frustration in Congress after the Justice Department 
    sent a list of politically exposed persons in the millions of files related 
    to its probes of Jeffrey Epstein. Several lawmakers argue the Justice Department 
    is intentionally muddying the waters on who was a predator versus someone 
    simply mentioned in an email or perhaps an article.
    """

    # Store transcript chunks
    embed_and_store(
        transcript=sample_transcript,
        metadata={
            "video_id": "yt_abc123",
            "channel": "CNN",
            "date": "2025-02-26"
        }
    )

    # Query
    results = query_similar_chunks("DOJ redacting Epstein files")
    for r in results:
        print(f"\nScore: {r['score']:.3f} | Channel: {r.get('channel')}")
        print(f"Text: {r['text'][:200]}")