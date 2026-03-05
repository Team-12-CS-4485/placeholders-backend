from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download("punkt_tab")
import numpy as np

# Upgraded from all-MiniLM-L6-v2 (384 dims) to all-mpnet-base-v2 (768 dims)
# Better semantic understanding, more accurate chunk boundaries
model = SentenceTransformer('all-mpnet-base-v2')

def semantic_chunk(text, threshold=0.5, min_words=40):
    """
    Splits a transcript into semantically coherent chunks.

    Args:
        text (str): Full transcript text
        threshold (float): Cosine similarity cutoff — below this triggers a new chunk
        min_words (int): Minimum words a chunk must have before it can be split

    Returns:
        List[str]: List of text chunks
    """
    sentences = nltk.sent_tokenize(text)

    if not sentences:
        return []

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # Compare embedding of current chunk centroid to next sentence
        # Using centroid instead of just previous sentence = more stable boundaries
        current_embedding = np.mean(
            [embeddings[j] for j in range(i - len(current_chunk), i)], axis=0
        )
        similarity = cosine_similarity(
            [current_embedding],
            [embeddings[i]]
        )[0][0]

        current_word_count = len(" ".join(current_chunk).split())

        if similarity < threshold and current_word_count >= min_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Don't lose the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks