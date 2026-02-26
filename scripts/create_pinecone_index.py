from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "quickstart-py"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-2",
        embed={
            # NOTE: sparse model (not llama-text-embed-v2)
            "model": "pinecone-sparse-english-v0",
            "field_map": {"text": "chunk_text"},
        },
    )
