"""
 Vector Database Module
Stores chunk embeddings in a FAISS index and configures it for
fast similarity search. Also persists the index + chunk text to disk.
"""
import os
import pickle
import faiss
import numpy as np

from ingest import load_document
from chunker import chunk_text
from embedder import embed_chunks

INDEX_PATH = "../data/faiss_index.bin"
CHUNKS_PATH = "../data/chunks.pkl"


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for similarity search.
    Uses IndexFlatL2 — exact search via L2 (Euclidean) distance.
    Simple and accurate; fine for small-to-medium document sets.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: list[str]):
    """Persist the FAISS index and the corresponding chunk texts."""
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index to {INDEX_PATH} and chunks to {CHUNKS_PATH}")


def load_index():
    """Load a previously saved FAISS index + chunks from disk."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("No saved index found. Run build first.")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


if __name__ == "__main__":
    test_file = "../data/CRAG_research_paper.pdf"
    text = load_document(test_file)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    embeddings = embed_chunks(chunks)

    index = build_index(embeddings)
    print(f"FAISS index built. Total vectors stored: {index.ntotal}")

    save_index(index, chunks)

    # sanity check: reload from disk and confirm it matches
    loaded_index, loaded_chunks = load_index()
    print(f"Reloaded index has {loaded_index.ntotal} vectors and {len(loaded_chunks)} chunks")