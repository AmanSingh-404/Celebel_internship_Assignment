"""
Embedding Module
Converts text chunks into vector representations using a pre-trained
sentence-transformers model.
"""
from sentence_transformers import SentenceTransformer
from ingest import load_document
from chunker import chunk_text

# Load once, reuse across calls (loading the model is the slow part)
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} (first run downloads it)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: list[str]):
    """
    Convert a list of text chunks into a 2D numpy array of embeddings.
    Shape: (num_chunks, embedding_dim) — for MiniLM-L6-v2, dim = 384.
    """
    model = get_model()
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def embed_query(query: str):
    """Convert a single query string into its embedding vector."""
    model = get_model()
    return model.encode([query], convert_to_numpy=True)[0]


if __name__ == "__main__":
    test_file = "../data/CRAG_research_paper.pdf"
    text = load_document(test_file)
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    embeddings = embed_chunks(chunks)

    print(f"\nNumber of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Example embedding (first 10 dims of chunk 0): {embeddings[0][:10]}")