"""
Query Processing + Retrieval Module
Converts a user question into a query vector, then searches the
FAISS index to retrieve the most contextually relevant chunks.
"""
from embedder import embed_query
from vector_store import load_index


def retrieve(query: str, top_k: int = 3):
    """
    Step 5: convert query -> embedding
    Step 6: search FAISS index -> return top_k most relevant chunks

    Returns a list of (chunk_text, distance) tuples, sorted by
    relevance (lower L2 distance = more similar).
    """
    index, chunks = load_index()

    query_vector = embed_query(query)
    query_vector = query_vector.reshape(1, -1)  # FAISS expects 2D input

    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(dist)))

    return results


if __name__ == "__main__":
    query = "What is Corrective Retrieval Augmented Generation?"
    results = retrieve(query, top_k=3)

    print(f"Query: {query}\n")
    for i, (chunk, dist) in enumerate(results):
        print(f"--- Result {i+1} (distance: {dist:.4f}) ---")
        print(chunk[:300])
        print()