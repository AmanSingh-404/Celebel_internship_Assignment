"""
 Optimization Experiments
Extends the base retriever with:
  1. Sentence-aware chunking (alternative to fixed-size character chunking)
  2. Hybrid retrieval: keyword (BM25) + vector search combined
  3. A re-ranking pass using cross-encoder scoring

"""
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingest import load_document
from embedder import embed_query
from vector_store import load_index
from retriever import retrieve as vector_only_retrieve


# ---------------------------------------------------------------
# 8a. Alternative chunking strategy: sentence-aware
# ---------------------------------------------------------------
def chunk_text_by_sentences(text: str, max_chars: int = 500) -> list[str]:
    """
    Groups whole sentences into chunks up to max_chars, instead of
    cutting mid-sentence. Usually improves retrieval quality because
    chunks stay semantically coherent.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence
    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------------
# 8b. Hybrid search: BM25 (keyword) + FAISS (vector)
# ---------------------------------------------------------------
def hybrid_retrieve(query: str, top_k: int = 3, vector_weight: float = 0.5):
    """
    Combines keyword-based BM25 ranking with vector similarity.
    Both scores are normalized to [0,1] and blended, so exact keyword
    matches AND semantically similar chunks both get a chance to surface.
    """
    index, chunks = load_index()

    # --- BM25 keyword scores ---
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(query.lower().split())
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()  # normalize to [0,1]

    # --- Vector similarity scores (convert FAISS L2 distance -> similarity) ---
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, len(chunks))
    vector_scores = [0.0] * len(chunks)
    max_dist = distances[0].max() if len(distances[0]) else 1.0
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            vector_scores[idx] = 1 - (dist / max_dist)  # higher = more similar

    # --- Blend and rank ---
    combined = [
        (chunks[i], vector_weight * vector_scores[i] + (1 - vector_weight) * bm25_scores[i])
        for i in range(len(chunks))
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]


# ---------------------------------------------------------------
# 8c. Re-ranking with a cross-encoder
# ---------------------------------------------------------------
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading cross-encoder re-ranker...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, candidates: list[str], top_k: int = 3):
    """
    Cross-encoders score (query, chunk) pairs jointly, which is more
    accurate than comparing separately-embedded vectors — but slower,
    so we only run it on a small candidate shortlist from initial
    retrieval, not the whole corpus.
    """
    reranker = get_reranker()
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    query = "What datasets were used in the experiments?"

    print("=" * 60)
    print("VECTOR-ONLY RETRIEVAL (baseline)")
    print("=" * 60)
    for chunk, dist in vector_only_retrieve(query, top_k=3):
        print(f"[dist={dist:.4f}] {chunk[:150]}...\n")

    print("=" * 60)
    print("HYBRID RETRIEVAL (BM25 + vector)")
    print("=" * 60)
    hybrid_results = hybrid_retrieve(query, top_k=5)
    for chunk, score in hybrid_results:
        print(f"[score={score:.4f}] {chunk[:150]}...\n")

    print("=" * 60)
    print("RE-RANKED (cross-encoder over hybrid candidates)")
    print("=" * 60)
    candidate_chunks = [c for c, _ in hybrid_results]
    reranked = rerank(query, candidate_chunks, top_k=3)
    for chunk, score in reranked:
        print(f"[score={score:.4f}] {chunk[:150]}...\n")