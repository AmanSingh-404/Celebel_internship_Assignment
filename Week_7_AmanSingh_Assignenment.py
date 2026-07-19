"""
RAG Document Question Answering System
=======================================
A single-file Retrieval-Augmented Generation pipeline that answers
questions from custom documents (PDF/TXT).

Pipeline stages (per assignment spec):
  1. Document Ingestion    - load PDF/TXT, clean raw text
  2. Text Chunking         - split into overlapping chunks
  3. Embedding Creation     - sentence-transformers (all-MiniLM-L6-v2)
  4. Vector Database        - FAISS index, persisted to disk
  5. Query Processing       - embed the user's question
  6. Context Retrieval      - similarity search over the index
  7. Answer Generation      - grounded prompt -> local LLM (flan-t5-base)
  8. Optimizations          - sentence-aware chunking, hybrid BM25+vector
                              search, cross-encoder re-ranking

Usage:
    python rag_pipeline.py --build data/document.pdf
    python rag_pipeline.py --ask "What is this document about?"
    python rag_pipeline.py --ask "..." --hybrid --rerank   # use optimizations

Requirements:
    pip install sentence-transformers faiss-cpu pypdf transformers torch
    numpy rank_bm25 sentencepiece
"""

import os
import re
import pickle
import argparse

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi


# =========================================================
# Config
# =========================================================
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embed_model = None
_gen_tokenizer = None
_gen_model = None
_reranker = None


# =========================================================
# Step 1: Document Ingestion
# =========================================================
def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def load_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        raw = load_pdf(file_path)
    elif ext == ".txt":
        raw = load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")
    return clean_text(raw)


# =========================================================
# Step 2: Text Chunking (+ Step 8a optimization: sentence-aware)
# =========================================================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Fixed-size character chunking with overlap (baseline method)."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_text_by_sentences(text: str, max_chars: int = 500) -> list:
    """Sentence-aware chunking (Step 8 optimization) - avoids cutting
    sentences in half, usually improves retrieval coherence."""
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


# =========================================================
# Step 3: Embeddings
# =========================================================
def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed_chunks(chunks: list) -> np.ndarray:
    model = get_embed_model()
    return model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)


def embed_query(query: str) -> np.ndarray:
    model = get_embed_model()
    return model.encode([query], convert_to_numpy=True)[0]


# =========================================================
# Step 4: Vector Database (FAISS)
# =========================================================
def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: list):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index to {INDEX_PATH} and chunks to {CHUNKS_PATH}")


def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("No saved index found. Run --build first.")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# =========================================================
# Steps 5 & 6: Query Processing + Retrieval
# =========================================================
def vector_retrieve(query: str, top_k: int = 3):
    index, chunks = load_index()
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append((chunks[idx], float(dist)))
    return results


# ---- Step 8b: Hybrid retrieval (BM25 keyword + vector) ----
def hybrid_retrieve(query: str, top_k: int = 3, vector_weight: float = 0.5):
    index, chunks = load_index()

    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(query.lower().split())
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, len(chunks))
    vector_scores = [0.0] * len(chunks)
    max_dist = distances[0].max() if len(distances[0]) else 1.0
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            vector_scores[idx] = 1 - (dist / max_dist)

    combined = [
        (chunks[i], vector_weight * vector_scores[i] + (1 - vector_weight) * bm25_scores[i])
        for i in range(len(chunks))
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]


# ---- Step 8c: Cross-encoder re-ranking ----
def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading re-ranker: {RERANK_MODEL_NAME} ...")
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates: list, top_k: int = 3):
    reranker = get_reranker()
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# =========================================================
# Step 7: Answer Generation
# =========================================================
def get_generator():
    global _gen_tokenizer, _gen_model
    if _gen_model is None:
        print(f"Loading generation model: {GEN_MODEL_NAME} ...")
        _gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        _gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    return _gen_tokenizer, _gen_model


def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    return (
        "Answer the question using only the information in the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate_answer(query: str, top_k: int = 3, use_hybrid: bool = False, use_rerank: bool = False) -> str:
    """Full RAG flow: retrieve -> (optional hybrid/rerank) -> prompt -> generate."""
    if use_hybrid:
        results = hybrid_retrieve(query, top_k=top_k * 2 if use_rerank else top_k)
    else:
        results = vector_retrieve(query, top_k=top_k * 2 if use_rerank else top_k)

    context_chunks = [chunk for chunk, _score in results]

    if use_rerank:
        reranked = rerank(query, context_chunks, top_k=top_k)
        context_chunks = [chunk for chunk, _score in reranked]

    prompt = build_prompt(query, context_chunks)

    tokenizer, model = get_generator()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================================================
# Build & Ask entry points
# =========================================================
def build(file_path: str, chunk_size: int = 500, overlap: int = 50, sentence_aware: bool = False):
    print(f"Loading document: {file_path}")
    text = load_document(file_path)
    print(f"Loaded {len(text)} characters")

    if sentence_aware:
        chunks = chunk_text_by_sentences(text, max_chars=chunk_size)
    else:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    print(f"Generated embeddings: {embeddings.shape}")

    index = build_index(embeddings)
    save_index(index, chunks)
    print("Pipeline build complete. Index is ready for querying.")


def ask(query: str, top_k: int = 3, use_hybrid: bool = False, use_rerank: bool = False):
    answer = generate_answer(query, top_k=top_k, use_hybrid=use_hybrid, use_rerank=use_rerank)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Document Question Answering pipeline")
    parser.add_argument("--build", type=str, help="Path to document to ingest and index")
    parser.add_argument("--ask", type=str, help="Question to ask against the indexed document")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--sentence_aware", action="store_true", help="Use sentence-aware chunking when building")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid BM25+vector retrieval when asking")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder re-ranking when asking")
    args = parser.parse_args()

    if args.build:
        build(args.build, sentence_aware=args.sentence_aware)
    elif args.ask:
        ask(args.ask, top_k=args.top_k, use_hybrid=args.hybrid, use_rerank=args.rerank)
    else:
        print("Specify --build <file_path> or --ask '<question>'. Use -h for help.")
