"""
pipeline.py — single entry point for the full RAG system.

Usage:
    python pipeline.py --build              # ingest + chunk + embed + index a document
    python pipeline.py --ask "your question"  # ask a question against the indexed doc
"""
import argparse

from ingest import load_document
from chunker import chunk_text
from embedder import embed_chunks
from vector_store import build_index, save_index
from generator import generate_answer


def build(file_path: str, chunk_size: int = 500, overlap: int = 50):
    print(f"Loading document: {file_path}")
    text = load_document(file_path)
    print(f"Loaded {len(text)} characters")

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    print(f"Generated embeddings: {embeddings.shape}")

    index = build_index(embeddings)
    save_index(index, chunks)
    print("Pipeline build complete. Index is ready for querying.")


def ask(query: str, top_k: int = 3):
    answer = generate_answer(query, top_k=top_k)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline CLI")
    parser.add_argument("--build", type=str, help="Path to document to ingest and index")
    parser.add_argument("--ask", type=str, help="Question to ask against the indexed document")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if args.build:
        build(args.build)
    elif args.ask:
        ask(args.ask, top_k=args.top_k)
    else:
        print("Specify --build <file_path> or --ask '<question>'. Use -h for help.")