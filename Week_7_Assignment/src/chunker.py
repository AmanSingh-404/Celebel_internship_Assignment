"""
 Text Chunking Module
Splits raw text into smaller overlapping chunks for embedding + retrieval.
"""
from ingest import load_document


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into chunks of ~chunk_size characters, with `overlap`
    characters repeated between consecutive chunks so context isn't
    lost at chunk boundaries.

    Simple character-based chunking — clean and easy to reason about
    for a first pass. (We'll experiment with smarter strategies in
    Step 8: sentence-aware / token-aware chunking.)
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # slide window forward, keep overlap

    return chunks


if __name__ == "__main__":
    test_file = "../data/CRAG_research_paper.pdf"
    text = load_document(test_file)

    chunks = chunk_text(text, chunk_size=500, overlap=50)

    print(f"Total characters: {len(text)}")
    print(f"Total chunks created: {len(chunks)}")
    print("\n--- Chunk 0 ---")
    print(chunks[0])
    print("\n--- Chunk 1 ---")
    print(chunks[1])