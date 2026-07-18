# Accepts PDFs and raw text files, returns cleaned raw text.

import os
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_txt(file_path: str) -> str:
    """Load raw text from a .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, strip weird line breaks."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    return "\n".join(lines)


def load_document(file_path: str) -> str:
    """
    Main entry point: detects file type by extension and returns
    cleaned raw text ready for chunking.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        raw = load_pdf(file_path)
    elif ext == ".txt":
        raw = load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")

    return clean_text(raw)


if __name__ == "__main__":
    # Quick manual test — change this path to your file in data/
    test_file = "../data/CRAG_research_paper.pdf"  # or sample.txt
    text = load_document(test_file)
    print(f"Loaded {len(text)} characters from {test_file}")
    print("\n--- First 500 characters ---\n")
    print(text[:500])