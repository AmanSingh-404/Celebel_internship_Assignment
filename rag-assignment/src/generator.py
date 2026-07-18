"""
Answer Generation Module
Combines retrieved context chunks + the original query into a single
prompt, and feeds it to a local language model to produce a grounded
answer.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import retrieve

GEN_MODEL_NAME = "google/flan-t5-base"
_tokenizer = None
_model = None


def get_generator():
    """Load tokenizer + model directly (avoids transformers pipeline
    task-registry issues across versions)."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading generation model: {GEN_MODEL_NAME} (first run downloads it)...")
        _tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    return _tokenizer, _model


def build_prompt(query: str, context_chunks: list[str]) -> str:
    """
    Step 7 core: unify retrieved context + query into one prompt.
    Clear instructions keep the model grounded in the given context
    rather than falling back on its own internal knowledge.
    """
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the question using only the information in the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt


def generate_answer(query: str, top_k: int = 3) -> str:
    """Full RAG answer: retrieve -> build prompt -> generate."""
    results = retrieve(query, top_k=top_k)
    context_chunks = [chunk for chunk, _dist in results]

    prompt = build_prompt(query, context_chunks)

    tokenizer, model = get_generator()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    query = "What is Corrective Retrieval Augmented Generation?"
    answer = generate_answer(query, top_k=3)

    print(f"Query: {query}\n")
    print(f"Answer: {answer}")