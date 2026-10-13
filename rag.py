"""A minimal RAG (Retrieval-Augmented Generation) pipeline.

Flow: load .txt files -> split into chunks -> embed chunks -> for a question,
retrieve the most similar chunks -> ask an LLM to answer using only those chunks.
"""
import os
from pathlib import Path

import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")

_embedder = None


def embed(texts):
    """Turn a list of strings into normalized vectors (one row per string)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True)


def load_documents(folder):
    """Read every .txt file in a folder. Returns a list of (filename, text)."""
    return [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(Path(folder).glob("*.txt"))
    ]


def chunk_text(text, size=120, overlap=20):
    """Split text into chunks of `size` words, overlapping by `overlap` words.

    Overlap stops a sentence from being cut in half and losing its context.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


class RAG:
    def __init__(self, folder="data"):
        self.chunks = []  # list of (source_filename, chunk_text)
        for name, text in load_documents(folder):
            for chunk in chunk_text(text):
                self.chunks.append((name, chunk))
        if not self.chunks:
            raise ValueError(f"No .txt documents found in '{folder}'.")
        self.vectors = embed([chunk for _, chunk in self.chunks])

    def retrieve(self, question, k=3):
        """Return the top-k most similar chunks as (source, text, score)."""
        query_vector = embed([question])[0]
        scores = self.vectors @ query_vector  # cosine similarity (vectors are normalized)
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i][0], self.chunks[i][1], float(scores[i])) for i in top]

    def answer(self, question, k=3):
        """Return (answer_text, retrieved_chunks)."""
        hits = self.retrieve(question, k)
        context = "\n\n".join(f"[{source}] {text}" for source, text, _ in hits)

        if not os.getenv("ANTHROPIC_API_KEY"):
            return (
                "(No ANTHROPIC_API_KEY set, so showing retrieved context only.)\n\n"
                + context,
                hits,
            )

        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=(
                "Answer the question using ONLY the provided context. "
                "If the context does not contain the answer, say you don't know."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return response.content[0].text, hits
