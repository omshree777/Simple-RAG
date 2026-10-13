import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


def fake_embed(texts, dim=256):
    """Bag-of-words hashing embedder: lets tests run offline, no model download."""
    vectors = np.zeros((len(texts), dim))
    for row, text in enumerate(texts):
        for word in text.lower().split():
            word = word.strip(".,?!")
            index = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            vectors[row, index] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


@pytest.fixture(autouse=True)
def patch_embedder(monkeypatch):
    monkeypatch.setattr(rag, "embed", fake_embed)


def test_chunk_text_overlaps():
    text = " ".join(str(i) for i in range(30))
    chunks = rag.chunk_text(text, size=10, overlap=3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert " ".join(chunks).count("29") >= 1


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        rag.chunk_text("a b c", size=5, overlap=5)


def test_retrieve_finds_relevant_chunk(tmp_path):
    (tmp_path / "cooking.txt").write_text("Boil the pasta in salted water and drain it.")
    (tmp_path / "space.txt").write_text("The telescope observed a distant galaxy and a nebula.")
    system = rag.RAG(str(tmp_path))
    top_source, _, _ = system.retrieve("How do I boil pasta?", k=1)[0]
    assert top_source == "cooking.txt"


def test_answer_without_api_key_returns_context(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "a.txt").write_text("Cosine similarity compares the angle between vectors.")
    system = rag.RAG(str(tmp_path))
    answer, hits = system.answer("What is cosine similarity?")
    assert "retrieved context only" in answer
    assert hits


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        rag.RAG(str(tmp_path))
