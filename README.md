# Simple RAG: Ask Questions About Your Documents

A small, readable implementation of Retrieval-Augmented Generation (RAG) in about 100 lines of Python. It reads text files, finds the passages most relevant to your question, and has an LLM answer using only those passages.

## How it works

```
.txt files -> split into chunks -> embed each chunk -> store vectors
                                                           |
question -> embed question -> find most similar chunks ----+
                                      |
                       LLM answers using only those chunks
                                      |
                          answer + which files were used
```

1. **Chunking**: each document is split into overlapping word windows so sentences are not cut off from their context.
2. **Embedding**: every chunk becomes a vector using `sentence-transformers` (`all-MiniLM-L6-v2`), which runs locally and is free.
3. **Retrieval**: the question is embedded the same way, and cosine similarity picks the top 3 chunks.
4. **Generation**: the chunks go to an LLM (Claude) with an instruction to answer only from that context, or say it doesn't know.

If no API key is set, the app still runs and shows the retrieved chunks, so you can see the retrieval step on its own.

## Setup

```bash
git clone https://github.com/<your-username>/simple-rag.git
cd simple-rag
pip install -r requirements.txt
cp .env.example .env      # then add your ANTHROPIC_API_KEY (optional)
python main.py
```

The first run downloads the embedding model (about 80 MB).

## Example

```
Ask: Why does RAG reduce hallucination?

Because the model is told to answer only from the retrieved passages, it is
less likely to invent facts. If the passages lack the answer, it says so.

Sources: rag_basics.txt
```

## Use your own documents

Drop any `.txt` files into the `data/` folder and run `python main.py` again.

## Tests

```bash
pip install pytest
pytest
```

The tests use a small fake embedder, so they run offline without downloading a model.

## Project structure

```
simple-rag/
  rag.py            # chunking, embedding, retrieval, answer generation
  main.py           # command-line chat loop
  data/             # sample documents
  tests/test_rag.py
  requirements.txt
  .env.example
```

## Limitations and next steps

- Only `.txt` files are supported. PDF and DOCX loading would be a natural addition.
- Vectors are kept in memory and rebuilt on every start. A vector database such as FAISS or Chroma would help with larger collections.
- No re-ranking or evaluation yet. Measuring retrieval quality on a set of test questions would be a good next step.
- Chunk size and overlap are fixed defaults and could be tuned per document type.
