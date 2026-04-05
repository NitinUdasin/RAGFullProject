# Embedding

Converts text chunks into dense vector representations stored in ChromaDB. Also embeds queries at retrieval time using the same model.

## Supported Models

| Provider | Model | Dims | Notes |
|---|---|---|---|
| OpenAI | `text-embedding-3-small` | 1536 | Cost-effective default |
| OpenAI | `text-embedding-3-large` | 3072 | Higher accuracy |
| Anthropic | — | — | No embedding API; use OpenAI or local |
| HuggingFace | `BAAI/bge-small-en-v1.5` | 384 | Fast local option |
| HuggingFace | `sentence-transformers/all-mpnet-base-v2` | 768 | Strong local baseline |
| Ollama | any pulled model | varies | Local inference via Ollama |

Selected via `EMBEDDING_MODEL` and `EMBEDDING_PROVIDER` env vars.

## Pipeline Position

```
[Chunking] → [Embedding] → [VectorStore]
                ↑
        also used at query time
```

## Key Rules

- **Never mix embedding models** across ingestion and query time — vectors become incomparable
- **Re-embed all chunks** when switching models; ChromaDB collections should be versioned by model name (e.g., `rag_docs_v2`)
- Batch embed for ingestion; single embed at query time

## Batching

```python
# Batch to stay within rate limits and model max-batch-size
def embed_chunks(chunks: list[str], batch_size: int = 100) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings.extend(embedding_model.embed_documents(batch))
    return embeddings
```

## Dimensionality & Distance Metric

ChromaDB collection must be created with a distance metric matching the model:
- `cosine` — for normalized models (most sentence transformers, OpenAI)
- `l2` — for unnormalized models
