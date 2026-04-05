# VectorStore

Manages ChromaDB collections — creation, upsert, similarity search, and deletion. Acts as the persistence layer for all embedded chunks.

## Pipeline Position

```
[Embedding] → [VectorStore] ← [Retrieval]
```

## ChromaDB Setup

```python
import chromadb

# HTTP client (for running ChromaDB as a server)
client = chromadb.HttpClient(host="localhost", port=8001)

# In-process (for local dev/testing only)
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}   # match your embedding model
)
```

## Upsert Pattern

```python
collection.upsert(
    ids=[doc.metadata["doc_id"] for doc in chunks],
    embeddings=embeddings,
    documents=[doc.page_content for doc in chunks],
    metadatas=[doc.metadata for doc in chunks],
)
```

Use `upsert` (not `add`) to safely re-ingest without duplicates.

## Similarity Search

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"source": "report_2024.pdf"},   # optional metadata filter
    include=["documents", "metadatas", "distances"]
)
```

## Collection Versioning

Name collections with the embedding model to prevent vector space mismatch:

```
rag_docs_openai-text-embedding-3-small
rag_docs_bge-small-en-v1.5
```

## Environment Variables

```
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION=rag_docs
```
