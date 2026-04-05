# Indexing

The offline pipeline that takes raw documents through loading → chunking → embedding → storage. Indexing strategy directly determines what is retrievable at query time.

## Pipeline

```
[Raw Documents]
      ↓
[DocumentLoading]   — parse & normalize
      ↓
[Chunking]          — split into retrievable units
      ↓
[Embedding]         — encode to dense vectors
      ↓
[VectorStore]       — upsert into ChromaDB
```

## Indexing Strategies

### Basic Flat Index
Each chunk is a standalone document. Simple, fast, and sufficient for most use cases.

### Parent-Child Index
Index small chunks for high-precision retrieval; store a reference to the larger parent chunk. At retrieval time, return the parent for richer context.

```python
# small chunk stored in vectorstore
chunk.metadata["parent_id"] = parent_doc.metadata["doc_id"]

# at retrieval: fetch parent from docstore using parent_id
parent = docstore.get(chunk.metadata["parent_id"])
```

### Summary Index
Store an LLM-generated summary of each document alongside (or instead of) the raw chunks. Useful for high-level routing or when documents are long and heterogeneous.

### Multi-Vector Index
Store multiple representations per chunk (original text + summary + hypothetical questions) to improve recall across different query styles.

```python
# For each chunk, generate and embed:
# 1. The chunk itself
# 2. An LLM-generated summary
# 3. N hypothetical questions the chunk answers
```

## Incremental Indexing

Use ChromaDB `upsert` with a stable `doc_id` (e.g., hash of source + chunk index) to support re-ingestion without duplicates.

```python
import hashlib

def make_chunk_id(source: str, chunk_index: int) -> str:
    return hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()
```

## Environment Variables

```
CHROMA_COLLECTION=rag_docs
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```
