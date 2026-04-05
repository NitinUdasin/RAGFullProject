# Chunking

Splits loaded documents into smaller pieces that fit within embedding model context windows and preserve semantic coherence.

## Strategies

| Strategy | Best For |
|---|---|
| **Fixed-size** (`CharacterTextSplitter`) | Simple, fast; ignores structure |
| **Recursive character** (`RecursiveCharacterTextSplitter`) | General-purpose; respects paragraphs → sentences → words |
| **Semantic** (embed + cosine threshold) | Groups sentences by topic shift; best semantic coherence |
| **Markdown / code splitter** | Structure-aware splitting for `.md` and source code |
| **Parent-child** | Indexes small chunks, retrieves larger parent for context |
| **Sentence window** | Stores surrounding sentences in metadata for context expansion at query time |

## Pipeline Position

```
[Document Loading] → [Chunking] → [Embedding] → [VectorStore]
```

## Critical Parameters

```python
RecursiveCharacterTextSplitter(
    chunk_size=512,        # tokens or chars depending on length_function
    chunk_overlap=64,      # overlap to avoid cutting context at boundaries
    length_function=len,   # swap for tiktoken counter to measure in tokens
)
```

- **chunk_size** — smaller = higher precision retrieval, lower context per chunk
- **chunk_overlap** — prevents facts split across boundaries from being missed
- Measure chunk size in **tokens**, not characters, to match embedding model limits

## Metadata Propagation

Chunkers must forward parent document metadata to each chunk and add:

```python
chunk.metadata["chunk_index"] = i
chunk.metadata["total_chunks"] = n
```

This enables parent-document retrieval and provenance tracking.
