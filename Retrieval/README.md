# Retrieval

Fetches the most relevant chunks from the vector store (and optionally other sources) given the translated query. Retrieval quality is the single biggest lever on final answer quality.

## Pipeline Position

```
[Routing] → [Retrieval] → [Ranking] → [ContextAssembly] → [Generation]
```

## Retrieval Methods

### 1. Dense Retrieval (Vector Search)
Embed the query and find nearest neighbors in ChromaDB using cosine similarity.

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
)
```

### 2. Sparse Retrieval (BM25 / Keyword)
Lexical matching. Catches exact terms that dense search may miss (product codes, names, acronyms).

```python
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi([doc.split() for doc in corpus])
scores = bm25.get_scores(query.split())
```

### 3. Hybrid Retrieval
Combine dense + sparse scores for higher recall. Merge via RRF or weighted sum.

```python
def hybrid_retrieve(query: str, alpha: float = 0.5) -> list[Document]:
    dense_results = dense_search(query)
    sparse_results = sparse_search(query)
    return reciprocal_rank_fusion([dense_results, sparse_results])
```

### 4. Self-Query Retrieval
LLM extracts metadata filters from the query and passes them to ChromaDB's `where` clause alongside the vector search.

```python
# Query: "PDFs about refunds from 2024"
# Extracted: {"source_type": "pdf", "year": 2024, "topic": "refunds"}
collection.query(
    query_embeddings=[...],
    where={"year": {"$eq": 2024}, "source_type": "pdf"},
)
```
See `QueryStructuring/` for filter extraction details.

### 5. Contextual Compression
After retrieval, extract only the relevant portions of each chunk using an LLM compressor — reduces noise in context.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)
```

## Key Parameters

| Parameter | Typical Value | Effect |
|---|---|---|
| `n_results` (k) | 5–20 | More = higher recall, more noise |
| `score_threshold` | 0.7 | Drop low-relevance results |
| `fetch_k` | 50 | Candidates for MMR diversity filtering |

## Retrieval with MMR (Diversity)

Maximum Marginal Relevance balances relevance with diversity to avoid returning near-duplicate chunks.

```python
collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["embeddings", "documents"],
    # post-process with MMR
)
```
