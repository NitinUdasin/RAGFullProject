# Ranking

Re-orders retrieved documents before they are assembled into context. The retriever optimizes for recall; the ranker optimizes for precision — putting the most relevant chunks at the top.

## Pipeline Position

```
[Retrieval] → [Ranking] → [ContextAssembly] → [Generation]
```

## Techniques

### 1. Reciprocal Rank Fusion (RRF)
Parameter-free rank merging for results from multiple retrieval sources (dense + sparse, multi-query).

```python
def reciprocal_rank_fusion(
    result_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for ranked in result_lists:
        for rank, doc in enumerate(ranked):
            id_ = doc.metadata["doc_id"]
            scores[id_] = scores.get(id_, 0) + 1 / (rank + k)
            doc_map[id_] = doc
    return sorted(doc_map.values(), key=lambda d: scores[d.metadata["doc_id"]], reverse=True)
```

### 2. Cross-Encoder Reranking
A cross-encoder scores each (query, chunk) pair jointly — more accurate than bi-encoder cosine similarity but slower.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [(query, doc.page_content) for doc in candidates]
scores = reranker.predict(pairs)
ranked = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]
```

**Models:** `cross-encoder/ms-marco-*` family for English; Cohere Rerank API for a hosted option.

### 3. Cohere Rerank (API)

```python
import cohere

co = cohere.Client(api_key)
response = co.rerank(
    model="rerank-english-v3.0",
    query=query,
    documents=[doc.page_content for doc in candidates],
    top_n=5,
)
ranked = [candidates[r.index] for r in response.results]
```

### 4. LLM-based Reranking
Ask the LLM to score or sort candidate chunks by relevance. Accurate but expensive — use only when top-k is small.

```python
rank_prompt = """
Score each document 0-10 for relevance to the query. Return JSON list of {{"id": ..., "score": ...}}.

Query: {query}
Documents: {docs}
"""
```

### 5. Lost-in-the-Middle Mitigation
LLMs attend better to content at the start and end of the context window. After ranking, reorder so the highest-scored chunks appear first and last, not in the middle.

```python
def reorder_for_llm(docs: list[Document]) -> list[Document]:
    result = []
    for i, doc in enumerate(docs):
        if i % 2 == 0:
            result.append(doc)
        else:
            result.insert(0, doc)
    return result
```

## Choosing a Ranker

| Scenario | Recommended |
|---|---|
| Multi-source merge | RRF |
| Best accuracy, latency acceptable | Cross-encoder |
| Hosted, easy integration | Cohere Rerank |
| Very small candidate set | LLM-based |
