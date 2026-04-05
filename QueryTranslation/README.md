# Query Translation

Transforms raw user input into better queries before retrieval to bridge the vocabulary gap between questions and indexed documents.

## Techniques Covered

| Technique | Description |
|---|---|
| **Query Rewriting** | Makes a vague query explicit and self-contained |
| **Multi-Query** | Generates N alternative phrasings, merges results |
| **RAG-Fusion** | Multi-query + Reciprocal Rank Fusion (RRF) reranking |
| **Step-Back** | Abstracts to a broader question to retrieve background context |
| **HyDE** | Embeds a hypothetical answer instead of the raw query |
| **Decomposition** | Breaks multi-hop questions into ordered sub-questions |

## Pipeline Position

```
[User Input] → [Query Translation] → [Routing] → [Retrieval]
```

## Key Files

- `query_rewrite.py` — LLM-based rewriting chain
- `multi_query.py` — multi-query generation + deduplication
- `rag_fusion.py` — RRF merging logic
- `step_back.py` — step-back abstraction chain
- `hyde.py` — hypothetical document embedding
- `decomposition.py` — sub-question decomposition

## See Also

- `docs/QueryTranslation.md` — detailed technique explanations with code samples
