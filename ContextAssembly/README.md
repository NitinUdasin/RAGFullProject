# Context Assembly

Combines ranked retrieved chunks into a single coherent context string passed to the LLM. Poor assembly wastes the context window and degrades answer quality.

## Pipeline Position

```
[Ranking] → [ContextAssembly] → [Generation]
```

## Responsibilities

1. **Deduplication** — remove near-identical chunks (e.g., from multi-query retrieval)
2. **Truncation** — ensure total context fits within the LLM's context window
3. **Formatting** — structure chunks clearly so the LLM can attribute and reference them
4. **Ordering** — apply lost-in-the-middle mitigation (see `Ranking/`)
5. **Provenance** — attach source metadata so the LLM can cite references

## Basic Assembly

```python
def assemble_context(docs: list[Document], max_tokens: int = 3000) -> str:
    parts = []
    total = 0
    for i, doc in enumerate(docs):
        snippet = f"[{i+1}] Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        tokens = count_tokens(snippet)
        if total + tokens > max_tokens:
            break
        parts.append(snippet)
        total += tokens
    return "\n\n---\n\n".join(parts)
```

## Deduplication

```python
def deduplicate(docs: list[Document], threshold: float = 0.95) -> list[Document]:
    seen_embeddings = []
    unique = []
    for doc in docs:
        emb = embed(doc.page_content)
        if all(cosine_sim(emb, s) < threshold for s in seen_embeddings):
            unique.append(doc)
            seen_embeddings.append(emb)
    return unique
```

For cheaper dedup, hash normalized text instead of comparing embeddings.

## Context Window Budget

| Component | Typical Token Budget |
|---|---|
| System prompt | 200–500 |
| Retrieved context | 2000–6000 |
| Chat history | 500–2000 |
| User query | 50–200 |
| Response (output) | 500–2000 |

Always measure with the same tokenizer as the target model (`tiktoken` for OpenAI, `anthropic` tokenizer for Claude).

## Prompt Template

```python
CONTEXT_TEMPLATE = """Use the following context to answer the question.
If the answer is not in the context, say so — do not fabricate.

Context:
{context}

Question: {question}
"""
```

## Provenance / Citations

Include `source` and `page` in each chunk header so the LLM can cite:

```
[1] Source: annual_report_2024.pdf (page 12)
Revenue grew 23% year-over-year...

[2] Source: press_release_q3.html
The company announced expansion into three new markets...
```
