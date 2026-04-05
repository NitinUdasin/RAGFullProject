# Query Structuring for Metadata Filters

Extracts structured filters from a natural language query so that vector search can be narrowed using ChromaDB's `where` clause. Combines semantic search with exact metadata matching.

## Pipeline Position

```
[Query Translation] → [Query Structuring] → [Retrieval (with filters)]
```

## Why It Matters

Vector search alone cannot handle:
- "Show me PDFs from Q3 2024"
- "Find Python tutorials rated above 4 stars"
- "Retrieve internal documents tagged 'confidential'"

These constraints belong in metadata filters, not the query embedding.

## Approach: Structured Output from LLM

Define a Pydantic schema matching your ChromaDB metadata fields, then use `with_structured_output` to extract filters.

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

class DocumentFilter(BaseModel):
    topic: Optional[str] = Field(None, description="Main topic or category")
    source_type: Optional[str] = Field(None, description="File type: pdf, html, docx")
    year: Optional[int] = Field(None, description="Publication year")
    author: Optional[str] = Field(None, description="Document author")

structured_llm = llm.with_structured_output(DocumentFilter)

filter_prompt = """
Extract any metadata constraints from the query.
Only populate fields that are explicitly mentioned.

Query: {query}
"""

filters: DocumentFilter = structured_llm.invoke(filter_prompt.format(query=query))
```

## Translating to ChromaDB `where` Clause

```python
def build_where_clause(filters: DocumentFilter) -> dict:
    where = {}
    if filters.year:
        where["year"] = {"$eq": filters.year}
    if filters.source_type:
        where["source_type"] = {"$eq": filters.source_type}
    if filters.topic:
        where["topic"] = {"$eq": filters.topic}
    return where if where else None

collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where=build_where_clause(filters),
)
```

## ChromaDB Filter Operators

| Operator | Example |
|---|---|
| `$eq` | `{"year": {"$eq": 2024}}` |
| `$ne` | `{"status": {"$ne": "draft"}}` |
| `$gt`, `$gte` | `{"rating": {"$gt": 4}}` |
| `$lt`, `$lte` | `{"page_count": {"$lt": 50}}` |
| `$in` | `{"tag": {"$in": ["rag", "llm"]}}` |
| `$and`, `$or` | `{"$and": [{...}, {...}]}` |

## Metadata Schema Requirements

Metadata fields used in filters must be:
1. Stored at chunk upsert time (set in `DocumentLoading` or `Chunking`)
2. Indexed — ChromaDB indexes all metadata fields automatically
3. Consistent types — do not mix string `"2024"` and int `2024` for the same field
