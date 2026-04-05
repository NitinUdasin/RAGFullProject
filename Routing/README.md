# Routing

Decides which retrieval path, data source, or pipeline branch to use based on the translated query. Prevents routing every query through a single expensive pipeline.

## Types of Routing

### 1. Logical Routing (LLM-based)
The LLM classifies the query and returns a destination label.

```python
from pydantic import BaseModel
from typing import Literal

class RouteQuery(BaseModel):
    destination: Literal["vectorstore", "web_search", "sql_database"]

structured_llm = llm.with_structured_output(RouteQuery)
route = structured_llm.invoke(query)
```

**Use when:** destinations have clear semantic differences (internal docs vs. live web vs. structured DB).

### 2. Semantic Routing (Embedding-based)
Embed the query and compare cosine similarity against prompt/description embeddings for each route. Pick the closest.

```python
# Pre-embed route descriptions once at startup
route_embeddings = {
    "billing": embed("questions about invoices, payments, and subscriptions"),
    "technical": embed("questions about bugs, APIs, and integrations"),
}

def semantic_route(query: str) -> str:
    q_emb = embed(query)
    return max(route_embeddings, key=lambda k: cosine_sim(q_emb, route_embeddings[k]))
```

**Use when:** routes differ in topic but may share vocabulary; no LLM call needed.

## Pipeline Position

```
[Query Translation] → [Routing] → [VectorStore Retrieval]
                                → [Web Search]
                                → [SQL Query]
                                → [Graph DB]
```

## LangGraph Integration

Routing maps to a **conditional edge** in the graph:

```python
graph.add_conditional_edges(
    "route_query",
    lambda state: state["route"],
    {
        "vectorstore": "retrieve_vectorstore",
        "web_search": "retrieve_web",
        "sql_database": "retrieve_sql",
    }
)
```
