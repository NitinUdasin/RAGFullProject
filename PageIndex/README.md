# Page Index (End-to-End RAG)

The `PageIndex` component ties together the full RAG pipeline into a single runnable LangGraph graph. It is the entry point for query-time execution.

## Full Pipeline

```
[User Query]
     ↓
[Query Translation]     — rewrite / expand query
     ↓
[Query Structuring]     — extract metadata filters
     ↓
[Routing]               — choose retrieval source
     ↓
[Retrieval]             — vector search with filters
     ↓
[Ranking]               — rerank candidates
     ↓
[Context Assembly]      — deduplicate, truncate, format
     ↓
[Generation]            — LLM answer + citations
     ↓
[Response]
```

## LangGraph Graph Definition

```python
from langgraph.graph import StateGraph, END
from app.state import RAGState

graph = StateGraph(RAGState)

graph.add_node("translate_query",    translate_query_node)
graph.add_node("structure_query",    structure_query_node)
graph.add_node("route_query",        route_query_node)
graph.add_node("retrieve",           retrieval_node)
graph.add_node("rank",               ranking_node)
graph.add_node("assemble_context",   context_assembly_node)
graph.add_node("generate",           generation_node)

graph.set_entry_point("translate_query")
graph.add_edge("translate_query",  "structure_query")
graph.add_edge("structure_query",  "route_query")
graph.add_conditional_edges(
    "route_query",
    lambda s: s["route"],
    {"vectorstore": "retrieve", "web_search": "retrieve_web"},
)
graph.add_edge("retrieve",         "rank")
graph.add_edge("rank",             "assemble_context")
graph.add_edge("assemble_context", "generate")
graph.add_edge("generate",         END)

rag_graph = graph.compile()
```

## State Schema

```python
from typing import TypedDict, Optional

class RAGState(TypedDict):
    query: str                          # original user query
    translated_query: str               # after query translation
    metadata_filters: dict              # extracted from query structuring
    route: str                          # routing decision
    documents: list                     # retrieved + ranked chunks
    context: str                        # assembled context string
    answer: str                         # final generated answer
    sources: list[str]                  # cited sources
```

## FastAPI Integration

```python
# app/api/query.py
@router.post("/query")
async def query_endpoint(request: QueryRequest):
    result = await rag_graph.ainvoke({"query": request.question})
    return {"answer": result["answer"], "sources": result["sources"]}

@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    async def generate():
        async for event in rag_graph.astream_events({"query": request.question}, version="v2"):
            if event["event"] == "on_chat_model_stream":
                yield f"data: {event['data']['chunk'].content}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Evaluation Metrics

| Metric | Tool | What It Measures |
|---|---|---|
| Faithfulness | RAGAS | Answer grounded in retrieved context |
| Answer Relevancy | RAGAS | Answer addresses the question |
| Context Precision | RAGAS | Retrieved chunks are relevant |
| Context Recall | RAGAS | All relevant chunks were retrieved |
