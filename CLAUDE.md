# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) system built with LangGraph for orchestration, FastAPI for serving, ChromaDB as the vector store, and support for multiple LLM providers (Anthropic Claude, OpenAI, local models via Ollama/LlamaCpp).It is a collection of demo projects for RAG in details, we will cover following in this project : 
1. Query Translation
2. Routing
3. Indexing
4. Retrival
5. Genration
6. Ranking
7. Query structuring for metadata filters
8. Page Index (end of RAG)


## Common Commands

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Add a new dependency
uv add fastapi chromadb langgraph

# Add a dev-only dependency
uv add --dev pytest pytest-cov ruff mypy

# Run the FastAPI server (development)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_retriever.py

# Run a single test by name
uv run pytest tests/test_retriever.py::test_function_name -v

# Run tests with coverage
uv run pytest --cov=app --cov-report=html

# Lint
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy app/
```

Dependencies are declared in `pyproject.toml`. The lockfile is `uv.lock` — commit it. Do not use `pip` or `requirements.txt`.

## Architecture

### Core Pipeline (LangGraph)

The RAG pipeline is modeled as a stateful LangGraph graph. Each node in the graph is a discrete step:

1. **Ingestion graph** — chunking, embedding, and upserting documents into ChromaDB
2. **Retrieval graph** — query rewriting → vector search → optional reranking
3. **Generation graph** — context assembly → LLM call → response streaming

Graph definitions live in `app/graphs/`. State schemas (TypedDicts) are defined alongside each graph.

### LLM Provider Abstraction

`app/llm/` contains a provider factory pattern. The active provider is selected via `LLM_PROVIDER` env var (`anthropic`, `openai`, `local`). All providers expose a common interface so graph nodes stay provider-agnostic.

### Vector Store (ChromaDB)

ChromaDB client and collection management are in `app/vectorstore/`. Collections are created per-document-set or per-tenant. Embeddings are generated separately (via the embedding model configured in `EMBEDDING_MODEL`) before being upserted.

### FastAPI Layer

`app/main.py` is the entry point. Routers in `app/api/` expose:
- `POST /ingest` — trigger document ingestion
- `POST /query` — run the RAG query pipeline
- `GET /health` — health check

### Key Environment Variables

```
LLM_PROVIDER=anthropic          # anthropic | openai | local
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
EMBEDDING_MODEL=...             # e.g. text-embedding-3-small or a local model name
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION=rag_docs
LOCAL_MODEL_PATH=...            # path to GGUF or similar for local inference
```

## Testing Conventions

- Unit tests mock the ChromaDB client and LLM provider; do not hit live APIs in `tests/unit/`
- Integration tests in `tests/integration/` require a running ChromaDB instance and real API keys
- Use `pytest.mark.integration` to tag integration tests; skip them in CI with `-m "not integration"`

## LangGraph Notes

- Graphs are compiled once at startup and reused across requests (thread-safe reads, stateless nodes)
- Streaming responses use LangGraph's `astream_events` interface piped through a FastAPI `StreamingResponse`
- Checkpointers (for conversation memory) use an in-memory store by default; swap to a persistent backend via `LANGGRAPH_CHECKPOINTER` env var
