# RAGFullProject

A comprehensive, hands-on collection of Retrieval-Augmented Generation (RAG) demos built with **LangGraph**, **FastAPI**, **ChromaDB**, and support for **Anthropic Claude**, **OpenAI**, and **local models** via Ollama.

Each folder is a self-contained demo module covering a specific component of the RAG pipeline — from raw document ingestion to final answer generation.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION                               │
│  DocumentLoading → Chunking → Embedding → VectorStore           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         QUERY TIME                              │
│  QueryTranslation → QueryStructuring → Routing                  │
│       → Retrieval → Ranking → ContextAssembly → Generation      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Modules

| Module | Description |
|---|---|
| [`DocumentLoading`](DocumentLoading/) | Load PDFs, web pages, and text files into Documents |
| [`Chunking`](Chunking/) | Fixed-size, recursive, semantic, parent-child, sentence-window |
| [`Embedding`](Embedding/) | OpenAI, HuggingFace, Ollama — with factory and batching |
| [`VectorStore`](VectorStore/) | ChromaDB upsert, search, metadata filtering |
| [`Indexing`](Indexing/) | End-to-end ingestion orchestrator (flat, parent-child, multi-vector) |
| [`QueryTranslation`](QueryTranslation/) | Rewriting, multi-query, RAG-Fusion, step-back, HyDE, decomposition |
| [`QueryStructuring`](QueryStructuring/) | Extract ChromaDB metadata filters from natural language |
| [`Routing`](Routing/) | Logical (LLM) and semantic (embedding) routing |
| [`Retrieval`](Retrieval/) | Dense, sparse (BM25), hybrid, self-query |
| [`Ranking`](Ranking/) | RRF, cross-encoder, Cohere Rerank, LLM reranking |
| [`ContextAssembly`](ContextAssembly/) | Dedup, truncation, lost-in-the-middle reordering |
| [`Generation`](Generation/) | Anthropic, OpenAI, local (Ollama) providers + streaming |
| [`PageIndex`](PageIndex/) | Full end-to-end LangGraph pipeline wiring all modules |

---

## Tech Stack

- **Orchestration** — [LangGraph](https://github.com/langchain-ai/langgraph)
- **Framework** — [FastAPI](https://fastapi.tiangolo.com/)
- **Vector Store** — [ChromaDB](https://www.trychroma.com/)
- **LLM Providers** — Anthropic Claude, OpenAI GPT, Ollama (local)
- **Package Manager** — [uv](https://github.com/astral-sh/uv)
- **Testing** — pytest

---

## Getting Started

### 1. Install dependencies

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

### 2. Set environment variables

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
```

### 3. Start ChromaDB

```bash
docker run -p 8001:8000 chromadb/chroma
```

### 4. Run the API server

```bash
uv run uvicorn app.main:app --reload --port 8000
```

### 5. Run tests

```bash
uv run pytest                              # all tests
uv run pytest tests/unit/                 # unit only
uv run pytest -m "not integration"        # skip integration tests
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` / `openai` / `local` | `openai` |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `EMBEDDING_PROVIDER` | `openai` / `huggingface` / `ollama` | `openai` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `CHROMA_HOST` | ChromaDB host | `localhost` |
| `CHROMA_PORT` | ChromaDB port | `8001` |
| `CHROMA_COLLECTION` | Collection name | `rag_docs` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

---

## Project Structure

```
RAGFullProject/
├── DocumentLoading/       # Ingestion: load raw sources
├── Chunking/              # Ingestion: split into chunks
├── Embedding/             # Ingestion + query: encode text to vectors
├── VectorStore/           # Ingestion + query: ChromaDB interface
├── Indexing/              # Ingestion: full pipeline orchestrator
├── QueryTranslation/      # Query: improve/expand the user query
├── QueryStructuring/      # Query: extract metadata filters
├── Routing/               # Query: select retrieval source
├── Retrieval/             # Query: fetch relevant chunks
├── Ranking/               # Query: reorder candidates
├── ContextAssembly/       # Query: build context string for LLM
├── Generation/            # Query: generate final answer
├── PageIndex/             # Full end-to-end LangGraph graph
└── docs/                  # Detailed technique write-ups
```
