"""
RAG Pipeline State
------------------
TypedDict shared across all LangGraph nodes in the full RAG pipeline.
Each node reads from and writes to this state.
"""

from typing import TypedDict, Optional
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    query: str                          # original user question

    # ── Query Translation ──────────────────────────────────────────────────────
    translated_query: str               # rewritten / expanded query

    # ── Query Structuring ──────────────────────────────────────────────────────
    metadata_filters: dict              # ChromaDB where clause

    # ── Routing ────────────────────────────────────────────────────────────────
    route: str                          # routing destination
    route_reasoning: str                # explanation for the routing decision

    # ── Retrieval ──────────────────────────────────────────────────────────────
    documents: list[Document]           # retrieved + ranked chunks
    step_back_docs: list[Document]      # for step-back retrieval strategy

    # ── Context Assembly ───────────────────────────────────────────────────────
    context: str                        # formatted context string for generation
    sources: list[str]                  # deduplicated source paths

    # ── Generation ─────────────────────────────────────────────────────────────
    answer: str                         # final LLM-generated answer

    # ── Decomposition (optional) ───────────────────────────────────────────────
    sub_questions: list[str]
    sub_answers: list[str]
