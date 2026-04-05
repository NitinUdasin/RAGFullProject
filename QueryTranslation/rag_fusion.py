"""
RAG-Fusion
----------
Extends multi-query retrieval by re-ranking merged results using
Reciprocal Rank Fusion (RRF) — a parameter-free rank aggregation algorithm.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).
"""

from collections import defaultdict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .multi_query import generate_query_variants


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Merge multiple ranked document lists using Reciprocal Rank Fusion.

    RRF score for document d across lists:
        score(d) = Σ  1 / (k + rank(d, list_i))

    A higher k reduces the impact of top-ranked documents.
    k=60 is the standard default from the original paper.

    Args:
        result_lists: List of ranked document lists (one per query variant).
        k:            RRF constant (default 60).

    Returns:
        Single merged and re-ranked list of Documents (highest RRF score first).
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = _doc_id(doc)
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    return sorted(doc_map.values(), key=lambda d: scores[_doc_id(d)], reverse=True)


def _doc_id(doc: Document) -> str:
    """Stable document identifier: prefer metadata['doc_id'], fall back to content hash."""
    return doc.metadata.get("doc_id") or str(hash(doc.page_content.strip()))


# ── Main retrieval function ───────────────────────────────────────────────────

def rag_fusion_retrieve(
    query: str,
    retriever,
    n: int = 4,
    k: int = 60,
    llm=None,
) -> list[Document]:
    """
    Multi-query retrieval followed by RRF re-ranking.

    Args:
        query:     Original user question.
        retriever: Any LangChain retriever.
        n:         Number of query variants to generate.
        k:         RRF constant.
        llm:       Optional LangChain chat model for variant generation.

    Returns:
        RRF-merged and re-ranked list of Documents.

    Example:
        >>> docs = rag_fusion_retrieve("What is attention in transformers?", retriever)
    """
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    variants = generate_query_variants(query, n=n, llm=llm)

    result_lists: list[list[Document]] = []
    for variant in variants:
        results = retriever.invoke(variant)
        result_lists.append(results)

    return reciprocal_rank_fusion(result_lists, k=k)


# ── LangGraph node ────────────────────────────────────────────────────────────

def rag_fusion_node(state: dict, retriever, n: int = 4, k: int = 60, llm=None) -> dict:
    """LangGraph node: populates state['documents'] using RAG-Fusion."""
    docs = rag_fusion_retrieve(
        state.get("translated_query", state["query"]), retriever, n, k, llm
    )
    return {**state, "documents": docs}


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demonstrate RRF merging with mock ranked lists
    from langchain_core.documents import Document

    list_a = [
        Document(page_content="Doc A", metadata={"doc_id": "a"}),
        Document(page_content="Doc B", metadata={"doc_id": "b"}),
        Document(page_content="Doc C", metadata={"doc_id": "c"}),
    ]
    list_b = [
        Document(page_content="Doc B", metadata={"doc_id": "b"}),
        Document(page_content="Doc D", metadata={"doc_id": "d"}),
        Document(page_content="Doc A", metadata={"doc_id": "a"}),
    ]

    merged = reciprocal_rank_fusion([list_a, list_b])
    print("RRF merged ranking:")
    for i, doc in enumerate(merged, 1):
        print(f"  {i}. {doc.page_content} (id={doc.metadata['doc_id']})")
