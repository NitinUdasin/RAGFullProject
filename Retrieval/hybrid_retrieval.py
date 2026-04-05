"""
Hybrid Retrieval
----------------
Combines dense (vector) and sparse (BM25) retrieval using
Reciprocal Rank Fusion for higher recall than either method alone.
"""

from langchain_core.documents import Document

from .dense_retrieval import DenseRetriever
from .sparse_retrieval import BM25Retriever


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists using RRF (shared with QueryTranslation)."""
    from collections import defaultdict

    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc.metadata.get("doc_id") or str(hash(doc.page_content.strip()))
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    return sorted(doc_map.values(), key=lambda d: scores[
        d.metadata.get("doc_id") or str(hash(d.page_content.strip()))
    ], reverse=True)


class HybridRetriever:
    """
    Combines dense and sparse retrieval with RRF merging.

    Args:
        dense_retriever:  DenseRetriever instance.
        sparse_retriever: BM25Retriever instance (must be pre-built with same corpus).
        k:                Number of final results to return.
        rrf_k:            RRF constant (default 60).

    Example:
        >>> hybrid = HybridRetriever(dense, sparse, k=5)
        >>> docs = hybrid.retrieve("API rate limit exceeded")
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        k: int = 5,
        rrf_k: int = 60,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.k = k
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> list[Document]:
        dense_results = self.dense.retrieve(query)
        sparse_results = self.sparse.retrieve(query)
        merged = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)
        return merged[: self.k]

    def as_langchain_retriever(self):
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun

        hybrid = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                return hybrid.retrieve(query)

        return _Retriever()


# ── LangGraph node ─────────────────────────────────────────────────────────────

def hybrid_retrieval_node(state: dict, retriever: HybridRetriever) -> dict:
    query = state.get("translated_query", state["query"])
    docs = retriever.retrieve(query)
    return {**state, "documents": docs}


if __name__ == "__main__":
    from langchain_core.documents import Document
    from VectorStore import ChromaStore
    from Embedding import get_embedder

    corpus = [
        Document(page_content="ChromaDB stores embeddings for vector search.", metadata={"source": "docs", "doc_id": "1"}),
        Document(page_content="BM25 is a keyword-based ranking algorithm.", metadata={"source": "docs", "doc_id": "2"}),
        Document(page_content="Hybrid retrieval combines dense and sparse methods.", metadata={"source": "docs", "doc_id": "3"}),
    ]

    embedder = get_embedder()
    store = ChromaStore("hybrid_test", embedding_fn=embedder.embed_documents)
    store.upsert(corpus)

    dense = DenseRetriever(store, embedder, k=3)
    sparse = BM25Retriever(corpus, k=3)
    hybrid = HybridRetriever(dense, sparse, k=3)

    results = hybrid.retrieve("vector search embeddings")
    print("Hybrid results:")
    for r in results:
        print(f"  {r.page_content[:80]}")
