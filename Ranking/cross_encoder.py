"""
Cross-Encoder Reranking
-----------------------
Scores each (query, document) pair jointly using a cross-encoder model.
More accurate than bi-encoder cosine similarity but slower — use on a
small candidate set (10–50 docs) after initial retrieval.

Requires: pip install sentence-transformers
"""

from langchain_core.documents import Document


CROSS_ENCODER_MODELS = {
    "mini":   "cross-encoder/ms-marco-MiniLM-L-6-v2",    # fastest
    "medium": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "large":  "cross-encoder/ms-marco-electra-base",      # most accurate
}


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.

    Args:
        model:   Model alias ("mini", "medium", "large") or a full HuggingFace model name.
        top_n:   Number of top results to return after reranking.
        device:  "cpu", "cuda", or "mps".

    Example:
        >>> reranker = CrossEncoderReranker(model="mini", top_n=5)
        >>> reranked = reranker.rerank("How does RAG work?", candidate_docs)
    """

    def __init__(
        self,
        model: str = "mini",
        top_n: int = 5,
        device: str = "cpu",
    ):
        from sentence_transformers import CrossEncoder

        model_name = CROSS_ENCODER_MODELS.get(model, model)
        self.model = CrossEncoder(model_name, device=device)
        self.top_n = top_n

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Score all (query, doc) pairs and return top_n docs sorted by score.

        Args:
            query:     The user query.
            documents: Candidate documents from initial retrieval.

        Returns:
            Top-N documents sorted by cross-encoder score (highest first).
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[: self.top_n]:
            doc.metadata["_rerank_score"] = float(score)
            results.append(doc)
        return results


# ── LangGraph node ─────────────────────────────────────────────────────────────

def cross_encoder_node(state: dict, reranker: CrossEncoderReranker) -> dict:
    """LangGraph node: reranks state['documents'] using a cross-encoder."""
    query = state.get("translated_query", state["query"])
    reranked = reranker.rerank(query, state.get("documents", []))
    return {**state, "documents": reranked}


if __name__ == "__main__":
    from langchain_core.documents import Document

    docs = [
        Document(page_content="RAG combines retrieval and generation for factual answers.", metadata={"doc_id": "1"}),
        Document(page_content="The Eiffel Tower is in Paris, France.", metadata={"doc_id": "2"}),
        Document(page_content="Vector databases store dense embeddings for semantic search.", metadata={"doc_id": "3"}),
    ]
    reranker = CrossEncoderReranker(model="mini", top_n=2)
    results = reranker.rerank("How does RAG work?", docs)
    for r in results:
        print(f"[{r.metadata['_rerank_score']:.4f}] {r.page_content}")
