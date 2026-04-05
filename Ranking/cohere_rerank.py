"""
Cohere Rerank API
-----------------
Hosted reranking via Cohere's Rerank endpoint.
No local model needed — simple API call with high accuracy.
Requires: pip install cohere, COHERE_API_KEY env var.
"""

import os
from langchain_core.documents import Document


class CohereReranker:
    """
    Reranks documents using Cohere's Rerank API.

    Args:
        model:  Cohere rerank model (default: rerank-english-v3.0).
        top_n:  Number of top results to return.
        api_key: Cohere API key (defaults to COHERE_API_KEY env var).

    Example:
        >>> reranker = CohereReranker(top_n=5)
        >>> reranked = reranker.rerank("what is RAG?", candidate_docs)
    """

    MODELS = {
        "english-v3": "rerank-english-v3.0",
        "multilingual-v3": "rerank-multilingual-v3.0",
        "english-v2": "rerank-english-v2.0",
    }

    def __init__(
        self,
        model: str = "english-v3",
        top_n: int = 5,
        api_key: str | None = None,
    ):
        import cohere

        self.top_n = top_n
        model_name = self.MODELS.get(model, model)
        self.model = model_name
        self.client = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Call Cohere Rerank API and return top_n re-ordered documents.

        Args:
            query:     User query string.
            documents: Candidate documents from retrieval.

        Returns:
            Top-N documents sorted by Cohere relevance score.
        """
        if not documents:
            return []

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=self.top_n,
        )

        results = []
        for hit in response.results:
            doc = documents[hit.index]
            doc.metadata["_rerank_score"] = hit.relevance_score
            results.append(doc)
        return results


# ── LangGraph node ─────────────────────────────────────────────────────────────

def cohere_rerank_node(state: dict, reranker: CohereReranker) -> dict:
    """LangGraph node: reranks state['documents'] using Cohere."""
    query = state.get("translated_query", state["query"])
    reranked = reranker.rerank(query, state.get("documents", []))
    return {**state, "documents": reranked}
