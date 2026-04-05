"""
Semantic Router (Embedding-based)
----------------------------------
Routes queries by comparing the query embedding against pre-embedded route
descriptions using cosine similarity. No LLM call needed — fast and cheap.
"""

import numpy as np
from dataclasses import dataclass, field
from langchain_openai import OpenAIEmbeddings


# ── Route definition ──────────────────────────────────────────────────────────

@dataclass
class Route:
    """A named route with example phrases that describe its content domain."""
    name: str
    descriptions: list[str]          # representative phrases — more = better coverage
    embedding: np.ndarray = field(default=None, repr=False)  # set at build time


# ── Router ────────────────────────────────────────────────────────────────────

class SemanticRouter:
    """
    Embedding-based router. At init, embeds all route descriptions and averages
    them into a centroid. At query time, picks the route with highest cosine
    similarity to the query embedding.

    Args:
        routes:          List of Route objects.
        embedding_model: LangChain embeddings instance (default: OpenAIEmbeddings).
        threshold:       Minimum similarity to accept a route. If all routes fall
                         below this, returns "fallback".

    Example:
        >>> routes = [
        ...     Route("billing",   ["invoice", "payment", "subscription", "refund"]),
        ...     Route("technical", ["bug", "API", "integration", "error", "code"]),
        ...     Route("general",   ["help", "question", "information"]),
        ... ]
        >>> router = SemanticRouter(routes)
        >>> router.route("how do I cancel my subscription?")
        'billing'
    """

    def __init__(
        self,
        routes: list[Route],
        embedding_model=None,
        threshold: float = 0.4,
    ):
        self.routes = routes
        self.threshold = threshold
        self._model = embedding_model or OpenAIEmbeddings()
        self._build(routes)

    def _build(self, routes: list[Route]) -> None:
        """Pre-compute centroid embeddings for each route."""
        for route in routes:
            embeddings = self._model.embed_documents(route.descriptions)
            matrix = np.array(embeddings)
            centroid = matrix.mean(axis=0)
            route.embedding = centroid / (np.linalg.norm(centroid) + 1e-10)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def route(self, query: str) -> str:
        """
        Return the name of the best matching route for the query,
        or "fallback" if no route exceeds the similarity threshold.
        """
        query_emb = np.array(self._model.embed_query(query))
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        best_route = max(self.routes, key=lambda r: self._cosine_similarity(query_emb, r.embedding))
        best_score = self._cosine_similarity(query_emb, best_route.embedding)

        return best_route.name if best_score >= self.threshold else "fallback"

    def route_with_scores(self, query: str) -> list[tuple[str, float]]:
        """Return all routes with their similarity scores, sorted descending."""
        query_emb = np.array(self._model.embed_query(query))
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        scores = [
            (route.name, self._cosine_similarity(query_emb, route.embedding))
            for route in self.routes
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)


# ── LangGraph node ────────────────────────────────────────────────────────────

def semantic_router_node(state: dict, router: SemanticRouter) -> dict:
    """LangGraph node: sets state['route'] using semantic routing."""
    destination = router.route(state.get("translated_query", state["query"]))
    return {**state, "route": destination}


if __name__ == "__main__":
    routes = [
        Route("billing",   ["invoice", "payment", "subscription", "charge", "refund", "billing"]),
        Route("technical", ["bug", "API", "integration", "error", "code", "SDK", "library"]),
        Route("general",   ["what is", "how does", "explain", "overview", "introduction"]),
    ]
    router = SemanticRouter(routes)

    queries = [
        "I was charged twice this month",
        "getting a 401 error from your API",
        "can you explain what embeddings are?",
    ]
    for q in queries:
        scores = router.route_with_scores(q)
        best = scores[0]
        print(f"Query: {q}")
        print(f"Route: {best[0]} (score={best[1]:.3f})\n")
