"""
Dense Retrieval
---------------
Nearest-neighbour vector search using cosine similarity in ChromaDB.
The standard retrieval method — fast and accurate for semantic queries.
"""

from langchain_core.documents import Document
from VectorStore import ChromaStore
from Embedding import BaseEmbedder, get_embedder


class DenseRetriever:
    """
    Retrieves documents using dense vector similarity search.

    Args:
        store:     ChromaStore instance.
        embedder:  BaseEmbedder for query embedding.
        k:         Number of results to return (default 5).
        where:     Optional ChromaDB metadata filter.

    Example:
        >>> retriever = DenseRetriever(store, embedder, k=5)
        >>> docs = retriever.retrieve("How does attention work?")
    """

    def __init__(
        self,
        store: ChromaStore,
        embedder: BaseEmbedder | None = None,
        k: int = 5,
        where: dict | None = None,
    ):
        self.store = store
        self.embedder = embedder or get_embedder()
        self.k = k
        self.where = where

    def retrieve(self, query: str) -> list[Document]:
        """Embed query and return top-k nearest documents."""
        query_embedding = self.embedder.embed_query(query)
        return self.store.search(query_embedding, n_results=self.k, where=self.where)

    def as_langchain_retriever(self):
        """Return a LangChain BaseRetriever-compatible object."""
        return self.store.as_langchain_retriever(k=self.k, where=self.where)


# ── LangGraph node ─────────────────────────────────────────────────────────────

def dense_retrieval_node(state: dict, retriever: DenseRetriever) -> dict:
    """LangGraph node: retrieves documents using dense search."""
    query = state.get("translated_query", state["query"])
    docs = retriever.retrieve(query)
    return {**state, "documents": docs}


if __name__ == "__main__":
    from DocumentLoading import StringLoader
    from Indexing import Indexer

    indexer = Indexer("dense_test")
    loader = StringLoader("ChromaDB is a vector database used in RAG systems.", source="test")
    indexer.ingest_from_loader(loader)

    retriever = DenseRetriever(indexer.store, indexer.embedder)
    results = retriever.retrieve("what is ChromaDB?")
    for r in results:
        print(r.page_content[:100])
