"""
ChromaDB Vector Store
----------------------
Manages ChromaDB collections: creation, upsert, similarity search,
metadata filtering, and deletion.
"""

import os
from langchain_core.documents import Document
import chromadb
from chromadb import Collection


def get_chroma_client(host: str | None = None, port: int | None = None) -> chromadb.ClientAPI:
    """
    Return a ChromaDB client.
    Uses HTTP client if host is provided, otherwise in-process (dev/test only).
    """
    host = host or os.getenv("CHROMA_HOST")
    port = int(port or os.getenv("CHROMA_PORT", 8001))

    if host:
        return chromadb.HttpClient(host=host, port=port)
    return chromadb.Client()


class ChromaStore:
    """
    Wrapper around a single ChromaDB collection for upsert and retrieval.

    Args:
        collection_name: Name of the ChromaDB collection.
        embedding_fn:    Callable that accepts list[str] and returns list[list[float]].
        distance:        Distance metric — "cosine" or "l2" (default: "cosine").
        host:            ChromaDB host (overrides CHROMA_HOST env var).
        port:            ChromaDB port (overrides CHROMA_PORT env var).

    Example:
        >>> from Embedding import get_embedder
        >>> embedder = get_embedder()
        >>> store = ChromaStore("rag_docs", embedding_fn=embedder.embed_documents)
        >>> store.upsert(chunks)
        >>> results = store.search("what is RAG?", query_embedding=embedder.embed_query("what is RAG?"))
    """

    def __init__(
        self,
        collection_name: str,
        embedding_fn,
        distance: str = "cosine",
        host: str | None = None,
        port: int | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.client = get_chroma_client(host, port)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert(self, documents: list[Document], batch_size: int = 100) -> None:
        """
        Embed and upsert documents into the collection.
        Uses upsert (not add) — safe for re-ingestion.

        Args:
            documents:  List of Document objects with doc_id in metadata.
            batch_size: Number of documents to embed and upsert per batch.
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            ids = [doc.metadata["doc_id"] for doc in batch]
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            embeddings = self.embedding_fn(texts)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[Document]:
        """
        Nearest-neighbour search using a pre-computed query embedding.

        Args:
            query_embedding: Embedded query vector.
            n_results:       Number of results to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            List of Document objects ordered by similarity (closest first).
        """
        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        docs: list[Document] = []
        for content, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            metadata["_distance"] = distance
            docs.append(Document(page_content=content, metadata=metadata))

        return docs

    def search_text(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[Document]:
        """Convenience wrapper: embed query text then search."""
        from Embedding import get_embedder
        embedder = get_embedder()
        query_embedding = embedder.embed_query(query_text)
        return self.search(query_embedding, n_results=n_results, where=where)

    # ── Management ────────────────────────────────────────────────────────────

    def delete(self, doc_ids: list[str]) -> None:
        """Delete documents by their IDs."""
        self.collection.delete(ids=doc_ids)

    def delete_where(self, where: dict) -> None:
        """Delete all documents matching a metadata filter."""
        self.collection.delete(where=where)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def drop(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)

    def as_langchain_retriever(self, k: int = 5, where: dict | None = None):
        """
        Return a minimal LangChain-compatible retriever backed by this store.
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from Embedding import get_embedder

        store = self
        embedder = get_embedder()

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                q_emb = embedder.embed_query(query)
                return store.search(q_emb, n_results=k, where=where)

        return _Retriever()


if __name__ == "__main__":
    from Embedding import get_embedder

    embedder = get_embedder("openai")
    store = ChromaStore(
        collection_name="rag_test",
        embedding_fn=embedder.embed_documents,
    )

    docs = [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "test", "doc_id": "1"}),
        Document(page_content="ChromaDB is a vector database.", metadata={"source": "test", "doc_id": "2"}),
    ]
    store.upsert(docs)
    print(f"Collection count: {store.count()}")

    q_emb = embedder.embed_query("what is RAG?")
    results = store.search(q_emb, n_results=2)
    for r in results:
        print(f"  [{r.metadata['_distance']:.3f}] {r.page_content}")
