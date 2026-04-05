"""
Base Embedder
-------------
Abstract wrapper around LangChain embedding models.
Adds batching, token counting, and a consistent interface across providers.
"""

from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings


class BaseEmbedder(ABC):
    """Thin wrapper that enforces a consistent embedding interface."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    def get_model(self) -> Embeddings:
        """Return the underlying LangChain Embeddings instance."""
        ...

    def embed_documents(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embed a list of texts in batches."""
        model = self.get_model()
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            all_embeddings.extend(model.embed_documents(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.get_model().embed_query(text)

    def collection_name(self, base: str) -> str:
        """Generate a versioned ChromaDB collection name."""
        safe = self.model_name.replace("/", "-").replace(":", "-")
        return f"{base}_{safe}"
