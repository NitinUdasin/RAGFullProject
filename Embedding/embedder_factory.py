"""
Embedder Factory
----------------
Selects and returns the correct BaseEmbedder based on EMBEDDING_PROVIDER
and EMBEDDING_MODEL environment variables.
"""

import os
from .base_embedder import BaseEmbedder


def get_embedder(
    provider: str | None = None,
    model: str | None = None,
) -> BaseEmbedder:
    """
    Factory function for embedding models.

    Reads EMBEDDING_PROVIDER and EMBEDDING_MODEL from env if not passed directly.

    Providers:
        - "openai"       → OpenAIEmbedder
        - "huggingface"  → HuggingFaceEmbedder
        - "ollama"       → OllamaEmbedder

    Args:
        provider: Embedding provider name.
        model:    Model name (provider-specific).

    Returns:
        A BaseEmbedder instance.

    Example:
        >>> embedder = get_embedder()           # reads from env
        >>> embedder = get_embedder("openai", "text-embedding-3-small")
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "openai")).lower()
    model = model or os.getenv("EMBEDDING_MODEL")

    if provider == "openai":
        from .openai_embeddings import OpenAIEmbedder
        return OpenAIEmbedder(model or "text-embedding-3-small")

    if provider == "huggingface":
        from .huggingface_embeddings import HuggingFaceEmbedder
        return HuggingFaceEmbedder(model or "BAAI/bge-small-en-v1.5")

    if provider == "ollama":
        from .ollama_embeddings import OllamaEmbedder
        return OllamaEmbedder(model or "nomic-embed-text")

    raise ValueError(
        f"Unknown embedding provider: '{provider}'. "
        "Choose from: openai, huggingface, ollama"
    )


if __name__ == "__main__":
    embedder = get_embedder("openai")
    print(f"Provider  : openai")
    print(f"Model     : {embedder.model_name}")
    print(f"Dimensions: {embedder.dimensions}")
    print(f"Collection: {embedder.collection_name('rag_docs')}")
