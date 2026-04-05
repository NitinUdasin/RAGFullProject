"""
Ollama Embeddings
-----------------
Local embeddings served via Ollama. Requires Ollama running at OLLAMA_HOST
with the target model already pulled (e.g. `ollama pull nomic-embed-text`).
"""

import os
from langchain_ollama import OllamaEmbeddings
from .base_embedder import BaseEmbedder


OLLAMA_MODELS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


class OllamaEmbedder(BaseEmbedder):
    """
    Args:
        model:    Ollama model name (default: nomic-embed-text).
        base_url: Ollama server URL (default: http://localhost:11434).

    Example:
        >>> embedder = OllamaEmbedder("nomic-embed-text")
        >>> vec = embedder.embed_query("hello world")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
    ):
        self._model_name = model
        self._dimensions = OLLAMA_MODELS.get(model, 768)
        self._model = OllamaEmbeddings(
            model=model,
            base_url=base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def get_model(self) -> OllamaEmbeddings:
        return self._model


if __name__ == "__main__":
    embedder = OllamaEmbedder()
    vec = embedder.embed_query("test query")
    print(f"Model     : {embedder.model_name}")
    print(f"Vector len: {len(vec)}")
