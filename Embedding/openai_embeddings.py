"""
OpenAI Embeddings
-----------------
Wraps OpenAI's text-embedding-3-small / text-embedding-3-large models.
Requires OPENAI_API_KEY environment variable.
"""

from langchain_openai import OpenAIEmbeddings
from .base_embedder import BaseEmbedder


OPENAI_MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """
    Args:
        model: OpenAI embedding model name (default: text-embedding-3-small).

    Example:
        >>> embedder = OpenAIEmbedder()
        >>> vectors = embedder.embed_documents(["hello world", "RAG pipelines"])
        >>> query_vec = embedder.embed_query("what is RAG?")
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        if model not in OPENAI_MODELS:
            raise ValueError(f"Unknown OpenAI embedding model: {model}. Choose from {list(OPENAI_MODELS)}")
        self._model_name = model
        self._dimensions = OPENAI_MODELS[model]
        self._model = OpenAIEmbeddings(model=model)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def get_model(self) -> OpenAIEmbeddings:
        return self._model


if __name__ == "__main__":
    embedder = OpenAIEmbedder("text-embedding-3-small")
    vec = embedder.embed_query("What is retrieval-augmented generation?")
    print(f"Model     : {embedder.model_name}")
    print(f"Dimensions: {embedder.dimensions}")
    print(f"Vector len: {len(vec)}")
    print(f"Collection: {embedder.collection_name('rag_docs')}")
