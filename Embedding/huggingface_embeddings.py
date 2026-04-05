"""
HuggingFace Embeddings
-----------------------
Local sentence-transformer models via langchain-huggingface.
No API key required — runs on CPU or GPU.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from .base_embedder import BaseEmbedder


HF_MODELS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Local embedding model using HuggingFace sentence-transformers.

    Args:
        model:      HuggingFace model name or path (default: BAAI/bge-small-en-v1.5).
        device:     "cpu", "cuda", or "mps" (default: "cpu").
        normalize:  L2-normalize embeddings (recommended for cosine similarity).

    Example:
        >>> embedder = HuggingFaceEmbedder("BAAI/bge-small-en-v1.5")
        >>> vectors = embedder.embed_documents(["hello world"])
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self._model_name = model
        self._dimensions = HF_MODELS.get(model, 768)
        self._model = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize},
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def get_model(self) -> HuggingFaceEmbeddings:
        return self._model


if __name__ == "__main__":
    embedder = HuggingFaceEmbedder()
    vec = embedder.embed_query("What is semantic search?")
    print(f"Model     : {embedder.model_name}")
    print(f"Dimensions: {embedder.dimensions}")
    print(f"Vector len: {len(vec)}")
