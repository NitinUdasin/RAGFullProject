"""
Semantic Chunking
-----------------
Groups sentences into chunks by detecting topic shifts using cosine similarity
between consecutive sentence embeddings. Splits when similarity drops below
a threshold — preserving topical coherence better than fixed-size chunking.
"""

import numpy as np
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from .base_chunker import BaseChunker, ChunkConfig


class SemanticTextChunker(BaseChunker):
    """
    Splits documents at semantic breakpoints detected via embedding similarity.

    Three breakpoint strategies (set via breakpoint_threshold_type):
      - "percentile"   : split at the Nth percentile drop in similarity (default)
      - "standard_deviation": split when drop exceeds mean - N*std
      - "interquartile": split using IQR of similarity scores

    Args:
        embedding_model:          LangChain embeddings instance (default: OpenAIEmbeddings).
        breakpoint_threshold_type: Strategy for detecting breakpoints.
        breakpoint_threshold_amount: Threshold value (percentile % or std multiplier).
        config:                   ChunkConfig (chunk_size/overlap ignored — semantic splitter
                                  determines boundaries automatically).

    Example:
        >>> chunker = SemanticTextChunker()
        >>> chunks = chunker.split(docs)
    """

    def __init__(
        self,
        embedding_model=None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        config: ChunkConfig | None = None,
    ):
        super().__init__(config)
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self._splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        return self._add_chunk_metadata(chunks)


class ManualSemanticChunker(BaseChunker):
    """
    Manual semantic chunker that does not require langchain_experimental.
    Splits sentences and groups them when cosine similarity drops below a threshold.

    Args:
        embedding_model: LangChain embeddings instance.
        threshold:       Cosine similarity threshold below which a new chunk starts (0–1).
        min_sentences:   Minimum sentences per chunk.

    Example:
        >>> chunker = ManualSemanticChunker(threshold=0.7)
        >>> chunks = chunker.split(docs)
    """

    def __init__(
        self,
        embedding_model=None,
        threshold: float = 0.75,
        min_sentences: int = 2,
        config: ChunkConfig | None = None,
    ):
        super().__init__(config)
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.threshold = threshold
        self.min_sentences = min_sentences

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s]

    def split(self, documents: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []

        for doc in documents:
            sentences = self._sentence_split(doc.page_content)
            if len(sentences) <= self.min_sentences:
                all_chunks.append(doc)
                continue

            embeddings = self.embedding_model.embed_documents(sentences)
            embeddings = [np.array(e) for e in embeddings]

            groups: list[list[str]] = [[sentences[0]]]
            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                if sim >= self.threshold or len(groups[-1]) < self.min_sentences:
                    groups[-1].append(sentences[i])
                else:
                    groups.append([sentences[i]])

            for group in groups:
                chunk = Document(
                    page_content=" ".join(group),
                    metadata=dict(doc.metadata),
                )
                all_chunks.append(chunk)

        return self._add_chunk_metadata(all_chunks)


if __name__ == "__main__":
    from langchain_core.documents import Document

    text = (
        "The transformer architecture uses self-attention mechanisms. "
        "Attention allows the model to weigh the importance of different tokens. "
        "Paris is the capital of France. The Eiffel Tower is located there. "
        "RAG combines retrieval with generation for better factual accuracy. "
        "ChromaDB is a popular open-source vector store."
    )
    docs = [Document(page_content=text, metadata={"source": "test", "doc_id": "abc"})]
    chunker = ManualSemanticChunker(threshold=0.6)
    chunks = chunker.split(docs)
    print(f"Semantic split → {len(chunks)} chunks")
    for c in chunks:
        print(f"  [{c.metadata['chunk_index']}] {c.page_content[:100]}")
