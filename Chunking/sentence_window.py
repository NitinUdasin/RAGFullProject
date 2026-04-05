"""
Sentence Window Chunking
-------------------------
Indexes individual sentences (small, precise units for retrieval), but stores
a window of surrounding sentences in metadata for richer context at generation time.

At retrieval time:
  1. Single sentence is matched via vector search.
  2. The surrounding window (e.g., 2 sentences before and after) is read from
     metadata['window'] and passed to ContextAssembly instead of the sentence alone.
"""

import re
from langchain_core.documents import Document

from .base_chunker import BaseChunker, ChunkConfig


class SentenceWindowChunker(BaseChunker):
    """
    Splits text into individual sentences. Each chunk stores:
      - page_content: the single sentence (used for embedding/retrieval)
      - metadata['window']: surrounding context (used for generation)
      - metadata['sentence_index']: position in original document

    Args:
        window_size: Number of sentences to include on each side of the target sentence.
        config:      ChunkConfig (chunk_size/overlap unused — sentence boundaries determine splits).

    Example:
        >>> chunker = SentenceWindowChunker(window_size=2)
        >>> chunks = chunker.split(docs)
        >>> # At retrieval: use chunk.metadata['window'] as context instead of page_content
    """

    def __init__(self, window_size: int = 2, config: ChunkConfig | None = None):
        super().__init__(config)
        self.window_size = window_size

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def split(self, documents: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []

        for doc in documents:
            sentences = self._split_sentences(doc.page_content)
            n = len(sentences)

            for i, sentence in enumerate(sentences):
                start = max(0, i - self.window_size)
                end = min(n, i + self.window_size + 1)
                window = " ".join(sentences[start:end])

                chunk = Document(
                    page_content=sentence,
                    metadata={
                        **doc.metadata,
                        "window": window,
                        "sentence_index": i,
                        "total_sentences": n,
                        "window_size": self.window_size,
                        "chunk_index": i,
                        "total_chunks": n,
                        "parent_id": doc.metadata.get("doc_id", "unknown"),
                    },
                )
                all_chunks.append(chunk)

        return all_chunks

    @staticmethod
    def get_context(chunk: Document) -> str:
        """
        Return the window context for generation instead of the raw sentence.
        Use this in ContextAssembly when the retriever returns sentence-window chunks.
        """
        return chunk.metadata.get("window", chunk.page_content)


if __name__ == "__main__":
    from langchain_core.documents import Document

    text = (
        "RAG stands for Retrieval-Augmented Generation. "
        "It combines a retrieval system with a language model. "
        "The retriever fetches relevant documents from a vector store. "
        "The generator then uses those documents to produce an answer. "
        "This reduces hallucination compared to using the LLM alone."
    )
    docs = [Document(page_content=text, metadata={"source": "intro.txt", "doc_id": "abc"})]
    chunker = SentenceWindowChunker(window_size=1)
    chunks = chunker.split(docs)

    print(f"Total sentence chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"Sentence : {c.page_content}")
        print(f"Window   : {c.metadata['window']}\n")
