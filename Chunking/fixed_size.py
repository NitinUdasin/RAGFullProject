"""
Fixed-Size Chunking
--------------------
Splits documents into chunks of a fixed character or token count.
Simple and fast; does not respect sentence or paragraph boundaries.
Use RecursiveChunker for structure-aware splitting.
"""

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from .base_chunker import BaseChunker, ChunkConfig


class FixedSizeChunker(BaseChunker):
    """
    Splits documents by character count with optional overlap.

    Args:
        config: ChunkConfig with chunk_size (chars) and chunk_overlap.

    Example:
        >>> chunker = FixedSizeChunker(ChunkConfig(chunk_size=500, chunk_overlap=50))
        >>> chunks = chunker.split(docs)
    """

    def __init__(self, config: ChunkConfig | None = None):
        super().__init__(config)
        self._splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separator="\n\n",
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        return self._add_chunk_metadata(chunks)


class TokenFixedSizeChunker(BaseChunker):
    """
    Splits documents by token count (tiktoken) rather than character count.
    Preferred when targeting models with token-based context windows.

    Args:
        config:    ChunkConfig with chunk_size (tokens) and chunk_overlap.
        model:     tiktoken model name for token counting (default: gpt-4o).

    Example:
        >>> chunker = TokenFixedSizeChunker(ChunkConfig(chunk_size=256, chunk_overlap=32))
        >>> chunks = chunker.split(docs)
    """

    def __init__(self, config: ChunkConfig | None = None, model: str = "gpt-4o"):
        super().__init__(config)
        enc = tiktoken.encoding_for_model(model)

        self._splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=enc.name,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        return self._add_chunk_metadata(chunks)


if __name__ == "__main__":
    from langchain_core.documents import Document

    sample = [Document(page_content="word " * 300, metadata={"source": "test", "doc_id": "abc"})]
    chunker = FixedSizeChunker()
    chunks = chunker.split(sample)
    print(f"Split into {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  chunk {c.metadata['chunk_index']}: {len(c.page_content)} chars")
