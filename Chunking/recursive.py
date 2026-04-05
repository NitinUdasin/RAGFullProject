"""
Recursive Character Text Splitter
-----------------------------------
General-purpose splitter that tries to keep paragraphs → sentences → words
together. Falls back to smaller separators only when needed.
This is the recommended default chunker for most document types.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from .base_chunker import BaseChunker, ChunkConfig


class RecursiveChunker(BaseChunker):
    """
    Splits text by trying a priority list of separators:
    ["\\n\\n", "\\n", " ", ""] — preserving semantic boundaries where possible.

    Args:
        config: ChunkConfig (chunk_size in chars, chunk_overlap).

    Example:
        >>> chunker = RecursiveChunker(ChunkConfig(chunk_size=512, chunk_overlap=64))
        >>> chunks = chunker.split(docs)
    """

    def __init__(self, config: ChunkConfig | None = None):
        super().__init__(config)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        return self._add_chunk_metadata(chunks)


class MarkdownChunker(BaseChunker):
    """
    Structure-aware splitter for Markdown documents.
    Splits on headers (##, ###) before falling back to paragraph breaks.

    Example:
        >>> chunker = MarkdownChunker()
        >>> chunks = chunker.split(md_docs)
    """

    def __init__(self, config: ChunkConfig | None = None):
        super().__init__(config)
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        return self._add_chunk_metadata(chunks)


class CodeChunker(BaseChunker):
    """
    Language-aware splitter for source code files.
    Splits on function/class boundaries before falling back to line breaks.

    Args:
        language: One of langchain_text_splitters.Language enum values.
                  e.g. Language.PYTHON, Language.JS, Language.GO

    Example:
        >>> from langchain_text_splitters import Language
        >>> chunker = CodeChunker(language=Language.PYTHON)
        >>> chunks = chunker.split(code_docs)
    """

    def __init__(self, language: Language = Language.PYTHON, config: ChunkConfig | None = None):
        super().__init__(config)
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        return self._add_chunk_metadata(chunks)


if __name__ == "__main__":
    from langchain_core.documents import Document

    md_text = """# Introduction\n\nThis is the intro.\n\n## Section 1\n\nContent here.\n\n## Section 2\n\nMore content."""
    docs = [Document(page_content=md_text, metadata={"source": "test.md", "doc_id": "abc"})]

    chunker = MarkdownChunker(ChunkConfig(chunk_size=100, chunk_overlap=10))
    chunks = chunker.split(docs)
    print(f"Split into {len(chunks)} markdown chunks")
    for c in chunks:
        print(f"  [{c.metadata['chunk_index']}] {c.page_content[:80]!r}")
