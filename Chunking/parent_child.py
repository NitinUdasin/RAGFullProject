"""
Parent-Child Chunking
----------------------
Indexes small child chunks for high-precision retrieval, but returns the
larger parent chunk to the LLM for richer context.

At retrieval time:
  1. Small child chunks are searched in the vector store.
  2. The parent chunk is fetched from a doc store using parent_id.
  3. The parent is passed to ContextAssembly instead of the child.

This gives the precision of small chunks with the context richness of large ones.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker, ChunkConfig


class ParentChildChunker(BaseChunker):
    """
    Produces two sets of chunks from the same documents:
      - parent_chunks: larger chunks stored in a doc store (not embedded)
      - child_chunks:  smaller chunks embedded in the vector store,
                       each with metadata['parent_id'] pointing to its parent

    Args:
        parent_config: ChunkConfig for the larger parent chunks.
        child_config:  ChunkConfig for the smaller child chunks.

    Example:
        >>> chunker = ParentChildChunker(
        ...     parent_config=ChunkConfig(chunk_size=1024, chunk_overlap=128),
        ...     child_config=ChunkConfig(chunk_size=256, chunk_overlap=32),
        ... )
        >>> parents, children = chunker.split_parent_child(docs)
        >>> # Embed children → vector store
        >>> # Store parents in a dict keyed by doc_id → docstore
    """

    def __init__(
        self,
        parent_config: ChunkConfig | None = None,
        child_config: ChunkConfig | None = None,
    ):
        super().__init__(parent_config or ChunkConfig(chunk_size=1024, chunk_overlap=128))
        self.child_config = child_config or ChunkConfig(chunk_size=256, chunk_overlap=32)

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_config.chunk_size,
            chunk_overlap=self.child_config.chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Returns only child chunks (for use as a BaseChunker)."""
        _, children = self.split_parent_child(documents)
        return children

    def split_parent_child(
        self, documents: list[Document]
    ) -> tuple[list[Document], list[Document]]:
        """
        Split documents into (parent_chunks, child_chunks).

        Each child chunk has:
          - metadata['parent_id']: doc_id of its parent chunk
          - metadata['chunk_index'], metadata['total_chunks']

        Returns:
            Tuple of (parent_chunks, child_chunks).
        """
        import hashlib

        parent_chunks: list[Document] = []
        child_chunks: list[Document] = []

        parents = self._parent_splitter.split_documents(documents)

        for i, parent in enumerate(parents):
            source = parent.metadata.get("source", "unknown")
            parent_id = hashlib.md5(f"{source}:parent:{i}".encode()).hexdigest()
            parent.metadata["doc_id"] = parent_id
            parent.metadata["chunk_index"] = i
            parent.metadata["chunk_type"] = "parent"
            parent_chunks.append(parent)

            children = self._child_splitter.split_documents([parent])
            for j, child in enumerate(children):
                child.metadata["parent_id"] = parent_id
                child.metadata["chunk_index"] = j
                child.metadata["total_chunks"] = len(children)
                child.metadata["chunk_type"] = "child"
                child_chunks.append(child)

        return parent_chunks, child_chunks

    @staticmethod
    def build_docstore(parent_chunks: list[Document]) -> dict[str, Document]:
        """
        Build an in-memory docstore keyed by doc_id.
        At retrieval time, fetch the parent using child.metadata['parent_id'].
        """
        return {doc.metadata["doc_id"]: doc for doc in parent_chunks}

    @staticmethod
    def fetch_parent(child: Document, docstore: dict[str, Document]) -> Document | None:
        """Retrieve parent document for a given child chunk."""
        parent_id = child.metadata.get("parent_id")
        return docstore.get(parent_id)


if __name__ == "__main__":
    from langchain_core.documents import Document

    docs = [Document(
        page_content="word " * 500,
        metadata={"source": "big_doc.txt", "doc_id": "orig"}
    )]
    chunker = ParentChildChunker()
    parents, children = chunker.split_parent_child(docs)
    print(f"Parents : {len(parents)}")
    print(f"Children: {len(children)}")
    print(f"First child parent_id: {children[0].metadata['parent_id']}")
    docstore = ParentChildChunker.build_docstore(parents)
    parent = ParentChildChunker.fetch_parent(children[0], docstore)
    print(f"Fetched parent length: {len(parent.page_content)} chars")
