"""
Indexer
-------
Orchestrates the full ingestion pipeline:
  DocumentLoading → Chunking → Embedding → VectorStore upsert

Supports flat, parent-child, and multi-vector indexing strategies.
"""

import os
from enum import Enum
from langchain_core.documents import Document

from DocumentLoading import BaseLoader
from Chunking import RecursiveChunker, ParentChildChunker, ChunkConfig
from Embedding import get_embedder, BaseEmbedder
from VectorStore import ChromaStore


class IndexStrategy(str, Enum):
    FLAT = "flat"
    PARENT_CHILD = "parent_child"
    MULTI_VECTOR = "multi_vector"


class Indexer:
    """
    End-to-end document ingestion pipeline.

    Args:
        collection_name: ChromaDB collection name.
        embedder:        BaseEmbedder instance (default: from env vars).
        strategy:        IndexStrategy (default: FLAT).
        chunk_config:    ChunkConfig for the primary chunker.

    Example:
        >>> indexer = Indexer("rag_docs")
        >>> indexer.ingest_from_loader(PDFLoader("report.pdf"))
        >>> indexer.ingest_documents(my_docs)
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embedder: BaseEmbedder | None = None,
        strategy: IndexStrategy = IndexStrategy.FLAT,
        chunk_config: ChunkConfig | None = None,
    ):
        self.embedder = embedder or get_embedder()
        self.strategy = strategy
        self.chunk_config = chunk_config or ChunkConfig(chunk_size=512, chunk_overlap=64)
        collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", self.embedder.collection_name("rag_docs")
        )
        self.store = ChromaStore(
            collection_name=collection_name,
            embedding_fn=self.embedder.embed_documents,
        )
        self.docstore: dict[str, Document] = {}  # for parent-child strategy

    # ── Ingestion entry points ────────────────────────────────────────────────

    def ingest_from_loader(self, loader: BaseLoader) -> int:
        """Load documents from a loader and ingest them. Returns chunk count."""
        docs = loader.load_and_validate()
        return self.ingest_documents(docs)

    def ingest_documents(self, documents: list[Document]) -> int:
        """Chunk, embed, and upsert documents. Returns number of chunks ingested."""
        if self.strategy == IndexStrategy.FLAT:
            return self._ingest_flat(documents)
        elif self.strategy == IndexStrategy.PARENT_CHILD:
            return self._ingest_parent_child(documents)
        elif self.strategy == IndexStrategy.MULTI_VECTOR:
            return self._ingest_multi_vector(documents)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    # ── Flat indexing ─────────────────────────────────────────────────────────

    def _ingest_flat(self, documents: list[Document]) -> int:
        chunker = RecursiveChunker(self.chunk_config)
        chunks = chunker.split(documents)
        self.store.upsert(chunks)
        return len(chunks)

    # ── Parent-child indexing ─────────────────────────────────────────────────

    def _ingest_parent_child(self, documents: list[Document]) -> int:
        child_config = ChunkConfig(
            chunk_size=self.chunk_config.chunk_size // 2,
            chunk_overlap=self.chunk_config.chunk_overlap // 2,
        )
        chunker = ParentChildChunker(
            parent_config=self.chunk_config,
            child_config=child_config,
        )
        parents, children = chunker.split_parent_child(documents)
        self.docstore.update(ParentChildChunker.build_docstore(parents))
        self.store.upsert(children)
        return len(children)

    # ── Multi-vector indexing ─────────────────────────────────────────────────

    def _ingest_multi_vector(self, documents: list[Document]) -> int:
        """
        For each chunk, index:
          1. The chunk itself
          2. An LLM-generated summary
        Both point back to the original chunk via parent_id.
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import hashlib

        chunker = RecursiveChunker(self.chunk_config)
        chunks = chunker.split(documents)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        summary_chain = (
            ChatPromptTemplate.from_template("Summarize this in 1-2 sentences:\n\n{text}")
            | llm
            | StrOutputParser()
        )

        all_docs: list[Document] = []
        for chunk in chunks:
            all_docs.append(chunk)
            summary_text = summary_chain.invoke({"text": chunk.page_content})
            summary_doc = Document(
                page_content=summary_text,
                metadata={
                    **chunk.metadata,
                    "doc_id": hashlib.md5(f"summary:{chunk.metadata['doc_id']}".encode()).hexdigest(),
                    "doc_type": "summary",
                    "parent_id": chunk.metadata["doc_id"],
                },
            )
            all_docs.append(summary_doc)

        self.store.upsert(all_docs)
        return len(all_docs)

    # ── Retriever factory ─────────────────────────────────────────────────────

    def as_retriever(self, k: int = 5, where: dict | None = None):
        """Return a LangChain-compatible retriever backed by this index."""
        return self.store.as_langchain_retriever(k=k, where=where)

    def collection_count(self) -> int:
        return self.store.count()
