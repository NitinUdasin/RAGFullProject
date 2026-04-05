"""
Base Chunker
------------
Abstract base class for all chunking strategies.
Handles chunk metadata injection (chunk_index, total_chunks, parent_id).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64


class BaseChunker(ABC):
    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        ...

    @staticmethod
    def _add_chunk_metadata(chunks: list[Document]) -> list[Document]:
        """
        Group chunks by parent doc_id and inject chunk_index / total_chunks.
        Preserves all existing metadata from the parent document.
        """
        from collections import defaultdict
        groups: dict[str, list[Document]] = defaultdict(list)

        for chunk in chunks:
            parent_id = chunk.metadata.get("doc_id", "unknown")
            groups[parent_id].append(chunk)

        result: list[Document] = []
        for parent_id, group in groups.items():
            total = len(group)
            for i, chunk in enumerate(group):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = total
                chunk.metadata["parent_id"] = parent_id
                result.append(chunk)

        return result
