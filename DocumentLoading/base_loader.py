"""
Base Loader
-----------
Abstract base class that all document loaders must implement.
Ensures a consistent Document output schema across all source types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from langchain_core.documents import Document


@dataclass
class LoaderConfig:
    """Common configuration shared across all loaders."""
    encoding: str = "utf-8"
    metadata_extras: dict = field(default_factory=dict)


class BaseLoader(ABC):
    """
    Abstract base for all document loaders.
    Subclasses must implement `load()` and return a list of Document objects
    with at minimum 'source' and 'doc_id' populated in metadata.
    """

    def __init__(self, config: LoaderConfig | None = None):
        self.config = config or LoaderConfig()

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return a list of Documents."""
        ...

    def load_and_validate(self) -> list[Document]:
        """Load documents and enforce metadata schema."""
        docs = self.load()
        for doc in docs:
            self._validate(doc)
            doc.metadata.update(self.config.metadata_extras)
        return docs

    @staticmethod
    def _validate(doc: Document) -> None:
        if not doc.page_content or not doc.page_content.strip():
            raise ValueError(f"Empty page_content in document: {doc.metadata}")
        if "source" not in doc.metadata:
            raise ValueError(f"Missing 'source' in metadata: {doc.metadata}")

    @staticmethod
    def make_doc_id(source: str, index: int) -> str:
        """Stable document ID for deduplication in ChromaDB."""
        import hashlib
        return hashlib.md5(f"{source}:{index}".encode()).hexdigest()
