"""
Text & Markdown Loader
-----------------------
Loads plain text (.txt) and Markdown (.md) files into Documents.
Also supports loading from a raw string (useful in tests and pipelines).
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader as LCTextLoader

from .base_loader import BaseLoader, LoaderConfig


class PlainTextLoader(BaseLoader):
    """
    Load a single .txt or .md file as one Document.

    Args:
        file_path: Path to the text file.
        config:    Optional LoaderConfig (encoding respected here).

    Example:
        >>> loader = PlainTextLoader("notes/overview.md")
        >>> docs = loader.load_and_validate()
    """

    def __init__(self, file_path: str, config: LoaderConfig | None = None):
        super().__init__(config)
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        loader = LCTextLoader(str(self.file_path), encoding=self.config.encoding)
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata["source"] = str(self.file_path)
            doc.metadata["doc_id"] = self.make_doc_id(str(self.file_path), i)
            doc.metadata["file_type"] = self.file_path.suffix.lstrip(".")

        return docs


class TextDirectoryLoader(BaseLoader):
    """
    Load all .txt and .md files from a directory.

    Args:
        directory: Root directory path.
        glob:      File glob pattern (default: **/*.{txt,md}).

    Example:
        >>> loader = TextDirectoryLoader("./knowledge_base")
        >>> docs = loader.load_and_validate()
    """

    PATTERNS = ["**/*.txt", "**/*.md"]

    def __init__(self, directory: str, config: LoaderConfig | None = None):
        super().__init__(config)
        self.directory = Path(directory)

    def load(self) -> list[Document]:
        all_docs: list[Document] = []
        for pattern in self.PATTERNS:
            for file_path in self.directory.glob(pattern):
                loader = PlainTextLoader(str(file_path), config=self.config)
                all_docs.extend(loader.load())
        return all_docs


class StringLoader(BaseLoader):
    """
    Create a Document from a raw string. Useful for testing and inline content.

    Args:
        text:     Raw text content.
        source:   Identifier string for metadata['source'].

    Example:
        >>> loader = StringLoader("LangGraph is a library for building stateful agents.", source="manual")
        >>> docs = loader.load_and_validate()
    """

    def __init__(self, text: str, source: str = "inline", config: LoaderConfig | None = None):
        super().__init__(config)
        self.text = text
        self.source = source

    def load(self) -> list[Document]:
        doc = Document(
            page_content=self.text,
            metadata={
                "source": self.source,
                "doc_id": self.make_doc_id(self.source, 0),
                "file_type": "text",
            },
        )
        return [doc]


if __name__ == "__main__":
    loader = StringLoader("This is a test document about RAG pipelines.", source="test")
    docs = loader.load_and_validate()
    print(docs[0])
