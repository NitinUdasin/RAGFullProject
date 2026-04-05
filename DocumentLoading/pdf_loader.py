"""
PDF Loader
----------
Loads PDF files into Document objects using PyPDFLoader (page-level)
or PDFMinerLoader (layout-aware, better for multi-column PDFs).
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader

from .base_loader import BaseLoader, LoaderConfig


class PDFLoader(BaseLoader):
    """
    Loads a PDF file, one Document per page.

    Args:
        file_path: Path to the PDF file.
        use_miner: Use PDFMiner instead of PyPDF (better for complex layouts).
        config:    Optional LoaderConfig.

    Example:
        >>> loader = PDFLoader("docs/report.pdf")
        >>> docs = loader.load_and_validate()
    """

    def __init__(self, file_path: str, use_miner: bool = False, config: LoaderConfig | None = None):
        super().__init__(config)
        self.file_path = Path(file_path)
        self.use_miner = use_miner

    def load(self) -> list[Document]:
        loader_cls = PDFMinerLoader if self.use_miner else PyPDFLoader
        loader = loader_cls(str(self.file_path))
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata["source"] = str(self.file_path)
            doc.metadata["doc_id"] = self.make_doc_id(str(self.file_path), i)
            doc.metadata.setdefault("page", i)
            doc.metadata["file_type"] = "pdf"

        return docs


class PDFDirectoryLoader(BaseLoader):
    """
    Loads all PDFs from a directory recursively.

    Args:
        directory: Path to the directory.
        glob:      Glob pattern (default: **/*.pdf).

    Example:
        >>> loader = PDFDirectoryLoader("./documents")
        >>> all_docs = loader.load_and_validate()
    """

    def __init__(self, directory: str, glob: str = "**/*.pdf", config: LoaderConfig | None = None):
        super().__init__(config)
        self.directory = Path(directory)
        self.glob = glob

    def load(self) -> list[Document]:
        all_docs: list[Document] = []
        pdf_files = list(self.directory.glob(self.glob))

        for pdf_path in pdf_files:
            loader = PDFLoader(str(pdf_path), config=self.config)
            all_docs.extend(loader.load())

        return all_docs


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    loader = PDFLoader(path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {path}")
    for doc in docs[:2]:
        print(f"  Page {doc.metadata.get('page')}: {doc.page_content[:100]}...")
