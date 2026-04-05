"""
Web Loader
----------
Loads content from URLs using WebBaseLoader (single or list of URLs)
or RecursiveUrlLoader (crawls a site up to a depth limit).
"""

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup

from .base_loader import BaseLoader, LoaderConfig


class SingleWebLoader(BaseLoader):
    """
    Load one or more URLs into Documents.

    Args:
        urls:   A single URL string or list of URL strings.
        config: Optional LoaderConfig.

    Example:
        >>> loader = SingleWebLoader("https://example.com/docs/intro")
        >>> docs = loader.load_and_validate()
    """

    def __init__(self, urls: str | list[str], config: LoaderConfig | None = None):
        super().__init__(config)
        self.urls = [urls] if isinstance(urls, str) else urls

    def load(self) -> list[Document]:
        loader = WebBaseLoader(self.urls)
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata.setdefault("source", self.urls[i] if i < len(self.urls) else "web")
            doc.metadata["doc_id"] = self.make_doc_id(doc.metadata["source"], 0)
            doc.metadata["file_type"] = "html"

        return docs


class RecursiveWebLoader(BaseLoader):
    """
    Recursively crawl a website up to a given depth.

    Args:
        url:       Root URL to start crawling.
        max_depth: How many levels deep to follow links (default 2).
        config:    Optional LoaderConfig.

    Example:
        >>> loader = RecursiveWebLoader("https://docs.example.com", max_depth=2)
        >>> docs = loader.load_and_validate()
    """

    def __init__(self, url: str, max_depth: int = 2, config: LoaderConfig | None = None):
        super().__init__(config)
        self.url = url
        self.max_depth = max_depth

    @staticmethod
    def _extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def load(self) -> list[Document]:
        loader = RecursiveUrlLoader(
            url=self.url,
            max_depth=self.max_depth,
            extractor=self._extract_text,
        )
        docs = loader.load()

        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", self.url)
            doc.metadata["source"] = source
            doc.metadata["doc_id"] = self.make_doc_id(source, i)
            doc.metadata["file_type"] = "html"

        return docs


if __name__ == "__main__":
    loader = SingleWebLoader("https://python.langchain.com/docs/introduction/")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s)")
    print(f"Source : {docs[0].metadata['source']}")
    print(f"Content: {docs[0].page_content[:200]}...")
