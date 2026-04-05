"""
Sparse Retrieval (BM25)
-----------------------
Lexical keyword-based retrieval using BM25 (Best Match 25).
Catches exact terms, product codes, and named entities that dense search may miss.
Requires: pip install rank-bm25
"""

import math
from collections import defaultdict
from langchain_core.documents import Document


class BM25Retriever:
    """
    In-memory BM25 retriever over a document corpus.

    BM25 parameters:
        k1 (1.5): Controls term frequency saturation.
        b  (0.75): Controls document length normalization.

    Args:
        documents: Corpus of Document objects to index.
        k1:        BM25 k1 parameter.
        b:         BM25 b parameter.
        k:         Number of results to return.

    Example:
        >>> retriever = BM25Retriever(docs, k=5)
        >>> results = retriever.retrieve("refund policy 2024")
    """

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
        k: int = 5,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.k = k
        self._build_index(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\b\w+\b", text.lower())

    def _build_index(self, documents: list[Document]) -> None:
        self._corpus = [self._tokenize(doc.page_content) for doc in documents]
        n = len(self._corpus)

        # Term → document frequency
        df: dict[str, int] = defaultdict(int)
        for tokens in self._corpus:
            for term in set(tokens):
                df[term] += 1

        self._idf: dict[str, float] = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }
        avg_dl = sum(len(t) for t in self._corpus) / max(n, 1)
        self._avg_dl = avg_dl

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        tf: dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf[t] += 1

        dl = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in self._idf:
                continue
            idf = self._idf[term]
            f = tf[term]
            score += idf * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            )
        return score

    def retrieve(self, query: str) -> list[Document]:
        """Return top-k documents ranked by BM25 score."""
        query_tokens = self._tokenize(query)
        scores = [
            (i, self._score(query_tokens, doc_tokens))
            for i, doc_tokens in enumerate(self._corpus)
        ]
        top = sorted(scores, key=lambda x: x[1], reverse=True)[: self.k]
        results = []
        for i, score in top:
            doc = self.documents[i]
            doc.metadata["_bm25_score"] = score
            results.append(doc)
        return results

    @classmethod
    def from_langchain_retriever(cls, documents: list[Document], k: int = 5):
        """Convenience constructor that also returns a LangChain-compatible retriever."""
        instance = cls(documents, k=k)
        return instance.as_langchain_retriever()

    def as_langchain_retriever(self):
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun

        bm25 = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                return bm25.retrieve(query)

        return _Retriever()


if __name__ == "__main__":
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Refund policy: customers can request refunds within 30 days.", metadata={"source": "policy", "doc_id": "1"}),
        Document(page_content="API rate limits are 60 requests per minute.", metadata={"source": "api_docs", "doc_id": "2"}),
        Document(page_content="Subscription plans include monthly and annual options.", metadata={"source": "pricing", "doc_id": "3"}),
    ]
    retriever = BM25Retriever(docs, k=2)
    results = retriever.retrieve("refund 30 days")
    for r in results:
        print(f"[{r.metadata['_bm25_score']:.3f}] {r.page_content}")
