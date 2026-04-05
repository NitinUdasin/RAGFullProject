"""
LLM-based Reranking
--------------------
Asks the LLM to score or sort a small set of candidate documents by relevance.
Accurate but expensive — suitable only when the candidate set is small (< 10 docs).
"""

import json
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# ── Prompts ───────────────────────────────────────────────────────────────────

SCORE_PROMPT = ChatPromptTemplate.from_template(
    """You are a relevance judge. Score each document 0–10 for how well it answers the query.
Return a JSON array of objects with "index" (0-based) and "score" fields. Nothing else.

Query: {query}

Documents:
{documents}

Scores:"""
)

SORT_PROMPT = ChatPromptTemplate.from_template(
    """You are a relevance judge. Return the indices of the documents sorted from most
to least relevant to the query. Return a JSON array of integers (0-based). Nothing else.

Query: {query}

Documents:
{documents}

Sorted indices:"""
)


# ── Reranker ──────────────────────────────────────────────────────────────────

class LLMReranker:
    """
    Reranks a small set of documents using LLM scoring.

    Args:
        llm:    Optional LangChain chat model.
        top_n:  Number of top results to return.
        mode:   "score" (returns 0–10 scores) or "sort" (returns sorted indices).

    Example:
        >>> reranker = LLMReranker(top_n=3)
        >>> reranked = reranker.rerank("what is embeddings?", candidate_docs)
    """

    def __init__(self, llm=None, top_n: int = 5, mode: str = "score"):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.top_n = top_n
        self.mode = mode

    @staticmethod
    def _format_documents(documents: list[Document]) -> str:
        return "\n\n".join(
            f"[{i}] {doc.page_content[:500]}" for i, doc in enumerate(documents)
        )

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        formatted = self._format_documents(documents)

        if self.mode == "sort":
            return self._rerank_sort(query, formatted, documents)
        return self._rerank_score(query, formatted, documents)

    def _rerank_score(self, query: str, formatted: str, documents: list[Document]) -> list[Document]:
        chain = SCORE_PROMPT | self.llm | StrOutputParser()
        raw = chain.invoke({"query": query, "documents": formatted})

        try:
            scored = json.loads(raw)
        except json.JSONDecodeError:
            return documents[: self.top_n]

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = []
        for item in scored[: self.top_n]:
            idx = item["index"]
            if 0 <= idx < len(documents):
                documents[idx].metadata["_rerank_score"] = item["score"]
                results.append(documents[idx])
        return results

    def _rerank_sort(self, query: str, formatted: str, documents: list[Document]) -> list[Document]:
        chain = SORT_PROMPT | self.llm | StrOutputParser()
        raw = chain.invoke({"query": query, "documents": formatted})

        try:
            indices = json.loads(raw)
        except json.JSONDecodeError:
            return documents[: self.top_n]

        results = []
        for idx in indices[: self.top_n]:
            if 0 <= idx < len(documents):
                results.append(documents[idx])
        return results


# ── LangGraph node ─────────────────────────────────────────────────────────────

def llm_rerank_node(state: dict, reranker: LLMReranker) -> dict:
    """LangGraph node: reranks state['documents'] using LLM scoring."""
    query = state.get("translated_query", state["query"])
    reranked = reranker.rerank(query, state.get("documents", []))
    return {**state, "documents": reranked}


if __name__ == "__main__":
    from langchain_core.documents import Document

    docs = [
        Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"doc_id": "1"}),
        Document(page_content="The Eiffel Tower was built in 1889.", metadata={"doc_id": "2"}),
        Document(page_content="LangGraph is used to build stateful RAG pipelines.", metadata={"doc_id": "3"}),
    ]
    reranker = LLMReranker(top_n=2)
    results = reranker.rerank("Tell me about RAG pipelines", docs)
    for r in results:
        print(f"[score={r.metadata.get('_rerank_score', '?')}] {r.page_content}")
