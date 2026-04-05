"""
Self-Query Retrieval
---------------------
Uses an LLM to extract structured metadata filters from the natural language query,
then passes both the semantic query and the filters to ChromaDB.

Enables queries like "PDFs about refunds from 2024" to use metadata filters
alongside vector search — more precise than vector search alone.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from VectorStore import ChromaStore
from Embedding import BaseEmbedder, get_embedder


# ── Filter schema ─────────────────────────────────────────────────────────────

class MetadataFilter(BaseModel):
    """Extracted metadata constraints from a natural language query."""
    semantic_query: str = Field(description="The semantic part of the query for vector search.")
    source_type: Optional[str] = Field(None, description="File type: pdf, html, txt, docx")
    year: Optional[int] = Field(None, description="Year of document publication")
    author: Optional[str] = Field(None, description="Document author name")
    topic: Optional[str] = Field(None, description="Document topic or category")


# ── Prompt ────────────────────────────────────────────────────────────────────

SELF_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract metadata filters from the user query for a document search system.
The metadata fields available are: source_type, year, author, topic.
Only populate fields that are explicitly or strongly implied in the query.
Also extract the core semantic query (without the filter constraints).
Return JSON only.""",
    ),
    ("human", "{query}"),
])


# ── Retriever ─────────────────────────────────────────────────────────────────

class SelfQueryRetriever:
    """
    Extracts filters from queries and runs filtered vector search.

    Args:
        store:    ChromaStore instance.
        embedder: BaseEmbedder for query embedding.
        llm:      Optional LangChain chat model.
        k:        Number of results.

    Example:
        >>> retriever = SelfQueryRetriever(store, embedder)
        >>> docs = retriever.retrieve("PDF reports about climate from 2023")
    """

    def __init__(
        self,
        store: ChromaStore,
        embedder: BaseEmbedder | None = None,
        llm=None,
        k: int = 5,
    ):
        self.store = store
        self.embedder = embedder or get_embedder()
        self.k = k
        llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.chain = SELF_QUERY_PROMPT | llm.with_structured_output(MetadataFilter)

    def extract_filters(self, query: str) -> MetadataFilter:
        return self.chain.invoke({"query": query})

    @staticmethod
    def build_where_clause(filters: MetadataFilter) -> dict | None:
        """Convert a MetadataFilter into a ChromaDB where clause."""
        conditions: list[dict[str, Any]] = []

        if filters.year:
            conditions.append({"year": {"$eq": filters.year}})
        if filters.source_type:
            conditions.append({"source_type": {"$eq": filters.source_type}})
        if filters.author:
            conditions.append({"author": {"$eq": filters.author}})
        if filters.topic:
            conditions.append({"topic": {"$eq": filters.topic}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve(self, query: str) -> list[Document]:
        """Extract filters, embed semantic query, and search with filters."""
        filters = self.extract_filters(query)
        where = self.build_where_clause(filters)
        query_embedding = self.embedder.embed_query(filters.semantic_query)
        return self.store.search(query_embedding, n_results=self.k, where=where)


# ── LangGraph node ─────────────────────────────────────────────────────────────

def self_query_node(state: dict, retriever: SelfQueryRetriever) -> dict:
    query = state.get("translated_query", state["query"])
    docs = retriever.retrieve(query)
    return {**state, "documents": docs}


if __name__ == "__main__":
    queries = [
        "PDF reports about climate change from 2023",
        "documents written by John Smith about machine learning",
        "what is the refund policy?",
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = SELF_QUERY_PROMPT | llm.with_structured_output(MetadataFilter)

    for q in queries:
        result = chain.invoke({"query": q})
        where = SelfQueryRetriever.build_where_clause(result)
        print(f"Query  : {q}")
        print(f"Semantic: {result.semantic_query}")
        print(f"Filters : {where}\n")
