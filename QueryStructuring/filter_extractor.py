"""
Query Structuring — Metadata Filter Extraction
-----------------------------------------------
Parses natural language queries into structured ChromaDB `where` clauses
using LLM structured output. Works in tandem with Retrieval/self_query.py.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ── Filter schemas ────────────────────────────────────────────────────────────

class DocumentFilter(BaseModel):
    """Structured metadata filter extracted from a natural language query."""
    semantic_query: str = Field(description="Core semantic query for vector search (stripped of filter constraints).")
    source_type: Optional[str] = Field(None, description="File type: pdf, html, txt, docx, csv")
    year: Optional[int] = Field(None, description="Publication year (4-digit integer)")
    author: Optional[str] = Field(None, description="Document author or creator")
    topic: Optional[str] = Field(None, description="Topic or category tag")
    language: Optional[str] = Field(None, description="Document language (e.g. 'en', 'fr')")


class ChromaWhereClause(BaseModel):
    """The resulting ChromaDB where clause dict and the cleaned semantic query."""
    semantic_query: str
    where: Optional[dict] = None


# ── Prompt ────────────────────────────────────────────────────────────────────

FILTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a metadata filter extractor for a document retrieval system.
Given a user query, extract any explicit metadata constraints into structured fields.
Available metadata fields: source_type, year, author, topic, language.
Only populate fields that are clearly stated or strongly implied.
Also extract the semantic_query — the core question without filter constraints.
Return JSON only.""",
    ),
    ("human", "{query}"),
])


# ── Extractor ─────────────────────────────────────────────────────────────────

class FilterExtractor:
    """
    Extracts ChromaDB-compatible metadata filters from natural language queries.

    Args:
        llm: Optional LangChain chat model.

    Example:
        >>> extractor = FilterExtractor()
        >>> result = extractor.extract("Find PDFs about climate change from 2023")
        >>> result.semantic_query   # "climate change"
        >>> result.where            # {"$and": [{"source_type": {"$eq": "pdf"}}, {"year": {"$eq": 2023}}]}
    """

    def __init__(self, llm=None):
        llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.chain = FILTER_PROMPT | llm.with_structured_output(DocumentFilter)

    def extract(self, query: str) -> ChromaWhereClause:
        """Extract filters and return a ChromaWhereClause with semantic_query and where dict."""
        filters = self.chain.invoke({"query": query})
        where = self._build_where(filters)
        return ChromaWhereClause(semantic_query=filters.semantic_query, where=where)

    @staticmethod
    def _build_where(filters: DocumentFilter) -> dict[str, Any] | None:
        conditions: list[dict] = []

        if filters.year is not None:
            conditions.append({"year": {"$eq": filters.year}})
        if filters.source_type:
            conditions.append({"source_type": {"$eq": filters.source_type.lower()}})
        if filters.author:
            conditions.append({"author": {"$eq": filters.author}})
        if filters.topic:
            conditions.append({"topic": {"$eq": filters.topic}})
        if filters.language:
            conditions.append({"language": {"$eq": filters.language}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ── ChromaDB filter operators reference ──────────────────────────────────────

def build_range_filter(field: str, gte: Any = None, lte: Any = None) -> dict:
    """Helper for building range filters (e.g., year between 2020 and 2024)."""
    clause: dict = {}
    if gte is not None:
        clause["$gte"] = gte
    if lte is not None:
        clause["$lte"] = lte
    return {field: clause}


def build_in_filter(field: str, values: list) -> dict:
    """Helper for building $in filters (e.g., topic in ['rag', 'llm'])."""
    return {field: {"$in": values}}


# ── LangGraph node ─────────────────────────────────────────────────────────────

def filter_extraction_node(state: dict, extractor: FilterExtractor | None = None) -> dict:
    """
    LangGraph node: extracts metadata filters from the query.
    Sets state['metadata_filters'] and state['translated_query'] (semantic part).
    """
    extractor = extractor or FilterExtractor()
    result = extractor.extract(state.get("translated_query", state["query"]))
    return {
        **state,
        "translated_query": result.semantic_query,
        "metadata_filters": result.where or {},
    }


if __name__ == "__main__":
    extractor = FilterExtractor()
    queries = [
        "Find PDF reports about climate change from 2023",
        "Show me documents written by Jane Doe about machine learning",
        "What is the refund policy?",
        "French language documents about cooking from 2022",
    ]
    for q in queries:
        result = extractor.extract(q)
        print(f"Query   : {q}")
        print(f"Semantic: {result.semantic_query}")
        print(f"Where   : {result.where}\n")
