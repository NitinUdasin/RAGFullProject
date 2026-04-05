"""
Multi-Query Generation
----------------------
Generates N alternative phrasings of the user query, retrieves documents
for each, then deduplicates and returns the merged result set.

Higher recall than single-query retrieval at the cost of N vector searches.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ── Prompt ────────────────────────────────────────────────────────────────────

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are a RAG query generator. Given a user question, produce {n} different
versions of that question that capture the same information need from different angles.
Vary vocabulary, structure, and specificity.

Return a JSON array of strings and nothing else.

Question: {query}
Variants:"""
)

# ── Query generation ──────────────────────────────────────────────────────────

def generate_query_variants(query: str, n: int = 3, llm=None) -> list[str]:
    """
    Generate N alternative phrasings for a query.

    Args:
        query: Original user question.
        n:     Number of variants to generate.
        llm:   Optional LangChain chat model.

    Returns:
        List of query strings (including the original).
    """
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "n": n})

    try:
        variants = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: split by newline if JSON parse fails
        variants = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]

    # Always include the original query
    all_queries = [query] + [v for v in variants if v != query]
    return all_queries[:n + 1]


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_documents(docs: list[Document]) -> list[Document]:
    """Remove duplicate documents by page_content hash."""
    seen: set[int] = set()
    unique: list[Document] = []
    for doc in docs:
        h = hash(doc.page_content.strip())
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


# ── Main retrieval function ───────────────────────────────────────────────────

def multi_query_retrieve(
    query: str,
    retriever,
    n: int = 3,
    llm=None,
) -> list[Document]:
    """
    Generate multiple query variants, retrieve for each, and return deduplicated results.

    Args:
        query:     Original user question.
        retriever: Any LangChain retriever (e.g. vectorstore.as_retriever()).
        n:         Number of query variants.
        llm:       Optional LangChain chat model for variant generation.

    Returns:
        Deduplicated list of retrieved Document objects.

    Example:
        >>> from langchain_chroma import Chroma
        >>> vectorstore = Chroma(...)
        >>> docs = multi_query_retrieve("How do I reset my password?", vectorstore.as_retriever())
    """
    variants = generate_query_variants(query, n=n, llm=llm)
    all_docs: list[Document] = []

    for variant in variants:
        results = retriever.invoke(variant)
        all_docs.extend(results)

    return deduplicate_documents(all_docs)


# ── LangGraph node ────────────────────────────────────────────────────────────

def multi_query_node(state: dict, retriever, n: int = 3, llm=None) -> dict:
    """LangGraph node: populates state['documents'] using multi-query retrieval."""
    docs = multi_query_retrieve(state.get("translated_query", state["query"]), retriever, n, llm)
    return {**state, "documents": docs}


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = "How does vector search work in RAG systems?"
    variants = generate_query_variants(query, n=3)
    print("Query variants:")
    for i, v in enumerate(variants):
        print(f"  {i+1}. {v}")
