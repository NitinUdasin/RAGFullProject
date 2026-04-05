"""
Query Rewriting
---------------
Reformulates a vague or short user query into a more explicit,
retrieval-friendly version using an LLM call.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ── Prompt ────────────────────────────────────────────────────────────────────

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a search query optimizer for a RAG system.
Rewrite the user query to be more specific, self-contained, and retrieval-friendly.
Fix grammar, expand abbreviations, and add implicit context where needed.
Return ONLY the rewritten query — no explanation.

Original query: {query}
Rewritten query:"""
)

# ── Chain ─────────────────────────────────────────────────────────────────────

def build_rewrite_chain(llm=None):
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return REWRITE_PROMPT | llm | StrOutputParser()


def rewrite_query(query: str, llm=None) -> str:
    """
    Rewrite a user query to improve retrieval quality.

    Args:
        query: Raw user input.
        llm:   Optional LangChain chat model. Defaults to gpt-4o-mini.

    Returns:
        Rewritten query string.

    Example:
        >>> rewrite_query("what about refunds?")
        "What is the refund policy for digital purchases?"
    """
    chain = build_rewrite_chain(llm)
    return chain.invoke({"query": query})


# ── LangGraph node ────────────────────────────────────────────────────────────

def rewrite_query_node(state: dict) -> dict:
    """LangGraph node: rewrites state['query'] and stores in state['translated_query']."""
    rewritten = rewrite_query(state["query"])
    return {**state, "translated_query": rewritten}


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "what about refunds?",
        "how does it work?",
        "pricing?",
        "tell me about the API limits",
    ]
    chain = build_rewrite_chain()
    for q in test_queries:
        rewritten = chain.invoke({"query": q})
        print(f"Original : {q}")
        print(f"Rewritten: {rewritten}\n")
