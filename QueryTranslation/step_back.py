"""
Step-Back Prompting
-------------------
Generates a broader, more abstract version of the original query to retrieve
background/conceptual context, then retrieves for both the original and
step-back queries and merges the results.

Reference: Zheng et al., "Take a Step Back: Evoking Reasoning via Abstraction
in Large Language Models" (2023).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .multi_query import deduplicate_documents


# ── Prompt ────────────────────────────────────────────────────────────────────

STEP_BACK_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at broadening specific questions into general ones.
Given a specific question, generate a more abstract, high-level question that
captures the underlying concept or principle. This broader question will help
retrieve useful background knowledge.

Return ONLY the step-back question — no explanation.

Specific question: {query}
Step-back question:"""
)

# ── Step-back query generation ────────────────────────────────────────────────

def generate_step_back_query(query: str, llm=None) -> str:
    """
    Generate a broader, more abstract version of the input query.

    Args:
        query: Specific user question.
        llm:   Optional LangChain chat model.

    Returns:
        Step-back (abstract) query string.

    Example:
        >>> generate_step_back_query("Why did Lehman Brothers collapse in 2008?")
        "What are the common causes of investment bank insolvency?"
    """
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = STEP_BACK_PROMPT | llm | StrOutputParser()
    return chain.invoke({"query": query})


# ── Main retrieval function ───────────────────────────────────────────────────

def step_back_retrieve(
    query: str,
    retriever,
    llm=None,
    include_original: bool = True,
) -> tuple[list[Document], list[Document]]:
    """
    Retrieve using both the original query and its step-back abstraction.

    Args:
        query:            Original user question.
        retriever:        Any LangChain retriever.
        llm:              Optional LangChain chat model.
        include_original: Whether to also retrieve for the original query.

    Returns:
        Tuple of (step_back_docs, original_docs).
        Callers can merge both or pass them separately to the generation prompt.

    Example:
        >>> step_docs, orig_docs = step_back_retrieve("How does RLHF work in GPT-4?", retriever)
    """
    step_back_query = generate_step_back_query(query, llm)
    step_back_docs = retriever.invoke(step_back_query)

    original_docs: list[Document] = []
    if include_original:
        original_docs = retriever.invoke(query)

    return step_back_docs, original_docs


def step_back_retrieve_merged(
    query: str,
    retriever,
    llm=None,
) -> list[Document]:
    """
    Convenience wrapper: retrieve for both queries and return a deduplicated merge.

    Step-back docs appear first (background context), followed by specific docs.
    """
    step_back_docs, original_docs = step_back_retrieve(query, retriever, llm)
    return deduplicate_documents(step_back_docs + original_docs)


# ── Generation prompt with dual context ──────────────────────────────────────

STEP_BACK_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant. Use the background context and the specific
context below to answer the question thoroughly and accurately.
If the answer is not in the context, say so.

Background context (general principles):
{step_back_context}

Specific context:
{specific_context}

Question: {question}
Answer:"""
)


def build_step_back_generation_chain(llm=None):
    """Chain that accepts step_back_context, specific_context, and question."""
    llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
    return STEP_BACK_GENERATION_PROMPT | llm | StrOutputParser()


# ── LangGraph node ────────────────────────────────────────────────────────────

def step_back_node(state: dict, retriever, llm=None) -> dict:
    """
    LangGraph node: adds step-back docs and original docs to state.
    Sets state['step_back_docs'] and state['documents'].
    """
    step_back_docs, original_docs = step_back_retrieve(
        state.get("translated_query", state["query"]), retriever, llm
    )
    return {
        **state,
        "step_back_docs": step_back_docs,
        "documents": original_docs,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "Why did the 2008 financial crisis happen?",
        "How does the attention mechanism in transformers work?",
        "What caused the Space Shuttle Challenger disaster?",
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    for q in test_cases:
        sb = generate_step_back_query(q, llm)
        print(f"Specific  : {q}")
        print(f"Step-back : {sb}\n")
