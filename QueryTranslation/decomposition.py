"""
Decomposition (Least-to-Most)
------------------------------
Breaks a complex, multi-hop query into a sequence of simpler sub-questions.
Each sub-question is answered in order, with previous answers passed as
context to subsequent ones — enabling multi-step reasoning across documents.

Two strategies:
  1. Sequential (answer-then-retrieve): answer sub-question N using retrieved
     docs + all prior answers, then feed that answer into sub-question N+1.
  2. Parallel (retrieve-then-merge): retrieve for all sub-questions independently,
     merge documents, and answer in a single generation call.

Reference: Zhou et al., "Least-to-Most Prompting Enables Complex Reasoning
in Large Language Models" (2023).
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .multi_query import deduplicate_documents


# ── Prompts ───────────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at breaking complex questions into simpler sub-questions.
Decompose the following question into {n} ordered sub-questions that build on
each other and together answer the original question.

Return a JSON array of strings and nothing else.

Question: {query}
Sub-questions:"""
)

SEQUENTIAL_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the sub-question using the retrieved context and any prior answers.
Be concise and factual. If the answer is not in the context, say so.

Prior answers:
{prior_answers}

Retrieved context:
{context}

Sub-question: {sub_question}
Answer:"""
)

FINAL_SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """Using the answers to the sub-questions below, write a comprehensive answer
to the original question.

Original question: {original_query}

Sub-questions and answers:
{qa_pairs}

Final answer:"""
)


# ── Decomposition ─────────────────────────────────────────────────────────────

def decompose_query(query: str, n: int = 3, llm=None) -> list[str]:
    """
    Break a complex query into ordered sub-questions.

    Args:
        query: Complex multi-hop user question.
        n:     Number of sub-questions to generate.
        llm:   Optional LangChain chat model.

    Returns:
        Ordered list of sub-question strings.

    Example:
        >>> decompose_query("How does our pricing compare to competitors given recent changes?")
        [
          "What is our current pricing structure?",
          "What is the competitor's pricing?",
          "What recent market changes are relevant to pricing?"
        ]
    """
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = DECOMPOSE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "n": n})

    try:
        sub_questions = json.loads(raw)
    except json.JSONDecodeError:
        sub_questions = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in raw.splitlines()
            if line.strip()
        ]
    return sub_questions


# ── Strategy 1: Sequential (recommended for multi-hop) ───────────────────────

def decompose_and_retrieve_sequential(
    query: str,
    retriever,
    n: int = 3,
    llm=None,
) -> dict:
    """
    Answer sub-questions sequentially, feeding prior answers as context.

    Args:
        query:     Complex user question.
        retriever: Any LangChain retriever.
        n:         Number of sub-questions.
        llm:       Optional LangChain chat model.

    Returns:
        Dict with keys:
          - 'sub_questions': list of sub-question strings
          - 'sub_answers':   list of answers (aligned with sub_questions)
          - 'all_documents': all retrieved documents across sub-questions
          - 'final_answer':  synthesized answer to the original query
    """
    llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
    sub_questions = decompose_query(query, n=n, llm=llm)

    answer_chain = SEQUENTIAL_ANSWER_PROMPT | llm | StrOutputParser()
    synthesis_chain = FINAL_SYNTHESIS_PROMPT | llm | StrOutputParser()

    sub_answers: list[str] = []
    all_docs: list[Document] = []

    for sub_q in sub_questions:
        docs = retriever.invoke(sub_q)
        all_docs.extend(docs)

        context = "\n\n".join(doc.page_content for doc in docs)
        prior = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(sub_questions, sub_answers)
        ) or "None"

        answer = answer_chain.invoke({
            "sub_question": sub_q,
            "context": context,
            "prior_answers": prior,
        })
        sub_answers.append(answer)

    qa_pairs = "\n\n".join(
        f"Q{i+1}: {q}\nA{i+1}: {a}"
        for i, (q, a) in enumerate(zip(sub_questions, sub_answers))
    )
    final_answer = synthesis_chain.invoke({
        "original_query": query,
        "qa_pairs": qa_pairs,
    })

    return {
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "all_documents": deduplicate_documents(all_docs),
        "final_answer": final_answer,
    }


# ── Strategy 2: Parallel retrieval (faster, less reasoning depth) ─────────────

def decompose_and_retrieve_parallel(
    query: str,
    retriever,
    n: int = 3,
    llm=None,
) -> list[Document]:
    """
    Retrieve documents for all sub-questions in parallel and return merged results.
    Faster than sequential but doesn't chain intermediate answers.

    Returns:
        Deduplicated list of Documents from all sub-question retrievals.
    """
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sub_questions = decompose_query(query, n=n, llm=llm)

    all_docs: list[Document] = []
    for sub_q in sub_questions:
        all_docs.extend(retriever.invoke(sub_q))
    return deduplicate_documents(all_docs)


# ── Public alias ──────────────────────────────────────────────────────────────

def decompose_and_retrieve(query: str, retriever, n: int = 3, llm=None) -> list[Document]:
    """Default decomposition retrieval (parallel strategy)."""
    return decompose_and_retrieve_parallel(query, retriever, n=n, llm=llm)


# ── LangGraph node ────────────────────────────────────────────────────────────

def decomposition_node(state: dict, retriever, n: int = 3, llm=None) -> dict:
    """
    LangGraph node: sequential decomposition.
    Populates state['documents'], state['sub_questions'], state['sub_answers'],
    and state['answer'].
    """
    result = decompose_and_retrieve_sequential(
        state.get("translated_query", state["query"]), retriever, n=n, llm=llm
    )
    return {
        **state,
        "documents": result["all_documents"],
        "sub_questions": result["sub_questions"],
        "sub_answers": result["sub_answers"],
        "answer": result["final_answer"],
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = "How does retrieval-augmented generation compare to fine-tuning for domain adaptation?"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sub_qs = decompose_query(query, n=3, llm=llm)
    print(f"Original: {query}\n")
    print("Sub-questions:")
    for i, q in enumerate(sub_qs, 1):
        print(f"  {i}. {q}")
