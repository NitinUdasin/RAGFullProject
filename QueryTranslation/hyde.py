"""
HyDE — Hypothetical Document Embeddings
----------------------------------------
Instead of embedding the raw query, ask the LLM to write a *hypothetical*
answer, then embed that answer and use it for nearest-neighbour search.

The generated text is semantically closer to real documents in the corpus
than a short query, which improves retrieval recall — especially when
queries are brief but documents are long and detailed.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (ACL 2023).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ── Prompt ────────────────────────────────────────────────────────────────────

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """Write a short, factual paragraph that would be a plausible answer to the
question below. Write confidently as if the information is known — this is
used only to improve document retrieval, not shown to the user.
Do NOT say you don't know. Keep it to 3-5 sentences.

Question: {query}
Hypothetical answer:"""
)

# ── Hypothetical document generation ─────────────────────────────────────────

def generate_hypothetical_document(query: str, llm=None) -> str:
    """
    Generate a hypothetical answer paragraph for a given query.

    Args:
        query: User question.
        llm:   Optional LangChain chat model.

    Returns:
        Hypothetical answer string (NOT shown to the user).

    Example:
        >>> generate_hypothetical_document("What is RAG?")
        "RAG stands for Retrieval-Augmented Generation. It is a technique that
         combines a retrieval system with a language model..."
    """
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    chain = HYDE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"query": query})


# ── Main retrieval function ───────────────────────────────────────────────────

def hyde_retrieve(
    query: str,
    retriever,
    llm=None,
) -> list[Document]:
    """
    Retrieve documents using a hypothetical document embedding instead of
    the raw query embedding.

    Note: This function passes the hypothetical document as the search query
    to the retriever. For vector stores that accept raw text (e.g., Chroma,
    FAISS via LangChain), this works out of the box — the retriever embeds it.

    Args:
        query:     Original user question.
        retriever: Any LangChain retriever.
        llm:       Optional LangChain chat model for hypothesis generation.

    Returns:
        List of retrieved Document objects.

    Example:
        >>> docs = hyde_retrieve("Explain RLHF", vectorstore.as_retriever())
    """
    hypothetical_doc = generate_hypothetical_document(query, llm)
    return retriever.invoke(hypothetical_doc)


# ── Direct embedding variant (for custom vector stores) ──────────────────────

def hyde_retrieve_with_embeddings(
    query: str,
    collection,           # ChromaDB collection
    embedding_model=None,
    llm=None,
    n_results: int = 5,
) -> list[Document]:
    """
    HyDE retrieval using direct embedding of the hypothetical document.
    Use this when you have direct access to the ChromaDB collection.

    Args:
        query:           User question.
        collection:      ChromaDB collection object.
        embedding_model: LangChain embeddings model (default: OpenAIEmbeddings).
        llm:             Optional chat model.
        n_results:       Number of results to return.

    Returns:
        List of Document objects.
    """
    embedding_model = embedding_model or OpenAIEmbeddings()
    hypothetical_doc = generate_hypothetical_document(query, llm)

    # Embed the hypothetical document
    hyp_embedding = embedding_model.embed_query(hypothetical_doc)

    results = collection.query(
        query_embeddings=[hyp_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


# ── LangGraph node ────────────────────────────────────────────────────────────

def hyde_node(state: dict, retriever, llm=None) -> dict:
    """LangGraph node: retrieves documents via HyDE and stores in state['documents']."""
    docs = hyde_retrieve(state.get("translated_query", state["query"]), retriever, llm)
    return {**state, "documents": docs}


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is the difference between RAG and fine-tuning?",
        "How does ChromaDB store embeddings?",
        "What are the limitations of transformer attention?",
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    for q in test_queries:
        hyp = generate_hypothetical_document(q, llm)
        print(f"Query     : {q}")
        print(f"Hypothesis: {hyp[:200]}...\n")
