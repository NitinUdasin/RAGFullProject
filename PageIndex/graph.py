"""
Full RAG Pipeline Graph
------------------------
Assembles all pipeline components into a single LangGraph StateGraph.

Pipeline:
  translate_query → structure_query → route_query
    → [vectorstore] retrieve → rank → assemble_context → generate
    → [web_search]  retrieve_web → assemble_context → generate
"""

from functools import partial
from langgraph.graph import StateGraph, END

from .state import RAGState
from QueryTranslation.query_rewrite import rewrite_query_node
from QueryStructuring import filter_extraction_node
from Routing import logical_router_node
from ContextAssembly import context_assembly_node, ContextAssembler
from Generation import RAGGenerator


def build_rag_graph(
    retriever,
    reranker=None,
    generator: RAGGenerator | None = None,
    assembler: ContextAssembler | None = None,
    enable_reranking: bool = True,
):
    """
    Build and compile the full RAG LangGraph pipeline.

    Args:
        retriever:        LangChain-compatible retriever (from Indexer.as_retriever()).
        reranker:         Optional reranker (CrossEncoderReranker, CohereReranker, etc.).
        generator:        RAGGenerator instance (default: reads from LLM_PROVIDER env).
        assembler:        ContextAssembler instance (default: max_tokens=3000).
        enable_reranking: Whether to include the ranking node.

    Returns:
        Compiled LangGraph CompiledGraph.

    Example:
        >>> from Indexing import Indexer
        >>> indexer = Indexer("rag_docs")
        >>> graph = build_rag_graph(indexer.as_retriever())
        >>> result = await graph.ainvoke({"query": "What is RAG?"})
    """
    generator = generator or RAGGenerator()
    assembler = assembler or ContextAssembler()

    graph = StateGraph(RAGState)

    # ── Nodes ──────────────────────────────────────────────────────────────────

    graph.add_node("translate_query", rewrite_query_node)

    graph.add_node("structure_query", filter_extraction_node)

    graph.add_node("route_query", logical_router_node)

    def retrieve_node(state: RAGState) -> RAGState:
        query = state.get("translated_query", state["query"])
        filters = state.get("metadata_filters") or None
        docs = retriever.invoke(query)
        return {**state, "documents": docs}

    graph.add_node("retrieve", retrieve_node)

    def web_search_node(state: RAGState) -> RAGState:
        """Placeholder — swap in a real web search tool."""
        from langchain_community.tools import DuckDuckGoSearchRun
        tool = DuckDuckGoSearchRun()
        result = tool.invoke(state.get("translated_query", state["query"]))
        from langchain_core.documents import Document
        doc = Document(page_content=result, metadata={"source": "web_search"})
        return {**state, "documents": [doc]}

    graph.add_node("retrieve_web", web_search_node)

    if enable_reranking and reranker:
        def rank_node(state: RAGState) -> RAGState:
            query = state.get("translated_query", state["query"])
            reranked = reranker.rerank(query, state.get("documents", []))
            return {**state, "documents": reranked}
        graph.add_node("rank", rank_node)

    def assemble_node(state: RAGState) -> RAGState:
        return context_assembly_node(state, assembler)

    graph.add_node("assemble_context", assemble_node)

    import asyncio

    def generate_node(state: RAGState) -> RAGState:
        answer = asyncio.run(generator.generate(
            context=state.get("context", ""),
            question=state.get("query", ""),
        ))
        return {**state, "answer": answer}

    graph.add_node("generate", generate_node)

    # ── Edges ──────────────────────────────────────────────────────────────────

    graph.set_entry_point("translate_query")
    graph.add_edge("translate_query", "structure_query")
    graph.add_edge("structure_query", "route_query")

    graph.add_conditional_edges(
        "route_query",
        lambda s: s.get("route", "vectorstore"),
        {
            "vectorstore": "retrieve",
            "web_search":  "retrieve_web",
            "sql_database": "retrieve",   # extend with SQL node as needed
        },
    )

    retrieve_next = "rank" if (enable_reranking and reranker) else "assemble_context"
    graph.add_edge("retrieve", retrieve_next)
    graph.add_edge("retrieve_web", "assemble_context")

    if enable_reranking and reranker:
        graph.add_edge("rank", "assemble_context")

    graph.add_edge("assemble_context", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ── Convenience async runner ───────────────────────────────────────────────────

async def run_rag_pipeline(query: str, graph) -> dict:
    """
    Run the compiled graph on a single query.

    Returns:
        Final state dict with 'answer' and 'sources'.
    """
    result = await graph.ainvoke({"query": query})
    return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
