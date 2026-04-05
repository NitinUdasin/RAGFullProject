"""
Context Assembly
----------------
Deduplicates, truncates, orders, and formats retrieved documents
into a context string ready to be injected into the generation prompt.
"""

import hashlib
from langchain_core.documents import Document


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken (falls back to word count if unavailable)."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


class ContextAssembler:
    """
    Assembles a ranked list of documents into a context string for the LLM.

    Responsibilities:
      1. Deduplicate near-identical chunks
      2. Apply lost-in-the-middle reordering (best chunks first + last)
      3. Truncate to token budget
      4. Format with source attribution

    Args:
        max_tokens:     Maximum tokens for the assembled context block.
        dedup:          Whether to deduplicate by content hash.
        reorder:        Apply lost-in-the-middle mitigation.
        include_source: Include source metadata in each chunk header.

    Example:
        >>> assembler = ContextAssembler(max_tokens=3000)
        >>> context = assembler.assemble(ranked_docs)
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        dedup: bool = True,
        reorder: bool = True,
        include_source: bool = True,
    ):
        self.max_tokens = max_tokens
        self.dedup = dedup
        self.reorder = reorder
        self.include_source = include_source

    def assemble(self, documents: list[Document]) -> str:
        """
        Assemble documents into a single context string.

        Args:
            documents: Ranked list of retrieved Document objects.

        Returns:
            Formatted context string for injection into the generation prompt.
        """
        docs = documents

        if self.dedup:
            docs = self._deduplicate(docs)

        if self.reorder:
            docs = self._lost_in_middle_reorder(docs)

        return self._format_and_truncate(docs)

    def assemble_with_sources(self, documents: list[Document]) -> tuple[str, list[str]]:
        """
        Assemble context and return (context_string, list_of_source_strings).
        Useful when the generation step needs to cite sources separately.
        """
        docs = documents
        if self.dedup:
            docs = self._deduplicate(docs)
        if self.reorder:
            docs = self._lost_in_middle_reorder(docs)

        context = self._format_and_truncate(docs)
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})
        return context, sources

    # ── Deduplication ─────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(documents: list[Document]) -> list[Document]:
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in documents:
            content_hash = hashlib.md5(doc.page_content.strip().encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)
        return unique

    # ── Lost-in-the-middle mitigation ─────────────────────────────────────────

    @staticmethod
    def _lost_in_middle_reorder(documents: list[Document]) -> list[Document]:
        """
        Place highest-ranked documents at the beginning and end of the context.
        LLMs attend better to content at the edges of the context window.
        """
        if len(documents) <= 2:
            return documents

        reordered: list[Document] = []
        left, right = [], []
        for i, doc in enumerate(documents):
            if i % 2 == 0:
                right.append(doc)
            else:
                left.insert(0, doc)
        return left + right

    # ── Formatting & truncation ───────────────────────────────────────────────

    def _format_and_truncate(self, documents: list[Document]) -> str:
        parts: list[str] = []
        total_tokens = 0

        for i, doc in enumerate(documents, start=1):
            header = self._make_header(i, doc) if self.include_source else f"[{i}]"
            snippet = f"{header}\n{doc.page_content.strip()}"
            tokens = count_tokens(snippet)

            if total_tokens + tokens > self.max_tokens:
                break

            parts.append(snippet)
            total_tokens += tokens

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _make_header(index: int, doc: Document) -> str:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_str = f" (page {page})" if page is not None else ""
        return f"[{index}] Source: {source}{page_str}"


# ── LangGraph node ─────────────────────────────────────────────────────────────

def context_assembly_node(
    state: dict,
    assembler: ContextAssembler | None = None,
) -> dict:
    """
    LangGraph node: assembles state['documents'] into state['context'].
    Also sets state['sources'] for citation.
    """
    assembler = assembler or ContextAssembler()
    context, sources = assembler.assemble_with_sources(state.get("documents", []))
    return {**state, "context": context, "sources": sources}


if __name__ == "__main__":
    from langchain_core.documents import Document

    docs = [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "intro.pdf", "page": 1, "doc_id": "1"}),
        Document(page_content="ChromaDB stores dense embeddings.", metadata={"source": "db_guide.html", "doc_id": "2"}),
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "intro.pdf", "page": 1, "doc_id": "1-dup"}),  # duplicate
        Document(page_content="LangGraph builds stateful pipelines.", metadata={"source": "langgraph.md", "doc_id": "3"}),
    ]

    assembler = ContextAssembler(max_tokens=500)
    context, sources = assembler.assemble_with_sources(docs)
    print("=== Context ===")
    print(context)
    print("\n=== Sources ===")
    print(sources)
