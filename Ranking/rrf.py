"""
Reciprocal Rank Fusion (RRF)
-----------------------------
Parameter-free algorithm for merging multiple ranked document lists.
Used after multi-source retrieval (dense + sparse, multi-query, etc.).
"""

from collections import defaultdict
from langchain_core.documents import Document


def get_doc_id(doc: Document) -> str:
    return doc.metadata.get("doc_id") or str(hash(doc.page_content.strip()))


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
    top_n: int | None = None,
) -> list[Document]:
    """
    Merge ranked document lists using RRF.

    Score for document d:  Σ  1 / (k + rank(d, list_i))

    Args:
        result_lists: One ranked list per retrieval source.
        k:            RRF constant (default 60). Higher = less weight on top ranks.
        top_n:        Return only the top N results (default: all).

    Returns:
        Merged and re-ranked Document list.

    Example:
        >>> merged = reciprocal_rank_fusion([dense_results, sparse_results])
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = get_doc_id(doc)
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    ranked = sorted(doc_map.values(), key=lambda d: scores[get_doc_id(d)], reverse=True)
    return ranked[:top_n] if top_n else ranked


def rrf_with_scores(
    result_lists: list[list[Document]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Same as reciprocal_rank_fusion but returns (document, rrf_score) tuples.
    Useful for debugging or downstream score weighting.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = get_doc_id(doc)
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    return sorted(
        [(doc, scores[get_doc_id(doc)]) for doc in doc_map.values()],
        key=lambda x: x[1],
        reverse=True,
    )


if __name__ == "__main__":
    from langchain_core.documents import Document

    list_a = [
        Document(page_content="Doc A", metadata={"doc_id": "a"}),
        Document(page_content="Doc B", metadata={"doc_id": "b"}),
        Document(page_content="Doc C", metadata={"doc_id": "c"}),
    ]
    list_b = [
        Document(page_content="Doc B", metadata={"doc_id": "b"}),
        Document(page_content="Doc D", metadata={"doc_id": "d"}),
        Document(page_content="Doc A", metadata={"doc_id": "a"}),
    ]

    for doc, score in rrf_with_scores([list_a, list_b]):
        print(f"  {doc.metadata['doc_id']}  rrf={score:.4f}")
