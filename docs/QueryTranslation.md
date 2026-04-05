# Query Translation

Query Translation transforms a user's raw input into one or more improved queries before hitting the vector store. The goal is to bridge the gap between how users phrase questions and how relevant documents are indexed.

---

## Why It Matters

Raw user queries are often:
- Too vague or short ("summarize payments")
- Ambiguous across multiple topics
- Phrased differently from the source documents
- Missing implicit context the user assumes

Query translation improves recall without changing the retrieval infrastructure.

---

## Techniques

### 1. Query Rewriting

Reformulate the query to be more explicit and self-contained using an LLM.

**Use when:** queries are short, colloquial, or missing context.

```
User: "what about refunds?"
Rewritten: "What is the refund policy for digital purchases?"
```

```python
from langchain_core.prompts import ChatPromptTemplate

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer. Rewrite the following query to be 
more specific and retrieval-friendly. Return only the rewritten query.

Query: {query}
""")

chain = rewrite_prompt | llm | StrOutputParser()
rewritten = chain.invoke({"query": user_query})
```

---

### 2. Multi-Query Generation

Generate N alternative phrasings of the same question, retrieve for each, then deduplicate results.

**Use when:** a single query may miss relevant documents due to vocabulary mismatch.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
docs = retriever.invoke(user_query)
```

**LangGraph node example:**

```python
def multi_query_node(state: RAGState) -> RAGState:
    queries = generate_variants(state["query"], n=3)
    all_docs = []
    for q in queries:
        all_docs.extend(vectorstore.similarity_search(q))
    state["documents"] = deduplicate(all_docs)
    return state
```

---

### 3. RAG-Fusion

Extends multi-query by re-ranking the merged results with **Reciprocal Rank Fusion (RRF)** — a parameter-free rank merging algorithm.

**Use when:** you want multi-query benefits with better result ordering.

```python
def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60) -> list[Document]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in results:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.metadata["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
            doc_map[doc_id] = doc

    return sorted(doc_map.values(), key=lambda d: scores[d.metadata["id"]], reverse=True)
```

---

### 4. Step-Back Prompting

Generate a more abstract, higher-level question from the specific query, retrieve on both, then combine context.

**Use when:** the original query is too narrow and background knowledge is needed for a good answer.

```
Specific: "Why did the 2008 Lehman Brothers collapse happen?"
Step-back: "What causes investment bank insolvency?"
```

```python
step_back_prompt = ChatPromptTemplate.from_template("""
Given the specific question below, generate a broader, more abstract question
that would help retrieve useful background context.

Question: {query}
Step-back question:
""")
```

Retrieve for both questions, merge context, then generate the answer.

---

### 5. HyDE (Hypothetical Document Embeddings)

Ask the LLM to write a *hypothetical* answer to the query, then embed and retrieve based on that generated answer rather than the raw query.

**Use when:** the query is short but answers in the corpus are long and detailed. The hypothetical answer is semantically closer to real documents than the query itself.

```python
hyde_prompt = ChatPromptTemplate.from_template("""
Write a short paragraph that would be a plausible answer to the following question.
Do not say you don't know — write as if it were true.

Question: {query}
Hypothetical answer:
""")

hypothetical_doc = (hyde_prompt | llm | StrOutputParser()).invoke({"query": query})
docs = vectorstore.similarity_search(hypothetical_doc)
```

---

### 6. Decomposition (Least-to-Most)

Break a complex query into ordered sub-questions. Answer each sub-question sequentially, feeding prior answers as context for the next.

**Use when:** the question requires multi-hop reasoning across different document sections.

```
Complex: "How does our pricing compare to competitor X given recent market changes?"

Sub-questions:
  1. What is our current pricing structure?
  2. What is competitor X's pricing?
  3. What recent market changes are relevant?
  4. How do these compare?
```

```python
decompose_prompt = ChatPromptTemplate.from_template("""
Break the following question into simpler sub-questions that must be answered in order.
Return a JSON array of strings.

Question: {query}
""")
```

---

## Choosing a Technique

| Scenario | Recommended Technique |
|---|---|
| Short / vague query | Query Rewriting |
| Vocabulary mismatch likely | Multi-Query or RAG-Fusion |
| Needs background knowledge | Step-Back |
| Short query, long answer documents | HyDE |
| Multi-hop / complex reasoning | Decomposition |
| High-recall requirement | RAG-Fusion |

---

## LangGraph Integration

Query translation nodes sit between the **input** and **retrieval** nodes in the graph:

```
[user_input] → [translate_query] → [retrieve] → [grade_documents] → [generate]
```

The `translate_query` node is swappable — select the strategy via config or a router node that inspects query complexity.

```python
def route_query_translation(state: RAGState) -> str:
    if state["query_complexity"] == "simple":
        return "rewrite"
    elif state["query_complexity"] == "multi_hop":
        return "decompose"
    else:
        return "multi_query"
```
