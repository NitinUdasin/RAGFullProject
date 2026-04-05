"""
Logical Router (LLM-based)
--------------------------
Uses a structured LLM output to classify the query and select the correct
retrieval source. The LLM understands semantic differences between routes.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ── Route schema ──────────────────────────────────────────────────────────────

class RouteQuery(BaseModel):
    """Route classification result."""
    destination: Literal["vectorstore", "web_search", "sql_database"] = Field(
        description="The data source best suited to answer the query."
    )
    reasoning: str = Field(description="One-sentence explanation for the routing decision.")


# ── Prompt ────────────────────────────────────────────────────────────────────

LOGICAL_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query router for a RAG system. Given a user question, decide
which data source to query:

- vectorstore: for questions about internal documents, policies, product documentation
- web_search:  for questions requiring recent or real-time information
- sql_database: for questions about structured data, metrics, statistics, or reports

Return a JSON object with 'destination' and 'reasoning'.""",
    ),
    ("human", "{query}"),
])


# ── Router ────────────────────────────────────────────────────────────────────

class LogicalRouter:
    """
    Routes queries to a retrieval source using LLM classification.

    Args:
        llm: Optional LangChain chat model. Defaults to gpt-4o-mini.

    Example:
        >>> router = LogicalRouter()
        >>> result = router.route("What were our Q3 sales figures?")
        >>> result.destination  # "sql_database"
    """

    def __init__(self, llm=None):
        llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.chain = LOGICAL_ROUTER_PROMPT | llm.with_structured_output(RouteQuery)

    def route(self, query: str) -> RouteQuery:
        """Classify query and return route decision."""
        return self.chain.invoke({"query": query})

    def get_destination(self, query: str) -> str:
        """Return just the destination string."""
        return self.route(query).destination


# ── LangGraph node ────────────────────────────────────────────────────────────

def logical_router_node(state: dict, llm=None) -> dict:
    """LangGraph node: sets state['route'] based on logical routing."""
    router = LogicalRouter(llm)
    result = router.route(state.get("translated_query", state["query"]))
    return {**state, "route": result.destination, "route_reasoning": result.reasoning}


if __name__ == "__main__":
    router = LogicalRouter()
    queries = [
        "What is our refund policy?",
        "What is the current Bitcoin price?",
        "How many users signed up last month?",
    ]
    for q in queries:
        result = router.route(q)
        print(f"Query      : {q}")
        print(f"Destination: {result.destination}")
        print(f"Reasoning  : {result.reasoning}\n")
