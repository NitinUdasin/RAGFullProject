"""
Generator
---------
Combines the provider factory with the RAG generation prompt.
Takes an assembled context string + user query and returns an answer.
"""

import os
from .base_provider import BaseLLMProvider


RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY
the provided context. If the answer is not contained in the context, say
"I don't have enough information to answer that." Do not fabricate facts."""

RAG_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}
Answer:"""


def get_provider(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """
    Factory: return the correct provider based on LLM_PROVIDER env var or argument.

    Providers: "anthropic", "openai", "local"
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model, **kwargs)

    if provider == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model=model, **kwargs)

    if provider == "local":
        from .local_provider import LocalProvider
        return LocalProvider(model=model, **kwargs)

    raise ValueError(f"Unknown LLM provider: '{provider}'. Choose from: anthropic, openai, local")


class RAGGenerator:
    """
    Generates answers from context + query using any LLM provider.

    Args:
        provider:     Provider name or BaseLLMProvider instance.
        model:        Model name (provider-specific).
        system_prompt: Override the default RAG system prompt.

    Example:
        >>> generator = RAGGenerator(provider="anthropic")
        >>> answer = await generator.generate(context="...", question="What is RAG?")
    """

    def __init__(
        self,
        provider: str | BaseLLMProvider | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ):
        if isinstance(provider, BaseLLMProvider):
            self.llm = provider
        else:
            self.llm = get_provider(provider, model)

        self.system_prompt = system_prompt or RAG_SYSTEM_PROMPT

    async def generate(self, context: str, question: str) -> str:
        """Generate a grounded answer from context."""
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        return await self.llm.generate(prompt, system=self.system_prompt)

    async def stream(self, context: str, question: str):
        """Stream a grounded answer from context."""
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        async for chunk in self.llm.stream(prompt, system=self.system_prompt):
            yield chunk


# ── LangGraph node ─────────────────────────────────────────────────────────────

def generation_node(state: dict, generator: RAGGenerator | None = None):
    """
    LangGraph node: generates an answer from state['context'] and state['query'].
    Sets state['answer'] and state['sources'].

    Note: This is a sync wrapper. For async graphs use an async node directly.
    """
    import asyncio
    generator = generator or RAGGenerator()
    answer = asyncio.run(generator.generate(
        context=state.get("context", ""),
        question=state.get("query", ""),
    ))
    return {**state, "answer": answer}


if __name__ == "__main__":
    import asyncio

    async def main():
        generator = RAGGenerator(provider="openai", model="gpt-4o-mini")
        context = "RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a language model to produce factually grounded answers."
        answer = await generator.generate(context=context, question="What does RAG stand for?")
        print(answer)

    asyncio.run(main())
