"""
Local LLM Provider (Ollama)
----------------------------
LLM provider using locally running models via Ollama.
Requires: Ollama running at OLLAMA_HOST with the model pulled.
e.g.  ollama pull llama3.2
"""

import os
from typing import AsyncIterator
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from .base_provider import BaseLLMProvider


class LocalProvider(BaseLLMProvider):
    """
    Local Ollama provider for generation and streaming.

    Args:
        model:       Ollama model name (default: llama3.2).
        temperature: Sampling temperature.
        base_url:    Ollama server URL (default: http://localhost:11434).

    Example:
        >>> provider = LocalProvider(model="llama3.2")
        >>> answer = await provider.generate("What is RAG?")
    """

    DEFAULT_MODEL = "llama3.2"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        base_url: str | None = None,
    ):
        self.model = model or os.getenv("LOCAL_MODEL", self.DEFAULT_MODEL)
        self.temperature = temperature
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a complete response from a local model."""
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        response = await self._llm.ainvoke(messages)
        return response.content

    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        """Stream response tokens from a local model."""
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        async for chunk in self._llm.astream(messages):
            if chunk.content:
                yield chunk.content


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = LocalProvider()
        print(f"Using model: {provider.model}")
        answer = await provider.generate("What is RAG in one sentence?")
        print(answer)

    asyncio.run(main())
