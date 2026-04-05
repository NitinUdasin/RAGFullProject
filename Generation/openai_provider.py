"""
OpenAI Provider
---------------
LLM provider using the OpenAI Chat Completions API.
Requires: pip install openai, OPENAI_API_KEY env var.
"""

import os
from typing import AsyncIterator
from openai import AsyncOpenAI

from .base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT provider for generation and streaming.

    Args:
        model:       OpenAI model name (default: gpt-4o).
        max_tokens:  Maximum output tokens.
        temperature: Sampling temperature.

    Example:
        >>> provider = OpenAIProvider(model="gpt-4o-mini")
        >>> answer = await provider.generate("Explain embeddings.", system="Be concise.")
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a complete response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        """Stream response tokens."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = OpenAIProvider(model="gpt-4o-mini")
        answer = await provider.generate("What is RAG in one sentence?")
        print(answer)

    asyncio.run(main())
