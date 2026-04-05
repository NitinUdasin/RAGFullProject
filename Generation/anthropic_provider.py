"""
Anthropic Claude Provider
--------------------------
LLM provider using the Anthropic Claude API.
Requires: pip install anthropic, ANTHROPIC_API_KEY env var.
"""

import os
from typing import AsyncIterator
import anthropic

from .base_provider import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """
    Claude provider for generation and streaming.

    Args:
        model:       Claude model ID (default: claude-sonnet-4-6).
        max_tokens:  Maximum output tokens.
        temperature: Sampling temperature (0 = deterministic).

    Example:
        >>> provider = AnthropicProvider()
        >>> answer = await provider.generate("What is RAG?", system="You are a helpful assistant.")
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.model = model or os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a complete response from Claude."""
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text

    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        """Stream response tokens from Claude."""
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = AnthropicProvider()
        answer = await provider.generate("What is RAG in one sentence?")
        print(answer)

    asyncio.run(main())
