"""
Base LLM Provider
-----------------
Abstract interface for all LLM providers (Anthropic, OpenAI, local).
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLMProvider(ABC):
    """All providers must implement generate and stream."""

    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a complete response."""
        ...

    @abstractmethod
    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        """Stream response tokens."""
        ...

    def generate_sync(self, prompt: str, system: str = "") -> str:
        """Synchronous wrapper around generate (for non-async contexts)."""
        import asyncio
        return asyncio.run(self.generate(prompt, system))
