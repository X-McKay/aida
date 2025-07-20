"""Simple LLM interface for AIDA."""

from collections.abc import AsyncGenerator

from aida.config.llm_profiles import Purpose

from .manager import LLMManager

# Global manager instance
_manager: LLMManager = None


def get_llm() -> LLMManager:
    """Get LLM manager."""
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager


async def chat(
    message: str, purpose: Purpose = Purpose.DEFAULT, stream: bool = False
) -> str | AsyncGenerator[str, None]:
    """Simple chat interface."""
    return await get_llm().chat(message, purpose, stream)
