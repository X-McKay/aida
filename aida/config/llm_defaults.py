"""Default LLM configuration and auto-setup utilities."""

from typing import Any


async def auto_configure_llm_providers():
    """Auto-configure LLM providers - compatibility function.

    The new LLM system auto-configures based on llm_profiles.py,
    so this is just a compatibility shim.
    """
    # The LLMManager automatically sets up Ollama-based models
    # from the profiles, so we don't need to do anything here
    from aida.llm import get_llm

    # Just verify the manager is initialized
    manager = get_llm()

    # Check if we have any purposes available
    purposes = manager.list_purposes()

    if not purposes:
        raise Exception("No LLM purposes configured. Check llm_profiles.py")

    return len(purposes)


def get_default_llm_config() -> dict[str, Any]:
    """Get default LLM configuration."""
    return {
        "default_provider": "ollama",
        "providers": {
            "ollama": {
                "type": "ollama",
                "api_url": "http://localhost:11434",
                "model": "llama3.2:latest",
                "timeout": 120,
            }
        },
    }
