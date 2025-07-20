"""Configuration for LLM response tool."""

from typing import Any

from aida.config.llm_profiles import Purpose


class LLMResponseConfig:
    """Configuration for LLM response tool operations."""

    # LLM Configuration
    LLM_PURPOSE = Purpose.DEFAULT

    # Response Configuration
    DEFAULT_MAX_LENGTH = 2000
    ABSOLUTE_MAX_LENGTH = 10000
    MIN_LENGTH = 100

    # Prompt Configuration
    CONTEXT_PREFIX = "Previous conversation context:"
    QUESTION_PREFIX = "Current question:"
    LENGTH_INSTRUCTION = (
        "Please provide a helpful, informative response (max {max_length} characters)."
    )

    # Processing Configuration
    ENABLE_RESPONSE_CACHE = False  # Disabled by default for dynamic responses
    CACHE_TTL_SECONDS = 300  # 5 minutes if enabled

    # Truncation Configuration
    TRUNCATION_SUFFIX = "..."
    TRUNCATION_BUFFER = 3  # Characters to reserve for suffix

    @classmethod
    def build_prompt(cls, question: str, context: str = "", max_length: int = None) -> str:
        """Build the prompt for LLM."""
        if max_length is None:
            max_length = cls.DEFAULT_MAX_LENGTH

        if context:
            prompt = f"{cls.CONTEXT_PREFIX}\n{context}\n\n{cls.QUESTION_PREFIX} {question}"
        else:
            prompt = question

        prompt += f"\n\n{cls.LENGTH_INSTRUCTION.format(max_length=max_length)}"

        return prompt

    @classmethod
    def get_mcp_config(cls) -> dict[str, Any]:
        """Get MCP server configuration."""
        return {
            "server_name": "aida-llm-response",
            "max_concurrent_requests": 20,
            "timeout_seconds": 30,
            "supported_models": ["default", "quick", "reasoning"],
        }

    @classmethod
    def get_observability_config(cls) -> dict[str, Any]:
        """Get OpenTelemetry configuration."""
        return {
            "service_name": "aida-llm-response-tool",
            "trace_enabled": True,
            "metrics_enabled": True,
            "export_endpoint": "http://localhost:4317",
        }
