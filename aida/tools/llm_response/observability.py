"""OpenTelemetry observability for LLM response tool."""

import logging
from typing import Any

from aida.tools.base_observability import SimpleObservability

logger = logging.getLogger(__name__)


class LLMResponseObservability(SimpleObservability):
    """OpenTelemetry observability for LLM response operations."""

    def __init__(self, llm_response_tool, config: dict[str, Any]):
        """Initialize LLM response observability with custom metrics."""
        custom_metrics = {
            "response_count": {
                "type": "counter",
                "description": "Total number of LLM responses",
                "unit": "1",
            },
            "response_length": {
                "type": "histogram",
                "description": "Length of LLM responses",
                "unit": "characters",
            },
            "question_length": {
                "type": "histogram",
                "description": "Length of questions asked",
                "unit": "characters",
            },
        }
        super().__init__(llm_response_tool, config, custom_metrics)
