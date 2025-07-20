"""OpenTelemetry observability for thinking tool."""

import logging
from typing import Any

from aida.tools.base_observability import SimpleObservability

logger = logging.getLogger(__name__)


class ThinkingObservability(SimpleObservability):
    """OpenTelemetry observability for thinking operations."""

    def __init__(self, thinking_tool, config: dict[str, Any]):
        custom_metrics = {
            "analysis_count": {
                "type": "counter",
                "description": "Total number of thinking analyses",
                "unit": "1",
            },
            "analysis_depth": {
                "type": "histogram",
                "description": "Depth of analysis performed",
                "unit": "level",
            },
            "cache_hits": {"type": "counter", "description": "Number of cache hits", "unit": "1"},
            "reasoning_type_usage": {
                "type": "counter",
                "description": "Usage count by reasoning type",
                "unit": "1",
            },
        }
        super().__init__(thinking_tool, config, custom_metrics)
