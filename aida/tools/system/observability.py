"""OpenTelemetry observability for system tool."""

import logging
from typing import Dict, Any

from aida.tools.base_observability import SimpleObservability

logger = logging.getLogger(__name__)


class SystemObservability(SimpleObservability):
    """OpenTelemetry observability for system operations."""
    
    def __init__(self, system_tool, config: Dict[str, Any]):
        custom_metrics = {
            "commands_executed": {
                "type": "counter",
                "description": "Total number of commands executed",
                "unit": "1"
            },
            "command_duration": {
                "type": "histogram",
                "description": "Command execution duration",
                "unit": "s"
            },
            "command_timeouts": {
                "type": "counter",
                "description": "Number of command timeouts",
                "unit": "1"
            },
            "processes_monitored": {
                "type": "gauge",
                "description": "Number of processes being monitored",
                "unit": "1"
            },
            "system_cpu_usage": {
                "type": "gauge",
                "description": "System CPU usage percentage",
                "unit": "percent"
            },
            "system_memory_usage": {
                "type": "gauge",
                "description": "System memory usage percentage",
                "unit": "percent"
            }
        }
        super().__init__(system_tool, config, custom_metrics)