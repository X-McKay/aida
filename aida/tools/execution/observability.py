"""OpenTelemetry observability for execution tool."""

import logging
from typing import Dict, Any

from aida.tools.base_observability import SimpleObservability

logger = logging.getLogger(__name__)


class ExecutionObservability(SimpleObservability):
    """OpenTelemetry observability for execution operations."""
    
    def __init__(self, execution_tool, config: Dict[str, Any]):
        custom_metrics = {
            "executions_total": {
                "type": "counter",
                "description": "Total number of code executions",
                "unit": "1"
            },
            "execution_duration": {
                "type": "histogram",
                "description": "Code execution duration",
                "unit": "s"
            },
            "execution_timeouts": {
                "type": "counter",
                "description": "Number of execution timeouts",
                "unit": "1"
            },
            "language_usage": {
                "type": "counter",
                "description": "Usage count by programming language",
                "unit": "1"
            },
            "memory_usage": {
                "type": "histogram",
                "description": "Memory usage during execution",
                "unit": "bytes"
            }
        }
        super().__init__(execution_tool, config, custom_metrics)
    
    @property
    def execution_counter(self):
        """Get execution counter metric for compatibility."""
        return self.custom_metrics.get("executions_total")
    
    def trace_execution(self, language: str, timeout: int):
        """Create a trace span for code execution."""
        return self.trace_operation("execute", language=language, timeout=timeout)
    
    def record_execution(self, language: str, duration: float, success: bool):
        """Record execution metrics."""
        self.record_operation(
            "execute",
            duration,
            success,
            executions_total=1,
            execution_duration=duration,
            language_usage=1
        )
        
        if not success:
            # Record timeout if applicable
            if hasattr(self, 'custom_metrics') and 'execution_timeouts' in self.custom_metrics:
                self.custom_metrics['execution_timeouts'].add(1, {"language": language})