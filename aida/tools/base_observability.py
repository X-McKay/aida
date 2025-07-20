"""Base OpenTelemetry observability implementation for AIDA tools."""

import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseObservability:
    """Base OpenTelemetry observability implementation."""
    
    def __init__(self, tool, config: Dict[str, Any]):
        """Initialize observability with tool reference and config."""
        self.tool = tool
        self.config = config
        self.enabled = config.get("trace_enabled", True)
        self.tracer = None
        self.meter = None
        
        # Common metrics
        self.operation_counter = None
        self.operation_duration = None
        self.operation_errors = None
        
        if self.enabled:
            self._setup_observability()
    
    def _setup_observability(self):
        """Setup OpenTelemetry components."""
        try:
            from opentelemetry import trace, metrics
            
            # Setup tracer
            service_name = self.config.get("service_name", f"aida-{self.tool.name}")
            self.tracer = trace.get_tracer(service_name)
            
            # Setup metrics if enabled
            if self.config.get("metrics_enabled", True):
                self.meter = metrics.get_meter(service_name)
                self._setup_common_metrics()
                self._setup_custom_metrics()
            
            logger.debug(f"OpenTelemetry observability enabled for {self.tool.name}")
            
        except ImportError:
            logger.warning(
                "OpenTelemetry not available. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk"
            )
            self.enabled = False
    
    def _setup_common_metrics(self):
        """Setup common metrics for all tools."""
        if not self.meter:
            return
        
        # Operation counter
        self.operation_counter = self.meter.create_counter(
            f"{self.tool.name}_operations_total",
            description=f"Total number of {self.tool.name} operations",
            unit="1"
        )
        
        # Operation duration histogram
        self.operation_duration = self.meter.create_histogram(
            f"{self.tool.name}_operation_duration",
            description=f"Duration of {self.tool.name} operations",
            unit="s"
        )
        
        # Error counter
        self.operation_errors = self.meter.create_counter(
            f"{self.tool.name}_errors_total",
            description=f"Total number of {self.tool.name} errors",
            unit="1"
        )
    
    def _setup_custom_metrics(self):
        """Override in subclasses to setup tool-specific metrics."""
        pass
    
    @contextmanager
    def trace_operation(self, operation: str, **attributes):
        """Create a trace span for tool operation."""
        if not self.enabled or not self.tracer:
            yield None
            return
            
        with self.tracer.start_as_current_span(
            f"{self.tool.name}.{operation}",
            attributes={
                "tool.name": self.tool.name,
                "tool.version": self.tool.version,
                "tool.operation": operation,
                **attributes
            }
        ) as span:
            yield span
    
    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool,
        **custom_metrics
    ):
        """Record common operation metrics."""
        if not self.enabled:
            return
        
        # Common labels
        labels = {"operation": operation}
        
        # Increment operation counter
        if self.operation_counter:
            self.operation_counter.add(1, labels)
        
        # Record duration
        if self.operation_duration:
            self.operation_duration.record(duration, labels)
        
        # Record errors
        if not success and self.operation_errors:
            self.operation_errors.add(1, labels)
        
        # Record custom metrics
        self._record_custom_metrics(operation, custom_metrics)
    
    def _record_custom_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Override in subclasses to record tool-specific metrics."""
        pass


class SimpleObservability(BaseObservability):
    """Simple observability for tools without complex metrics."""
    
    def __init__(self, tool, config: Dict[str, Any], custom_metrics: Optional[Dict[str, Any]] = None):
        """Initialize with optional custom metric definitions.
        
        Args:
            tool: The tool instance
            config: Observability configuration
            custom_metrics: Dict mapping metric names to their definitions
                Each definition should have:
                - type: "counter", "histogram", or "gauge"
                - description: Metric description
                - unit: Metric unit (optional)
        """
        self.custom_metric_definitions = custom_metrics or {}
        self.custom_metrics = {}
        super().__init__(tool, config)
    
    def _setup_custom_metrics(self):
        """Setup custom metrics from definitions."""
        if not self.meter:
            return
        
        for metric_name, definition in self.custom_metric_definitions.items():
            metric_type = definition["type"]
            description = definition["description"]
            unit = definition.get("unit", "1")
            
            full_name = f"{self.tool.name}_{metric_name}"
            
            if metric_type == "counter":
                self.custom_metrics[metric_name] = self.meter.create_counter(
                    full_name,
                    description=description,
                    unit=unit
                )
            elif metric_type == "histogram":
                self.custom_metrics[metric_name] = self.meter.create_histogram(
                    full_name,
                    description=description,
                    unit=unit
                )
            elif metric_type == "gauge":
                self.custom_metrics[metric_name] = self.meter.create_gauge(
                    full_name,
                    description=description,
                    unit=unit
                )
    
    def _record_custom_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Record custom metrics."""
        for metric_name, value in metrics.items():
            if metric_name in self.custom_metrics:
                metric = self.custom_metrics[metric_name]
                labels = {"operation": operation}
                
                if hasattr(metric, "add"):
                    metric.add(value, labels)
                elif hasattr(metric, "record"):
                    metric.record(value, labels)
                elif hasattr(metric, "set"):
                    metric.set(value, labels)