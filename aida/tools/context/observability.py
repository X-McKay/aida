"""OpenTelemetry observability for context tool."""

from contextlib import contextmanager
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextObservability:
    """OpenTelemetry observability for context operations."""

    def __init__(self, context_tool, config: dict[str, Any]):
        """Initialize context observability with tool instance and configuration."""
        self.context_tool = context_tool
        self.config = config
        self.enabled = config.get("trace_enabled", True)
        self.tracer = None
        self.meter = None

        if self.enabled:
            self._setup_observability()

    def _setup_observability(self):
        """Setup OpenTelemetry components."""
        try:
            from opentelemetry import metrics, trace

            # Setup tracer
            self.tracer = trace.get_tracer(self.config.get("service_name", "aida-context-tool"))

            # Setup metrics if enabled
            if self.config.get("metrics_enabled", True):
                self.meter = metrics.get_meter(self.config.get("service_name", "aida-context-tool"))
                self._setup_metrics()

            logger.debug("OpenTelemetry observability enabled for context tool")

        except ImportError:
            logger.warning(
                "OpenTelemetry not available. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk"
            )
            self.enabled = False

    def _setup_metrics(self):
        """Setup custom metrics."""
        if not self.meter:
            return

        # Counter for operations
        self.operation_counter = self.meter.create_counter(
            "context_operations_total", description="Total number of context operations", unit="1"
        )

        # Histogram for compression ratios
        self.compression_ratio_histogram = self.meter.create_histogram(
            "context_compression_ratio", description="Compression ratios achieved", unit="ratio"
        )

        # Histogram for processing time
        self.processing_time_histogram = self.meter.create_histogram(
            "context_processing_time",
            description="Processing time for context operations",
            unit="s",
        )

        # Gauge for content size
        self.content_size_gauge = self.meter.create_gauge(
            "context_content_size", description="Size of processed content"
        )

    @contextmanager
    def trace_operation(self, operation: str, content_size: int, **attributes):
        """Create a trace span for context operation."""
        if not self.enabled or not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(
            f"context_{operation}",
            attributes={
                "context.operation": operation,
                "context.content_size": content_size,
                "tool.name": self.context_tool.name,
                "tool.version": self.context_tool.version,
                **attributes,
            },
        ) as span:
            yield span

    def record_operation(
        self,
        operation: str,
        processing_time: float,
        content_size: int,
        compression_ratio: float | None = None,
    ):
        """Record metrics for an operation."""
        if not self.enabled:
            return

        # Update counter
        if self.operation_counter:
            self.operation_counter.add(1, {"operation": operation})

        # Update histograms
        if self.processing_time_histogram:
            self.processing_time_histogram.record(processing_time, {"operation": operation})

        if compression_ratio and self.compression_ratio_histogram:
            self.compression_ratio_histogram.record(compression_ratio, {"operation": operation})

        # Update gauge
        if self.content_size_gauge:
            self.content_size_gauge.set(content_size, {"operation": operation})

    @property
    def compression_ratio(self):
        """Get compression ratio metric for compatibility."""
        return self.compression_ratio_histogram

    @property
    def token_reduction(self):
        """Get token reduction metric for compatibility."""
        # Create a virtual metric for token reduction
        if not hasattr(self, "_token_reduction"):
            if self.meter:
                self._token_reduction = self.meter.create_histogram(
                    "context_token_reduction",
                    description="Token reduction percentage",
                    unit="percent",
                )
            else:
                self._token_reduction = None
        return self._token_reduction

    def record_compression(self, ratio: float):
        """Record compression ratio."""
        if self.compression_ratio_histogram:
            self.compression_ratio_histogram.record(ratio)

    def record_token_reduction(self, percentage: float):
        """Record token reduction percentage."""
        if self.token_reduction:
            self.token_reduction.record(percentage)
