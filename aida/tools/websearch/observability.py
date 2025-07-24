"""Observability for web search tool."""

import time
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .models import SearchCategory, SearchOperation
from .websearch import WebSearchTool


class WebSearchObservability:
    """Observability wrapper for web search tool."""

    def __init__(self, tool: WebSearchTool, config: dict[str, Any]):
        """Initialize observability with tool and config."""
        self.tool = tool
        self.config = config

        # Initialize tracer
        self.tracer = trace.get_tracer(
            "aida.tools.websearch", tool.version, schema_url="https://aida.dev/schemas/1.0.0"
        )

        # Initialize metrics
        meter = metrics.get_meter(
            "aida.tools.websearch", tool.version, schema_url="https://aida.dev/schemas/1.0.0"
        )

        # Counters
        self.search_counter = meter.create_counter(
            name="websearch.searches.total",
            unit="1",
            description="Total number of web searches performed",
        )

        self.error_counter = meter.create_counter(
            name="websearch.errors.total",
            unit="1",
            description="Total number of web search errors",
        )

        self.scraped_pages_counter = meter.create_counter(
            name="websearch.scraped_pages.total",
            unit="1",
            description="Total number of pages scraped",
        )

        # Histograms
        self.search_duration = meter.create_histogram(
            name="websearch.search.duration",
            unit="s",
            description="Duration of web search operations",
        )

        self.result_count = meter.create_histogram(
            name="websearch.results.count",
            unit="1",
            description="Number of results returned per search",
        )

        self.content_size = meter.create_histogram(
            name="websearch.content.size",
            unit="By",
            description="Size of scraped content in bytes",
        )

    async def execute_with_observability(self, **kwargs) -> Any:
        """Execute tool with observability."""
        # Extract operation info
        operation = kwargs.get("operation", SearchOperation.SEARCH)
        category = kwargs.get("category", SearchCategory.GENERAL)

        # Start span
        with self.tracer.start_as_current_span(
            f"websearch.{operation}", kind=trace.SpanKind.CLIENT
        ) as span:
            # Set span attributes
            span.set_attribute("websearch.operation", operation)
            span.set_attribute("websearch.category", category)

            if "query" in kwargs:
                span.set_attribute("websearch.query", kwargs["query"])
            if "url" in kwargs:
                span.set_attribute("websearch.url", kwargs["url"])

            start_time = time.time()

            try:
                # Execute the actual tool
                result = await self.tool.execute(**kwargs)

                # Record metrics
                duration = time.time() - start_time
                self.search_duration.record(
                    duration, {"operation": operation, "category": category}
                )

                # Record operation-specific metrics
                if operation == SearchOperation.SEARCH:
                    self.search_counter.add(1, {"category": category})

                    if result.result and "results" in result.result:
                        result_count = len(result.result["results"])
                        self.result_count.record(result_count, {"category": category})
                        span.set_attribute("websearch.result_count", result_count)

                    if kwargs.get("scrape_content") and "scraped_content" in result.result:
                        scraped_count = len(result.result["scraped_content"])
                        self.scraped_pages_counter.add(scraped_count, {"category": category})
                        span.set_attribute("websearch.scraped_pages", scraped_count)

                elif operation == SearchOperation.GET_WEBSITE:
                    if result.result and "website_content" in result.result:
                        content = result.result["website_content"]
                        content_size = len(content.get("content", "").encode("utf-8"))
                        self.content_size.record(content_size)
                        span.set_attribute("websearch.content_size", content_size)

                # Set success status
                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                # Record error
                self.error_counter.add(1, {"operation": operation, "error_type": type(e).__name__})

                # Set error status on span
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)

                # Re-raise the exception
                raise

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        # This is a placeholder - actual implementation would query the metrics backend
        return {
            "total_searches": 0,
            "total_errors": 0,
            "total_scraped_pages": 0,
            "average_search_duration": 0.0,
            "average_result_count": 0.0,
            "average_content_size": 0.0,
        }
