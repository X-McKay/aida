"""OpenTelemetry observability for file operations tool."""

import logging
from typing import Any

from aida.tools.base_observability import SimpleObservability

logger = logging.getLogger(__name__)


class FilesObservability(SimpleObservability):
    """OpenTelemetry observability for file operations."""

    def __init__(self, files_tool, config: dict[str, Any]):
        """Initialize file operations observability with custom metrics."""
        custom_metrics = {
            "file_operations_total": {
                "type": "counter",
                "description": "Total number of file operations",
                "unit": "1",
            },
            "bytes_read": {
                "type": "counter",
                "description": "Total bytes read from files",
                "unit": "bytes",
            },
            "bytes_written": {
                "type": "counter",
                "description": "Total bytes written to files",
                "unit": "bytes",
            },
            "files_processed": {
                "type": "histogram",
                "description": "Number of files processed per operation",
                "unit": "files",
            },
            "operation_size": {
                "type": "histogram",
                "description": "Size of data processed in operations",
                "unit": "bytes",
            },
        }
        super().__init__(files_tool, config, custom_metrics)
