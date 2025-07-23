"""Configuration for file operations tool."""

import os
from pathlib import Path
from typing import Any


class FilesConfig:
    """Configuration constants for file operations tool."""

    # File size limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_BATCH_SIZE = 100  # Maximum operations in a batch

    # Buffer sizes
    READ_BUFFER_SIZE = 8192
    WRITE_BUFFER_SIZE = 8192

    # Search limits
    MAX_SEARCH_RESULTS = 1000
    MAX_SEARCH_DEPTH = 10

    # Encoding
    DEFAULT_ENCODING = "utf-8"
    FALLBACK_ENCODINGS = ["utf-8", "latin-1", "ascii"]

    # File patterns to ignore
    IGNORE_PATTERNS = [
        "*.pyc",
        "__pycache__",
        "*.pyo",
        ".git",
        ".svn",
        ".hg",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        "node_modules",
        "venv",
        ".env",
    ]

    # Dangerous operations that require confirmation
    DANGEROUS_OPERATIONS = ["delete", "move", "batch"]

    # File type detection
    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".html",
        ".css",
        ".xml",
        ".json",
        ".yaml",
        ".yml",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".rs",
        ".go",
        ".rb",
        ".php",
        ".swift",
    }

    BINARY_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
    }

    # Security settings
    ALLOWED_BASE_PATHS = [os.path.expanduser("~"), "/tmp", "/var/tmp"]

    FORBIDDEN_PATHS = [
        "/etc",
        "/sys",
        "/proc",
        "/dev",
        "/boot",
        "/root",
        "/bin",
        "/sbin",
        "/usr/bin",
        "/usr/sbin",
    ]

    @classmethod
    def is_safe_path(cls, path: str) -> bool:
        """Check if a path is safe to access."""
        path_obj = Path(path).resolve()

        # Check forbidden paths
        for forbidden in cls.FORBIDDEN_PATHS:
            if str(path_obj).startswith(forbidden):
                return False

        # Check if in allowed base paths
        for allowed in cls.ALLOWED_BASE_PATHS:
            if str(path_obj).startswith(str(Path(allowed).resolve())):
                return True

        return False

    @classmethod
    def should_ignore(cls, path: str) -> bool:
        """Check if a path should be ignored."""
        path_obj = Path(path)
        name = path_obj.name

        for pattern in cls.IGNORE_PATTERNS:
            if pattern.startswith("*."):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True

        return False

    @classmethod
    def is_text_file(cls, path: str) -> bool:
        """Check if a file is likely text based on extension."""
        return Path(path).suffix.lower() in cls.TEXT_EXTENSIONS

    @classmethod
    def is_binary_file(cls, path: str) -> bool:
        """Check if a file is likely binary based on extension."""
        return Path(path).suffix.lower() in cls.BINARY_EXTENSIONS

    # MCP configuration
    @classmethod
    def get_mcp_config(cls) -> dict[str, Any]:
        """Get MCP server configuration."""
        return {
            "name": "aida-files",
            "description": "File and directory operations",
            "version": "2.0.0",
        }

    # Observability configuration
    @classmethod
    def get_observability_config(cls) -> dict[str, Any]:
        """Get observability configuration."""
        return {
            "service_name": "aida-files-tool",
            "trace_enabled": True,
            "metrics_enabled": True,
            "export_endpoint": "http://localhost:4317",
        }
