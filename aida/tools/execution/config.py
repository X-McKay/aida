"""Configuration for execution tool."""

from typing import Dict, List, Any
from aida.llm import Purpose


class ExecutionConfig:
    """Configuration constants for execution tool."""
    
    # Execution settings
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_TIMEOUT = 300  # 5 minutes
    DEFAULT_MEMORY_LIMIT = "512m"
    MAX_MEMORY_LIMIT = "2g"
    
    # Container settings
    CONTAINER_WORK_DIR = "/workspace"
    CONTAINER_REGISTRY = "docker.io"
    
    # Language configurations
    LANGUAGE_IMAGES = {
        "python": "python:3.11-slim",
        "javascript": "node:18-slim",
        "node": "node:18-slim",
        "bash": "alpine:latest",
        "go": "golang:1.21-alpine",
        "rust": "rust:1.73-slim",
        "java": "openjdk:17-slim"
    }
    
    LANGUAGE_COMMANDS = {
        "python": ["python", "-u"],
        "javascript": ["node"],
        "node": ["node"],
        "bash": ["sh", "-c"],
        "go": ["go", "run"],
        "rust": ["cargo", "run"],
        "java": ["java"]
    }
    
    LANGUAGE_FILE_EXTENSIONS = {
        "python": ".py",
        "javascript": ".js",
        "node": ".js",
        "bash": ".sh",
        "go": ".go",
        "rust": ".rs",
        "java": ".java"
    }
    
    # Package management
    PACKAGE_MANAGERS = {
        "python": {
            "command": ["pip", "install"],
            "file": "requirements.txt"
        },
        "javascript": {
            "command": ["npm", "install"],
            "file": "package.json"
        },
        "node": {
            "command": ["npm", "install"],
            "file": "package.json"
        },
        "go": {
            "command": ["go", "get"],
            "file": "go.mod"
        },
        "rust": {
            "command": ["cargo", "add"],
            "file": "Cargo.toml"
        }
    }
    
    # Security settings
    ALLOWED_ENV_VARS = [
        "HOME", "USER", "PATH", "LANG", "LC_ALL",
        "PYTHONPATH", "NODE_PATH", "GOPATH"
    ]
    
    BLOCKED_PATTERNS = [
        "subprocess.call", "subprocess.run", "os.system",
        "eval(", "exec(", "__import__",
        "open(", "file(",  # Restrict file operations
    ]
    
    # MCP configuration
    @classmethod
    def get_mcp_config(cls) -> Dict[str, Any]:
        """Get MCP server configuration."""
        return {
            "name": "aida-execution",
            "description": "Execute code in secure containers",
            "version": "2.0.0"
        }
    
    # Observability configuration
    @classmethod
    def get_observability_config(cls) -> Dict[str, Any]:
        """Get observability configuration."""
        return {
            "service_name": "aida-execution-tool",
            "trace_enabled": True,
            "metrics_enabled": True,
            "export_endpoint": "http://localhost:4317"
        }