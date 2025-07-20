"""Configuration for system tool."""

from typing import Dict, List, Any
import os
import sys


class SystemConfig:
    """Configuration constants for system tool."""
    
    # Execution settings
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_TIMEOUT = 300  # 5 minutes
    DEFAULT_SHELL = "/bin/bash" if sys.platform != "win32" else "cmd.exe"
    
    # Output limits
    MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB
    OUTPUT_TRUNCATE_MSG = "\n... [OUTPUT TRUNCATED] ..."
    
    # Allowed commands (whitelist)
    ALLOWED_COMMANDS = {
        # System info
        "uname", "hostname", "whoami", "date", "uptime",
        "df", "du", "free", "ps", "top", "htop",
        
        # File operations
        "ls", "pwd", "cat", "head", "tail", "wc",
        "grep", "find", "which", "file", "stat",
        
        # Network
        "ping", "curl", "wget", "netstat", "ss",
        "nslookup", "dig", "host",
        
        # Development
        "git", "python", "python3", "pip", "pip3",
        "node", "npm", "yarn", "go", "cargo",
        "gcc", "g++", "make", "cmake",
        
        # Text processing
        "echo", "printf", "sed", "awk", "cut",
        "sort", "uniq", "tr", "jq",
        
        # Archive
        "tar", "zip", "unzip", "gzip", "gunzip",
        
        # Shell interpreters (for script execution)
        "bash", "sh", "python", "python3"
    }
    
    # Forbidden commands (blacklist)
    FORBIDDEN_COMMANDS = {
        "rm", "rmdir", "dd", "mkfs", "fdisk",
        "shutdown", "reboot", "halt", "poweroff",
        "passwd", "useradd", "userdel", "usermod",
        "su", "sudo", "chown", "chmod",
        "kill", "killall", "pkill",
        "systemctl", "service", "init"
    }
    
    # Dangerous patterns in commands
    DANGEROUS_PATTERNS = [
        r">\s*/dev/",  # Writing to device files
        r";\s*rm\s+-rf",  # Command injection with rm -rf
        r"&&\s*rm\s+-rf",  # Command chaining with rm -rf
        r"\|\s*sudo",  # Piping to sudo
        r"`.*`",  # Command substitution
        r"\$\(.*\)",  # Command substitution
        r"curl.*\|\s*sh",  # Curl pipe to shell
        r"wget.*\|\s*sh",  # Wget pipe to shell
    ]
    
    # Environment variables to filter
    FILTERED_ENV_VARS = {
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "SECRET_KEY",
        "API_KEY",
        "PASSWORD",
        "PRIVATE_KEY"
    }
    
    # Safe environment variables
    SAFE_ENV_VARS = {
        "PATH", "HOME", "USER", "SHELL", "TERM",
        "LANG", "LC_ALL", "PWD", "TMPDIR",
        "PYTHONPATH", "NODE_PATH", "GOPATH"
    }
    
    @classmethod
    def is_command_allowed(cls, command: str) -> bool:
        """Check if a command is allowed."""
        # Extract base command
        base_command = command.split()[0] if command else ""
        base_command = os.path.basename(base_command)
        
        # Check blacklist first
        if base_command in cls.FORBIDDEN_COMMANDS:
            return False
        
        # Check whitelist
        return base_command in cls.ALLOWED_COMMANDS
    
    @classmethod
    def filter_env_vars(cls, env: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive environment variables."""
        filtered = {}
        for key, value in env.items():
            if key.upper() in cls.FILTERED_ENV_VARS:
                filtered[key] = "***FILTERED***"
            elif any(pattern in key.upper() for pattern in ["SECRET", "TOKEN", "KEY", "PASSWORD"]):
                filtered[key] = "***FILTERED***"
            else:
                filtered[key] = value
        return filtered
    
    # Process management
    ALLOWED_SIGNALS = ["TERM", "INT", "HUP", "USR1", "USR2"]
    
    # Script execution
    ALLOWED_INTERPRETERS = {
        "python": ["python", "python3"],
        "bash": ["bash", "sh"],
        "node": ["node"],
        "ruby": ["ruby"],
        "perl": ["perl"]
    }
    
    # MCP configuration
    @classmethod
    def get_mcp_config(cls) -> Dict[str, Any]:
        """Get MCP server configuration."""
        return {
            "name": "aida-system",
            "description": "Secure system command execution",
            "version": "2.0.0"
        }
    
    # Observability configuration
    @classmethod
    def get_observability_config(cls) -> Dict[str, Any]:
        """Get observability configuration."""
        return {
            "service_name": "aida-system-tool",
            "trace_enabled": True,
            "metrics_enabled": True,
            "export_endpoint": "http://localhost:4317"
        }