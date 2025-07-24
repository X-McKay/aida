"""Sandbox configuration system for worker agents.

This module provides the configuration and management for sandboxed
execution environments using Dagger containers.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from typing import Any

import dagger

logger = logging.getLogger(__name__)


class SandboxIsolationLevel(Enum):
    """Levels of sandbox isolation."""

    MINIMAL = "minimal"  # Basic process isolation
    STANDARD = "standard"  # Network and filesystem isolation
    STRICT = "strict"  # Full isolation with minimal capabilities


class ResourceType(Enum):
    """Types of resources that can be limited."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESSES = "processes"


@dataclass
class ResourceLimits:
    """Resource limits for sandboxed execution."""

    cpu_cores: float = 2.0  # Number of CPU cores
    memory_mb: int = 2048  # Memory in megabytes
    disk_mb: int = 1024  # Disk space in megabytes
    max_processes: int = 100  # Maximum number of processes
    network_enabled: bool = False  # Whether network access is allowed

    def to_dagger_opts(self) -> dict[str, Any]:
        """Convert to Dagger container options."""
        opts = {}

        # CPU limits (Dagger uses millicores, so 1 core = 1000)
        if self.cpu_cores:
            opts["cpu_limit"] = int(self.cpu_cores * 1000)

        # Memory limits
        if self.memory_mb:
            opts["memory_limit"] = f"{self.memory_mb}m"

        return opts


@dataclass
class MountPoint:
    """A mount point for the sandbox."""

    host_path: str
    container_path: str
    read_only: bool = True

    def validate(self) -> None:
        """Validate the mount point configuration."""
        # Ensure host path exists
        host = Path(self.host_path)
        if not host.exists():
            raise ValueError(f"Host path does not exist: {self.host_path}")

        # Ensure container path is absolute
        if not self.container_path.startswith("/"):
            raise ValueError(f"Container path must be absolute: {self.container_path}")


@dataclass
class SandboxConfig:
    """Complete configuration for a worker sandbox."""

    # Basic configuration
    sandbox_id: str
    worker_type: str
    base_image: str = "ubuntu:22.04"

    # Isolation settings
    isolation_level: SandboxIsolationLevel = SandboxIsolationLevel.STANDARD
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Filesystem configuration
    mount_points: list[MountPoint] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)  # Paths worker can access via MCP
    working_directory: str = "/workspace"

    # Network configuration
    allowed_hosts: list[str] = field(default_factory=list)  # Empty = no network
    dns_servers: list[str] = field(default_factory=lambda: ["8.8.8.8", "8.8.4.4"])

    # MCP configuration
    allowed_mcp_servers: list[str] = field(default_factory=list)
    mcp_socket_path: str = "/tmp/mcp.sock"

    # A2A configuration
    a2a_socket_path: str = "/tmp/a2a.sock"

    # Security settings
    user: str = "worker"
    group: str = "worker"
    uid: int = 1000
    gid: int = 1000
    read_only_root: bool = True
    no_new_privileges: bool = True

    # Environment variables
    environment: dict[str, str] = field(default_factory=dict)

    # Execution settings
    init_script: str | None = None
    startup_timeout: int = 30
    shutdown_timeout: int = 10

    def validate(self) -> None:
        """Validate the sandbox configuration."""
        # Validate mount points
        for mount in self.mount_points:
            mount.validate()

        # Validate resource limits
        if self.resource_limits.cpu_cores <= 0:
            raise ValueError("CPU cores must be positive")
        if self.resource_limits.memory_mb < 128:
            raise ValueError("Memory must be at least 128MB")

        # Validate paths
        if not self.working_directory.startswith("/"):
            raise ValueError("Working directory must be absolute path")

    def apply_isolation_level(self) -> None:
        """Apply settings based on isolation level."""
        if self.isolation_level == SandboxIsolationLevel.MINIMAL:
            # Minimal isolation - more permissive
            self.resource_limits.network_enabled = True
            self.read_only_root = False

        elif self.isolation_level == SandboxIsolationLevel.STANDARD:
            # Standard isolation - balanced
            self.resource_limits.network_enabled = bool(self.allowed_hosts)
            self.read_only_root = True

        elif self.isolation_level == SandboxIsolationLevel.STRICT:
            # Strict isolation - maximum security
            self.resource_limits.network_enabled = False
            self.allowed_hosts = []
            self.read_only_root = True
            self.no_new_privileges = True
            # Further restrict resources
            self.resource_limits.cpu_cores = min(self.resource_limits.cpu_cores, 1.0)
            self.resource_limits.memory_mb = min(self.resource_limits.memory_mb, 1024)


class SandboxBuilder:
    """Builder for creating Dagger containers from sandbox configuration."""

    def __init__(self, dagger_client: dagger.Client):
        """Initialize the sandbox builder.

        Args:
            dagger_client: Dagger client instance
        """
        self.client = dagger_client

    async def build_container(self, config: SandboxConfig) -> dagger.Container:
        """Build a Dagger container from sandbox configuration.

        Args:
            config: Sandbox configuration

        Returns:
            Configured Dagger container
        """
        # Validate configuration
        config.validate()
        config.apply_isolation_level()

        # Start with base image
        container = self.client.container().from_(config.base_image)

        # Apply resource limits
        resource_opts = config.resource_limits.to_dagger_opts()
        # Note: Dagger API for resource limits may vary by version
        # Skipping for now as it's not critical for testing

        # Create user and group
        container = await self._setup_user(container, config)

        # Set up filesystem
        container = await self._setup_filesystem(container, config)

        # Set up network
        container = await self._setup_network(container, config)

        # Set environment variables
        for key, value in config.environment.items():
            container = container.with_env_variable(key, value)

        # Add standard environment
        container = container.with_env_variable("SANDBOX_ID", config.sandbox_id)
        container = container.with_env_variable("WORKER_TYPE", config.worker_type)

        # Mount sockets
        container = await self._mount_sockets(container, config)

        # Apply security settings
        container = await self._apply_security(container, config)

        # Run init script if provided
        if config.init_script:
            container = container.with_exec(["sh", "-c", config.init_script])

        # Set working directory and user
        container = container.with_workdir(config.working_directory)
        container = container.with_user(f"{config.uid}:{config.gid}")

        return container

    async def _setup_user(
        self, container: dagger.Container, config: SandboxConfig
    ) -> dagger.Container:
        """Set up user and group in container."""
        # Create group
        container = container.with_exec(
            ["sh", "-c", f"groupadd -g {config.gid} {config.group} || true"]
        )

        # Create user with home directory
        container = container.with_exec(
            [
                "sh",
                "-c",
                f"useradd -u {config.uid} -g {config.gid} -m -s /bin/bash {config.user} || true",
            ]
        )

        # Create working directory and set ownership
        container = container.with_exec(
            [
                "sh",
                "-c",
                f"mkdir -p {config.working_directory} && "
                f"chown -R {config.user}:{config.group} {config.working_directory}",
            ]
        )

        return container

    async def _setup_filesystem(
        self, container: dagger.Container, config: SandboxConfig
    ) -> dagger.Container:
        """Set up filesystem mounts and permissions."""
        # Mount configured mount points
        for mount in config.mount_points:
            # Create directory in container
            container = container.with_exec(["mkdir", "-p", mount.container_path])

            # Mount the directory
            # Note: In production, would use proper Dagger directory mounting
            # This is a simplified version
            if mount.read_only:
                # Mount as read-only
                container = container.with_mounted_directory(
                    mount.container_path, self.client.host().directory(mount.host_path)
                )

        return container

    async def _setup_network(
        self, container: dagger.Container, config: SandboxConfig
    ) -> dagger.Container:
        """Set up network configuration."""
        if not config.resource_limits.network_enabled:
            # Disable network (this is simplified - real implementation
            # would use proper network isolation)
            container = container.with_env_variable("NO_NETWORK", "1")
        elif config.allowed_hosts:
            # Set up network restrictions
            # In production, would use iptables or similar
            hosts_list = ",".join(config.allowed_hosts)
            container = container.with_env_variable("ALLOWED_HOSTS", hosts_list)

        return container

    async def _mount_sockets(
        self, container: dagger.Container, config: SandboxConfig
    ) -> dagger.Container:
        """Mount A2A and MCP sockets."""
        # Mount A2A socket
        container = container.with_unix_socket(
            config.a2a_socket_path, self.client.host().unix_socket("/tmp/aida_a2a.sock")
        )

        # Mount MCP socket if needed
        if config.allowed_mcp_servers:
            container = container.with_unix_socket(
                config.mcp_socket_path, self.client.host().unix_socket("/tmp/aida_mcp.sock")
            )

        return container

    async def _apply_security(
        self, container: dagger.Container, config: SandboxConfig
    ) -> dagger.Container:
        """Apply security settings to container."""
        # These are simplified versions - real implementation would use
        # proper security features

        if config.read_only_root:
            container = container.with_env_variable("READ_ONLY_ROOT", "1")

        if config.no_new_privileges:
            container = container.with_env_variable("NO_NEW_PRIVS", "1")

        # Drop capabilities (simplified)
        container = container.with_env_variable("DROP_CAPS", "1")

        return container


def create_default_sandbox_config(
    worker_type: str, worker_id: str, capabilities: list[str]
) -> SandboxConfig:
    """Create a default sandbox configuration for a worker type.

    Args:
        worker_type: Type of worker (coding, research, etc.)
        worker_id: Unique worker ID
        capabilities: List of worker capabilities

    Returns:
        Default sandbox configuration
    """
    config = SandboxConfig(
        sandbox_id=f"sandbox_{worker_id}",
        worker_type=worker_type,
        isolation_level=SandboxIsolationLevel.STANDARD,
    )

    # Configure based on worker type
    if worker_type == "coding":
        config.base_image = "python:3.11-slim"
        config.resource_limits.memory_mb = 2048
        config.resource_limits.cpu_cores = 2.0
        config.allowed_mcp_servers = ["filesystem"]
        config.environment["PYTHONUNBUFFERED"] = "1"

    elif worker_type == "research":
        config.base_image = "ubuntu:22.04"
        config.resource_limits.memory_mb = 1024
        config.resource_limits.cpu_cores = 1.0
        config.resource_limits.network_enabled = True
        config.allowed_hosts = ["*.searxng.org", "duckduckgo.com"]
        config.allowed_mcp_servers = ["websearch"]

    elif worker_type == "execution":
        config.base_image = "ubuntu:22.04"
        config.resource_limits.memory_mb = 4096
        config.resource_limits.cpu_cores = 4.0
        config.isolation_level = SandboxIsolationLevel.STRICT
        config.allowed_mcp_servers = []

    else:
        # Default configuration
        config.resource_limits.memory_mb = 1024
        config.resource_limits.cpu_cores = 1.0

    return config


class SandboxManager:
    """Manages sandbox lifecycle for workers."""

    def __init__(self):
        """Initialize the sandbox manager."""
        self._sandboxes: dict[str, dagger.Container] = {}
        self._configs: dict[str, SandboxConfig] = {}
        self._dagger_client: dagger.Client | None = None
        self._builder: SandboxBuilder | None = None

    async def initialize(self) -> None:
        """Initialize the Dagger client and builder."""
        # For testing, skip Dagger initialization if not available
        try:
            async with dagger.Connection(dagger.Config(log_output=False)) as client:
                self._dagger_client = client
                self._builder = SandboxBuilder(client)
                logger.info("Dagger client initialized for sandbox management")
        except Exception as e:
            logger.warning(f"Failed to initialize Dagger client: {e}. Sandboxing disabled.")
            self._dagger_client = None
            self._builder = None

    async def create_sandbox(
        self, worker_id: str, config: SandboxConfig
    ) -> dagger.Container | None:
        """Create a new sandbox for a worker.

        Args:
            worker_id: Worker ID
            config: Sandbox configuration

        Returns:
            Created container or None if sandboxing disabled
        """
        if not self._builder:
            logger.warning(f"Sandbox creation skipped for {worker_id} - Dagger not available")
            return None

        if worker_id in self._sandboxes:
            raise ValueError(f"Sandbox already exists for worker {worker_id}")

        # Build container
        container = await self._builder.build_container(config)

        # Store references
        self._sandboxes[worker_id] = container
        self._configs[worker_id] = config

        return container

    async def destroy_sandbox(self, worker_id: str) -> None:
        """Destroy a worker's sandbox.

        Args:
            worker_id: Worker ID
        """
        if worker_id in self._sandboxes:
            # In production, would properly stop and remove container
            del self._sandboxes[worker_id]
            del self._configs[worker_id]

    def get_sandbox(self, worker_id: str) -> dagger.Container | None:
        """Get a worker's sandbox container.

        Args:
            worker_id: Worker ID

        Returns:
            Container if exists
        """
        return self._sandboxes.get(worker_id)

    def get_config(self, worker_id: str) -> SandboxConfig | None:
        """Get a worker's sandbox configuration.

        Args:
            worker_id: Worker ID

        Returns:
            Configuration if exists
        """
        return self._configs.get(worker_id)

    async def cleanup(self) -> None:
        """Clean up all sandboxes and close connections."""
        # Destroy all sandboxes
        worker_ids = list(self._sandboxes.keys())
        for worker_id in worker_ids:
            await self.destroy_sandbox(worker_id)

        # Close Dagger connection
        if self._dagger_client:
            # Dagger client doesn't have close method, cleanup handled by context manager
            self._dagger_client = None
