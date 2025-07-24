"""MCP Executor for managing and executing MCP tools across providers."""

import asyncio
import logging
from typing import Any

from aida.providers.mcp.base import MCPProvider
from aida.providers.mcp.filesystem_client import MCPFilesystemClient


class MCPExecutor:
    """Manages multiple MCP providers and routes tool calls to appropriate providers.

    This executor provides a unified interface for:
    - Managing MCP provider lifecycle (connect/disconnect)
    - Discovering available tools across all providers
    - Routing tool calls to the correct provider
    - Handling provider-specific capabilities
    """

    def __init__(self):
        """Initialize the MCP executor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.providers: dict[str, MCPProvider] = {}
        self.tool_registry: dict[str, str] = {}  # tool_name -> provider_name
        self.resource_registry: dict[str, str] = {}  # resource_uri -> provider_name
        self._connected = False
        self._lock = asyncio.Lock()

    async def add_provider(
        self, provider_name: str, provider_config: dict[str, Any] | None = None
    ) -> None:
        """Add and initialize an MCP provider.

        Args:
            provider_name: Name of the provider (e.g., "filesystem", "searxng_enhanced")
            provider_config: Optional configuration for the provider
        """
        async with self._lock:
            if provider_name in self.providers:
                self.logger.warning(f"Provider '{provider_name}' already exists, skipping")
                return

            # Create provider instance based on name
            provider = self._create_provider(provider_name, provider_config)
            if not provider:
                error_msg = (
                    f"Unknown MCP provider: '{provider_name}'. "
                    f"Available providers: {list(self._get_available_providers().keys())}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Connect to provider
            try:
                await provider.connect()
                self.providers[provider_name] = provider

                # Discover and register tools
                tools = await provider.list_tools()
                for tool in tools:
                    tool_name = tool.get("name", "")
                    if tool_name:
                        if tool_name in self.tool_registry:
                            self.logger.warning(
                                f"Tool '{tool_name}' already registered with provider "
                                f"'{self.tool_registry[tool_name]}', overriding with '{provider_name}'"
                            )
                        self.tool_registry[tool_name] = provider_name

                # Discover and register resources if supported
                if hasattr(provider, "list_resources"):
                    resources = await provider.list_resources()
                    for resource in resources:
                        uri = resource.get("uri", "")
                        if uri:
                            self.resource_registry[uri] = provider_name

                self.logger.info(f"Added provider '{provider_name}' with {len(tools)} tools")

                # Set connected flag if we have at least one provider
                if len(self.providers) > 0:
                    self._connected = True

            except Exception as e:
                self.logger.error(f"Failed to add provider '{provider_name}': {e}")
                raise

    async def remove_provider(self, provider_name: str) -> None:
        """Remove and disconnect an MCP provider.

        Args:
            provider_name: Name of the provider to remove
        """
        async with self._lock:
            if provider_name not in self.providers:
                self.logger.warning(f"Provider '{provider_name}' not found")
                return

            provider = self.providers[provider_name]

            # Disconnect provider
            try:
                await provider.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting provider '{provider_name}': {e}")

            # Remove from registries
            del self.providers[provider_name]

            # Remove tools registered by this provider
            tools_to_remove = [
                tool for tool, prov in self.tool_registry.items() if prov == provider_name
            ]
            for tool in tools_to_remove:
                del self.tool_registry[tool]

            # Remove resources registered by this provider
            resources_to_remove = [
                uri for uri, prov in self.resource_registry.items() if prov == provider_name
            ]
            for uri in resources_to_remove:
                del self.resource_registry[uri]

            self.logger.info(f"Removed provider '{provider_name}'")

    async def execute_mcp_tool(
        self, provider_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool from a specific provider.

        Args:
            provider_name: Name of the provider
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")

        provider = self.providers[provider_name]

        try:
            result = await provider.call_tool(tool_name, arguments)
            self.logger.debug(f"MCPExecutor received result for tool '{tool_name}': {result}")
            return result
        except Exception as e:
            self.logger.error(
                f"Error executing tool '{tool_name}' from provider '{provider_name}': {e}"
            )
            raise

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool, automatically finding the correct provider.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self.is_connected:
            raise RuntimeError(
                "MCP executor is not connected. Ensure at least one provider is connected."
            )

        if tool_name not in self.tool_registry:
            # List available tools for better error message
            available_tools = list(self.tool_registry.keys())
            if not available_tools:
                raise ValueError(
                    f"Tool '{tool_name}' not found. No tools are currently available. "
                    "Check that MCP providers are properly connected."
                )
            else:
                raise ValueError(
                    f"Tool '{tool_name}' not found. Available tools: {available_tools[:10]}"
                    f"{' and more...' if len(available_tools) > 10 else ''}"
                )

        provider_name = self.tool_registry[tool_name]
        return await self.execute_mcp_tool(provider_name, tool_name, arguments)

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource from the appropriate provider.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if uri not in self.resource_registry:
            raise ValueError(f"Resource '{uri}' not found")

        provider_name = self.resource_registry[uri]
        provider = self.providers[provider_name]

        if not hasattr(provider, "read_resource"):
            raise ValueError(f"Provider '{provider_name}' does not support resource reading")

        return await provider.read_resource(uri)

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools across all providers.

        Returns:
            List of tool descriptions
        """
        all_tools = []

        for provider_name, provider in self.providers.items():
            try:
                tools = await provider.list_tools()
                # Add provider info to each tool
                for tool in tools:
                    tool["provider"] = provider_name
                all_tools.extend(tools)
            except Exception as e:
                self.logger.error(f"Error listing tools from provider '{provider_name}': {e}")

        return all_tools

    async def list_resources(self) -> list[dict[str, Any]]:
        """List all available resources across providers that support them.

        Returns:
            List of resource descriptions
        """
        all_resources = []

        for provider_name, provider in self.providers.items():
            if hasattr(provider, "list_resources"):
                try:
                    resources = await provider.list_resources()
                    # Add provider info to each resource
                    for resource in resources:
                        resource["provider"] = provider_name
                    all_resources.extend(resources)
                except Exception as e:
                    self.logger.error(
                        f"Error listing resources from provider '{provider_name}': {e}"
                    )

        return all_resources

    async def get_capabilities(self) -> dict[str, Any]:
        """Get capabilities of all connected providers.

        Returns:
            Dictionary mapping provider names to their capabilities
        """
        capabilities = {}

        for provider_name, provider in self.providers.items():
            if hasattr(provider, "get_capabilities"):
                try:
                    caps = await provider.get_capabilities()
                    capabilities[provider_name] = caps
                except Exception as e:
                    self.logger.error(
                        f"Error getting capabilities from provider '{provider_name}': {e}"
                    )
                    capabilities[provider_name] = {"error": str(e)}
            else:
                capabilities[provider_name] = {"supported": ["tools"]}

        return capabilities

    async def connect_all(self) -> None:
        """Connect all providers that are not yet connected."""
        async with self._lock:
            for provider_name, provider in self.providers.items():
                if not provider._connected:
                    try:
                        await provider.connect()
                        self.logger.info(f"Connected provider '{provider_name}'")
                    except Exception as e:
                        self.logger.error(f"Failed to connect provider '{provider_name}': {e}")

            self._connected = True

    async def disconnect_all(self) -> None:
        """Disconnect all connected providers."""
        async with self._lock:
            for provider_name, provider in self.providers.items():
                if provider._connected:
                    try:
                        await provider.disconnect()
                        self.logger.info(f"Disconnected provider '{provider_name}'")
                    except Exception as e:
                        self.logger.error(f"Error disconnecting provider '{provider_name}': {e}")

            self._connected = False
            self.providers.clear()
            self.tool_registry.clear()
            self.resource_registry.clear()

    def _get_available_providers(self) -> dict[str, type]:
        """Get map of available provider names to classes.

        Returns:
            Dictionary mapping provider names to their classes
        """
        return {
            "filesystem": MCPFilesystemClient,
            # Add more providers as they are implemented:
            # "searxng_enhanced": MCPSearchClient,
            # "github": MCPGitHubClient,
            # etc.
        }

    def _create_provider(
        self, provider_name: str, provider_config: dict[str, Any] | None = None
    ) -> MCPProvider | None:
        """Create a provider instance based on name.

        Args:
            provider_name: Name of the provider
            provider_config: Optional configuration

        Returns:
            Provider instance or None if unknown
        """
        provider_map = self._get_available_providers()

        provider_class = provider_map.get(provider_name)
        if not provider_class:
            return None

        # Create provider with config if provided
        if provider_config:
            return provider_class(**provider_config)
        else:
            return provider_class()

    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """Get detailed information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information or None if not found
        """
        if tool_name not in self.tool_registry:
            return None

        provider_name = self.tool_registry[tool_name]
        provider = self.providers.get(provider_name)

        if not provider:
            return None

        # This would require providers to have a get_tool_info method
        # For now, return basic info
        return {"name": tool_name, "provider": provider_name, "available": provider._connected}

    @property
    def is_connected(self) -> bool:
        """Check if executor has any connected providers."""
        return self._connected and len(self.providers) > 0

    @property
    def provider_count(self) -> int:
        """Get number of registered providers."""
        return len(self.providers)

    @property
    def tool_count(self) -> int:
        """Get total number of available tools."""
        return len(self.tool_registry)

    def __repr__(self) -> str:
        """String representation of the executor."""
        return (
            f"MCPExecutor(providers={list(self.providers.keys())}, "
            f"tools={self.tool_count}, connected={self.is_connected})"
        )
