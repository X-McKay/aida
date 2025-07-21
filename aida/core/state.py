"""State management for AIDA agents."""

import asyncio
from collections.abc import Callable
from datetime import datetime
from enum import Enum
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent status enumeration."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentState(BaseModel):
    """Agent state model."""

    agent_id: str
    status: AgentStatus = AgentStatus.INITIALIZING
    started_at: datetime | None = None
    last_heartbeat: datetime | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    memory_usage: dict[str, Any] = Field(default_factory=dict)
    error_count: int = 0
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class SystemState(BaseModel):
    """System-wide state model."""

    system_id: str
    agents: dict[str, AgentState] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    active_connections: int = 0
    system_metrics: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class StateStore:
    """Abstract state store interface."""

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        raise NotImplementedError

    async def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        raise NotImplementedError

    async def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys matching pattern."""
        raise NotImplementedError

    async def clear(self) -> None:
        """Clear all data."""
        raise NotImplementedError


class MemoryStateStore(StateStore):
    """In-memory state store implementation."""

    def __init__(self):
        """Initialize in-memory state store with thread-safe access."""
        self._data: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        async with self._lock:
            return self._data.get(key)

    async def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        async with self._lock:
            self._data[key] = value

    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        async with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            return key in self._data

    async def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys matching pattern."""
        async with self._lock:
            if pattern is None:
                return list(self._data.keys())

            # Simple pattern matching (supports * wildcard)
            import fnmatch

            return [key for key in self._data if fnmatch.fnmatch(key, pattern)]

    async def clear(self) -> None:
        """Clear all data."""
        async with self._lock:
            self._data.clear()


class RedisStateStore(StateStore):
    """Redis-based state store implementation."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis state store.

        Args:
            redis_url: Redis connection URL (default: redis://localhost:6379)
        """
        self.redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis  # type: ignore[import]

                self._redis = await redis.from_url(self.redis_url)
            except ImportError:
                raise ImportError(
                    "Redis support requires 'redis' package. Install with: pip install redis"
                )
        return self._redis

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        redis = await self._get_redis()
        data = await redis.get(key)
        if data:
            return json.loads(data)
        return None

    async def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        redis = await self._get_redis()
        data = json.dumps(value, default=str)
        await redis.set(key, data)

    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        redis = await self._get_redis()
        result = await redis.delete(key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        redis = await self._get_redis()
        return await redis.exists(key) > 0

    async def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys matching pattern."""
        redis = await self._get_redis()
        pattern = pattern or "*"
        keys = await redis.keys(pattern)
        return [key.decode() for key in keys]

    async def clear(self) -> None:
        """Clear all data."""
        redis = await self._get_redis()
        await redis.flushdb()


class StateManager:
    """State manager for AIDA system."""

    def __init__(self, store: StateStore | None = None):
        """Initialize state manager with specified store.

        Args:
            store: State store implementation (defaults to MemoryStateStore)
        """
        self.store = store or MemoryStateStore()
        self._agent_states: dict[str, AgentState] = {}
        self._system_state: SystemState | None = None
        self._lock = asyncio.Lock()

        # Change tracking
        self._change_subscribers: list[Callable] = []
        self._last_updated: dict[str, datetime] = {}

    async def initialize_system(self, system_id: str) -> None:
        """Initialize system state."""
        async with self._lock:
            self._system_state = SystemState(system_id=system_id)
            await self.store.set(f"system:{system_id}", self._system_state.dict())
            logger.info(f"System state initialized: {system_id}")

    async def register_agent(self, agent_state: AgentState) -> None:
        """Register an agent state."""
        async with self._lock:
            self._agent_states[agent_state.agent_id] = agent_state
            await self.store.set(f"agent:{agent_state.agent_id}", agent_state.dict())

            # Update system state
            if self._system_state:
                self._system_state.agents[agent_state.agent_id] = agent_state
                await self.store.set(
                    f"system:{self._system_state.system_id}", self._system_state.dict()
                )

            self._last_updated[agent_state.agent_id] = datetime.utcnow()
            await self._notify_change("agent_registered", agent_state.agent_id)

            logger.info(f"Agent state registered: {agent_state.agent_id}")

    async def update_agent_state(self, agent_id: str, updates: dict[str, Any]) -> None:
        """Update agent state."""
        async with self._lock:
            agent_state = self._agent_states.get(agent_id)
            if not agent_state:
                logger.error(f"Agent state not found: {agent_id}")
                return

            # Apply updates
            for key, value in updates.items():
                if hasattr(agent_state, key):
                    setattr(agent_state, key, value)

            # Persist changes
            await self.store.set(f"agent:{agent_id}", agent_state.dict())

            # Update system state
            if self._system_state and agent_id in self._system_state.agents:
                self._system_state.agents[agent_id] = agent_state
                await self.store.set(
                    f"system:{self._system_state.system_id}", self._system_state.dict()
                )

            self._last_updated[agent_id] = datetime.utcnow()
            await self._notify_change("agent_updated", agent_id)

    async def get_agent_state(self, agent_id: str) -> AgentState | None:
        """Get agent state."""
        async with self._lock:
            # Try memory first
            if agent_id in self._agent_states:
                return self._agent_states[agent_id]

            # Try persistent store
            data = await self.store.get(f"agent:{agent_id}")
            if data and isinstance(data, dict):
                # Create with explicit agent_id to satisfy type checker
                agent_state = AgentState(
                    agent_id=data.get("agent_id", agent_id),
                    status=data.get("status", AgentStatus.INITIALIZING),
                    started_at=data.get("started_at"),
                    last_heartbeat=data.get("last_heartbeat"),
                    tasks_completed=data.get("tasks_completed", 0),
                    tasks_failed=data.get("tasks_failed", 0),
                    memory_usage=data.get("memory_usage", {}),
                    error_count=data.get("error_count", 0),
                    last_error=data.get("last_error"),
                    metadata=data.get("metadata", {}),
                )
                self._agent_states[agent_id] = agent_state
                return agent_state

            return None

    async def get_system_state(self) -> SystemState | None:
        """Get system state."""
        async with self._lock:
            if self._system_state:
                return self._system_state

            # Try to load from store
            keys = await self.store.keys("system:*")
            if keys:
                data = await self.store.get(keys[0])
                if data and isinstance(data, dict):
                    # Extract system_id from key or use default
                    system_id = (
                        keys[0].replace("system:", "")
                        if keys[0].startswith("system:")
                        else "default"
                    )
                    # Create with explicit system_id to satisfy type checker
                    self._system_state = SystemState(
                        system_id=data.get("system_id", system_id),
                        agents=data.get("agents", {}),
                        started_at=data.get("started_at", datetime.utcnow()),
                        total_tasks_completed=data.get("total_tasks_completed", 0),
                        total_tasks_failed=data.get("total_tasks_failed", 0),
                        active_connections=data.get("active_connections", 0),
                        system_metrics=data.get("system_metrics", {}),
                    )
                    return self._system_state

            return None

    async def get_all_agent_states(self) -> list[AgentState]:
        """Get all agent states."""
        async with self._lock:
            # Load any missing states from store
            agent_keys = await self.store.keys("agent:*")
            for key in agent_keys:
                agent_id = key.split(":", 1)[1]
                if agent_id not in self._agent_states:
                    data = await self.store.get(key)
                    if data and isinstance(data, dict):
                        # Create with explicit agent_id to satisfy type checker
                        self._agent_states[agent_id] = AgentState(
                            agent_id=data.get("agent_id", agent_id),
                            status=data.get("status", AgentStatus.INITIALIZING),
                            started_at=data.get("started_at"),
                            last_heartbeat=data.get("last_heartbeat"),
                            tasks_completed=data.get("tasks_completed", 0),
                            tasks_failed=data.get("tasks_failed", 0),
                            memory_usage=data.get("memory_usage", {}),
                            error_count=data.get("error_count", 0),
                            last_error=data.get("last_error"),
                            metadata=data.get("metadata", {}),
                        )

            return list(self._agent_states.values())

    async def remove_agent(self, agent_id: str) -> bool:
        """Remove agent state."""
        async with self._lock:
            # Remove from memory
            removed = self._agent_states.pop(agent_id, None) is not None

            # Remove from store
            await self.store.delete(f"agent:{agent_id}")

            # Update system state
            if self._system_state and agent_id in self._system_state.agents:
                del self._system_state.agents[agent_id]
                await self.store.set(
                    f"system:{self._system_state.system_id}", self._system_state.dict()
                )

            if removed:
                self._last_updated.pop(agent_id, None)
                await self._notify_change("agent_removed", agent_id)
                logger.info(f"Agent state removed: {agent_id}")

            return removed

    async def update_system_metrics(self, metrics: dict[str, Any]) -> None:
        """Update system metrics."""
        async with self._lock:
            if not self._system_state:
                return

            self._system_state.system_metrics.update(metrics)
            await self.store.set(
                f"system:{self._system_state.system_id}", self._system_state.dict()
            )
            await self._notify_change("system_metrics_updated", self._system_state.system_id)

    async def get_agent_stats(self) -> dict[str, Any]:
        """Get aggregated agent statistics."""
        agent_states = await self.get_all_agent_states()

        stats = {
            "total_agents": len(agent_states),
            "agents_by_status": {},
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "total_errors": 0,
        }

        for state in agent_states:
            # Count by status
            status = state.status.value
            stats["agents_by_status"][status] = stats["agents_by_status"].get(status, 0) + 1

            # Aggregate metrics
            stats["total_tasks_completed"] += state.tasks_completed
            stats["total_tasks_failed"] += state.tasks_failed
            stats["total_errors"] += state.error_count

        return stats

    def subscribe_to_changes(self, callback: Callable) -> None:
        """Subscribe to state changes."""
        self._change_subscribers.append(callback)

    def unsubscribe_from_changes(self, callback: Callable) -> None:
        """Unsubscribe from state changes."""
        if callback in self._change_subscribers:
            self._change_subscribers.remove(callback)

    async def _notify_change(self, change_type: str, entity_id: str) -> None:
        """Notify subscribers of state changes."""
        for callback in self._change_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(change_type, entity_id)
                else:
                    callback(change_type, entity_id)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")


# Global state manager instance
_global_state_manager: StateManager | None = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = StateManager()
    # Type narrowing with cast
    from typing import cast

    return cast(StateManager, _global_state_manager)
