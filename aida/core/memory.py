"""Memory and context management for AIDA agents."""

import asyncio
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Memory type enumeration."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class Priority(int, Enum):
    """Memory priority levels."""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class MemoryEntry:
    """Memory entry data structure."""
    id: str
    content: Any
    memory_type: MemoryType
    priority: Priority
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    tags: Set[str] = None
    metadata: Dict[str, Any] = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.metadata is None:
            self.metadata = {}


class Context(BaseModel):
    """Context object for agent interactions."""
    
    id: str
    agent_id: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)
    
    def add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add an entry to context history."""
        entry["timestamp"] = datetime.utcnow().isoformat()
        self.history.append(entry)
        self.updated_at = datetime.utcnow()
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent history entries."""
        return self.history[-limit:]
    
    def is_expired(self) -> bool:
        """Check if context is expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at


class MemoryStore:
    """Abstract memory store interface."""
    
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        raise NotImplementedError
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        raise NotImplementedError
    
    async def search(
        self, 
        query: str = None,
        memory_type: MemoryType = None,
        tags: Set[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memory entries."""
        raise NotImplementedError
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        raise NotImplementedError
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memory entries."""
        raise NotImplementedError
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """In-memory implementation of memory store."""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._entries: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()
    
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        async with self._lock:
            # Enforce size limit
            if len(self._entries) >= self.max_entries:
                await self._evict_oldest()
            
            self._entries[entry.id] = entry
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        async with self._lock:
            entry = self._entries.get(memory_id)
            if entry:
                entry.accessed_at = datetime.utcnow()
                entry.access_count += 1
            return entry
    
    async def search(
        self, 
        query: str = None,
        memory_type: MemoryType = None,
        tags: Set[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memory entries."""
        async with self._lock:
            results = []
            
            for entry in self._entries.values():
                # Filter by memory type
                if memory_type and entry.memory_type != memory_type:
                    continue
                
                # Filter by tags
                if tags and not tags.intersection(entry.tags):
                    continue
                
                # Simple text search in content
                if query:
                    content_str = str(entry.content).lower()
                    if query.lower() not in content_str:
                        continue
                
                results.append(entry)
                
                if len(results) >= limit:
                    break
            
            # Sort by priority and recency
            results.sort(key=lambda x: (x.priority.value, x.accessed_at), reverse=True)
            return results
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        async with self._lock:
            return self._entries.pop(memory_id, None) is not None
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memory entries."""
        async with self._lock:
            now = datetime.utcnow()
            expired_ids = [
                entry_id for entry_id, entry in self._entries.items()
                if entry.expires_at and now > entry.expires_at
            ]
            
            for entry_id in expired_ids:
                del self._entries[entry_id]
            
            return len(expired_ids)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        async with self._lock:
            stats = {
                "total_entries": len(self._entries),
                "max_entries": self.max_entries,
                "memory_usage": sum(len(str(entry.content)) for entry in self._entries.values()),
                "by_type": {},
                "by_priority": {}
            }
            
            # Count by type and priority
            for entry in self._entries.values():
                # By type
                type_key = entry.memory_type.value
                stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1
                
                # By priority
                priority_key = entry.priority.name
                stats["by_priority"][priority_key] = stats["by_priority"].get(priority_key, 0) + 1
            
            return stats
    
    async def _evict_oldest(self) -> None:
        """Evict the oldest, least accessed entries."""
        if not self._entries:
            return
        
        # Sort by access count and time
        entries_by_score = sorted(
            self._entries.items(),
            key=lambda x: (x[1].access_count, x[1].accessed_at)
        )
        
        # Remove lowest scoring entries
        evict_count = max(1, len(self._entries) // 10)  # Remove 10%
        for i in range(evict_count):
            if i < len(entries_by_score):
                entry_id = entries_by_score[i][0]
                del self._entries[entry_id]


class MemoryManager:
    """Memory manager for AIDA agents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Memory store
        store_type = self.config.get("store_type", "memory")
        if store_type == "memory":
            max_entries = self.config.get("max_entries", 10000)
            self.store = InMemoryStore(max_entries)
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
        
        # Context management
        self._contexts: Dict[str, Context] = {}
        self._context_lock = asyncio.Lock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Configuration
        self.cleanup_interval = timedelta(minutes=self.config.get("cleanup_interval_minutes", 30))
        self.default_context_ttl = timedelta(hours=self.config.get("default_context_ttl_hours", 24))
        
        # Statistics
        self._stats = {
            "memories_created": 0,
            "memories_retrieved": 0,
            "memories_deleted": 0,
            "contexts_created": 0,
            "contexts_expired": 0
        }
    
    async def start(self) -> None:
        """Start the memory manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.debug("Memory manager started")
    
    async def stop(self) -> None:
        """Stop the memory manager."""
        self._shutdown_event.set()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.debug("Memory manager stopped")
    
    async def store_memory(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.WORKING,
        priority: Priority = Priority.MEDIUM,
        tags: Set[str] = None,
        metadata: Dict[str, Any] = None,
        ttl: Optional[timedelta] = None
    ) -> str:
        """Store a memory entry."""
        import uuid
        
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            priority=priority,
            created_at=now,
            accessed_at=now,
            tags=tags or set(),
            metadata=metadata or {},
            expires_at=now + ttl if ttl else None
        )
        
        await self.store.store(entry)
        self._stats["memories_created"] += 1
        
        logger.debug(f"Memory stored: {memory_id} (type: {memory_type.value})")
        return memory_id
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry."""
        entry = await self.store.retrieve(memory_id)
        if entry:
            self._stats["memories_retrieved"] += 1
        return entry
    
    async def search_memories(
        self,
        query: str = None,
        memory_type: MemoryType = None,
        tags: Set[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memory entries."""
        return await self.store.search(query, memory_type, tags, limit)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        deleted = await self.store.delete(memory_id)
        if deleted:
            self._stats["memories_deleted"] += 1
        return deleted
    
    async def create_context(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        ttl: Optional[timedelta] = None
    ) -> Context:
        """Create a new context."""
        import uuid
        
        context_id = str(uuid.uuid4())
        ttl = ttl or self.default_context_ttl
        
        context = Context(
            id=context_id,
            agent_id=agent_id,
            session_id=session_id,
            conversation_id=conversation_id,
            expires_at=datetime.utcnow() + ttl if ttl else None
        )
        
        async with self._context_lock:
            self._contexts[context_id] = context
        
        self._stats["contexts_created"] += 1
        logger.debug(f"Context created: {context_id} for agent {agent_id}")
        
        return context
    
    async def get_context(self, context_id: str) -> Optional[Context]:
        """Get a context by ID."""
        async with self._context_lock:
            context = self._contexts.get(context_id)
            
            if context and context.is_expired():
                del self._contexts[context_id]
                self._stats["contexts_expired"] += 1
                return None
            
            return context
    
    async def update_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """Update a context."""
        async with self._context_lock:
            context = self._contexts.get(context_id)
            if not context or context.is_expired():
                return False
            
            for key, value in updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)
            
            context.updated_at = datetime.utcnow()
            return True
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete a context."""
        async with self._context_lock:
            return self._contexts.pop(context_id, None) is not None
    
    async def get_agent_contexts(self, agent_id: str) -> List[Context]:
        """Get all contexts for an agent."""
        async with self._context_lock:
            return [
                ctx for ctx in self._contexts.values()
                if ctx.agent_id == agent_id and not ctx.is_expired()
            ]
    
    async def compact_conversation(
        self,
        context_id: str,
        compression_ratio: float = 0.5
    ) -> bool:
        """Compact conversation history in a context."""
        context = await self.get_context(context_id)
        if not context:
            return False
        
        history_len = len(context.history)
        if history_len <= 10:  # Don't compress small histories
            return True
        
        # Keep most recent entries and important ones
        keep_count = max(5, int(history_len * compression_ratio))
        
        # Sort by importance (can be enhanced with ML)
        important_entries = sorted(
            context.history,
            key=lambda x: len(str(x.get("content", ""))),  # Simple: longer = more important
            reverse=True
        )[:keep_count//2]
        
        # Keep most recent entries
        recent_entries = context.history[-(keep_count//2):]
        
        # Combine and sort by timestamp
        combined = list(set(important_entries + recent_entries))
        combined.sort(key=lambda x: x.get("timestamp", ""))
        
        context.history = combined
        context.updated_at = datetime.utcnow()
        
        logger.info(f"Context {context_id} history compressed: {history_len} -> {len(combined)}")
        return True
    
    async def cleanup(self) -> Dict[str, int]:
        """Clean up expired memories and contexts."""
        # Clean up expired memories
        expired_memories = await self.store.cleanup_expired()
        
        # Clean up expired contexts
        async with self._context_lock:
            now = datetime.utcnow()
            expired_context_ids = [
                ctx_id for ctx_id, ctx in self._contexts.items()
                if ctx.expires_at and now > ctx.expires_at
            ]
            
            for ctx_id in expired_context_ids:
                del self._contexts[ctx_id]
                self._stats["contexts_expired"] += 1
        
        expired_contexts = len(expired_context_ids)
        
        logger.debug(f"Cleanup completed: {expired_memories} memories, {expired_contexts} contexts")
        
        return {
            "expired_memories": expired_memories,
            "expired_contexts": expired_contexts
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        store_stats = await self.store.get_stats()
        
        async with self._context_lock:
            context_stats = {
                "total_contexts": len(self._contexts),
                "contexts_by_agent": {},
            }
            
            for ctx in self._contexts.values():
                agent_id = ctx.agent_id
                context_stats["contexts_by_agent"][agent_id] = \
                    context_stats["contexts_by_agent"].get(agent_id, 0) + 1
        
        return {
            **self._stats,
            "store": store_stats,
            "contexts": context_stats
        }
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                await self.cleanup()
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait a minute on error