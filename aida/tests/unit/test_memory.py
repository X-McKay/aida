"""Tests for memory module."""

from datetime import datetime, timedelta

import pytest

from aida.core.memory import (
    Context,
    InMemoryStore,
    MemoryEntry,
    MemoryManager,
    MemoryType,
    Priority,
)


class TestMemoryType:
    """Test MemoryType enum."""

    def test_memory_type_values(self):
        """Test memory type enum values."""
        assert MemoryType.SHORT_TERM.value == "short_term"
        assert MemoryType.LONG_TERM.value == "long_term"
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"


class TestPriority:
    """Test Priority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert Priority.LOW == 1
        assert Priority.MEDIUM == 5
        assert Priority.HIGH == 8
        assert Priority.CRITICAL == 10

    def test_priority_comparison(self):
        """Test priority comparison."""
        assert Priority.LOW < Priority.MEDIUM
        assert Priority.MEDIUM < Priority.HIGH
        assert Priority.HIGH < Priority.CRITICAL


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        now = datetime.utcnow()
        entry = MemoryEntry(
            id="test_123",
            content={"data": "test"},
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.MEDIUM,
            created_at=now,
            accessed_at=now,
            access_count=0,
            tags={"test", "unit"},
            metadata={"source": "test"},
            expires_at=now + timedelta(hours=1),
        )

        assert entry.id == "test_123"
        assert entry.content == {"data": "test"}
        assert entry.memory_type == MemoryType.SHORT_TERM
        assert entry.priority == Priority.MEDIUM
        assert entry.created_at == now
        assert entry.accessed_at == now
        assert entry.access_count == 0
        assert entry.tags == {"test", "unit"}
        assert entry.metadata == {"source": "test"}
        assert entry.expires_at == now + timedelta(hours=1)

    def test_memory_entry_defaults(self):
        """Test memory entry default values."""
        now = datetime.utcnow()
        entry = MemoryEntry(
            id="test",
            content="data",
            memory_type=MemoryType.WORKING,
            priority=Priority.LOW,
            created_at=now,
            accessed_at=now,
            tags=set(),  # Explicitly set defaults
            metadata={},
        )

        assert entry.access_count == 0
        assert entry.tags == set()
        assert entry.metadata == {}
        assert entry.expires_at is None


class TestContext:
    """Test Context class."""

    def test_context_creation(self):
        """Test creating a context."""
        now = datetime.utcnow()
        context = Context(
            id="ctx_123",
            agent_id="agent_456",
            session_id="session_789",
            conversation_id="conv_abc",
            variables={"key": "value"},
            history=[{"action": "test"}],
            metadata={"source": "test"},
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=1),
        )

        assert context.id == "ctx_123"
        assert context.agent_id == "agent_456"
        assert context.session_id == "session_789"
        assert context.conversation_id == "conv_abc"
        assert context.variables == {"key": "value"}
        assert context.history == [{"action": "test"}]
        assert context.metadata == {"source": "test"}
        assert context.created_at == now
        assert context.updated_at == now
        assert context.expires_at == now + timedelta(hours=1)

    def test_context_defaults(self):
        """Test context default values."""
        context = Context(id="ctx_123", agent_id="agent_456")

        assert context.session_id is None
        assert context.conversation_id is None
        assert context.variables == {}
        assert context.history == []
        assert context.metadata == {}
        assert context.expires_at is None

    def test_set_variable(self):
        """Test setting a context variable."""
        context = Context(id="ctx", agent_id="agent")
        original_updated = context.updated_at

        # Wait a tiny bit to ensure updated_at changes
        import time

        time.sleep(0.001)

        context.set_variable("key", "value")

        assert context.variables["key"] == "value"
        assert context.updated_at > original_updated

    def test_get_variable(self):
        """Test getting a context variable."""
        context = Context(id="ctx", agent_id="agent", variables={"existing": "value"})

        assert context.get_variable("existing") == "value"
        assert context.get_variable("missing") is None
        assert context.get_variable("missing", "default") == "default"

    def test_add_to_history(self):
        """Test adding to context history."""
        context = Context(id="ctx", agent_id="agent")
        original_updated = context.updated_at

        # Wait a tiny bit
        import time

        time.sleep(0.001)

        entry = {"action": "test", "data": "value"}
        context.add_to_history(entry)

        assert len(context.history) == 1
        assert context.history[0]["action"] == "test"
        assert context.history[0]["data"] == "value"
        assert "timestamp" in context.history[0]
        assert context.updated_at > original_updated

    def test_get_recent_history(self):
        """Test getting recent history."""
        context = Context(id="ctx", agent_id="agent")

        # Add multiple entries
        for i in range(15):
            context.add_to_history({"index": i})

        # Get recent 10
        recent = context.get_recent_history(10)
        assert len(recent) == 10
        assert recent[0]["index"] == 5  # Should start from index 5
        assert recent[-1]["index"] == 14  # Should end at index 14

        # Get recent 5
        recent = context.get_recent_history(5)
        assert len(recent) == 5
        assert recent[0]["index"] == 10
        assert recent[-1]["index"] == 14

    def test_is_expired(self):
        """Test checking if context is expired."""
        # Not expired - no expiry set
        context = Context(id="ctx", agent_id="agent")
        assert context.is_expired() is False

        # Not expired - future expiry
        context.expires_at = datetime.utcnow() + timedelta(hours=1)
        assert context.is_expired() is False

        # Expired - past expiry
        context.expires_at = datetime.utcnow() - timedelta(hours=1)
        assert context.is_expired() is True

    def test_context_json_encoding(self):
        """Test context JSON encoding."""
        now = datetime.utcnow()
        context = Context(id="ctx", agent_id="agent", created_at=now)

        # Should be able to convert to dict with datetime encoding
        context_dict = context.model_dump()
        assert isinstance(context_dict["created_at"], datetime)

        # Should be able to convert to JSON
        context_json = context.model_dump_json()
        assert isinstance(context_json, str)
        assert "ctx" in context_json
        assert "agent" in context_json


class TestInMemoryStore:
    """Test InMemoryStore class."""

    @pytest.fixture
    def store(self):
        """Create an in-memory store."""
        return InMemoryStore(max_entries=100)

    @pytest.fixture
    def sample_entry(self):
        """Create a sample memory entry."""
        return MemoryEntry(
            id="test_123",
            content="test content",
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.MEDIUM,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            tags=set(),
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_store_entry(self, store, sample_entry):
        """Test storing an entry."""
        await store.store(sample_entry)

        # Should be stored
        assert sample_entry.id in store._entries

    @pytest.mark.asyncio
    async def test_retrieve_by_id(self, store, sample_entry):
        """Test retrieving entry by ID."""
        await store.store(sample_entry)

        retrieved = await store.retrieve(sample_entry.id)
        assert retrieved is not None
        assert retrieved.id == sample_entry.id
        assert retrieved.content == sample_entry.content
        assert retrieved.access_count == 1  # Should increment

        # Non-existent ID
        assert await store.retrieve("non_existent") is None

    @pytest.mark.asyncio
    async def test_search_by_type(self, store):
        """Test searching by memory type."""
        # Add entries of different types
        for i in range(3):
            entry = MemoryEntry(
                id=f"short_{i}",
                content=f"content_{i}",
                memory_type=MemoryType.SHORT_TERM,
                priority=Priority.MEDIUM,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
            )
            await store.store(entry)

        for i in range(2):
            entry = MemoryEntry(
                id=f"long_{i}",
                content=f"content_{i}",
                memory_type=MemoryType.LONG_TERM,
                priority=Priority.HIGH,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
            )
            await store.store(entry)

        # Search by type
        short_term = await store.search(memory_type=MemoryType.SHORT_TERM)
        assert len(short_term) == 3
        assert all(e.memory_type == MemoryType.SHORT_TERM for e in short_term)

        long_term = await store.search(memory_type=MemoryType.LONG_TERM)
        assert len(long_term) == 2
        assert all(e.memory_type == MemoryType.LONG_TERM for e in long_term)

    @pytest.mark.asyncio
    async def test_search_by_tags(self, store):
        """Test searching by tags."""
        # Add entries with different tags
        entry1 = MemoryEntry(
            id="entry1",
            content="content1",
            memory_type=MemoryType.WORKING,
            priority=Priority.MEDIUM,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            tags={"python", "code"},
        )

        entry2 = MemoryEntry(
            id="entry2",
            content="content2",
            memory_type=MemoryType.WORKING,
            priority=Priority.MEDIUM,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            tags={"python", "test"},
        )

        entry3 = MemoryEntry(
            id="entry3",
            content="content3",
            memory_type=MemoryType.WORKING,
            priority=Priority.MEDIUM,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            tags={"javascript", "code"},
        )

        await store.store(entry1)
        await store.store(entry2)
        await store.store(entry3)

        # Search by single tag
        python_entries = await store.search(tags={"python"})
        assert len(python_entries) == 2
        assert all("python" in e.tags for e in python_entries)

        # Search by multiple tags (any match - intersection behavior)
        python_code = await store.search(tags={"python", "code"})
        assert len(python_code) == 3  # All entries match at least one tag
        # entry1 has both, entry2 has python, entry3 has code

    @pytest.mark.asyncio
    async def test_search_with_limit(self, store):
        """Test searching with result limit."""
        # Add 10 entries
        for i in range(10):
            entry = MemoryEntry(
                id=f"entry_{i}",
                content=f"content_{i}",
                memory_type=MemoryType.WORKING,
                priority=Priority.MEDIUM,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
            )
            await store.store(entry)

        # Search with limit
        results = await store.search(memory_type=MemoryType.WORKING, limit=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_delete_entry(self, store, sample_entry):
        """Test deleting an entry."""
        await store.store(sample_entry)

        # Delete
        success = await store.delete(sample_entry.id)
        assert success is True

        # Should be gone
        assert sample_entry.id not in store._entries

        # Delete non-existent
        success = await store.delete("non_existent")
        assert success is False

    # @pytest.mark.asyncio
    # async def test_clear_store(self, store):
    #     """Test clearing all entries."""
    #     # Note: clear() method doesn't exist in current implementation
    #     pass

    @pytest.mark.asyncio
    async def test_eviction_when_full(self, store):
        """Test eviction when store is full."""
        store.max_entries = 3

        # Add entries with different priorities
        low_priority = MemoryEntry(
            id="low",
            content="low",
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.LOW,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
        )

        medium_priority = MemoryEntry(
            id="medium",
            content="medium",
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.MEDIUM,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
        )

        high_priority = MemoryEntry(
            id="high",
            content="high",
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.HIGH,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
        )

        await store.store(low_priority)
        await store.store(medium_priority)
        await store.store(high_priority)

        # Store is full, add one more
        new_entry = MemoryEntry(
            id="new",
            content="new",
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.MEDIUM,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
        )

        await store.store(new_entry)

        # Low priority should be evicted
        assert "low" not in store._entries
        assert "medium" in store._entries
        assert "high" in store._entries
        assert "new" in store._entries

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, store):
        """Test cleaning up expired entries."""
        # Add expired entry
        expired = MemoryEntry(
            id="expired",
            content="expired",
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.MEDIUM,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            expires_at=datetime.utcnow() - timedelta(hours=1),
            tags=set(),
            metadata={},
        )

        # Add non-expired entry
        valid = MemoryEntry(
            id="valid",
            content="valid",
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.MEDIUM,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            tags=set(),
            metadata={},
        )

        await store.store(expired)
        await store.store(valid)

        # Cleanup
        await store.cleanup_expired()

        # Expired should be gone
        assert "expired" not in store._entries
        assert "valid" in store._entries


class TestMemoryManager:
    """Test MemoryManager class."""

    @pytest.fixture
    def manager(self):
        """Create a memory manager."""
        return MemoryManager()

    @pytest.fixture
    def sample_context(self):
        """Create a sample context."""
        return Context(id="ctx_123", agent_id="agent_456")

    def test_initialization(self, manager):
        """Test memory manager initialization."""
        assert isinstance(manager.store, InMemoryStore)
        assert manager._contexts == {}
        # Note: _lock is on the store, not the manager

    @pytest.mark.asyncio
    async def test_store_memory(self, manager):
        """Test storing memory."""
        memory_id = await manager.store_memory(
            content="test content",
            memory_type=MemoryType.SHORT_TERM,
            priority=Priority.MEDIUM,
            tags={"test"},
            metadata={"source": "unit_test"},
        )

        assert memory_id is not None

        # Retrieve to verify
        memory = await manager.retrieve_memory(memory_id)
        assert memory is not None
        assert memory.content == "test content"
        assert memory.memory_type == MemoryType.SHORT_TERM
        assert memory.priority == Priority.MEDIUM
        assert "test" in memory.tags
        assert memory.metadata["source"] == "unit_test"

    @pytest.mark.asyncio
    async def test_store_memory_with_expiry(self, manager):
        """Test storing memory with expiry."""
        memory_id = await manager.store_memory(
            content="expires soon", memory_type=MemoryType.SHORT_TERM, ttl=timedelta(seconds=3600)
        )

        memory = await manager.retrieve_memory(memory_id)
        assert memory.expires_at is not None
        assert memory.expires_at > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, manager):
        """Test recalling memory."""
        # Store a memory
        memory_id = await manager.store_memory(
            content="recall test", memory_type=MemoryType.WORKING
        )

        # Recall
        memory = await manager.retrieve_memory(memory_id)
        assert memory is not None
        assert memory.content == "recall test"
        assert memory.access_count == 1

        # Recall again
        memory = await manager.retrieve_memory(memory_id)
        assert memory.access_count == 2

        # Non-existent
        assert await manager.retrieve_memory("non_existent") is None

    @pytest.mark.asyncio
    async def test_search_memories(self, manager):
        """Test searching memories."""
        # Store memories with different attributes
        await manager.store_memory(
            content="python code", memory_type=MemoryType.SEMANTIC, tags={"python", "code"}
        )

        await manager.store_memory(
            content="python test", memory_type=MemoryType.SEMANTIC, tags={"python", "test"}
        )

        await manager.store_memory(
            content="javascript code", memory_type=MemoryType.SEMANTIC, tags={"javascript", "code"}
        )

        # Search by type
        semantic = await manager.search_memories(memory_type=MemoryType.SEMANTIC)
        assert len(semantic) == 3

        # Search by tags
        python_memories = await manager.search_memories(tags={"python"})
        assert len(python_memories) == 2

        # Search by type and tags
        python_semantic = await manager.search_memories(
            memory_type=MemoryType.SEMANTIC, tags={"python"}
        )
        assert len(python_semantic) == 2

    @pytest.mark.asyncio
    async def test_delete_memory(self, manager):
        """Test forgetting memory."""
        # Store and forget
        memory_id = await manager.store_memory(
            content="forget me", memory_type=MemoryType.SHORT_TERM
        )

        success = await manager.delete_memory(memory_id)
        assert success is True

        # Should be gone
        assert await manager.retrieve_memory(memory_id) is None

        # Forget non-existent
        success = await manager.delete_memory("non_existent")
        assert success is False

    # @pytest.mark.asyncio
    # async def test_clear_memories(self, manager):
    #     """Test clearing memories."""
    #     # Store multiple memories
    #     for i in range(5):
    #         await manager.store_memory(
    #             content=f"memory_{i}",
    #             memory_type=MemoryType.WORKING
    #         )
    #
    #     # Clear specific type
    #     # Note: clear_memories method doesn't exist in current implementation
    #     # await manager.clear_memories(memory_type=MemoryType.WORKING)
    #
    #     # Should be gone
    #     # working_memories = await manager.search_memories(memory_type=MemoryType.WORKING)
    #     # assert len(working_memories) == 0

    @pytest.mark.asyncio
    async def test_create_context(self, manager):
        """Test creating context."""
        context = await manager.create_context(agent_id="agent_123", session_id="session_456")

        assert context.id is not None
        assert context.agent_id == "agent_123"
        assert context.session_id == "session_456"

        # Should be stored
        assert context.id in manager._contexts

    @pytest.mark.asyncio
    async def test_get_context(self, manager, sample_context):
        """Test getting context."""
        # Add context
        manager._contexts[sample_context.id] = sample_context

        # Get existing
        context = await manager.get_context(sample_context.id)
        assert context is sample_context

        # Get non-existent
        assert await manager.get_context("non_existent") is None

    @pytest.mark.asyncio
    async def test_update_context(self, manager, sample_context):
        """Test updating context."""
        # Add context
        manager._contexts[sample_context.id] = sample_context

        # Update
        success = await manager.update_context(
            sample_context.id, {"variables": {"new_var": "value"}, "metadata": {"updated": True}}
        )

        assert success is True
        assert sample_context.variables["new_var"] == "value"
        assert sample_context.metadata["updated"] is True

        # Update non-existent
        success = await manager.update_context("non_existent", {"variables": {"test": "value"}})
        assert success is False

    @pytest.mark.asyncio
    async def test_delete_context(self, manager, sample_context):
        """Test deleting context."""
        # Add and delete
        manager._contexts[sample_context.id] = sample_context

        success = await manager.delete_context(sample_context.id)
        assert success is True
        assert sample_context.id not in manager._contexts

        # Delete non-existent
        success = await manager.delete_context("non_existent")
        assert success is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_contexts(self, manager):
        """Test cleaning up expired contexts."""
        # Add expired context
        expired = Context(
            id="expired", agent_id="agent", expires_at=datetime.utcnow() - timedelta(hours=1)
        )

        # Add valid context
        valid = Context(
            id="valid", agent_id="agent", expires_at=datetime.utcnow() + timedelta(hours=1)
        )

        manager._contexts["expired"] = expired
        manager._contexts["valid"] = valid

        # Cleanup
        cleanup_stats = await manager.cleanup()
        assert cleanup_stats["expired_contexts"] == 1

        # Expired should be gone
        assert "expired" not in manager._contexts
        assert "valid" in manager._contexts


class TestGlobalMemoryManager:
    """Test global memory manager."""

    def test_memory_manager_creation(self):
        """Test creating memory manager instances."""
        manager1 = MemoryManager()
        manager2 = MemoryManager()
        # Each instance is independent
        assert manager1 is not manager2


if __name__ == "__main__":
    pytest.main([__file__])
