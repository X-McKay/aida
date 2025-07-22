"""Tests for the event system - fixed version."""

import asyncio
from datetime import datetime
from unittest.mock import Mock

import pytest

from aida.core.events import (
    Event,
    EventBus,
    EventFilter,
    EventSubscription,
    get_event_bus,
)


class TestEvent:
    """Test Event class functionality."""

    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            type="task_started",
            source="test_source",
            data={"task": "test_task"},
            metadata={"priority": "high"},
        )

        assert event.type == "task_started"
        assert event.source == "test_source"
        assert event.data == {"task": "test_task"}
        assert event.metadata == {"priority": "high"}
        assert isinstance(event.timestamp, datetime)
        assert event.id is not None
        assert event.correlation_id is None

    def test_event_defaults(self):
        """Test event default values."""
        event = Event(type="task_started", source="test")

        assert event.data == {}
        assert event.metadata == {}
        assert event.correlation_id is None

    def test_event_with_correlation_id(self):
        """Test event with correlation ID."""
        event = Event(
            type="task_started",
            source="test",
            correlation_id="corr_123",
        )

        assert event.correlation_id == "corr_123"

    def test_event_unique_ids(self):
        """Test that events have unique IDs."""
        event1 = Event(type="test", source="test")
        event2 = Event(type="test", source="test")

        assert event1.id != event2.id

    def test_event_json_serialization(self):
        """Test event JSON serialization."""
        event = Event(
            type="task_started",
            source="test",
            data={"key": "value"},
        )

        # Test that event can be serialized to JSON
        json_str = event.model_dump_json()
        assert "task_started" in json_str
        assert "test" in json_str

        # Test that timestamp is properly encoded
        event_dict = event.model_dump()
        assert isinstance(event_dict["timestamp"], datetime)


class TestEventFilter:
    """Test EventFilter functionality."""

    def test_event_filter_creation(self):
        """Test creating an event filter."""
        filter = EventFilter(
            event_types=["task_started", "task_completed"],
            sources=["agent_1", "agent_2"],
            metadata_filters={"priority": "high"},
            correlation_id="corr_123",
        )

        assert filter.event_types == ["task_started", "task_completed"]
        assert filter.sources == ["agent_1", "agent_2"]
        assert filter.metadata_filters == {"priority": "high"}
        assert filter.correlation_id == "corr_123"

    def test_event_filter_defaults(self):
        """Test event filter default values."""
        filter = EventFilter()

        assert filter.event_types is None
        assert filter.sources is None
        assert filter.metadata_filters == {}
        assert filter.correlation_id is None


class TestEventSubscription:
    """Test EventSubscription functionality."""

    def test_event_subscription_creation(self):
        """Test creating an event subscription."""
        handler = Mock()
        filter = EventFilter(event_types=["task_started"])

        subscription = EventSubscription(
            handler=handler,
            event_filter=filter,
            subscription_id="sub_123",
        )

        assert subscription.id == "sub_123"
        assert subscription.handler == handler
        assert subscription.filter == filter

    def test_subscription_matches_event(self):
        """Test subscription matching logic."""
        handler = Mock()

        # Test with event type filter
        filter = EventFilter(event_types=["task_started"])
        subscription = EventSubscription(handler, filter, "sub_1")

        matching_event = Event(type="task_started", source="test")
        non_matching_event = Event(type="task_completed", source="test")

        assert subscription.matches(matching_event) is True
        assert subscription.matches(non_matching_event) is False

        # Test with source filter
        filter = EventFilter(sources=["agent_1"])
        subscription = EventSubscription(handler, filter, "sub_2")

        matching_event = Event(type="any", source="agent_1")
        non_matching_event = Event(type="any", source="agent_2")

        assert subscription.matches(matching_event) is True
        assert subscription.matches(non_matching_event) is False

    def test_subscription_no_filter(self):
        """Test subscription without filter matches all events."""
        handler = Mock()
        subscription = EventSubscription(handler, None, "sub_1")

        event1 = Event(type="any", source="test1")
        event2 = Event(type="other", source="test2")

        assert subscription.matches(event1) is True
        assert subscription.matches(event2) is True


class TestEventBus:
    """Test EventBus functionality."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for testing."""
        return EventBus()

    def test_subscribe(self, event_bus):
        """Test subscribing to events."""
        handler = Mock()

        # Subscribe with event types
        subscription_id = event_bus.subscribe("task_started", handler)
        assert subscription_id is not None
        assert len(event_bus._subscriptions) == 1

    def test_subscribe_with_filter(self, event_bus):
        """Test subscribing with custom filter."""
        handler = Mock()
        filter = EventFilter(
            event_types=["task_started"], sources=["agent_1"], metadata_filters={"priority": "high"}
        )

        subscription_id = event_bus.subscribe("task_started", handler, filter)
        assert subscription_id is not None
        assert len(event_bus._subscriptions) == 1

    def test_subscribe_multiple_types(self, event_bus):
        """Test subscribing to multiple event types."""
        handler = Mock()

        subscription_id = event_bus.subscribe(["task_started", "task_completed"], handler)
        assert subscription_id is not None
        assert len(event_bus._subscriptions) == 1

    def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        handler = Mock()
        subscription_id = event_bus.subscribe("task_started", handler)

        # Unsubscribe
        success = event_bus.unsubscribe(subscription_id)
        assert success is True
        assert len(event_bus._subscriptions) == 0

        # Try unsubscribing again
        success = event_bus.unsubscribe(subscription_id)
        assert success is False

    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus):
        """Test publishing events to subscribers."""
        handler1_called = False
        handler2_called = False

        async def handler1(event):
            nonlocal handler1_called
            handler1_called = True

        async def handler2(event):
            nonlocal handler2_called
            handler2_called = True

        # Start the event bus
        await event_bus.start()

        # Subscribe handlers
        event_bus.subscribe("task_started", handler1)
        event_bus.subscribe("task_completed", handler2)

        # Publish matching event for handler1
        event = Event(type="task_started", source="test")
        await event_bus.emit(event)

        # Allow async handlers to complete
        await asyncio.sleep(0.01)

        assert handler1_called is True
        assert handler2_called is False

        # Stop the event bus
        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_publish_to_multiple_subscribers(self, event_bus):
        """Test publishing to multiple matching subscribers."""
        call_count = 0

        async def handler(event):
            nonlocal call_count
            call_count += 1

        # Start the event bus
        await event_bus.start()

        # Subscribe multiple handlers to same event type
        event_bus.subscribe("task_started", handler)
        event_bus.subscribe("task_started", handler)

        # Publish event
        event = Event(type="task_started", source="test")
        await event_bus.emit(event)

        # Allow async handlers to complete
        await asyncio.sleep(0.01)

        assert call_count == 2

        # Stop the event bus
        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_publish_with_sync_handler(self, event_bus):
        """Test publishing to synchronous handlers."""
        handler_called = False

        def sync_handler(event):
            nonlocal handler_called
            handler_called = True

        # Start the event bus
        await event_bus.start()

        event_bus.subscribe("task_started", sync_handler)

        # Publish event
        event = Event(type="task_started", source="test")
        await event_bus.emit(event)

        # Allow handlers to complete
        await asyncio.sleep(0.01)

        assert handler_called is True

        # Stop the event bus
        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_error_handling_in_handler(self, event_bus):
        """Test that errors in handlers don't break event bus."""
        good_handler_called = False

        async def failing_handler(event):
            raise Exception("Handler error")

        async def good_handler(event):
            nonlocal good_handler_called
            good_handler_called = True

        # Start the event bus
        await event_bus.start()

        # Subscribe both handlers
        event_bus.subscribe("test", failing_handler)
        event_bus.subscribe("test", good_handler)

        # Publish event - should not raise
        event = Event(type="test", source="test")
        await event_bus.emit(event)

        # Allow handlers to complete
        await asyncio.sleep(0.01)

        # Good handler should still be called
        assert good_handler_called is True

        # Stop the event bus
        await event_bus.stop()

    def test_clear_subscriptions(self, event_bus):
        """Test clearing all subscriptions."""
        # Note: EventBus doesn't have a clear() method
        # We can test that unsubscribe works for individual subscriptions
        handler1 = Mock()
        handler2 = Mock()

        sub1 = event_bus.subscribe("event1", handler1)
        sub2 = event_bus.subscribe("event2", handler2)

        assert len(event_bus._subscriptions) == 2

        # Unsubscribe both
        event_bus.unsubscribe(sub1)
        event_bus.unsubscribe(sub2)
        assert len(event_bus._subscriptions) == 0


class TestGlobalEventBus:
    """Test global event bus functionality."""

    def test_get_event_bus_singleton(self):
        """Test that get_event_bus returns singleton."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_clear_and_reuse(self):
        """Test clearing and reusing the global event bus."""
        # Get bus and add subscription
        bus = get_event_bus()
        handler = Mock()
        sub_id = bus.subscribe("test_event", handler)
        initial_sub_count = len(bus._subscriptions)

        # Unsubscribe to clear
        bus.unsubscribe(sub_id)

        # Should have one less subscription
        assert len(bus._subscriptions) == initial_sub_count - 1
        assert initial_sub_count > 0  # Verify we had subscriptions before


if __name__ == "__main__":
    pytest.main([__file__])
