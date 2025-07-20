"""Event system for AIDA agents."""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
import uuid
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class Event(BaseModel):
    """Event object for the AIDA event system."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class EventFilter(BaseModel):
    """Event filter for subscriptions."""
    
    event_types: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None


EventHandler = Callable[[Event], Any]


class EventSubscription:
    """Event subscription with handler and filter."""
    
    def __init__(
        self, 
        handler: EventHandler, 
        event_filter: Optional[EventFilter] = None,
        subscription_id: Optional[str] = None
    ):
        self.id = subscription_id or str(uuid.uuid4())
        self.handler = handler
        self.filter = event_filter
        self.created_at = datetime.utcnow()
        self.event_count = 0
        self.last_event_at: Optional[datetime] = None
    
    def matches(self, event: Event) -> bool:
        """Check if event matches subscription filter."""
        if not self.filter:
            return True
        
        # Check event types
        if self.filter.event_types and event.type not in self.filter.event_types:
            return False
        
        # Check sources
        if self.filter.sources and event.source not in self.filter.sources:
            return False
        
        # Check correlation ID
        if self.filter.correlation_id and event.correlation_id != self.filter.correlation_id:
            return False
        
        # Check metadata filters
        for key, value in self.filter.metadata_filters.items():
            if event.metadata.get(key) != value:
                return False
        
        return True
    
    async def handle_event(self, event: Event) -> None:
        """Handle an event."""
        try:
            self.event_count += 1
            self.last_event_at = datetime.utcnow()
            
            # Call handler
            if asyncio.iscoroutinefunction(self.handler):
                await self.handler(event)
            else:
                self.handler(event)
                
        except Exception as e:
            logger.error(f"Error in event handler {self.id}: {e}")


class EventBus:
    """Event bus for AIDA system."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Subscriptions
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._type_subscriptions: Dict[str, Set[str]] = {}  # event_type -> subscription_ids
        
        # Event history
        self._event_history: List[Event] = []
        
        # Statistics
        self._stats = {
            "events_emitted": 0,
            "events_handled": 0,
            "handler_errors": 0,
            "subscriptions_count": 0
        }
        
        # Async handling
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the event bus."""
        if self._processor_task is None:
            self._processor_task = asyncio.create_task(self._process_events())
            logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus."""
        if self._processor_task:
            self._shutdown_event.set()
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
            logger.info("Event bus stopped")
    
    async def emit(self, event: Event) -> None:
        """Emit an event."""
        try:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self.max_history:
                self._event_history.pop(0)
            
            # Queue for processing
            await self._event_queue.put(event)
            self._stats["events_emitted"] += 1
            
            logger.debug(f"Event emitted: {event.type} from {event.source}")
            
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
    
    def subscribe(
        self, 
        event_types: str | List[str], 
        handler: EventHandler,
        event_filter: Optional[EventFilter] = None,
        subscription_id: Optional[str] = None
    ) -> str:
        """Subscribe to events."""
        # Normalize event types
        if isinstance(event_types, str):
            event_types = [event_types]
        
        # Create filter if not provided
        if event_filter is None:
            event_filter = EventFilter(event_types=event_types)
        elif event_filter.event_types is None:
            event_filter.event_types = event_types
        
        # Create subscription
        subscription = EventSubscription(
            handler=handler,
            event_filter=event_filter,
            subscription_id=subscription_id
        )
        
        # Store subscription
        self._subscriptions[subscription.id] = subscription
        
        # Index by event types
        for event_type in event_types:
            if event_type not in self._type_subscriptions:
                self._type_subscriptions[event_type] = set()
            self._type_subscriptions[event_type].add(subscription.id)
        
        self._stats["subscriptions_count"] = len(self._subscriptions)
        
        logger.debug(f"Subscription {subscription.id} created for events: {event_types}")
        return subscription.id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            return False
        
        # Remove from type index
        if subscription.filter and subscription.filter.event_types:
            for event_type in subscription.filter.event_types:
                if event_type in self._type_subscriptions:
                    self._type_subscriptions[event_type].discard(subscription_id)
                    if not self._type_subscriptions[event_type]:
                        del self._type_subscriptions[event_type]
        
        # Remove subscription
        del self._subscriptions[subscription_id]
        self._stats["subscriptions_count"] = len(self._subscriptions)
        
        logger.info(f"Subscription {subscription_id} removed")
        return True
    
    def get_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all subscriptions."""
        return [
            {
                "id": sub.id,
                "filter": sub.filter.dict() if sub.filter else None,
                "created_at": sub.created_at.isoformat(),
                "event_count": sub.event_count,
                "last_event_at": sub.last_event_at.isoformat() if sub.last_event_at else None
            }
            for sub in self._subscriptions.values()
        ]
    
    def get_event_history(self, limit: Optional[int] = None) -> List[Event]:
        """Get event history."""
        if limit is None:
            return self._event_history.copy()
        else:
            return self._event_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return self._stats.copy()
    
    async def wait_for_event(
        self, 
        event_type: str, 
        timeout: float = 30.0,
        event_filter: Optional[EventFilter] = None
    ) -> Optional[Event]:
        """Wait for a specific event."""
        future = asyncio.get_event_loop().create_future()
        
        def handler(event: Event):
            if not future.done():
                future.set_result(event)
        
        # Subscribe temporarily
        subscription_id = self.subscribe(
            event_type, 
            handler, 
            event_filter
        )
        
        try:
            # Wait for event
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            return None
            
        finally:
            # Cleanup subscription
            self.unsubscribe(subscription_id)
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while not self._shutdown_event.is_set():
            try:
                # Get event from queue
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                # Find matching subscriptions
                matching_subscriptions = self._find_matching_subscriptions(event)
                
                # Handle event for each subscription
                for subscription in matching_subscriptions:
                    try:
                        await subscription.handle_event(event)
                        self._stats["events_handled"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error handling event in subscription {subscription.id}: {e}")
                        self._stats["handler_errors"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _find_matching_subscriptions(self, event: Event) -> List[EventSubscription]:
        """Find subscriptions that match the event."""
        matching = []
        
        # Get subscriptions by event type
        subscription_ids = self._type_subscriptions.get(event.type, set())
        
        # Also check subscriptions with no type filter
        for sub_id, subscription in self._subscriptions.items():
            if (subscription.filter is None or 
                subscription.filter.event_types is None or 
                sub_id in subscription_ids):
                
                if subscription.matches(event):
                    matching.append(subscription)
        
        return matching


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


async def emit_event(event: Event) -> None:
    """Emit an event to the global event bus."""
    event_bus = get_event_bus()
    await event_bus.emit(event)


def subscribe_to_events(
    event_types: str | List[str],
    handler: EventHandler,
    event_filter: Optional[EventFilter] = None
) -> str:
    """Subscribe to events on the global event bus."""
    event_bus = get_event_bus()
    return event_bus.subscribe(event_types, handler, event_filter)