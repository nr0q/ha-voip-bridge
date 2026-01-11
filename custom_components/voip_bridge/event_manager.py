"""Event queue management for VoIP Bridge."""
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    STORAGE_KEY,
    STORAGE_VERSION,
    DEFAULT_STARTUP_CALL_DELAY,
    PRIORITY_CRITICAL,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class OutboundEvent:
    """Represents an event that may trigger an outbound call."""
    event_id: str
    priority: int
    message: str
    timestamp: datetime | None = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventManager:
    """Manages the event queue and persistence."""
    
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize event manager."""
        self.hass = hass
        self.entry = entry
        self._active_events: dict[str, OutboundEvent] = {}
        self._call_in_progress = False
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}_{entry.entry_id}")
    
    async def async_setup(self) -> None:
        """Load events from storage and setup."""
        await self._restore_events()
        
        # Check if we should call after restore
        if self._should_call():
            _LOGGER.warning(
                f"Restored {len(self._active_events)} events, "
                f"will attempt call after startup delay"
            )
            # Schedule delayed call
            startup_delay = DEFAULT_STARTUP_CALL_DELAY
            
            async def delayed_call():
                await asyncio.sleep(startup_delay)
                if self._should_call():
                    _LOGGER.info("Attempting outbound call for pending events")
                    # TODO: Trigger call
            
            self.hass.async_create_task(delayed_call())
    
    async def _restore_events(self) -> None:
        """Restore events from storage."""
        stored_data = await self._store.async_load()
        
        if stored_data and "events" in stored_data:
            for event_data in stored_data["events"]:
                # Convert timestamp string back to datetime
                if isinstance(event_data.get("timestamp"), str):
                    event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                
                event = OutboundEvent(**event_data)
                self._active_events[event.event_id] = event
            
            _LOGGER.info(f"Restored {len(self._active_events)} events from storage")
    
    async def _persist_events(self) -> None:
        """Save events to storage."""
        events_data = []
        for event in self._active_events.values():
            event_dict = asdict(event)
            # Convert datetime to ISO string
            if event_dict["timestamp"]:
                event_dict["timestamp"] = event_dict["timestamp"].isoformat()
            events_data.append(event_dict)
        
        await self._store.async_save({"events": events_data})
    
    def _should_call(self) -> bool:
        """Determine if we should initiate an outbound call."""
        if self._call_in_progress:
            return False
        
        # Call if any priority 1 (critical) events
        if any(e.priority == PRIORITY_CRITICAL for e in self._active_events.values()):
            return True
        
        # Call if 3+ priority 2 (warning) events
        warning_count = sum(1 for e in self._active_events.values() if e.priority == 2)
        if warning_count >= 3:
            return True
        
        return False
    
    async def add_event(self, event: OutboundEvent) -> None:
        """Add event to queue."""
        _LOGGER.info(f"Adding event: {event.event_id} (priority {event.priority})")
        self._active_events[event.event_id] = event
        await self._persist_events()
        
        # Update sensor state
        await self._update_sensor()
        
        # Check if we should call
        if self._should_call():
            _LOGGER.warning(f"Event threshold met, initiating outbound call")
            # TODO: Initiate call
    
    async def clear_event(self, event_id: str) -> None:
        """Clear event from queue."""
        if event_id in self._active_events:
            event = self._active_events.pop(event_id)
            _LOGGER.info(f"Cleared event: {event_id}")
            await self._persist_events()
            await self._update_sensor()
    
    async def clear_all(self) -> None:
        """Clear all events from queue."""
        count = len(self._active_events)
        self._active_events.clear()
        _LOGGER.info(f"Cleared all {count} events")
        await self._persist_events()
        await self._update_sensor()
    
    def get_events(self) -> list[dict[str, Any]]:
        """Get list of active events."""
        return [
            {
                "event_id": e.event_id,
                "priority": e.priority,
                "message": e.message,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in self._active_events.values()
        ]
    
    async def _update_sensor(self) -> None:
        """Update the sensor state."""
        self.hass.states.async_set(
            f"sensor.voip_bridge_active_events",
            len(self._active_events),
            attributes={"events": self.get_events()}
        )
    
    async def async_shutdown(self) -> None:
        """Cleanup on shutdown."""
        _LOGGER.info("Event manager shutting down")
        # Final persist
        await self._persist_events()