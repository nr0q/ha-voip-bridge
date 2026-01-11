"""VoIP Bridge integration for Home Assistant."""
import asyncio
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType
import voluptuous as vol

from .const import (
    DOMAIN,
    CONF_SIP_SERVER,
    CONF_SIP_PORT,
    CONF_SIP_USERNAME,
    CONF_SIP_PASSWORD,
    CONF_SIP_EXTENSION,
    CONF_ASSIST_PIPELINE,
    CONF_REQUIRE_PIN_INTERNAL,
    CONF_REQUIRE_PIN_EXTERNAL,
    CONF_PIN,
    CONF_TRUSTED_CALLER_IDS,
    CONF_OUTBOUND_DESTINATIONS,
    CONF_CODEC,
    CONF_SAMPLE_RATE,
    CONF_VAD_AGGRESSIVENESS,
    CONF_SILENCE_TIMEOUT,
    SERVICE_ADD_EVENT,
    SERVICE_CLEAR_EVENT,
    SERVICE_CLEAR_ALL_EVENTS,
    SERVICE_TRIGGER_CALL,
    SERVICE_LIST_EVENTS,
    PRIORITY_CRITICAL,
    PRIORITY_WARNING,
    PRIORITY_INFO,
    DEFAULT_SIP_PORT,
    DEFAULT_CODEC,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VAD_AGGRESSIVENESS,
    DEFAULT_SILENCE_TIMEOUT,
)
from .event_manager import EventManager, OutboundEvent
from .sip_client import SIPClient
from .audio_bridge import AudioBridge
from .call_session import CallSession, CallSessionConfig

_LOGGER = logging.getLogger(__name__)

# Service schemas
SERVICE_ADD_EVENT_SCHEMA = vol.Schema({
    vol.Required("event_id"): cv.string,
    vol.Required("priority"): vol.In([PRIORITY_CRITICAL, PRIORITY_WARNING, PRIORITY_INFO]),
    vol.Required("message"): cv.string,
})

SERVICE_CLEAR_EVENT_SCHEMA = vol.Schema({
    vol.Required("event_id"): cv.string,
})

SERVICE_TRIGGER_CALL_SCHEMA = vol.Schema({
    vol.Optional("force"): cv.boolean,
})


class VoIPBridgeCoordinator:
    """Coordinator for VoIP Bridge components."""
    
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize coordinator."""
        self.hass = hass
        self.entry = entry
        
        # Components
        self.event_manager: EventManager | None = None
        self.sip_client: SIPClient | None = None
        self.audio_bridge: AudioBridge | None = None
        self.current_session: CallSession | None = None
        
    async def async_setup(self) -> None:
        """Set up all components."""
        # Initialize event manager
        self.event_manager = EventManager(self.hass, self.entry)
        await self.event_manager.async_setup()
        
        # Initialize audio bridge
        self.audio_bridge = AudioBridge(
            self.hass,
            sample_rate=int(self.entry.data.get(CONF_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)),
            codec=self.entry.data.get(CONF_CODEC, DEFAULT_CODEC),
            energy_threshold=self.entry.data.get(CONF_VAD_AGGRESSIVENESS, DEFAULT_VAD_AGGRESSIVENESS),
            silence_timeout=self.entry.data.get(CONF_SILENCE_TIMEOUT, DEFAULT_SILENCE_TIMEOUT),
        )
        
        # Set up audio callback
        self.audio_bridge.set_speech_complete_callback(self._on_speech_complete)
        
        # Initialize SIP client
        self.sip_client = SIPClient(
            self.hass,
            server=self.entry.data[CONF_SIP_SERVER],
            port=self.entry.data.get(CONF_SIP_PORT, DEFAULT_SIP_PORT),
            username=self.entry.data[CONF_SIP_USERNAME],
            password=self.entry.data[CONF_SIP_PASSWORD],
            extension=self.entry.data[CONF_SIP_EXTENSION],
            sample_rate=int(self.entry.data.get(CONF_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)),
        )
        
        # Set up SIP callbacks
        self.sip_client.set_callbacks(
            on_incoming_call=self._on_incoming_call,
            on_call_established=self._on_call_established,
            on_call_ended=self._on_call_ended,
            on_audio_received=self._on_audio_received,
            on_dtmf_received=self._on_dtmf_received,
            get_outbound_audio=self._get_outbound_audio,
        )
        
        # Start SIP client
        self.sip_client.start()
        
        _LOGGER.info("VoIP Bridge coordinator initialized")
    
    async def _on_incoming_call(self, caller_uri: str) -> None:
        """Handle incoming call."""
        _LOGGER.info(f"Incoming call from {caller_uri}")
        
        # Extract caller ID
        caller_id = self._extract_caller_id(caller_uri)
        
        # Determine if PIN required
        require_pin = self._should_require_pin(caller_id, is_internal=True)
        
        # Create call session
        session_config = CallSessionConfig(
            require_pin=require_pin,
            pin=self.entry.data.get(CONF_PIN, "1234"),
        )
        
        self.current_session = CallSession(
            self.hass,
            self.sip_client,
            self.audio_bridge,
            self.event_manager,
            session_config,
            caller_id,
            "inbound",
        )
        
        await self.current_session.start()
    
    async def _on_call_established(self) -> None:
        """Handle call established."""
        _LOGGER.info("Call established")
    
    async def _on_call_ended(self) -> None:
        """Handle call ended."""
        _LOGGER.info("Call ended")
        self.current_session = None
        await self.audio_bridge.clear_buffers()
    
    async def _on_audio_received(self, audio_data: bytes) -> None:
        """Handle audio received from call."""
        if self.current_session:
            await self.audio_bridge.process_inbound_audio(audio_data)
    
    async def _on_dtmf_received(self, digit: str) -> None:
        """Handle DTMF digit received."""
        if self.current_session:
            await self.current_session.handle_dtmf(digit)
    
    async def _on_speech_complete(self, audio_data: bytes) -> None:
        """Handle completed speech segment."""
        if self.current_session:
            await self.current_session.handle_speech(audio_data)
    
    def _get_outbound_audio(self, size: int) -> bytes | None:
        """Get outbound audio frame."""
        return self.audio_bridge.get_outbound_frame(size)
    
    def _extract_caller_id(self, uri: str) -> str:
        """Extract caller ID from SIP URI."""
        # Parse sip:user@host format
        if "@" in uri:
            user_part = uri.split("@")[0]
            if ":" in user_part:
                return user_part.split(":")[1]
            return user_part
        return uri
    
    def _should_require_pin(self, caller_id: str, is_internal: bool) -> bool:
        """Determine if PIN should be required for this caller."""
        # Check if caller is in trusted list
        trusted_ids = self.entry.data.get(CONF_TRUSTED_CALLER_IDS, [])
        if caller_id in trusted_ids:
            return False
        
        # Check internal/external settings
        if is_internal:
            return self.entry.data.get(CONF_REQUIRE_PIN_INTERNAL, False)
        else:
            return self.entry.data.get(CONF_REQUIRE_PIN_EXTERNAL, True)
    
    async def make_outbound_call(self, destination: str, force: bool = False) -> None:
        """Initiate outbound call.
        
        Args:
            destination: Phone number to call
            force: Force call even if no events pending
        """
        if self.current_session:
            _LOGGER.warning("Call already in progress")
            return
        
        # Check if we should call
        if not force and not self.event_manager._should_call():
            _LOGGER.info("No events meet threshold for calling")
            return
        
        _LOGGER.info(f"Initiating outbound call to {destination}")
        
        # Create call session (outbound doesn't need PIN)
        session_config = CallSessionConfig(
            require_pin=False,
            pin="",
        )
        
        self.current_session = CallSession(
            self.hass,
            self.sip_client,
            self.audio_bridge,
            self.event_manager,
            session_config,
            destination,
            "outbound",
        )
        
        # Make the call
        self.sip_client.make_call(destination)
        
        # Start session (will wait for call to be established)
        await self.current_session.start()
    
    async def trigger_alert_call(self, force: bool = False) -> None:
        """Trigger alert call to configured destinations."""
        destinations = self.entry.data.get(CONF_OUTBOUND_DESTINATIONS, [])
        
        if not destinations:
            _LOGGER.error("No outbound destinations configured")
            return
        
        # Try each destination in priority order
        for dest in sorted(destinations, key=lambda x: x.get("priority", 999)):
            if not dest.get("enabled", True):
                continue
            
            number = dest["number"]
            _LOGGER.info(f"Attempting call to {dest.get('name', number)}")
            
            try:
                await self.make_outbound_call(number, force)
                # If call initiated successfully, stop trying
                # TODO: Add retry logic if call fails
                break
            except Exception as e:
                _LOGGER.error(f"Failed to call {number}: {e}")
                continue
    
    async def async_shutdown(self) -> None:
        """Shutdown coordinator and cleanup."""
        _LOGGER.info("Shutting down VoIP Bridge coordinator")
        
        if self.current_session:
            await self.current_session.hangup()
        
        if self.sip_client:
            self.sip_client.stop()
        
        if self.event_manager:
            await self.event_manager.async_shutdown()


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the VoIP Bridge component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up VoIP Bridge from a config entry."""
    _LOGGER.info("Setting up VoIP Bridge integration")
    
    # Create and setup coordinator
    coordinator = VoIPBridgeCoordinator(hass, entry)
    await coordinator.async_setup()
    
    # Store in hass.data
    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
    }
    
    # Register services
    async def handle_add_event(call: ServiceCall) -> None:
        """Handle add_event service call."""
        event = OutboundEvent(
            event_id=call.data["event_id"],
            priority=call.data["priority"],
            message=call.data["message"],
        )
        await coordinator.event_manager.add_event(event)
    
    async def handle_clear_event(call: ServiceCall) -> None:
        """Handle clear_event service call."""
        await coordinator.event_manager.clear_event(call.data["event_id"])
    
    async def handle_clear_all_events(call: ServiceCall) -> None:
        """Handle clear_all_events service call."""
        await coordinator.event_manager.clear_all()
    
    async def handle_trigger_call(call: ServiceCall) -> None:
        """Handle trigger_call service call."""
        force = call.data.get("force", False)
        await coordinator.trigger_alert_call(force)
    
    async def handle_list_events(call: ServiceCall) -> None:
        """Handle list_events service call."""
        events = coordinator.event_manager.get_events()
        _LOGGER.info(f"Active events: {events}")
        # TODO: Return events via response
    
    # Register services
    hass.services.async_register(
        DOMAIN, SERVICE_ADD_EVENT, handle_add_event, schema=SERVICE_ADD_EVENT_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_CLEAR_EVENT, handle_clear_event, schema=SERVICE_CLEAR_EVENT_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_CLEAR_ALL_EVENTS, handle_clear_all_events
    )
    hass.services.async_register(
        DOMAIN, SERVICE_TRIGGER_CALL, handle_trigger_call, schema=SERVICE_TRIGGER_CALL_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_LIST_EVENTS, handle_list_events
    )
    
    _LOGGER.info("VoIP Bridge services registered")
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading VoIP Bridge integration")
    
    # Cleanup
    data = hass.data[DOMAIN].pop(entry.entry_id)
    coordinator: VoIPBridgeCoordinator = data["coordinator"]
    await coordinator.async_shutdown()
    
    # Unregister services
    hass.services.async_remove(DOMAIN, SERVICE_ADD_EVENT)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_EVENT)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_ALL_EVENTS)
    hass.services.async_remove(DOMAIN, SERVICE_TRIGGER_CALL)
    hass.services.async_remove(DOMAIN, SERVICE_LIST_EVENTS)
    
    return True