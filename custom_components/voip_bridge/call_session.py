"""Call session state machine for VoIP Bridge."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from homeassistant.core import HomeAssistant

from .const import (
    STATE_IDLE,
    STATE_RINGING,
    STATE_AUTH_REQUIRED,
    STATE_AUTHENTICATED,
    STATE_PRESENTING_EVENTS,
    STATE_IN_EVENT,
    STATE_GENERAL_QUERY,
    STATE_CLOSING,
    STATE_ENDED,
)

if TYPE_CHECKING:
    from .audio_bridge import AudioBridge
    from .event_manager import EventManager, OutboundEvent
    from .sip_client import SIPClient

_LOGGER = logging.getLogger(__name__)


@dataclass
class CallSessionConfig:
    """Configuration for call session."""
    require_pin: bool
    pin: str
    max_pin_attempts: int = 3
    conversation_timeout: int = 300  # 5 minutes


class CallState(str, Enum):
    """Call states."""
    IDLE = STATE_IDLE
    RINGING = STATE_RINGING
    AUTH_REQUIRED = STATE_AUTH_REQUIRED
    AUTHENTICATED = STATE_AUTHENTICATED
    PRESENTING_EVENTS = STATE_PRESENTING_EVENTS
    IN_EVENT = STATE_IN_EVENT
    GENERAL_QUERY = STATE_GENERAL_QUERY
    CLOSING = STATE_CLOSING
    ENDED = STATE_ENDED


class CallSession:
    """Manages state and logic for a single call session."""
    
    def __init__(
        self,
        hass: HomeAssistant,
        sip_client: SIPClient,
        audio_bridge: AudioBridge,
        event_manager: EventManager,
        config: CallSessionConfig,
        caller_id: str,
        direction: str,
    ) -> None:
        """Initialize call session."""
        self.hass = hass
        self.sip_client = sip_client
        self.audio_bridge = audio_bridge
        self.event_manager = event_manager
        self.config = config
        self.caller_id = caller_id
        self.direction = direction
        
        # State
        self.state = CallState.IDLE
        self.authenticated = False
        self.dtmf_buffer = ""
        self.pin_attempts = 0
        
        # Event navigation
        self.current_event_index = 0
        self.events_to_present: list[OutboundEvent] = []
        
        # Assist conversation
        self.conversation_id: str | None = None
        
        # Timers
        self._timeout_task: asyncio.Task | None = None
    
    async def start(self) -> None:
        """Start call session."""
        _LOGGER.info(f"Starting {self.direction} call session from {self.caller_id}")
        
        if self.direction == "inbound":
            # Answer the call
            #self.sip_client.answer_call()
            await self.transition_to(CallState.RINGING)
        else:
            # Outbound call
            await self.transition_to(CallState.RINGING)
        
        # Set timeout
        self._timeout_task = asyncio.create_task(self._handle_timeout())
    
    async def transition_to(self, new_state: CallState) -> None:
        """Transition to new state with appropriate actions."""
        old_state = self.state
        self.state = new_state
        _LOGGER.debug(f"Call state: {old_state} -> {new_state}")
        
        # Handle state entry actions
        if new_state == CallState.RINGING:
            if self.config.require_pin:
                await self.transition_to(CallState.AUTH_REQUIRED)
            else:
                self.authenticated = True
                await self.transition_to(CallState.AUTHENTICATED)
        
        elif new_state == CallState.AUTH_REQUIRED:
            await self._speak("Please enter your PIN")
        
        elif new_state == CallState.AUTHENTICATED:
            await self._speak("Authenticated")
            
            # Get pending events
            self.events_to_present = list(self.event_manager._active_events.values())
            
            if self.events_to_present:
                await self.transition_to(CallState.PRESENTING_EVENTS)
            else:
                await self._speak("No pending events. How can I help you?")
                await self.transition_to(CallState.GENERAL_QUERY)
        
        elif new_state == CallState.PRESENTING_EVENTS:
            await self._present_event_summary()
        
        elif new_state == CallState.IN_EVENT:
            await self._present_current_event()
        
        elif new_state == CallState.CLOSING:
            await self._speak("Goodbye")
            await asyncio.sleep(1)
            await self.hangup()
        
        elif new_state == CallState.ENDED:
            await self._cleanup()
    
    async def handle_dtmf(self, digit: str) -> None:
        """Handle DTMF digit received."""
        await self.audio_bridge.play_dtmf_acknowledgment()
        
        if self.state == CallState.AUTH_REQUIRED:
            self.dtmf_buffer += digit
            
            # Check if we have enough digits
            if len(self.dtmf_buffer) >= len(self.config.pin):
                if self.dtmf_buffer == self.config.pin:
                    self.authenticated = True
                    self.dtmf_buffer = ""
                    await self.transition_to(CallState.AUTHENTICATED)
                else:
                    self.pin_attempts += 1
                    self.dtmf_buffer = ""
                    
                    if self.pin_attempts >= self.config.max_pin_attempts:
                        await self._speak("Too many attempts. Goodbye.")
                        await self.transition_to(CallState.CLOSING)
                    else:
                        await self._speak("Invalid PIN. Try again.")
    
    async def handle_speech(self, audio_data: bytes) -> None:
        """Handle speech audio from caller.
        
        Args:
            audio_data: PCM16 audio data
        """
        if not self.authenticated:
            # Don't process speech before authentication
            return
        
        # Send to STT
        transcript = await self._speech_to_text(audio_data)
        
        if not transcript:
            return
        
        _LOGGER.info(f"User said: {transcript}")
        
        # Process based on state
        await self._process_user_command(transcript)
    
    async def _process_user_command(self, transcript: str) -> None:
        """Process user voice command based on current state."""
        lower = transcript.lower()
        
        # Check for event navigation commands first
        if self.state in (CallState.PRESENTING_EVENTS, CallState.IN_EVENT):
            if "next" in lower or "next event" in lower:
                await self._next_event()
                return
            
            elif "clear" in lower and "all" in lower:
                await self.event_manager.clear_all()
                await self._speak("All events cleared. Goodbye.")
                await self.transition_to(CallState.CLOSING)
                return
            
            elif "clear" in lower:
                await self._clear_current_event()
                await self._next_event()
                return
            
            elif "ignore" in lower or "skip" in lower:
                await self._next_event()
                return
            
            elif "repeat" in lower:
                await self._present_current_event()
                return
            
            elif "how many" in lower or "event count" in lower:
                count = len(self.events_to_present) - self.current_event_index
                await self._speak(f"You have {count} remaining events")
                return
        
        # Check for goodbye/hangup
        if any(word in lower for word in ["goodbye", "bye", "hang up", "that's all"]):
            await self.transition_to(CallState.CLOSING)
            return
        
        # General Assist query
        await self.transition_to(CallState.GENERAL_QUERY)
        response = await self._query_assist(transcript)
        await self._speak(response)
        
        # Return to events if we were presenting them
        if self.events_to_present and self.current_event_index < len(self.events_to_present):
            await self._speak("Returning to events.")
            await self.transition_to(CallState.IN_EVENT)
    
    async def _present_event_summary(self) -> None:
        """Present summary of pending events."""
        count = len(self.events_to_present)
        
        if count == 0:
            await self._speak("No pending events.")
            await self.transition_to(CallState.GENERAL_QUERY)
            return
        
        # Sort by priority
        self.events_to_present.sort(key=lambda e: e.priority)
        
        await self._speak(f"You have {count} event{'s' if count != 1 else ''}.")
        await asyncio.sleep(0.5)
        
        self.current_event_index = 0
        await self.transition_to(CallState.IN_EVENT)
    
    async def _present_current_event(self) -> None:
        """Present current event in list."""
        if self.current_event_index >= len(self.events_to_present):
            # All events presented
            await self._speak("All events reviewed. Say goodbye to hang up, or ask me anything.")
            await self.transition_to(CallState.GENERAL_QUERY)
            return
        
        event = self.events_to_present[self.current_event_index]
        position = self.current_event_index + 1
        total = len(self.events_to_present)
        
        priority_text = {1: "Critical", 2: "Warning", 3: "Info"}.get(event.priority, "")
        
        await self._speak(
            f"Event {position} of {total}. {priority_text}. {event.message}. "
            f"Say next, clear, or ignore."
        )
    
    async def _next_event(self) -> None:
        """Move to next event."""
        self.current_event_index += 1
        await self._present_current_event()
    
    async def _clear_current_event(self) -> None:
        """Clear current event from queue."""
        if self.current_event_index < len(self.events_to_present):
            event = self.events_to_present[self.current_event_index]
            await self.event_manager.clear_event(event.event_id)
            await self._speak("Cleared.")
    
    async def _speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech audio to text using Assist pipeline.
        
        Args:
            audio_data: PCM16 audio data
            
        Returns:
            Transcript text
        """
        # TODO: Implement Assist pipeline STT
        # For now, return dummy text
        _LOGGER.warning("STT not yet implemented")
        return ""
    
    async def _query_assist(self, text: str) -> str:
        """Query Assist pipeline with text.
        
        Args:
            text: User query text
            
        Returns:
            Response text
        """
        # TODO: Implement Assist pipeline query
        _LOGGER.warning("Assist query not yet implemented")
        return "I'm sorry, Assist integration is not yet implemented."
    
    async def _speak(self, text: str) -> None:
        """Convert text to speech and play to caller."""
        _LOGGER.info(f"TTS: {text}")

        try:
            from homeassistant.components import tts
            # Generate TTS
            url = await tts.async_get_url(
                self.hass,
                engine="tts.piper",
                message=text,
                language="en",
                options={},
            )

            _LOGGER.info(f"TTS URL: {url}")

            # Download the audio file
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:8123{url}") as response:
                    audio_data = await response.read()
                    _LOGGER.info(f"Downloaded {len(audio_data)} bytes")

                    # TODO: Convert from WAV/MP3 to PCM16 @ 8kHz
                    await self.audio_bridge.play_tone(440, 1.0)

        except Exception as e:
            _LOGGER.error(f"TTS error: {e}", exc_info=True)
            await self.audio_bridge.play_tone(880, 3.0)
        
    async def _handle_timeout(self) -> None:
        """Handle call timeout."""
        await asyncio.sleep(self.config.conversation_timeout)
        
        if self.state not in (CallState.ENDED, CallState.CLOSING):
            _LOGGER.info("Call timeout reached")
            await self._speak("Call timeout. Goodbye.")
            await self.transition_to(CallState.CLOSING)
    
    async def hangup(self) -> None:
        """Hangup the call."""
        self.sip_client.hangup_call()
        await self.transition_to(CallState.ENDED)
    
    async def _cleanup(self) -> None:
        """Cleanup call session."""
        if self._timeout_task:
            self._timeout_task.cancel()
        
        await self.audio_bridge.clear_buffers()
        
        _LOGGER.info("Call session ended")