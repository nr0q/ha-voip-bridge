"""SIP client using pyVoIP for VoIP Bridge integration."""
import asyncio
import logging
from typing import Callable, Optional
import threading

from pyVoIP.VoIP import VoIPPhone, CallState, InvalidStateError

_LOGGER = logging.getLogger(__name__)


class VoIPBridgePhone:
    """Wrapper for pyVoIP phone with async callback support."""
    
    def __init__(
        self,
        hass,
        server: str,
        port: int,
        username: str,
        password: str,
        extension: str,
        my_ip: str,
        sample_rate: int = 8000,
    ):
        """Initialize VoIP phone.
        
        Args:
            hass: Home Assistant instance
            server: SIP server address
            port: SIP server port
            username: SIP username
            password: SIP password
            extension: Extension number
            my_ip: Local IP address for RTP
            sample_rate: Audio sample rate (default 8000 for PCMU/PCMA)
        """
        self.hass = hass
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.extension = extension
        self.my_ip = my_ip
        self.sample_rate = sample_rate
        
        # pyVoIP phone instance
        self.phone: Optional[VoIPPhone] = None
        
        # Current active call
        self.current_call = None
        
        # Callbacks (async coroutines)
        self._on_incoming_call: Optional[Callable] = None
        self._on_call_established: Optional[Callable] = None
        self._on_call_ended: Optional[Callable] = None
        self._on_audio_received: Optional[Callable] = None
        self._on_dtmf_received: Optional[Callable] = None
        self._get_outbound_audio: Optional[Callable] = None
        
        # Audio handling
        self._audio_thread: Optional[threading.Thread] = None
        self._stop_audio = threading.Event()
        
        _LOGGER.info(
            f"VoIP phone initialized: {username}@{server}:{port} ext={extension}"
        )
    
    def set_callbacks(
        self,
        on_incoming_call: Optional[Callable] = None,
        on_call_established: Optional[Callable] = None,
        on_call_ended: Optional[Callable] = None,
        on_audio_received: Optional[Callable] = None,
        on_dtmf_received: Optional[Callable] = None,
        get_outbound_audio: Optional[Callable] = None,
    ):
        """Set callback functions for call events.
        
        All callbacks should be async coroutines except get_outbound_audio
        which is synchronous and returns bytes.
        """
        self._on_incoming_call = on_incoming_call
        self._on_call_established = on_call_established
        self._on_call_ended = on_call_ended
        self._on_audio_received = on_audio_received
        self._on_dtmf_received = on_dtmf_received
        self._get_outbound_audio = get_outbound_audio
    
    def start(self):
        """Start the VoIP phone."""
        try:
            # Create phone with callback
            self.phone = VoIPPhone(
                self.server,
                self.port,
                self.username,
                self.password,
                callCallback=self._handle_incoming_call_sync,
                myIP=self.my_ip,
                sipPort=5060,
                rtpPortLow=10000,
                rtpPortHigh=20000,
            )
            
            self.phone.start()
            _LOGGER.info("VoIP phone started successfully")
            
        except Exception as e:
            _LOGGER.error(f"Failed to start VoIP phone: {e}")
            raise
    
    def stop(self):
        """Stop the VoIP phone."""
        try:
            if self.current_call:
                try:
                    self.current_call.hangup()
                except InvalidStateError:
                    pass
            
            if self.phone:
                self.phone.stop()
            
            # Stop audio thread
            self._stop_audio.set()
            if self._audio_thread and self._audio_thread.is_alive():
                self._audio_thread.join(timeout=2)
            
            _LOGGER.info("VoIP phone stopped")
            
        except Exception as e:
            _LOGGER.error(f"Error stopping VoIP phone: {e}")
    
    def _handle_incoming_call_sync(self, call):
        """Synchronous callback for incoming calls (called by pyVoIP).
        
        This bridges from pyVoIP's synchronous callback to our async system.
        """
        _LOGGER.info(f"Incoming call from {call.request.headers['From']}")
        
        # Store the call
        self.current_call = call
        
        # Extract caller info
        caller_uri = call.request.headers['From']['uri']
        
        # Answer the call
        try:
            call.answer()
            _LOGGER.info("Call answered")
            
            # Start audio handling thread
            self._stop_audio.clear()
            self._audio_thread = threading.Thread(
                target=self._audio_loop,
                args=(call,),
                daemon=True
            )
            self._audio_thread.start()
            
            # Notify via async callback
            if self._on_incoming_call:
                asyncio.run_coroutine_threadsafe(
                    self._on_incoming_call(caller_uri),
                    self.hass.loop
                )
            
            if self._on_call_established:
                asyncio.run_coroutine_threadsafe(
                    self._on_call_established(),
                    self.hass.loop
                )
            
            # DON'T BLOCK - return immediately so pyVoIP can process RTP
            
        except InvalidStateError as e:
            _LOGGER.error(f"Invalid call state: {e}")
            
            # Notify call ended on error
            if self._on_call_ended:
                asyncio.run_coroutine_threadsafe(
                    self._on_call_ended(),
                    self.hass.loop
                )
            
            self.current_call = None
            self._stop_audio.set()
    
    def _audio_loop(self, call):
        """Audio handling loop (runs in separate thread).
        
        Handles bidirectional audio and DTMF detection.
        """
        _LOGGER.info("Audio loop started")
        
        try:
            while not self._stop_audio.is_set() and call.state.state == CallState.ANSWERED:
                # Get incoming audio (RTP packets)
                try:
                    audio_data = call.read_audio(length=160, blocking=False)  # 20ms at 8kHz
                    
                    if audio_data and self._on_audio_received:
                        # Send to async handler
                        asyncio.run_coroutine_threadsafe(
                            self._on_audio_received(audio_data),
                            self.hass.loop
                        )
                
                except:
                    pass  # No audio available
                
                # Get outgoing audio if callback provided
                if self._get_outbound_audio:
                    try:
                        outbound = self._get_outbound_audio(160)  # 20ms at 8kHz
                        if outbound:
                            call.write_audio(outbound)
                    except Exception as e:
                        _LOGGER.error(f"Error writing audio: {e}")
                
                # Small sleep to prevent busy loop
                asyncio.run_coroutine_threadsafe(
                    asyncio.sleep(0.01),
                    self.hass.loop
                ).result()
        
        except Exception as e:
            _LOGGER.error(f"Error in audio loop: {e}")
        
        finally:
            _LOGGER.info("Audio loop ended")
            
            # Notify call ended
            if self._on_call_ended:
                asyncio.run_coroutine_threadsafe(
                    self._on_call_ended(),
                    self.hass.loop
                )
            
            self.current_call = None
            self._stop_audio.set()
    
    def make_call(self, number: str):
        """Initiate outbound call.
        
        Args:
            number: Phone number or SIP URI to call
        """
        if self.current_call:
            _LOGGER.warning("Call already in progress")
            return
        
        try:
            _LOGGER.info(f"Making call to {number}")
            call = self.phone.call(number)
            self.current_call = call
            
            # Wait for call to be answered
            call.state.wait_for_state(CallState.ANSWERED)
            
            # Start audio thread
            self._stop_audio.clear()
            self._audio_thread = threading.Thread(
                target=self._audio_loop,
                args=(call,),
                daemon=True
            )
            self._audio_thread.start()
            
            # Notify established
            if self._on_call_established:
                asyncio.run_coroutine_threadsafe(
                    self._on_call_established(),
                    self.hass.loop
                )
            
            # Wait for call to end
            call.state.wait_for_state(CallState.ENDED)
            
        except InvalidStateError as e:
            _LOGGER.error(f"Call failed: {e}")
        
        finally:
            if self._on_call_ended:
                asyncio.run_coroutine_threadsafe(
                    self._on_call_ended(),
                    self.hass.loop
                )
            
            self.current_call = None
            self._stop_audio.set()
    
    def hangup_call(self):
        """Hang up current call."""
        if self.current_call:
            try:
                self.current_call.hangup()
                _LOGGER.info("Call hung up")
            except InvalidStateError:
                pass
    
    def send_dtmf(self, digit: str):
        """Send DTMF digit.
        
        Args:
            digit: DTMF digit (0-9, *, #, A-D)
        """
        if self.current_call:
            try:
                self.current_call.send_dtmf(digit)
                _LOGGER.debug(f"Sent DTMF: {digit}")
            except Exception as e:
                _LOGGER.error(f"Failed to send DTMF: {e}")


# Compatibility wrapper to match original interface
class SIPClient:
    """SIP client wrapper matching original interface."""
    
    def __init__(
        self,
        hass,
        server: str,
        port: int,
        username: str,
        password: str,
        extension: str,
        sample_rate: int = 8000,
    ):
        """Initialize SIP client."""
        self.hass = hass
        
        # Auto-detect local IP
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((server, port))
            my_ip = s.getsockname()[0]
        except:
            my_ip = "0.0.0.0"
        finally:
            s.close()
        
        self.phone = VoIPBridgePhone(
            hass,
            server,
            port,
            username,
            password,
            extension,
            my_ip,
            sample_rate,
        )
    
    def set_callbacks(self, **kwargs):
        """Set callbacks."""
        self.phone.set_callbacks(**kwargs)
    
    def start(self):
        """Start SIP client."""
        self.phone.start()
    
    def stop(self):
        """Stop SIP client."""
        self.phone.stop()
    
    def make_call(self, number: str):
        """Make outbound call."""
        # Run in executor since it's blocking
        asyncio.get_event_loop().run_in_executor(
            None,
            self.phone.make_call,
            number
        )
    
    def hangup_call(self):
        """Hang up call."""
        self.phone.hangup_call()
    
    def send_dtmf(self, digit: str):
        """Send DTMF."""
        self.phone.send_dtmf(digit)