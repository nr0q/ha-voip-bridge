"""SIP client wrapper using PJSUA2."""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

import pjsua2 as pj

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class SIPAccount(pj.Account):
    """PJSIP Account handler."""
    
    def __init__(self, sip_client: SIPClient) -> None:
        """Initialize account."""
        super().__init__()
        self.sip_client = sip_client
    
    def onRegState(self, prm: pj.OnRegStateParam) -> None:
        """Handle registration state changes."""
        info = self.getInfo()
        status = info.regStatus
        
        if status == pj.PJSIP_SC_OK:
            _LOGGER.info(f"SIP registration successful: {info.uri}")
            self.sip_client._on_registration_changed(True)
        else:
            _LOGGER.warning(f"SIP registration failed: {status} - {info.regStatusText}")
            self.sip_client._on_registration_changed(False)
    
    def onIncomingCall(self, prm: pj.OnIncomingCallParam) -> None:
        """Handle incoming call."""
        call = SIPCall(self.sip_client, self, prm.callId)
        call_info = call.getInfo()
        
        _LOGGER.info(f"Incoming call from {call_info.remoteUri}")
        self.sip_client._on_incoming_call(call, call_info)


class SIPCall(pj.Call):
    """PJSIP Call handler."""
    
    def __init__(self, sip_client: SIPClient, account: SIPAccount, call_id: int = pj.PJSUA_INVALID_ID) -> None:
        """Initialize call."""
        super().__init__(account, call_id)
        self.sip_client = sip_client
        self.audio_media: pj.AudioMedia | None = None
    
    def onCallState(self, prm: pj.OnCallStateParam) -> None:
        """Handle call state changes."""
        info = self.getInfo()
        state = info.state
        
        _LOGGER.debug(f"Call state: {state} ({info.stateText})")
        
        if state == pj.PJSIP_INV_STATE_DISCONNECTED:
            _LOGGER.info("Call disconnected")
            self.sip_client._on_call_ended(self)
        elif state == pj.PJSIP_INV_STATE_CONFIRMED:
            _LOGGER.info("Call established")
            self.sip_client._on_call_established(self)
    
    def onCallMediaState(self, prm: pj.OnCallMediaStateParam) -> None:
        """Handle media state changes."""
        info = self.getInfo()
        
        for mi in info.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO:
                if mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                    # Get the audio media
                    self.audio_media = self.getAudioMedia(mi.index)
                    _LOGGER.info("Audio media active")
                    self.sip_client._on_media_active(self, self.audio_media)
    
    def onDtmfDigit(self, prm: pj.OnDtmfDigitParam) -> None:
        """Handle DTMF digit received."""
        digit = prm.digit
        _LOGGER.debug(f"DTMF received: {digit}")
        self.sip_client._on_dtmf_received(self, digit)


class AudioFrameCallback(pj.AudioMediaPort):
    """Custom audio media port for capturing/playing audio."""
    
    def __init__(self, sip_client: SIPClient, sample_rate: int = 8000) -> None:
        """Initialize audio port."""
        super().__init__()
        self.sip_client = sip_client
        self.sample_rate = sample_rate
        self._createPort(
            "voip_bridge_audio",
            sample_rate,
            1,  # channels
            16,  # bits per sample
            sample_rate * 20 // 1000,  # samples per frame (20ms)
        )
    
    def onFrameRequested(self, frame: pj.MediaFrame) -> None:
        """Called when system needs audio to play (outbound audio)."""
        # Get audio from TTS queue
        audio_data = self.sip_client._get_outbound_audio(frame.size)
        if audio_data:
            # Copy to frame buffer
            frame.buf = audio_data
            frame.size = len(audio_data)
            frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
        else:
            # Silence
            frame.type = pj.PJMEDIA_FRAME_TYPE_NONE
    
    def onFrameReceived(self, frame: pj.MediaFrame) -> None:
        """Called when audio received from call (inbound audio)."""
        if frame.type == pj.PJMEDIA_FRAME_TYPE_AUDIO:
            # Send audio to STT processor
            audio_bytes = bytes(frame.buf[:frame.size])
            self.sip_client._on_audio_received(audio_bytes)


class SIPClient:
    """SIP client wrapper."""
    
    def __init__(
        self,
        hass: HomeAssistant,
        server: str,
        port: int,
        username: str,
        password: str,
        extension: str,
        sample_rate: int = 8000,
    ) -> None:
        """Initialize SIP client."""
        self.hass = hass
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.extension = extension
        self.sample_rate = sample_rate
        
        self._ep: pj.Endpoint | None = None
        self._account: SIPAccount | None = None
        self._current_call: SIPCall | None = None
        self._audio_port: AudioFrameCallback | None = None
        self._transport: pj.TransportConfig | None = None
        
        self._registered = False
        self._running = False
        self._thread: threading.Thread | None = None
        
        # Callbacks
        self._on_incoming_call_cb: Callable | None = None
        self._on_call_established_cb: Callable | None = None
        self._on_call_ended_cb: Callable | None = None
        self._on_audio_received_cb: Callable | None = None
        self._on_dtmf_received_cb: Callable | None = None
        self._get_outbound_audio_cb: Callable | None = None
    
    def set_callbacks(
        self,
        on_incoming_call: Callable | None = None,
        on_call_established: Callable | None = None,
        on_call_ended: Callable | None = None,
        on_audio_received: Callable | None = None,
        on_dtmf_received: Callable | None = None,
        get_outbound_audio: Callable | None = None,
    ) -> None:
        """Set callback functions."""
        self._on_incoming_call_cb = on_incoming_call
        self._on_call_established_cb = on_call_established
        self._on_call_ended_cb = on_call_ended
        self._on_audio_received_cb = on_audio_received
        self._on_dtmf_received_cb = on_dtmf_received
        self._get_outbound_audio_cb = get_outbound_audio
    
    def start(self) -> None:
        """Start SIP client in separate thread."""
        if self._running:
            _LOGGER.warning("SIP client already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_pjsip, daemon=True)
        self._thread.start()
        _LOGGER.info("SIP client thread started")
    
    def _run_pjsip(self) -> None:
        """Run PJSIP in dedicated thread."""
        try:
            # Create endpoint
            self._ep = pj.Endpoint()
            self._ep.libCreate()
            
            # Initialize endpoint
            ep_cfg = pj.EpConfig()
            ep_cfg.logConfig.level = 4
            ep_cfg.logConfig.consoleLevel = 4
            self._ep.libInit(ep_cfg)
            
            # Create UDP transport
            transport_cfg = pj.TransportConfig()
            transport_cfg.port = 0  # Any available port
            self._transport = self._ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
            
            # Start endpoint
            self._ep.libStart()
            _LOGGER.info("PJSIP endpoint started")
            
            # Create account
            acc_cfg = pj.AccountConfig()
            acc_cfg.idUri = f"sip:{self.username}@{self.server}"
            acc_cfg.regConfig.registrarUri = f"sip:{self.server}:{self.port}"
            
            cred = pj.AuthCredInfo()
            cred.scheme = "digest"
            cred.realm = "*"
            cred.username = self.username
            cred.data = self.password
            cred.dataType = pj.PJSIP_CRED_DATA_PLAIN_PASSWD
            acc_cfg.sipConfig.authCreds.append(cred)
            
            self._account = SIPAccount(self)
            self._account.create(acc_cfg)
            _LOGGER.info(f"SIP account created for {self.extension}")
            
            # Create audio port
            self._audio_port = AudioFrameCallback(self, self.sample_rate)
            
            # Keep thread alive
            while self._running:
                self._ep.libHandleEvents(50)  # 50ms timeout
            
        except Exception as e:
            _LOGGER.error(f"Error in PJSIP thread: {e}", exc_info=True)
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Cleanup PJSIP resources."""
        try:
            if self._account:
                self._account.delete()
            if self._ep:
                self._ep.libDestroy()
        except Exception as e:
            _LOGGER.error(f"Error during cleanup: {e}")
    
    def stop(self) -> None:
        """Stop SIP client."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        _LOGGER.info("SIP client stopped")
    
    def make_call(self, destination: str) -> None:
        """Initiate outbound call."""
        if not self._account:
            _LOGGER.error("Cannot make call: account not initialized")
            return
        
        try:
            call = SIPCall(self, self._account)
            call_prm = pj.CallOpParam()
            call_prm.opt.audioCount = 1
            call_prm.opt.videoCount = 0
            
            dest_uri = f"sip:{destination}@{self.server}"
            call.makeCall(dest_uri, call_prm)
            self._current_call = call
            
            _LOGGER.info(f"Initiating call to {destination}")
        except Exception as e:
            _LOGGER.error(f"Failed to make call: {e}", exc_info=True)
    
    def answer_call(self) -> None:
        """Answer incoming call."""
        if self._current_call:
            try:
                call_prm = pj.CallOpParam()
                call_prm.statusCode = 200
                self._current_call.answer(call_prm)
                _LOGGER.info("Answered incoming call")
            except Exception as e:
                _LOGGER.error(f"Failed to answer call: {e}")
    
    def hangup_call(self) -> None:
        """Hangup current call."""
        if self._current_call:
            try:
                call_prm = pj.CallOpParam()
                self._current_call.hangup(call_prm)
                _LOGGER.info("Hung up call")
            except Exception as e:
                _LOGGER.error(f"Failed to hangup call: {e}")
    
    def send_dtmf(self, digits: str) -> None:
        """Send DTMF digits."""
        if self._current_call:
            try:
                self._current_call.dialDtmf(digits)
                _LOGGER.debug(f"Sent DTMF: {digits}")
            except Exception as e:
                _LOGGER.error(f"Failed to send DTMF: {e}")
    
    # Internal callback handlers (called from PJSIP thread)
    def _on_registration_changed(self, registered: bool) -> None:
        """Handle registration state change."""
        self._registered = registered
    
    def _on_incoming_call(self, call: SIPCall, call_info: pj.CallInfo) -> None:
        """Handle incoming call."""
        self._current_call = call
        if self._on_incoming_call_cb:
            # Schedule callback in HA event loop
            asyncio.run_coroutine_threadsafe(
                self._on_incoming_call_cb(call_info.remoteUri),
                self.hass.loop
            )
    
    def _on_call_established(self, call: SIPCall) -> None:
        """Handle call established."""
        if self._on_call_established_cb:
            asyncio.run_coroutine_threadsafe(
                self._on_call_established_cb(),
                self.hass.loop
            )
    
    def _on_call_ended(self, call: SIPCall) -> None:
        """Handle call ended."""
        self._current_call = None
        if self._on_call_ended_cb:
            asyncio.run_coroutine_threadsafe(
                self._on_call_ended_cb(),
                self.hass.loop
            )
    
    def _on_media_active(self, call: SIPCall, audio_media: pj.AudioMedia) -> None:
        """Handle media becoming active."""
        # Connect our audio port to the call
        if self._audio_port:
            try:
                # Bidirectional audio
                audio_media.startTransmit(self._audio_port)
                self._audio_port.startTransmit(audio_media)
                _LOGGER.info("Audio streams connected")
            except Exception as e:
                _LOGGER.error(f"Failed to connect audio: {e}")
    
    def _on_audio_received(self, audio_data: bytes) -> None:
        """Handle received audio data."""
        if self._on_audio_received_cb:
            asyncio.run_coroutine_threadsafe(
                self._on_audio_received_cb(audio_data),
                self.hass.loop
            )
    
    def _on_dtmf_received(self, call: SIPCall, digit: str) -> None:
        """Handle DTMF digit."""
        if self._on_dtmf_received_cb:
            asyncio.run_coroutine_threadsafe(
                self._on_dtmf_received_cb(digit),
                self.hass.loop
            )
    
    def _get_outbound_audio(self, size: int) -> bytes | None:
        """Get audio to send to caller."""
        if self._get_outbound_audio_cb:
            return self._get_outbound_audio_cb(size)
        return None
    
    @property
    def is_registered(self) -> bool:
        """Check if registered to SIP server."""
        return self._registered
    
    @property
    def has_active_call(self) -> bool:
        """Check if there's an active call."""
        return self._current_call is not None