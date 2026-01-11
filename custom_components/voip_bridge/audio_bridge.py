"""Audio processing and bridging for VoIP Bridge."""
from __future__ import annotations

import asyncio
import audioop
import logging
from collections import deque
from typing import Callable

import numpy as np
import webrtcvad_wheels

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class AudioBuffer:
    """Circular buffer for audio data."""
    
    def __init__(self, max_duration_ms: int = 3000, sample_rate: int = 8000) -> None:
        """Initialize audio buffer.
        
        Args:
            max_duration_ms: Maximum buffer duration in milliseconds
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.max_samples = (max_duration_ms * sample_rate) // 1000
        self._buffer: deque[bytes] = deque(maxlen=self.max_samples // 160)  # 20ms frames
        self._lock = asyncio.Lock()
    
    async def add_frame(self, frame: bytes) -> None:
        """Add audio frame to buffer."""
        async with self._lock:
            self._buffer.append(frame)
    
    async def get_all(self) -> bytes:
        """Get all buffered audio and clear."""
        async with self._lock:
            if not self._buffer:
                return b''
            audio = b''.join(self._buffer)
            self._buffer.clear()
            return audio
    
    async def clear(self) -> None:
        """Clear buffer."""
        async with self._lock:
            self._buffer.clear()
    
    def get_duration_ms(self) -> int:
        """Get current buffer duration in milliseconds."""
        num_frames = len(self._buffer)
        return num_frames * 20  # 20ms per frame


class VoiceActivityDetector:
    """Voice activity detection using WebRTC VAD."""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 8000) -> None:
        """Initialize VAD.
        
        Args:
            aggressiveness: VAD aggressiveness (0-3)
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speech = False
    
    def process_frame(self, frame: bytes, frame_duration_ms: int = 20) -> tuple[bool, bool]:
        """Process audio frame and detect voice activity.
        
        Args:
            frame: Audio frame (PCM16)
            frame_duration_ms: Frame duration in milliseconds
            
        Returns:
            Tuple of (is_speech, speech_ended)
            - is_speech: True if current frame contains speech
            - speech_ended: True if speech segment just ended
        """
        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            _LOGGER.warning(f"VAD error: {e}")
            return False, False
        
        speech_ended = False
        
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            
            # Start of speech detected
            if not self._is_speech and self._speech_frames >= 3:
                self._is_speech = True
                _LOGGER.debug("Speech started")
        else:
            self._silence_frames += 1
            
            # End of speech detected (150ms of silence after speech)
            if self._is_speech and self._silence_frames >= 8:  # 160ms
                self._is_speech = False
                self._speech_frames = 0
                speech_ended = True
                _LOGGER.debug("Speech ended")
        
        return self._is_speech, speech_ended
    
    def reset(self) -> None:
        """Reset VAD state."""
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speech = False


class AudioCodec:
    """Audio codec conversion utilities."""
    
    @staticmethod
    def ulaw_to_pcm16(data: bytes) -> bytes:
        """Convert μ-law to PCM16."""
        return audioop.ulaw2lin(data, 2)
    
    @staticmethod
    def pcm16_to_ulaw(data: bytes) -> bytes:
        """Convert PCM16 to μ-law."""
        return audioop.lin2ulaw(data, 2)
    
    @staticmethod
    def alaw_to_pcm16(data: bytes) -> bytes:
        """Convert A-law to PCM16."""
        return audioop.alaw2lin(data, 2)
    
    @staticmethod
    def pcm16_to_alaw(data: bytes) -> bytes:
        """Convert PCM16 to A-law."""
        return audioop.lin2alaw(data, 2)
    
    @staticmethod
    def resample(data: bytes, from_rate: int, to_rate: int, channels: int = 1) -> bytes:
        """Resample audio data."""
        if from_rate == to_rate:
            return data
        
        # Convert to numpy array
        samples = np.frombuffer(data, dtype=np.int16)
        
        # Calculate new length
        new_length = int(len(samples) * to_rate / from_rate)
        
        # Resample using linear interpolation
        resampled = np.interp(
            np.linspace(0, len(samples), new_length),
            np.arange(len(samples)),
            samples
        ).astype(np.int16)
        
        return resampled.tobytes()
    
    @staticmethod
    def adjust_volume(data: bytes, factor: float) -> bytes:
        """Adjust audio volume."""
        if factor == 1.0:
            return data
        return audioop.mul(data, 2, factor)


class AudioBridge:
    """Bridge between SIP audio and Home Assistant Assist."""
    
    def __init__(
        self,
        hass: HomeAssistant,
        sample_rate: int = 8000,
        codec: str = "PCMU",
        vad_aggressiveness: int = 2,
        silence_timeout: float = 1.5,
    ) -> None:
        """Initialize audio bridge."""
        self.hass = hass
        self.sample_rate = sample_rate
        self.codec = codec
        self.silence_timeout = silence_timeout
        
        # Audio processing
        self._inbound_buffer = AudioBuffer(max_duration_ms=5000, sample_rate=sample_rate)
        self._outbound_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = VoiceActivityDetector(vad_aggressiveness, sample_rate)
        self._codec_converter = AudioCodec()
        
        # State
        self._is_processing = False
        self._silence_start: float | None = None
        self._speech_detected = False
        
        # Callbacks
        self._on_speech_complete_cb: Callable[[bytes], None] | None = None
    
    def set_speech_complete_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for when speech segment completes."""
        self._on_speech_complete_cb = callback
    
    async def process_inbound_audio(self, audio_data: bytes) -> None:
        """Process audio received from SIP call.
        
        Args:
            audio_data: Raw audio from SIP (codec-encoded)
        """
        # Convert from codec to PCM16
        if self.codec == "PCMU":
            pcm_data = self._codec_converter.ulaw_to_pcm16(audio_data)
        elif self.codec == "PCMA":
            pcm_data = self._codec_converter.alaw_to_pcm16(audio_data)
        elif self.codec == "opus":
            # TODO: Implement Opus decoding
            pcm_data = audio_data
        else:
            pcm_data = audio_data
        
        # Add to buffer
        await self._inbound_buffer.add_frame(pcm_data)
        
        # Process with VAD
        is_speech, speech_ended = self._vad.process_frame(pcm_data)
        
        if is_speech:
            self._speech_detected = True
            self._silence_start = None
        elif self._speech_detected and not is_speech:
            # Track silence duration
            if self._silence_start is None:
                self._silence_start = asyncio.get_event_loop().time()
            
            # Check if silence timeout reached
            silence_duration = asyncio.get_event_loop().time() - self._silence_start
            if silence_duration >= self.silence_timeout or speech_ended:
                await self._handle_speech_complete()
    
    async def _handle_speech_complete(self) -> None:
        """Handle completion of speech segment."""
        if not self._speech_detected:
            return
        
        # Get all buffered audio
        audio_data = await self._inbound_buffer.get_all()
        
        if len(audio_data) > 0 and self._on_speech_complete_cb:
            _LOGGER.debug(f"Speech complete, {len(audio_data)} bytes")
            
            # Call callback with PCM16 audio
            if asyncio.iscoroutinefunction(self._on_speech_complete_cb):
                await self._on_speech_complete_cb(audio_data)
            else:
                self._on_speech_complete_cb(audio_data)
        
        # Reset state
        self._speech_detected = False
        self._silence_start = None
        self._vad.reset()
    
    async def queue_outbound_audio(self, audio_data: bytes, sample_rate: int = 16000) -> None:
        """Queue audio to send to caller (from TTS).
        
        Args:
            audio_data: PCM16 audio data
            sample_rate: Sample rate of audio data
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_data = self._codec_converter.resample(
                audio_data,
                sample_rate,
                self.sample_rate
            )
        
        # Convert to codec
        if self.codec == "PCMU":
            encoded = self._codec_converter.pcm16_to_ulaw(audio_data)
        elif self.codec == "PCMA":
            encoded = self._codec_converter.pcm16_to_alaw(audio_data)
        elif self.codec == "opus":
            # TODO: Implement Opus encoding
            encoded = audio_data
        else:
            encoded = audio_data
        
        # Split into 20ms frames and queue
        frame_size = (self.sample_rate * 20) // 1000  # 20ms frame
        bytes_per_frame = frame_size * 2  # 16-bit samples
        
        for i in range(0, len(encoded), bytes_per_frame):
            frame = encoded[i:i + bytes_per_frame]
            if len(frame) < bytes_per_frame:
                # Pad last frame with silence
                frame += b'\x00' * (bytes_per_frame - len(frame))
            await self._outbound_queue.put(frame)
    
    def get_outbound_frame(self, size: int) -> bytes | None:
        """Get next outbound audio frame (called from SIP thread).
        
        Args:
            size: Requested frame size in bytes
            
        Returns:
            Audio frame or None if no audio available
        """
        try:
            # Non-blocking get
            return self._outbound_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    async def clear_buffers(self) -> None:
        """Clear all audio buffers."""
        await self._inbound_buffer.clear()
        
        # Clear outbound queue
        while not self._outbound_queue.empty():
            try:
                self._outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self._vad.reset()
        self._speech_detected = False
        self._silence_start = None
    
    async def play_tone(self, frequency: int = 440, duration_ms: int = 200) -> None:
        """Generate and queue a simple tone.
        
        Args:
            frequency: Tone frequency in Hz
            duration_ms: Duration in milliseconds
        """
        num_samples = (self.sample_rate * duration_ms) // 1000
        t = np.linspace(0, duration_ms / 1000, num_samples)
        
        # Generate sine wave
        samples = np.sin(2 * np.pi * frequency * t)
        
        # Convert to int16
        samples = (samples * 32767 * 0.3).astype(np.int16)  # 30% volume
        
        await self.queue_outbound_audio(samples.tobytes(), self.sample_rate)
    
    async def play_dtmf_acknowledgment(self) -> None:
        """Play short beep to acknowledge DTMF input."""
        await self.play_tone(800, 100)