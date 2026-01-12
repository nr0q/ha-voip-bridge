"""Audio bridge for VoIP integration."""
import asyncio
import logging
from typing import Callable, Optional
import numpy as np
from collections import deque
import time

from vad import EnergyVAD

_LOGGER = logging.getLogger(__name__)


class AudioBuffer:
    """Circular buffer for audio data."""
    
    def __init__(self, max_duration: float = 3.0, sample_rate: int = 8000):
        """Initialize audio buffer.
        
        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = deque(maxlen=self.max_samples)
    
    def add(self, audio_data: bytes):
        """Add audio data to buffer.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
        """
        # Convert bytes to int16 samples
        samples = np.frombuffer(audio_data, dtype=np.int16)
        self.buffer.extend(samples)
    
    def get_all(self) -> np.ndarray:
        """Get all buffered audio as numpy array."""
        return np.array(self.buffer, dtype=np.int16)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class VoiceActivityDetector:
    """Voice activity detection using energy-based VAD."""
    
    def __init__(
        self,
        sample_rate: int = 8000,
        energy_threshold: float = 0.05,
        silence_timeout: float = 1.5,
    ):
        """Initialize VAD.
        
        Args:
            sample_rate: Audio sample rate
            energy_threshold: Energy threshold for speech detection
            silence_timeout: Seconds of silence before ending speech
        """
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        
        # Initialize EnergyVAD
        self.vad = EnergyVAD(
            sample_rate=sample_rate,
            frame_length=25,  # ms
            frame_shift=20,   # ms
            energy_threshold=energy_threshold,
            pre_emphasis=0.95,
        )
        
        # State tracking
        self.is_speaking = False
        self.last_speech_time = 0
        self.speech_buffer = []
    
    def process_audio(self, audio_data: bytes) -> tuple[bool, Optional[bytes]]:
        """Process audio chunk and detect voice activity.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            
        Returns:
            Tuple of (is_speech_detected, completed_speech_audio)
            completed_speech_audio is None unless speech segment just ended
        """
        # Convert to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)
        
        # Run VAD on this chunk
        # VAD expects float32 normalized to [-1, 1]
        normalized = samples.astype(np.float32) / 32768.0
        voice_activity = self.vad(normalized)
        
        # Check if any frames in this chunk contain speech
        has_speech = np.any(voice_activity)
        current_time = time.time()
        
        if has_speech:
            # Speech detected
            self.last_speech_time = current_time
            
            if not self.is_speaking:
                # Start of new speech segment
                _LOGGER.debug("Speech started")
                self.is_speaking = True
                self.speech_buffer = []
            
            # Add to speech buffer
            self.speech_buffer.append(audio_data)
            return True, None
        
        elif self.is_speaking:
            # Currently in speech, but this chunk is silence
            silence_duration = current_time - self.last_speech_time
            
            # Add to buffer (capture trailing silence)
            self.speech_buffer.append(audio_data)
            
            if silence_duration >= self.silence_timeout:
                # End of speech segment
                _LOGGER.debug(f"Speech ended after {silence_duration:.2f}s silence")
                self.is_speaking = False
                
                # Combine all buffered audio
                complete_audio = b''.join(self.speech_buffer)
                self.speech_buffer = []
                
                return False, complete_audio
            
            return True, None  # Still in speech segment, waiting for timeout
        
        else:
            # Not speaking, no speech detected
            return False, None
    
    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.last_speech_time = 0
        self.speech_buffer = []


class AudioCodec:
    """Audio codec conversion utilities."""
    
    @staticmethod
    def pcm_to_mulaw(pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to μ-law (PCMU).
        
        Args:
            pcm_data: 16-bit PCM audio data
            
        Returns:
            μ-law encoded audio data
        """
        # Convert bytes to int16 array
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Normalize to [-1, 1]
        normalized = samples.astype(np.float32) / 32768.0
        
        # Apply μ-law companding
        mu = 255.0
        sign = np.sign(normalized)
        abs_val = np.abs(normalized)
        compressed = sign * np.log(1 + mu * abs_val) / np.log(1 + mu)
        
        # Scale to [0, 255]
        mulaw = ((compressed + 1) / 2 * 255).astype(np.uint8)
        
        return mulaw.tobytes()
    
    @staticmethod
    def mulaw_to_pcm(mulaw_data: bytes) -> bytes:
        """Convert μ-law (PCMU) to 16-bit PCM.
        
        Args:
            mulaw_data: μ-law encoded audio data
            
        Returns:
            16-bit PCM audio data
        """
        # Convert bytes to uint8 array
        mulaw = np.frombuffer(mulaw_data, dtype=np.uint8)
        
        # Normalize to [-1, 1]
        normalized = (mulaw.astype(np.float32) / 255.0) * 2 - 1
        
        # Apply μ-law expansion
        mu = 255.0
        sign = np.sign(normalized)
        abs_val = np.abs(normalized)
        expanded = sign * (np.power(1 + mu, abs_val) - 1) / mu
        
        # Scale to int16
        pcm = (expanded * 32768).astype(np.int16)
        
        return pcm.tobytes()


class AudioBridge:
    """Bridge between SIP audio and Home Assistant."""
    
    def __init__(
        self,
        hass,
        sample_rate: int = 8000,
        codec: str = "PCMU",
        energy_threshold: float = 0.05,
        silence_timeout: float = 1.5,
    ):
        """Initialize audio bridge.
        
        Args:
            hass: Home Assistant instance
            sample_rate: Audio sample rate (8000 for PCMU/PCMA)
            codec: Audio codec (PCMU or PCMA)
            energy_threshold: VAD energy threshold
            silence_timeout: Seconds of silence before ending speech
        """
        self.hass = hass
        self.sample_rate = sample_rate
        self.codec = codec
        
        # Initialize VAD
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            silence_timeout=silence_timeout,
        )
        
        # Audio buffers
        self.inbound_buffer = AudioBuffer(max_duration=3.0, sample_rate=sample_rate)
        self.outbound_buffer = AudioBuffer(max_duration=10.0, sample_rate=sample_rate)
        
        # Callbacks
        self._speech_complete_callback: Optional[Callable] = None
        
        _LOGGER.info(
            f"Audio bridge initialized: {sample_rate}Hz, codec={codec}, "
            f"threshold={energy_threshold}, timeout={silence_timeout}s"
        )
    
    def set_speech_complete_callback(self, callback: Callable):
        """Set callback for when speech segment completes.
        
        Args:
            callback: Async function to call with completed audio (bytes)
        """
        self._speech_complete_callback = callback
    
    async def process_inbound_audio(self, audio_data: bytes):
        """Process incoming audio from call.
        
        Args:
            audio_data: Raw audio data (codec-encoded)
        """
        # _LOGGER.info(f"Processing {len(audio_data)} bytes")
        # Decode from codec to PCM16
        if self.codec == "PCMU":
            pcm_data = AudioCodec.mulaw_to_pcm(audio_data)
        else:
            # Assume already PCM
            pcm_data = audio_data
        
        # Run VAD
        is_speech, completed_audio = self.vad.process_audio(pcm_data)
        
        # If speech segment completed, trigger callback
        if completed_audio and self._speech_complete_callback:
            await self._speech_complete_callback(completed_audio)
    
    async def queue_outbound_audio(self, audio_data: bytes):
        """Queue audio for outbound transmission.
        
        Args:
            audio_data: PCM16 audio data to send
        """
        # Add to outbound buffer
        self.outbound_buffer.add(audio_data)
    
    def get_outbound_frame(self, size: int) -> Optional[bytes]:
        """Get next outbound audio frame.
        
        Args:
            size: Number of samples needed
            
        Returns:
            Codec-encoded audio frame, or None if no audio available
        """
        # Get samples from buffer
        all_samples = self.outbound_buffer.get_all()
        
        if len(all_samples) < size:
            # Not enough audio
            return None
        
        # Extract frame
        frame_samples = all_samples[:size]
        
        # Remove from buffer (convert back to deque operations)
        for _ in range(size):
            if len(self.outbound_buffer.buffer) > 0:
                self.outbound_buffer.buffer.popleft()
        
        # Encode to codec
        pcm_bytes = frame_samples.tobytes()
        
        if self.codec == "PCMU":
            return AudioCodec.pcm_to_mulaw(pcm_bytes)
        else:
            return pcm_bytes
    
    async def clear_buffers(self):
        """Clear all audio buffers."""
        self.inbound_buffer.clear()
        self.outbound_buffer.clear()
        self.vad.reset()
    
    async def play_tone(self, frequency: int = 440, duration: float = 0.2):
        """Play a simple tone (for acknowledgments).
        
        Args:
            frequency: Tone frequency in Hz
            duration: Tone duration in seconds
        """
        # Generate sine wave
        num_samples = int(self.sample_rate * duration)
        _LOGGER.info(f"Generating tone: {num_samples} samples")
        t = np.linspace(0, duration, num_samples, False)
        tone = np.sin(2 * np.pi * frequency * t)
        
        # Scale to int16
        tone_int16 = (tone * 32767 * 0.3).astype(np.int16)  # 30% volume
        
        # Queue for output
        await self.queue_outbound_audio(tone_int16.tobytes())
        _LOGGER.info(f"Tone queued, buffer now has {len(self.outbound_buffer.get_all())} samples")