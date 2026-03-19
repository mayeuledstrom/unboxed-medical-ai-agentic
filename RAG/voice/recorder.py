"""
recorder.py — Audio recording via sounddevice.
Captures from the microphone until silence is detected, then saves as WAV PCM 16-bit.
"""

import os
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000        # Hz — optimal for Voxtral
CHANNELS = 1               # Mono
DTYPE = "float32"          # float32 internally, converted to int16 on write
SILENCE_THRESHOLD = 0.01   # RMS: silence threshold (adjustable per microphone)
SILENCE_DURATION = 2.0     # Consecutive seconds of silence to stop recording
MIN_SPEECH_DURATION = 0.5  # Minimum seconds before checking for silence
MAX_DURATION = 30.0        # Maximum recording duration in seconds
CHUNK_SIZE = 512            # Samples per detection frame


def _rms(data: np.ndarray) -> float:
    """Computes the Root Mean Square of an audio block."""
    return float(np.sqrt(np.mean(data ** 2)))


def record_until_silence(
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
    max_duration: float = MAX_DURATION,
) -> str:
    """
    Records from the microphone until silence is detected.
    Saves as WAV PCM 16-bit (optimal format for Voxtral).
    Returns the path to the created temporary WAV file.
    """
    print(f"[recorder] Listening... (silence > {silence_duration}s to stop, threshold={silence_threshold})")

    frames = []
    silence_frames = 0
    silence_limit = int(silence_duration * sample_rate / CHUNK_SIZE)
    min_frames = int(MIN_SPEECH_DURATION * sample_rate / CHUNK_SIZE)
    max_frames = int(max_duration * sample_rate / CHUNK_SIZE)
    started_speaking = False
    frame_count = 0

    stop_event = threading.Event()

    def callback(indata, frame_count_cb, time_info, status):
        nonlocal silence_frames, started_speaking, frame_count
        data = indata.copy()
        frames.append(data)
        level = _rms(data)
        frame_count += 1

        if level > silence_threshold:
            started_speaking = True
            silence_frames = 0
        elif started_speaking and frame_count > min_frames:
            # Only start counting silence after MIN_SPEECH_DURATION seconds
            silence_frames += 1
            if silence_frames >= silence_limit:
                stop_event.set()

        if frame_count >= max_frames:
            stop_event.set()

    with sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=CHUNK_SIZE,
        callback=callback,
    ):
        stop_event.wait()

    audio_float = np.concatenate(frames, axis=0)
    duration_s = len(audio_float) / sample_rate
    print(f"[recorder] Recording done ({duration_s:.1f}s, peak RMS={_rms(audio_float):.4f})")

    # Convert float32 → int16 PCM (standard format for STT APIs)
    audio_int16 = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_int16, sample_rate, subtype="PCM_16")
    print(f"[recorder] WAV PCM 16-bit → {tmp.name}")
    return tmp.name


def record_fixed_duration(duration: float = 5.0, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Alternative: records for a fixed duration.
    Useful for push-to-talk mode.
    """
    print(f"[recorder] Recording {duration}s...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_int16, sample_rate, subtype="PCM_16")
    print(f"[recorder] WAV PCM 16-bit ({duration}s) → {tmp.name}")
    return tmp.name


def calibrate_threshold(duration: float = 2.0, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Measures the ambient noise level and suggests an appropriate threshold.
    Useful for diagnosing silence detection issues.
    """
    print(f"[recorder] Calibrating ({duration}s of silence, do not speak)...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    noise_rms = _rms(audio)
    suggested = noise_rms * 3.0  # threshold = 3x background noise
    print(f"[recorder] Ambient noise RMS={noise_rms:.4f} → suggested threshold: {suggested:.4f}")
    print(f"[recorder] (Current: {SILENCE_THRESHOLD})")
    return suggested
