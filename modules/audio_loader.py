"""
Audio Loader and Synthetic Signal Generator Module
==================================================
Handles audio file I/O, format conversions, metadata extraction,
and generation of realistic synthetic audio + noise mixtures for DSP validation.
"""

import io
import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
import scipy.signal
import soundfile as sf


def get_audio_metadata(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Extract key time-domain and structural metadata from an audio array.
    """
    num_samples = len(audio) if audio.ndim == 1 else audio.shape[0]
    num_channels = 1 if audio.ndim == 1 else audio.shape[1]
    duration = float(num_samples) / float(sample_rate) if sample_rate > 0 else 0.0

    peak_amp = float(np.max(np.abs(audio))) if num_samples > 0 else 0.0
    rms_amp = float(np.sqrt(np.mean(np.square(audio)))) if num_samples > 0 else 0.0
    crest_factor = (peak_amp / (rms_amp + 1e-12)) if rms_amp > 1e-12 else 0.0
    crest_factor_db = 20.0 * math.log10(max(crest_factor, 1e-6))

    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": round(duration, 3),
        "num_samples": int(num_samples),
        "num_channels": int(num_channels),
        "peak_amplitude": round(peak_amp, 4),
        "rms_amplitude": round(rms_amp, 4),
        "crest_factor_db": round(crest_factor_db, 2),
    }


def load_audio(
    file_source: Any,
    target_sr: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Load an audio file from a path or file-like buffer (e.g. Streamlit UploadedFile).

    Parameters:
    -----------
    file_source : str, Path, or file-like buffer
        Audio file path or BytesIO buffer.
    target_sr : int, optional
        Target sample rate for resampling. If None, original sample rate is retained.
    mono : bool, default True
        If True, converts multi-channel audio to single-channel mono by averaging.

    Returns:
    --------
    audio : np.ndarray
        Audio signal as 1D (mono) float32 numpy array normalized to [-1.0, 1.0].
    sample_rate : int
        Effective sample rate in Hz.
    metadata : dict
        Dictionary of audio file properties.
    """
    # Read audio via soundfile
    if hasattr(file_source, "read"):
        file_source.seek(0)
        data, sr = sf.read(file_source, dtype="float32")
    else:
        data, sr = sf.read(str(file_source), dtype="float32")

    # Multi-channel to mono conversion
    if data.ndim > 1:
        if mono:
            data = np.mean(data, axis=1)
        else:
            data = data[:, 0]  # default to primary channel if multi-channel array not flattened

    # Optional resampling
    if target_sr is not None and target_sr != sr:
        gcd = math.gcd(int(target_sr), int(sr))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sr = int(target_sr)

    # Ensure float32 and shape (N,)
    data = np.ascontiguousarray(data, dtype=np.float32).ravel()

    meta = get_audio_metadata(data, sr)
    return data, sr, meta


def save_audio_to_bytes(audio: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> bytes:
    """
    Export audio array to in-memory WAV bytes buffer for download or playback.
    """
    clean_audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    clean_audio = np.clip(clean_audio, -1.0, 1.0).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, clean_audio, sample_rate, format="WAV", subtype=subtype)
    buf.seek(0)
    return buf.getvalue()


def save_audio(file_path: str, audio: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> None:
    """
    Write audio array to disk in WAV format safely.
    """
    clean_audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    clean_audio = np.clip(clean_audio, -1.0, 1.0).astype(np.float32)
    sf.write(file_path, clean_audio, sample_rate, format="WAV", subtype=subtype)


def generate_synthetic_speech_signal(
    duration: float = 3.5,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Synthesize realistic voiced harmonic speech-like formant signal.
    Combines a glottal fundamental frequency f0 ~ 130 Hz with vocal tract
    formants (F1=500 Hz, F2=1500 Hz, F3=2500 Hz) modulated by syllabic envelopes.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Formant frequencies and bandwidths typical of vowels /a/, /i/, /u/
    # Initial 0.40s is silent room background to provide reliable noise profile
    vowel_segments = [
        {"start": 0.40, "end": 1.25, "f0": 130.0, "formants": [(700, 0.4), (1220, 0.3), (2600, 0.15)]},
        {"start": 1.45, "end": 2.30, "f0": 125.0, "formants": [(300, 0.5), (2200, 0.35), (3000, 0.12)]},
        {"start": 2.50, "end": 3.35, "f0": 115.0, "formants": [(350, 0.45), (800, 0.3), (2250, 0.1)]},
    ]

    for seg in vowel_segments:
        idx = (t >= seg["start"]) & (t <= seg["end"])
        t_seg = t[idx] - seg["start"]
        seg_len = len(t_seg)
        if seg_len == 0:
            continue

        # Smooth envelope (Tukey/Hann window)
        envelope = np.sin(np.pi * np.linspace(0, 1, seg_len)) ** 2

        seg_signal = np.zeros(seg_len)
        f0 = seg["f0"]
        # Harmonic excitation
        for harmonic in range(1, 25):
            freq = harmonic * f0
            if freq >= sample_rate / 2:
                break
            # Formant shaping filter response
            formant_gain = 0.02
            for f_res, amp in seg["formants"]:
                # Lorentzian resonance curve
                bw = 90.0
                formant_gain += amp * (bw**2) / ((freq - f_res)**2 + bw**2)

            seg_signal += formant_gain * np.sin(2.0 * np.pi * freq * t_seg)

        signal[idx] += seg_signal * envelope

    # Normalize to [-0.8, 0.8]
    max_val = np.max(np.abs(signal))
    if max_val > 1e-6:
        signal = 0.8 * (signal / max_val)

    return signal.astype(np.float32)


def generate_synthetic_noise(
    noise_type: str,
    duration: float,
    sample_rate: int
) -> np.ndarray:
    """
    Generate realistic acoustic interference noise signals:
    - 'white': Flat spectrum Gaussian white noise.
    - 'hum': 50 Hz power-line electromagnetic hum with 100 Hz, 150 Hz, 250 Hz harmonics.
    - 'fan': Low-frequency colored blower noise (1/f shaped broadband rumble).
    - 'traffic': Pink/Brownian low-frequency rumble with irregular fluctuation.
    - 'mixed': Combination of hum, fan, and white noise.
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    rng = np.random.default_rng(seed=42)

    noise_type = noise_type.lower()

    if noise_type == "white":
        noise = rng.normal(0, 1.0, num_samples)

    elif noise_type in ("hum", "electrical_hum", "powerline"):
        # 50 Hz fundamental + strong harmonics
        noise = (
            0.60 * np.sin(2 * np.pi * 50.0 * t)
            + 0.30 * np.sin(2 * np.pi * 100.0 * t + 0.5)
            + 0.15 * np.sin(2 * np.pi * 150.0 * t + 1.2)
            + 0.08 * np.sin(2 * np.pi * 250.0 * t + 2.0)
            + 0.05 * rng.normal(0, 0.1, num_samples)
        )

    elif noise_type in ("fan", "fan_noise"):
        # White noise filtered by low-pass 2nd order Butterworth at 350 Hz
        raw = rng.normal(0, 1.0, num_samples)
        sos = scipy.signal.butter(3, 350.0, btype="lowpass", fs=sample_rate, output="sos")
        noise = scipy.signal.sosfilt(sos, raw)
        # Add a slight motor blade whistling tone
        noise += 0.25 * np.sin(2 * np.pi * 180.0 * t)

    elif noise_type in ("traffic", "traffic_noise"):
        # Brownian 1/f^2 noise: integrated white noise passed through bandpass
        raw = rng.normal(0, 1.0, num_samples)
        brown = np.cumsum(raw)
        brown = brown - np.mean(brown)
        sos = scipy.signal.butter(2, [40.0, 600.0], btype="bandpass", fs=sample_rate, output="sos")
        noise = scipy.signal.sosfilt(sos, brown)
        # Add intermittent road vehicle rumble modulation
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t) ** 2
        noise = noise * mod

    elif noise_type in ("mixed", "mixed_noise"):
        # Hum + Fan + Background White noise
        n_white = rng.normal(0, 0.3, num_samples)
        n_hum = 0.5 * np.sin(2 * np.pi * 50.0 * t) + 0.2 * np.sin(2 * np.pi * 150.0 * t)
        sos = scipy.signal.butter(2, 400.0, btype="lowpass", fs=sample_rate, output="sos")
        n_fan = scipy.signal.sosfilt(sos, rng.normal(0, 0.8, num_samples))
        noise = n_hum + n_fan + n_white

    else:
        # Default fallback: Gaussian white noise
        noise = rng.normal(0, 1.0, num_samples)

    # Normalize noise variance to unity
    noise_std = np.std(noise)
    if noise_std > 1e-6:
        noise = noise / noise_std

    return noise.astype(np.float32)


def create_noisy_mixture(
    clean_signal: np.ndarray,
    noise_type: str = "white",
    target_snr_db: float = 5.0,
    sample_rate: int = 16000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Mix a clean signal with noise at an exact specified Signal-to-Noise Ratio (SNR).

    Mathematical Foundation:
    ------------------------
    SNR_dB = 10 * log10( P_signal / P_noise )
    => P_noise_target = P_signal / (10 ^ (SNR_dB / 10))
    => scaling_factor = sqrt( P_noise_target / P_noise_actual )
    x[n] = s[n] + scaling_factor * d[n]

    Returns:
    --------
    noisy_signal : np.ndarray
        Mixed signal x[n] = s[n] + d_scaled[n].
    clean_signal : np.ndarray
        Original reference clean signal s[n].
    scaled_noise : np.ndarray
        Scaled additive noise signal d_scaled[n].
    actual_snr_db : float
        Calculated actual input SNR in dB.
    """
    clean_signal = np.ascontiguousarray(clean_signal, dtype=np.float32)
    duration = len(clean_signal) / float(sample_rate)

    raw_noise = generate_synthetic_noise(noise_type, duration, sample_rate)
    # Match length
    if len(raw_noise) < len(clean_signal):
        raw_noise = np.pad(raw_noise, (0, len(clean_signal) - len(raw_noise)))
    else:
        raw_noise = raw_noise[:len(clean_signal)]

    # Compute powers
    p_signal = np.mean(clean_signal ** 2)
    p_noise = np.mean(raw_noise ** 2)

    if p_signal < 1e-12:
        # Silent signal
        return raw_noise, clean_signal, raw_noise, 0.0

    target_snr_linear = 10.0 ** (target_snr_db / 10.0)
    target_p_noise = p_signal / target_snr_linear
    scale = math.sqrt(max(target_p_noise / (p_noise + 1e-12), 1e-12))

    scaled_noise = (raw_noise * scale).astype(np.float32)
    noisy_signal = (clean_signal + scaled_noise).astype(np.float32)

    # Actual achieved SNR
    actual_p_noise = np.mean(scaled_noise ** 2)
    actual_snr_db = 10.0 * math.log10(p_signal / (actual_p_noise + 1e-12))

    return noisy_signal, clean_signal, scaled_noise, round(float(actual_snr_db), 2)
