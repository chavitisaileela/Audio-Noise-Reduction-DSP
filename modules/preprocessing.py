"""
Audio Preprocessing and Signal Conditioning Module
==================================================
Performs validation, DC offset removal, amplitude normalization,
framing, windowing, clipping detection, and noise-profile segment estimation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def validate_signal(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validate input signal, sanitize NaN/Inf values, and assess signal integrity.
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Input audio signal is empty or None.")

    # Flatten if multi-dimensional
    signal = np.asarray(audio, dtype=np.float32).ravel()

    # Check for NaN / Inf
    nan_count = int(np.sum(np.isnan(signal)))
    inf_count = int(np.sum(np.isinf(signal)))
    if nan_count > 0 or inf_count > 0:
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)

    # Check for silence or flatline
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-7:
        status = "silent"
    else:
        status = "valid"

    # Check clipping count (samples reaching or exceeding +/-0.999)
    clipping_count = int(np.sum(np.abs(signal) >= 0.999))
    clipping_pct = (clipping_count / len(signal)) * 100.0

    report = {
        "status": status,
        "num_samples": len(signal),
        "nan_replaced": nan_count,
        "inf_replaced": inf_count,
        "clipping_samples": clipping_count,
        "clipping_percentage": round(clipping_pct, 3),
        "peak_amplitude": round(peak, 4),
    }
    return signal, report


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC bias (zero-frequency offset) from the signal:
    x_ac[n] = x[n] - mean(x)
    """
    signal = np.asarray(audio, dtype=np.float32)
    mean_val = np.mean(signal)
    return (signal - mean_val).astype(np.float32)


def normalize_signal(
    audio: np.ndarray,
    target_peak: float = 0.95,
    method: str = "peak"
) -> np.ndarray:
    """
    Normalize signal amplitude to avoid clipping while maximizing dynamic range.

    Parameters:
    -----------
    audio : np.ndarray
        Input audio array.
    target_peak : float, default 0.95
        Maximum absolute peak value allowed (typically 0.90 to 0.98 to preserve headroom).
    method : str
        'peak' for peak normalization, or 'rms' for RMS normalization.
    """
    signal = np.asarray(audio, dtype=np.float32)
    if len(signal) == 0:
        return signal

    if method == "peak":
        peak = np.max(np.abs(signal))
        if peak > 1e-8:
            return (signal * (target_peak / peak)).astype(np.float32)
        return signal

    elif method == "rms":
        rms = np.sqrt(np.mean(signal ** 2))
        target_rms = target_peak * 0.25  # ~12 dB crest factor headroom
        if rms > 1e-8:
            scaled = signal * (target_rms / rms)
            # Prevent hard clipping
            peak = np.max(np.abs(scaled))
            if peak > target_peak:
                scaled = scaled * (target_peak / peak)
            return scaled.astype(np.float32)
        return signal

    return signal


def frame_signal(
    audio: np.ndarray,
    frame_size: int = 512,
    hop_size: int = 256,
    window_type: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment 1D signal into overlapping windowed frames for Short-Time processing.

    Returns:
    --------
    frames : np.ndarray of shape (num_frames, frame_size)
        Windowed frames.
    window : np.ndarray of shape (frame_size,)
        Window analysis vector.
    """
    signal = np.asarray(audio, dtype=np.float32)
    sig_len = len(signal)

    if window_type.lower() == "hann":
        window = np.hanning(frame_size).astype(np.float32)
    elif window_type.lower() == "hamming":
        window = np.hamming(frame_size).astype(np.float32)
    elif window_type.lower() == "blackman":
        window = np.blackman(frame_size).astype(np.float32)
    else:
        window = np.ones(frame_size, dtype=np.float32)

    # Pad signal so last frame is complete
    num_frames = int(np.ceil(float(max(sig_len - frame_size, 0)) / hop_size)) + 1
    pad_len = (num_frames - 1) * hop_size + frame_size
    padded_signal = np.pad(signal, (0, max(0, pad_len - sig_len)), mode="constant")

    # Stride tricks / indexing for fast framing
    indices = (
        np.tile(np.arange(0, frame_size), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * hop_size, hop_size), (frame_size, 1)).T
    )
    frames = padded_signal[indices] * window

    return frames.astype(np.float32), window


def estimate_noise_profile(
    audio: np.ndarray,
    sample_rate: int,
    noise_duration_sec: float = 0.35,
    method: str = "initial"
) -> Tuple[np.ndarray, int, int]:
    """
    Extract a representative noise segment from the signal for DSP noise estimators.

    Methods:
    --------
    - 'initial': Extract the first `noise_duration_sec` seconds (standard assumption
      of lead-in recording room tone/noise before speaker talks).
    - 'minimum_energy': Scan signal for the lowest-energy window of duration `noise_duration_sec`.

    Returns:
    --------
    noise_segment : np.ndarray
        Array containing the isolated noise samples.
    start_idx : int
        Start sample index.
    end_idx : int
        End sample index.
    """
    signal = np.asarray(audio, dtype=np.float32)
    win_len = int(sample_rate * noise_duration_sec)
    win_len = min(win_len, len(signal))

    if win_len < 128:
        return signal, 0, len(signal)

    if method == "initial" or len(signal) <= win_len * 2:
        return signal[:win_len], 0, win_len

    # Minimum energy sliding search
    # Downsample energy search using block sums
    block_size = max(128, win_len // 10)
    num_blocks = len(signal) // block_size
    energies = [
        np.sum(signal[i * block_size : i * block_size + win_len] ** 2)
        for i in range(num_blocks - (win_len // block_size))
    ]

    if len(energies) > 0:
        min_block = int(np.argmin(energies))
        start_idx = min_block * block_size
        end_idx = start_idx + win_len
        return signal[start_idx:end_idx], start_idx, end_idx

    return signal[:win_len], 0, win_len
