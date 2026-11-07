"""
Audio Signal Reconstruction and Post-Processing Module
======================================================
Guarantees signal integrity before audio playback and export:
- Boundary de-clicking with smooth cosine taper
- Peak limiting and dynamic range scaling
- Anti-clipping protection
- Real-valued enforcement and finite-value sanitization
"""

from typing import Tuple, Dict, Any
import numpy as np


def reconstruct_and_sanitize(
    audio: np.ndarray,
    target_peak: float = 0.95,
    fade_samples: int = 64,
    soft_clip: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Sanitize and condition processed audio signal for flawless audio output.

    Parameters:
    -----------
    audio : np.ndarray
        Raw output from a DSP algorithm (may contain small boundary artifacts or exceed 1.0).
    target_peak : float, default 0.95
        Maximum peak amplitude allowed.
    fade_samples : int, default 64
        Small boundary fade to eliminate click/pop transient at start and end.
    soft_clip : bool, default True
        If True, applies smooth hyperbolic tangent compression to extreme spikes.

    Returns:
    --------
    clean_audio : np.ndarray
        Guaranteed valid float32 audio in range [-target_peak, target_peak].
    report : dict
        Integrity statistics.
    """
    if audio is None or len(audio) == 0:
        return np.array([0.0], dtype=np.float32), {"status": "empty"}

    # Enforce real values
    if np.iscomplexobj(audio):
        audio = np.real(audio)

    signal = np.asarray(audio, dtype=np.float32).ravel()

    # Sanitize NaN / Inf
    nan_count = int(np.sum(np.isnan(signal)))
    inf_count = int(np.sum(np.isinf(signal)))
    if nan_count > 0 or inf_count > 0:
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)

    # Boundary fade-in and fade-out to prevent boundary discontinuities
    N = len(signal)
    if fade_samples > 0 and N > fade_samples * 2:
        fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        fade_out = fade_in[::-1]
        signal[:fade_samples] *= fade_in
        signal[-fade_samples:] *= fade_out

    orig_peak = float(np.max(np.abs(signal))) if N > 0 else 0.0

    # Soft clipping with tanh if signal has large transients
    if soft_clip and orig_peak > 1.0:
        # Smooth compression above threshold
        signal = np.tanh(signal)
        orig_peak = float(np.max(np.abs(signal)))

    # Rescale to target_peak headroom
    if orig_peak > target_peak:
        signal = signal * (target_peak / orig_peak)

    final_peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2)))

    report = {
        "original_peak": round(orig_peak, 4),
        "final_peak": round(final_peak, 4),
        "final_rms": round(rms, 4),
        "nan_cleaned": nan_count,
        "inf_cleaned": inf_count,
        "samples": N,
    }

    return signal.astype(np.float32), report
