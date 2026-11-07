"""
LMS and Normalized LMS (NLMS) Adaptive Filter Module
====================================================
Implements adaptive noise cancellation (ANC) using:
- Standard Least Mean Squares (LMS)
- Normalized Least Mean Squares (NLMS)
- Reference noise input handling (external reference, hum generator, or delay line / ALE)
- Convergence trajectory tracking and coefficient inspection
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def generate_reference_noise_if_none(
    primary_signal: np.ndarray,
    sample_rate: int,
    reference_mode: str = "delay_line",
    hum_freq_hz: float = 50.0,
    delay_samples: int = 160
) -> np.ndarray:
    """
    Generate an appropriate reference noise signal when no secondary microphone is available.

    Modes:
    ------
    - 'hum_reference': Generates a synthetic reference tone containing the hum frequency
      and its harmonics (ideal for eliminating 50 Hz/60 Hz power-line interference).
    - 'delay_line' (Adaptive Line Enhancer): Uses a delayed version of the primary input:
      x[n] = d[n - Delta]. Delay decorrelates non-stationary speech while keeping correlated noise.
    - 'correlated_noise': Synthetic white or lowpass noise correlated with the primary signal.
    """
    N = len(primary_signal)
    t = np.linspace(0, float(N) / sample_rate, N, endpoint=False)

    if reference_mode == "hum_reference":
        ref = (
            0.7 * np.sin(2.0 * np.pi * hum_freq_hz * t)
            + 0.3 * np.sin(2.0 * np.pi * 2.0 * hum_freq_hz * t + 0.3)
            + 0.15 * np.sin(2.0 * np.pi * 3.0 * hum_freq_hz * t + 0.7)
        )
        return ref.astype(np.float32)

    elif reference_mode == "delay_line":
        # Delayed primary input
        delay = max(1, int(delay_samples))
        ref = np.zeros_like(primary_signal)
        ref[delay:] = primary_signal[:-delay]
        return ref.astype(np.float32)

    else:
        # Default fallback: correlated noise model
        rng = np.random.default_rng(seed=123)
        return rng.normal(0, 0.5, N).astype(np.float32)


def lms_adaptive_filter(
    primary_signal: np.ndarray,
    reference_signal: Optional[np.ndarray],
    sample_rate: int,
    filter_order: int = 32,
    mu: float = 0.01,
    normalized: bool = True,
    eps: float = 1e-6,
    reference_mode: str = "delay_line",
    hum_freq_hz: float = 50.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Adaptive Noise Cancellation (ANC) using LMS / Normalized LMS.

    DSP Mathematical Formulation:
    -----------------------------
    Primary input (Speech + Noise):
        d[n] = s[n] + n_0[n]

    Reference input (Noise correlated with n_0):
        x[n] = [x[n], x[n-1], ..., x[n-M+1]]^T

    Filter output (estimated noise):
        y[n] = w^T[n] * x[n] = sum_{k=0}^{M-1} w_k[n] * x[n-k]

    Error signal (enhanced speech):
        e[n] = d[n] - y[n]

    Coefficient update:
        Standard LMS:
            w[n+1] = w[n] + 2 * mu * e[n] * x[n]
        Normalized LMS (NLMS):
            w[n+1] = w[n] + (mu / (eps + ||x[n]||^2)) * e[n] * x[n]

    Parameters:
    -----------
    primary_signal : np.ndarray
        Noisy signal d[n].
    reference_signal : np.ndarray or None
        Reference noise signal x[n]. If None, generated via `reference_mode`.
    sample_rate : int
        Audio sampling rate in Hz.
    filter_order : int, default 32
        Filter tap length M (typically 16 to 128).
    mu : float, default 0.01
        Adaptation step size parameter.
    normalized : bool, default True
        Whether to use Normalized LMS (NLMS) for stability.
    eps : float, default 1e-6
        Regularization term preventing division by zero.
    reference_mode : str
        Method to generate reference if reference_signal is None.
    hum_freq_hz : float
        Frequency in Hz if using hum reference.

    Returns:
    --------
    enhanced_audio : np.ndarray
        Error signal e[n] representing enhanced audio.
    estimated_noise : np.ndarray
        Filter output y[n].
    squared_error_history : np.ndarray
        e^2[n] trajectory showing convergence over time.
    info : dict
        Execution metrics and final filter weights.
    """
    d = np.asarray(primary_signal, dtype=np.float32)
    N = len(d)
    M = int(np.clip(filter_order, 4, 256))

    if reference_signal is None or len(reference_signal) != N:
        x = generate_reference_noise_if_none(
            d, sample_rate, reference_mode=reference_mode, hum_freq_hz=hum_freq_hz
        )
    else:
        x = np.asarray(reference_signal, dtype=np.float32)

    # Initialize filter state and buffers
    w = np.zeros(M, dtype=np.float32)
    x_buffer = np.zeros(M, dtype=np.float32)

    y = np.zeros(N, dtype=np.float32)
    e = np.zeros(N, dtype=np.float32)
    e_sq = np.zeros(N, dtype=np.float32)

    # Step size limit check
    mu = float(np.clip(mu, 1e-5, 0.5))

    # Adaptive loop
    for n in range(N):
        # Shift input buffer
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = x[n]

        # Compute output: y[n] = w^T * x
        y_n = float(np.dot(w, x_buffer))
        y[n] = y_n

        # Compute error: e[n] = d[n] - y[n]
        e_n = float(d[n] - y_n)
        e[n] = e_n
        e_sq[n] = e_n ** 2

        # Weight update
        if normalized:
            norm_factor = eps + float(np.dot(x_buffer, x_buffer))
            w += (mu / norm_factor) * e_n * x_buffer
        else:
            w += (2.0 * mu) * e_n * x_buffer

    # Smoothing error energy for clear visualization
    window_len = min(512, N // 10) if N > 512 else 1
    if window_len > 1:
        smooth_kernel = np.ones(window_len, dtype=np.float32) / window_len
        e_sq_smoothed = np.convolve(e_sq, smooth_kernel, mode="same")
    else:
        e_sq_smoothed = e_sq

    # Prevent digital clipping in enhanced audio
    peak = np.max(np.abs(e))
    if peak > 0.99:
        e = 0.95 * (e / peak)

    info = {
        "algorithm": "Normalized LMS (NLMS)" if normalized else "Standard LMS",
        "filter_order": M,
        "step_size_mu": mu,
        "normalized": normalized,
        "reference_mode": reference_mode,
        "final_weights_norm": round(float(np.linalg.norm(w)), 4),
        "initial_error_energy": round(float(np.mean(e_sq[:min(1000, N)])), 5),
        "final_error_energy": round(float(np.mean(e_sq[-min(1000, N):])), 5),
    }

    return e.astype(np.float32), y.astype(np.float32), e_sq_smoothed.astype(np.float32), info
