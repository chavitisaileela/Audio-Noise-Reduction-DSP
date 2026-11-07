"""
Digital Butterworth Filter Module
=================================
Implements IIR Butterworth digital filters (Low-Pass, High-Pass, Band-Pass, Notch)
using Second-Order Sections (SOS) and zero-phase forward-backward filtering.
Provides frequency response calculation for Bode plots and ECE visualization.
"""

from typing import Tuple, Dict, Any
import numpy as np
import scipy.signal


def validate_filter_frequencies(
    filter_type: str,
    cutoff_hz: Any,
    sample_rate: int
) -> Tuple[bool, str]:
    """
    Validate that filter cutoff frequencies obey the Nyquist-Shannon criterion:
    0 < cutoff < fs / 2.
    """
    nyquist = sample_rate / 2.0

    if filter_type in ("lowpass", "highpass", "notch"):
        if not isinstance(cutoff_hz, (int, float)):
            return False, f"Cutoff frequency must be a single numeric value, got {type(cutoff_hz).__name__}."
        if cutoff_hz <= 0:
            return False, f"Cutoff frequency must be strictly positive (> 0 Hz), got {cutoff_hz} Hz."
        if cutoff_hz >= nyquist:
            return False, f"Cutoff frequency ({cutoff_hz} Hz) must be less than Nyquist limit ({nyquist} Hz)."

    elif filter_type == "bandpass":
        if not isinstance(cutoff_hz, (list, tuple)) or len(cutoff_hz) != 2:
            return False, "Bandpass filter requires a tuple of [low_cutoff, high_cutoff] in Hz."
        f_low, f_high = cutoff_hz
        if f_low <= 0:
            return False, f"Lower cutoff frequency must be positive, got {f_low} Hz."
        if f_high >= nyquist:
            return False, f"Upper cutoff frequency ({f_high} Hz) must be below Nyquist ({nyquist} Hz)."
        if f_low >= f_high:
            return False, f"Lower cutoff ({f_low} Hz) must be strictly less than upper cutoff ({f_high} Hz)."

    return True, "Valid"


def apply_butterworth_filter(
    audio: np.ndarray,
    sample_rate: int,
    filter_type: str = "lowpass",
    cutoff_hz: Any = 3000.0,
    order: int = 5,
    zero_phase: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply a digital Butterworth filter to the input audio signal.

    DSP Formulation:
    ----------------
    The Butterworth filter provides a maximally flat magnitude response in the passband:
        |H(j*w)|^2 = 1 / (1 + (w / w_c)^(2*N))

    Filtering is computed using Second-Order Sections (SOS) representation:
        H(z) = prod_{k=1}^K (b0_k + b1_k*z^-1 + b2_k*z^-2) / (1 + a1_k*z^-1 + a2_k*z^-2)

    Zero-phase filtering (filtfilt):
        y[n] = H(z) * H(z^-1) * x[n] => zero phase distortion, squared magnitude response.

    Parameters:
    -----------
    audio : np.ndarray
        1D input signal x[n].
    sample_rate : int
        Sampling frequency in Hz.
    filter_type : str
        'lowpass', 'highpass', 'bandpass', or 'notch'.
    cutoff_hz : float or tuple of float
        Cutoff frequency in Hz (single value for low/high/notch, tuple for bandpass).
    order : int, default 5
        Filter order N (1 to 10).
    zero_phase : bool, default True
        If True, applies forward-backward zero-phase filtering (no group delay).

    Returns:
    --------
    filtered_audio : np.ndarray
        Filtered audio signal y[n].
    sos : np.ndarray
        Second-order sections coefficients matrix.
    filter_info : dict
        Metadata and parameters of the applied filter.
    """
    signal = np.asarray(audio, dtype=np.float32)
    valid, msg = validate_filter_frequencies(filter_type, cutoff_hz, sample_rate)
    if not valid:
        raise ValueError(msg)

    order = int(np.clip(order, 1, 12))

    # Design filter coefficients in Second-Order Sections (SOS)
    if filter_type.lower() == "lowpass":
        sos = scipy.signal.butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    elif filter_type.lower() == "highpass":
        sos = scipy.signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    elif filter_type.lower() == "bandpass":
        sos = scipy.signal.butter(order, cutoff_hz, btype="bandpass", fs=sample_rate, output="sos")
    elif filter_type.lower() == "notch":
        # Notch filter designed using iirnotch
        f0 = float(cutoff_hz)
        q = 30.0  # Quality factor
        b, a = scipy.signal.iirnotch(f0, q, fs=sample_rate)
        sos = scipy.signal.tf2sos(b, a)
    else:
        raise ValueError(f"Unknown filter type '{filter_type}'. Must be lowpass, highpass, bandpass, or notch.")

    # Apply filter
    if zero_phase:
        # Zero-phase filtering: effective order is doubled, phase delay = 0
        filtered = scipy.signal.sosfiltfilt(sos, signal)
    else:
        filtered = scipy.signal.sosfilt(sos, signal)

    filter_info = {
        "filter_name": f"Butterworth {filter_type.capitalize()}",
        "filter_type": filter_type,
        "order": order,
        "effective_order": order * 2 if zero_phase else order,
        "cutoff_hz": cutoff_hz,
        "sample_rate": sample_rate,
        "zero_phase": zero_phase,
    }

    return filtered.astype(np.float32), sos, filter_info


def compute_filter_frequency_response(
    sos: np.ndarray,
    sample_rate: int,
    num_points: int = 1024
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the frequency response (Bode magnitude and phase) of the digital filter.

    Returns:
    --------
    frequencies : np.ndarray
        Frequency axis in Hz (0 to fs/2).
    magnitude_db : np.ndarray
        Magnitude response in decibels 20*log10(|H(w)|).
    phase_deg : np.ndarray
        Phase response in degrees.
    """
    w, h = scipy.signal.sosfreqz(sos, worN=num_points, fs=sample_rate)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-6))
    phase_deg = np.unwrap(np.angle(h)) * (180.0 / np.pi)

    return w, mag_db, phase_deg
