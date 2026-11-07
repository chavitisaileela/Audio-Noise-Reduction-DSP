"""
Discrete Fourier Transform (FFT) and Spectrogram Analysis Module
================================================================
Implements classical DSP spectral analysis:
- One-sided Discrete Fourier Transform (FFT)
- Magnitude spectrum (linear and dB scale)
- Phase spectrum (wrapped and unwrapped)
- Short-Time Fourier Transform (STFT) and Spectrogram computation
- Spectral features (Centroid, Rolloff, Dominant Frequencies)
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import scipy.signal


def compute_fft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the one-sided Discrete Fourier Transform of an audio signal.

    DSP Formulation:
    ----------------
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * k * n / N)
    Magnitude: |X[k]|
    Magnitude (dB): 20 * log10( |X[k]| / max(|X|) + eps )
    Phase: angle(X[k]) = arctan2( Im(X[k]), Re(X[k]) )

    Returns:
    --------
    frequencies : np.ndarray
        Frequency axis in Hertz, ranging from 0 to fs/2.
    magnitude_linear : np.ndarray
        One-sided magnitude spectrum normalized by N.
    magnitude_db : np.ndarray
        Magnitude spectrum in decibels normalized to 0 dB peak.
    phase_rad : np.ndarray
        Phase spectrum in radians [-pi, pi].
    """
    signal = np.asarray(audio, dtype=np.float32)
    N = len(signal) if n_fft is None else int(n_fft)

    if N < 2:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])

    # Next power of 2 for fast radix-2 FFT execution if not specified
    if n_fft is None and N > 4096:
        # Zero-pad or segment if very large to prevent memory overload
        # For analysis, 8192 or 16384 bins provides sub-Hz resolution
        fft_len = min(65536, int(2 ** np.ceil(np.log2(N))))
    else:
        fft_len = N

    # Apply Hann window to minimize spectral leakage
    window = np.hanning(min(len(signal), fft_len))
    windowed_sig = signal[:len(window)] * window

    # One-sided real FFT
    fft_complex = np.fft.rfft(windowed_sig, n=fft_len)
    frequencies = np.fft.rfftfreq(fft_len, d=1.0 / sample_rate)

    # Magnitude calculation normalized by coherent window gain
    coherent_gain = np.sum(window) if np.sum(window) > 0 else 1.0
    magnitude_linear = (2.0 * np.abs(fft_complex)) / coherent_gain
    # DC component correction
    magnitude_linear[0] = magnitude_linear[0] / 2.0

    # Decibel scaling normalized to peak (max 0 dB, noise floor bounded at -100 dB)
    peak_mag = np.max(magnitude_linear)
    if peak_mag > 1e-12:
        magnitude_db = 20.0 * np.log10(np.maximum(magnitude_linear / peak_mag, 1e-5))
    else:
        magnitude_db = np.full_like(magnitude_linear, -100.0)

    # Phase spectrum
    phase_rad = np.angle(fft_complex)

    return frequencies, magnitude_linear, magnitude_db, phase_rad


def find_dominant_frequencies(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    top_k: int = 5,
    min_distance_hz: float = 30.0
) -> List[Dict[str, float]]:
    """
    Locate the strongest harmonic frequency peaks in the spectrum.
    """
    if len(frequencies) < 2 or len(magnitude) < 2:
        return []

    freq_step = frequencies[1] - frequencies[0]
    distance_bins = max(1, int(min_distance_hz / (freq_step + 1e-12)))

    # Find local maxima
    peaks, props = scipy.signal.find_peaks(
        magnitude,
        distance=distance_bins,
        prominence=0.01 * np.max(magnitude) if np.max(magnitude) > 0 else None
    )

    if len(peaks) == 0:
        # Fallback to direct argmax
        top_idx = np.argsort(magnitude)[::-1][:top_k]
        peaks = np.sort(top_idx)

    # Rank by magnitude descending
    sorted_peaks = sorted(peaks, key=lambda idx: magnitude[idx], reverse=True)[:top_k]

    results = []
    for idx in sorted_peaks:
        results.append({
            "frequency_hz": round(float(frequencies[idx]), 1),
            "magnitude": round(float(magnitude[idx]), 4),
        })
    return results


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_size: int = 256,
    window_type: str = "hann"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) and decibel Spectrogram.

    Parameters:
    -----------
    audio : np.ndarray
        1D input signal.
    sample_rate : int
        Audio sampling rate in Hz.
    n_fft : int, default 512
        Analysis window / FFT length.
    hop_size : int, default 256
        Frame advance step (50% overlap standard).
    window_type : str
        Window weighting ('hann', 'hamming', 'blackman').

    Returns:
    --------
    times : np.ndarray
        Time points (seconds) corresponding to STFT frame centers.
    frequencies : np.ndarray
        Frequency bins (Hz) from 0 to fs/2.
    spectrogram_db : np.ndarray of shape (num_freq_bins, num_frames)
        Power spectral density in decibels.
    stft_complex : np.ndarray
        Raw complex STFT matrix.
    """
    signal = np.asarray(audio, dtype=np.float32)

    # Compute STFT using scipy.signal.stft
    f, t, Zxx = scipy.signal.stft(
        signal,
        fs=sample_rate,
        window=window_type,
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        nfft=n_fft,
    )

    # Power spectrogram
    power = np.abs(Zxx) ** 2
    max_pwr = np.max(power) if np.max(power) > 0 else 1.0

    # Convert to decibels relative to max power, floor at -80 dB
    spectrogram_db = 10.0 * np.log10(np.maximum(power / max_pwr, 1e-8))

    return t, f, spectrogram_db, Zxx


def compute_spectral_features(
    frequencies: np.ndarray,
    magnitude: np.ndarray
) -> Dict[str, float]:
    """
    Calculate fundamental DSP spectral descriptors:
    - Spectral Centroid: center of gravity of the spectrum.
    - Spectral Spread: spectral standard deviation.
    - Spectral Rolloff (85%): frequency below which 85% of total energy resides.
    - Spectral Flatness: Wiener entropy (tonality vs noise measure).
    """
    mag = np.maximum(magnitude, 1e-12)
    sum_mag = np.sum(mag)

    if sum_mag < 1e-12:
        return {"spectral_centroid_hz": 0.0, "spectral_rolloff_85_hz": 0.0, "spectral_flatness": 0.0}

    # Spectral Centroid
    centroid = float(np.sum(frequencies * mag) / sum_mag)

    # Spectral Spread
    spread = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * mag) / sum_mag))

    # Spectral Rolloff 85%
    cum_energy = np.cumsum(mag)
    rolloff_idx = np.where(cum_energy >= 0.85 * sum_mag)[0]
    rolloff_85 = float(frequencies[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(frequencies[-1])

    # Spectral Flatness = exp(mean(log(mag))) / mean(mag)
    geom_mean = np.exp(np.mean(np.log(mag)))
    arith_mean = np.mean(mag)
    flatness = float(geom_mean / (arith_mean + 1e-12))

    return {
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_spread_hz": round(spread, 1),
        "spectral_rolloff_85_hz": round(rolloff_85, 1),
        "spectral_flatness": round(flatness, 4),
    }
