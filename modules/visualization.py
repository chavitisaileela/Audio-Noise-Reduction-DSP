"""
Visualization Module for Audio Signals and DSP Spectra
======================================================
Generates publication-quality Matplotlib figures for:
- Time-domain waveforms (single, comparison, residual)
- Discrete Fourier Transform (FFT) spectra in decibels
- Short-Time Fourier Transform (STFT) spectrograms
- Digital filter frequency response (Bode magnitude and phase)
- Adaptive filter error convergence curves
- Algorithmic performance benchmark comparisons
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Set global clean formatting
matplotlib.rcParams["figure.autolayout"] = True
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Audio Waveform",
    color: str = "#1f77b4"
) -> plt.Figure:
    """
    Plot the continuous time-domain waveform of an audio signal.
    """
    signal = np.asarray(audio, dtype=np.float32).ravel()
    time_axis = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    ax.plot(time_axis, signal, color=color, linewidth=0.8, alpha=0.9, label="Signal Amplitude")
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.6, alpha=0.7)

    # Annotate peak and RMS
    peak = float(np.max(np.abs(signal))) if len(signal) > 0 else 0.0
    rms = float(np.sqrt(np.mean(signal ** 2))) if len(signal) > 0 else 0.0
    ax.axhline(rms, color="#2ca02c", linestyle=":", linewidth=1.0, label=f"+RMS ({rms:.3f})")
    ax.axhline(-rms, color="#2ca02c", linestyle=":", linewidth=1.0, label=f"-RMS (-{rms:.3f})")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Normalized Amplitude", fontsize=10)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, max(time_axis[-1], 0.01) if len(time_axis) > 0 else 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    return fig


def plot_waveform_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    title: str = "Waveform Comparison: Original (Noisy) vs Enhanced"
) -> plt.Figure:
    """
    Plot stacked before-and-after waveforms and the difference (attenuated noise).
    """
    min_len = min(len(original), len(enhanced))
    orig = original[:min_len]
    enh = enhanced[:min_len]
    diff = orig - enh

    time_axis = np.linspace(0, min_len / sample_rate, min_len, endpoint=False)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6.8), sharex=True, dpi=100)

    # Original
    ax1.plot(time_axis, orig, color="#d62728", linewidth=0.7, label="Original Noisy Signal")
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_title(f"{title} - Time Domain", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    # Enhanced
    ax2.plot(time_axis, enh, color="#2ca02c", linewidth=0.7, label="Enhanced Signal (DSP Output)")
    ax2.set_ylabel("Amplitude", fontsize=9)
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # Difference (Suppressed Noise)
    ax3.plot(time_axis, diff, color="#7f7f7f", linewidth=0.7, label="Suppressed Noise / Difference Component")
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylabel("Amplitude", fontsize=9)
    ax3.set_ylim(-1.05, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    return fig


def plot_fft_spectrum(
    frequencies: np.ndarray,
    magnitude_db: np.ndarray,
    title: str = "FFT Magnitude Spectrum",
    color: str = "#1f77b4"
) -> plt.Figure:
    """
    Plot the one-sided FFT spectrum in decibels.
    """
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=100)

    # Convert Hz to kHz if sample rate is high
    max_freq = frequencies[-1] if len(frequencies) > 0 else 8000.0
    use_khz = max_freq >= 4000.0
    x_data = frequencies / 1000.0 if use_khz else frequencies
    x_unit = "kHz" if use_khz else "Hz"

    ax.plot(x_data, magnitude_db, color=color, linewidth=1.0, alpha=0.9, label="Magnitude")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(f"Frequency ({x_unit})", fontsize=10)
    ax.set_ylabel("Magnitude (dB)", fontsize=10)
    ax.set_ylim(-80, 5)
    ax.set_xlim(0, x_data[-1] if len(x_data) > 0 else 8.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    return fig


def plot_fft_comparison(
    orig_freq: np.ndarray,
    orig_mag_db: np.ndarray,
    enh_freq: np.ndarray,
    enh_mag_db: np.ndarray,
    title: str = "Spectral Comparison: Original Noisy vs Enhanced Spectrum"
) -> plt.Figure:
    """
    Plot overlay comparison of original and enhanced FFT spectra to show noise attenuation.
    """
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=100)

    use_khz = orig_freq[-1] >= 4000.0
    x1 = orig_freq / 1000.0 if use_khz else orig_freq
    x2 = enh_freq / 1000.0 if use_khz else enh_freq
    unit = "kHz" if use_khz else "Hz"

    ax.plot(x1, orig_mag_db, color="#d62728", alpha=0.7, linewidth=1.0, label="Original (Noisy)")
    ax.plot(x2, enh_mag_db, color="#1f77b4", alpha=0.9, linewidth=1.1, label="Enhanced (Denoised)")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(f"Frequency ({unit})", fontsize=10)
    ax.set_ylabel("Normalized Magnitude (dB)", fontsize=10)
    ax.set_ylim(-80, 5)
    ax.set_xlim(0, max(x1[-1], x2[-1]))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    return fig


def plot_spectrogram(
    times: np.ndarray,
    frequencies: np.ndarray,
    spectrogram_db: np.ndarray,
    title: str = "Spectrogram (STFT)",
    cmap: str = "magma"
) -> plt.Figure:
    """
    Plot 2D Short-Time Fourier Transform spectrogram.
    """
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=100)

    use_khz = frequencies[-1] >= 4000.0
    y_data = frequencies / 1000.0 if use_khz else frequencies
    y_unit = "kHz" if use_khz else "Hz"

    c = ax.pcolormesh(
        times, y_data, spectrogram_db,
        shading="gouraud", cmap=cmap, vmin=-70, vmax=0
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel(f"Frequency ({y_unit})", fontsize=10)
    ax.set_ylim(0, y_data[-1] if len(y_data) > 0 else 8.0)

    cbar = fig.colorbar(c, ax=ax, pad=0.02)
    cbar.set_label("Power Spectral Density (dB)", fontsize=9)

    return fig


def plot_spectrogram_comparison(
    times1: np.ndarray, freqs1: np.ndarray, spec1: np.ndarray,
    times2: np.ndarray, freqs2: np.ndarray, spec2: np.ndarray,
    title1: str = "Original (Noisy) Spectrogram",
    title2: str = "Enhanced (Denoised) Spectrogram",
    cmap: str = "magma"
) -> plt.Figure:
    """
    Plot dual side-by-side or stacked spectrograms with synchronized axes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, sharey=True, dpi=100)

    use_khz = freqs1[-1] >= 4000.0
    y1 = freqs1 / 1000.0 if use_khz else freqs1
    y2 = freqs2 / 1000.0 if use_khz else freqs2
    y_unit = "kHz" if use_khz else "Hz"

    c1 = ax1.pcolormesh(times1, y1, spec1, shading="gouraud", cmap=cmap, vmin=-70, vmax=0)
    ax1.set_title(title1, fontsize=11, fontweight="bold")
    ax1.set_ylabel(f"Frequency ({y_unit})", fontsize=9)

    c2 = ax2.pcolormesh(times2, y2, spec2, shading="gouraud", cmap=cmap, vmin=-70, vmax=0)
    ax2.set_title(title2, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel(f"Frequency ({y_unit})", fontsize=9)

    # Colorbar
    cbar = fig.colorbar(c2, ax=[ax1, ax2], pad=0.02, location="right")
    cbar.set_label("Power Spectral Density (dB)", fontsize=9)

    return fig


def plot_filter_bode(
    frequencies: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    cutoff_hz: Any,
    title: str = "Butterworth Filter Frequency Response"
) -> plt.Figure:
    """
    Plot Bode magnitude and phase response curves for the designed digital filter.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, dpi=100)

    use_khz = frequencies[-1] >= 4000.0
    x = frequencies / 1000.0 if use_khz else frequencies
    unit = "kHz" if use_khz else "Hz"

    # Magnitude response
    ax1.plot(x, magnitude_db, color="#1f77b4", linewidth=1.5, label="|H(f)|")
    ax1.axhline(-3.0, color="#d62728", linestyle="--", linewidth=1.0, label="-3 dB Cutoff Threshold")

    # Mark cutoff lines
    if isinstance(cutoff_hz, (int, float)):
        cut_plot = cutoff_hz / 1000.0 if use_khz else cutoff_hz
        ax1.axvline(cut_plot, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"fc = {cutoff_hz} Hz")
    elif isinstance(cutoff_hz, (list, tuple)):
        for f_cut in cutoff_hz:
            cp = f_cut / 1000.0 if use_khz else f_cut
            ax1.axvline(cp, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"fc = {f_cut} Hz")

    ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Gain (dB)", fontsize=9)
    ax1.set_ylim(-60, 5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    # Phase response
    ax2.plot(x, phase_deg, color="#ff7f0e", linewidth=1.2, label="Phase")
    ax2.set_xlabel(f"Frequency ({unit})", fontsize=10)
    ax2.set_ylabel("Phase (degrees)", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    return fig


def plot_lms_convergence(
    error_sq_history: np.ndarray,
    sample_rate: int,
    title: str = "LMS Adaptive Filter Error Convergence Trajectory"
) -> plt.Figure:
    """
    Plot squared error e^2[n] trajectory showing convergence of adaptive filter.
    """
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)

    t = np.linspace(0, len(error_sq_history) / sample_rate, len(error_sq_history), endpoint=False)
    # Convert error to dB scale
    err_db = 10.0 * np.log10(np.maximum(error_sq_history, 1e-9))

    ax.plot(t, err_db, color="#9467bd", linewidth=0.9, label="Squared Error e^2[n] (dB)")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Error Energy (dB)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    return fig


def plot_metrics_comparison_chart(
    results_list: List[Dict[str, Any]]
) -> plt.Figure:
    """
    Generate horizontal bar chart comparing SNR Improvement or MSE across algorithms.
    """
    algorithms = [r["algorithm"] for r in results_list]

    has_ref = all(r.get("has_reference", False) for r in results_list)

    if has_ref:
        snr_improvements = [r.get("snr_improvement_db", 0.0) for r in results_list]
        fig, ax = plt.subplots(figsize=(8, max(2.5, len(algorithms) * 0.7)), dpi=100)
        bars = ax.barh(algorithms, snr_improvements, color="#2ca02c", alpha=0.85, height=0.5)
        ax.set_xlabel("SNR Improvement (dB) [Higher is Better]", fontsize=10, fontweight="bold")
        ax.set_title("Algorithm Performance Comparison (Ground-Truth Reference)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")

        # Value annotations on bars
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"+{w:.2f} dB", va="center", fontsize=9)
    else:
        attenuations = [r.get("estimated_noise_reduction_db", 0.0) for r in results_list]
        fig, ax = plt.subplots(figsize=(8, max(2.5, len(algorithms) * 0.7)), dpi=100)
        bars = ax.barh(algorithms, attenuations, color="#1f77b4", alpha=0.85, height=0.5)
        ax.set_xlabel("Estimated Noise Floor Attenuation (dB)", fontsize=10, fontweight="bold")
        ax.set_title("Algorithm Noise Floor Suppression (Blind Estimation)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.2f} dB", va="center", fontsize=9)

    return fig
