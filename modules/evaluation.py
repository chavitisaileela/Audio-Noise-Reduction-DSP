"""
Performance Evaluation and Metrics Module
=========================================
Computes objective DSP metrics for audio quality and noise reduction:
- Reference-based: Input SNR, Output SNR, SNR Improvement (Delta SNR), MSE, PSNR
- Non-reference: Estimated Noise Floor Attenuation, RMS changes, Crest factor
- Processing execution time
- Clear, pedagogical ECE explanations of each metric
"""

import time
import math
from typing import Dict, Any, Optional, Tuple
import numpy as np


def compute_reference_metrics(
    clean_reference: np.ndarray,
    noisy_input: np.ndarray,
    enhanced_output: np.ndarray
) -> Dict[str, Any]:
    """
    Compute objective ground-truth metrics when clean reference s[n] is available.

    Mathematical Formulation:
    -------------------------
    Input SNR (dB):
        SNR_in = 10 * log10( sum(s^2[n]) / sum((x[n] - s[n])^2) )

    Output SNR (dB):
        SNR_out = 10 * log10( sum(s^2[n]) / sum((s_hat[n] - s[n])^2) )

    SNR Improvement (dB):
        Delta_SNR = SNR_out - SNR_in

    Mean Squared Error (MSE):
        MSE = (1 / N) * sum((s[n] - s_hat[n])^2)

    Peak Signal-to-Noise Ratio (PSNR, dB):
        PSNR = 10 * log10( max(s[n])^2 / MSE )
    """
    s = np.asarray(clean_reference, dtype=np.float64).ravel()
    x = np.asarray(noisy_input, dtype=np.float64).ravel()
    s_hat = np.asarray(enhanced_output, dtype=np.float64).ravel()

    # Align lengths to minimum common length
    min_len = min(len(s), len(x), len(s_hat))
    if min_len < 1:
        return {"error": "Signals must have non-zero length"}

    s = s[:min_len]
    x = x[:min_len]
    s_hat = s_hat[:min_len]

    # Power of clean signal
    p_clean = float(np.mean(s ** 2))
    if p_clean < 1e-12:
        return {"error": "Clean reference signal is silent"}

    # Input noise power (x - s)
    in_noise = x - s
    p_in_noise = float(np.mean(in_noise ** 2))
    input_snr = 10.0 * math.log10(p_clean / max(p_in_noise, 1e-12))

    # Output error / distortion (s_hat - s)
    out_error = s_hat - s
    p_out_error = float(np.mean(out_error ** 2))
    output_snr = 10.0 * math.log10(p_clean / max(p_out_error, 1e-12))

    snr_improvement = output_snr - input_snr
    mse = float(np.mean(out_error ** 2))

    peak_clean = float(np.max(np.abs(s)))
    psnr = 10.0 * math.log10((peak_clean ** 2) / max(mse, 1e-12))

    # Normalized Cross Correlation
    norm_corr = float(np.sum(s * s_hat) / (np.sqrt(np.sum(s**2) * np.sum(s_hat**2)) + 1e-12))

    return {
        "has_reference": True,
        "input_snr_db": round(input_snr, 2),
        "output_snr_db": round(output_snr, 2),
        "snr_improvement_db": round(snr_improvement, 2),
        "mse": round(mse, 6),
        "psnr_db": round(psnr, 2),
        "correlation": round(norm_corr, 4),
        "clean_rms": round(float(np.sqrt(p_clean)), 4),
        "noisy_rms": round(float(np.sqrt(np.mean(x ** 2))), 4),
        "enhanced_rms": round(float(np.sqrt(np.mean(s_hat ** 2))), 4),
    }


def compute_blind_metrics(
    noisy_input: np.ndarray,
    enhanced_output: np.ndarray,
    noise_start_sample: int = 0,
    noise_end_sample: int = 4000
) -> Dict[str, Any]:
    """
    Compute non-reference (blind) metrics when no ground-truth clean audio exists.

    Calculates:
    - Estimated Noise Floor Attenuation in silence segment
    - RMS amplitude before vs after
    - Peak amplitude and crest factor changes
    """
    x = np.asarray(noisy_input, dtype=np.float64).ravel()
    s_hat = np.asarray(enhanced_output, dtype=np.float64).ravel()

    min_len = min(len(x), len(s_hat))
    x = x[:min_len]
    s_hat = s_hat[:min_len]

    rms_in = float(np.sqrt(np.mean(x ** 2)))
    rms_out = float(np.sqrt(np.mean(s_hat ** 2)))

    peak_in = float(np.max(np.abs(x)))
    peak_out = float(np.max(np.abs(s_hat)))

    # Estimate noise floor reduction from silence segment
    n_start = max(0, min(noise_start_sample, min_len - 100))
    n_end = min(noise_end_sample, min_len)

    if n_end > n_start:
        p_noise_in = float(np.mean(x[n_start:n_end] ** 2))
        p_noise_out = float(np.mean(s_hat[n_start:n_end] ** 2))
        noise_attenuation_db = 10.0 * math.log10(max(p_noise_in, 1e-12) / max(p_noise_out, 1e-12))
    else:
        noise_attenuation_db = 0.0

    return {
        "has_reference": False,
        "estimated_noise_reduction_db": round(noise_attenuation_db, 2),
        "noisy_rms": round(rms_in, 4),
        "enhanced_rms": round(rms_out, 4),
        "noisy_peak": round(peak_in, 4),
        "enhanced_peak": round(peak_out, 4),
        "rms_reduction_pct": round(max(0.0, (rms_in - rms_out) / (rms_in + 1e-12) * 100.0), 2),
    }


class BenchmarkTimer:
    """Utility context manager to measure DSP processing execution time."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_sec = time.perf_counter() - self.start
        self.elapsed_ms = self.elapsed_sec * 1000.0


METRIC_EXPLANATIONS = {
    "SNR (Signal-to-Noise Ratio)": (
        "Measures the ratio of desired speech signal power to background noise power in decibels (dB). "
        "A higher SNR indicates clearer speech with less background contamination."
    ),
    "SNR Improvement (Delta SNR)": (
        "Output SNR minus Input SNR (dB). Indicates the net gain in signal cleanliness achieved "
        "by the DSP algorithm against a known ground-truth clean reference."
    ),
    "MSE (Mean Squared Error)": (
        "Average squared difference between the enhanced audio and the ideal clean speech. "
        "Lower values represent greater fidelity and minimal signal distortion."
    ),
    "PSNR (Peak Signal-to-Noise Ratio)": (
        "Ratio between the maximum possible signal power and the corrupting noise/distortion power in dB. "
        "Widely used standard in multimedia signal processing."
    ),
    "Estimated Noise Floor Reduction": (
        "The attenuation (in dB) of acoustic energy within isolated background silence periods. "
        "Used when a ground-truth clean recording is unavailable."
    ),
    "Processing Time": (
        "Wall-clock execution duration in milliseconds required by the DSP algorithm to process "
        "the complete audio sequence, indicating real-time capability."
    ),
}
