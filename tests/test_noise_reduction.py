"""
Unit Tests for DSP Noise Reduction Algorithms
"""

import pytest
import numpy as np

from modules.audio_loader import generate_synthetic_speech_signal, create_noisy_mixture
from modules.spectral_subtraction import spectral_subtraction
from modules.wiener_filter import wiener_filter
from modules.adaptive_filter import lms_adaptive_filter
from modules.evaluation import compute_reference_metrics


def test_spectral_subtraction_runs_and_improves_snr():
    sr = 16000
    clean = generate_synthetic_speech_signal(duration=3.0, sample_rate=sr)
    noisy, _, _, in_snr = create_noisy_mixture(clean, "white", target_snr_db=5.0, sample_rate=sr)

    enhanced, info = spectral_subtraction(
        noisy, sr, noise_duration_sec=0.30, alpha=2.0, beta=0.02
    )

    assert len(enhanced) == len(noisy)
    assert not np.isnan(enhanced).any()
    assert not np.isinf(enhanced).any()
    assert np.max(np.abs(enhanced)) <= 1.0

    # Test SNR improvement
    metrics = compute_reference_metrics(clean, noisy, enhanced)
    assert metrics["snr_improvement_db"] > 0.0


def test_wiener_filter_runs_and_improves_snr():
    sr = 16000
    clean = generate_synthetic_speech_signal(duration=3.0, sample_rate=sr)
    noisy, _, _, in_snr = create_noisy_mixture(clean, "white", target_snr_db=5.0, sample_rate=sr)

    enhanced, info = wiener_filter(
        noisy, sr, noise_duration_sec=0.30, alpha_dd=0.95, exponent=1.0
    )

    assert len(enhanced) == len(noisy)
    assert not np.isnan(enhanced).any()
    assert not np.isinf(enhanced).any()
    assert np.max(np.abs(enhanced)) <= 1.0

    metrics = compute_reference_metrics(clean, noisy, enhanced)
    assert metrics["snr_improvement_db"] > 0.0


def test_nlms_adaptive_filter_with_hum():
    sr = 16000
    # Pure hum cancellation test
    from modules.audio_loader import generate_synthetic_noise
    hum = generate_synthetic_noise("hum", 2.0, sr)

    enhanced, est_noise, err_history, info = lms_adaptive_filter(
        hum,
        reference_signal=None,
        sample_rate=sr,
        filter_order=32,
        mu=0.02,
        normalized=True,
        reference_mode="hum_reference",
        hum_freq_hz=50.0
    )

    assert len(enhanced) == len(hum)
    assert len(est_noise) == len(hum)
    assert not np.isnan(enhanced).any()

    # Verify steady-state attenuation of 50 Hz hum (at least 6 dB reduction)
    initial_hum_power = float(np.mean(hum[:2000] ** 2))
    residual_power = float(np.mean(enhanced[-4000:] ** 2))
    attenuation_db = 10.0 * np.log10(initial_hum_power / max(residual_power, 1e-12))
    assert attenuation_db > 6.0


def test_nlms_adaptive_filter_with_direct_reference():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    # Interference
    ref_noise = 0.6 * np.sin(2 * np.pi * 120 * t + 0.4).astype(np.float32)
    noisy = clean + ref_noise

    enhanced, est_noise, err_history, info = lms_adaptive_filter(
        noisy,
        reference_signal=ref_noise,
        sample_rate=sr,
        filter_order=16,
        mu=0.01,
        normalized=True
    )

    # In steady state (second half), noise should be cancelled
    steady_clean = clean[sr // 2:]
    steady_enh = enhanced[sr // 2:]
    corr = np.corrcoef(steady_clean, steady_enh)[0, 1]
    assert corr > 0.90
