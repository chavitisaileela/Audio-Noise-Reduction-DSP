"""
Unit Tests for Digital Butterworth Filters
"""

import pytest
import numpy as np

from modules.filters import (
    validate_filter_frequencies,
    apply_butterworth_filter,
    compute_filter_frequency_response,
)


def test_validate_filter_frequencies():
    valid, _ = validate_filter_frequencies("lowpass", 3000, 16000)
    assert valid is True

    # Cutoff above Nyquist
    invalid, _ = validate_filter_frequencies("lowpass", 9000, 16000)
    assert invalid is False

    # Negative cutoff
    invalid, _ = validate_filter_frequencies("lowpass", -100, 16000)
    assert invalid is False

    # Valid Bandpass
    valid, _ = validate_filter_frequencies("bandpass", (300, 3400), 16000)
    assert valid is True

    # Inverted Bandpass
    invalid, _ = validate_filter_frequencies("bandpass", (3400, 300), 16000)
    assert invalid is False


def test_butterworth_lowpass_attenuation():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Low frequency tone (300 Hz) + high frequency noise tone (4000 Hz)
    sig_low = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    sig_high = np.sin(2 * np.pi * 4000 * t).astype(np.float32)
    combined = sig_low + sig_high

    filtered, sos, info = apply_butterworth_filter(
        combined, sr, filter_type="lowpass", cutoff_hz=1000.0, order=5
    )

    # Low frequency component should be preserved
    # High frequency component should be heavily attenuated (> 20 dB attenuation)
    high_power_before = np.mean(sig_high ** 2)
    diff = filtered - sig_low
    high_power_after = np.mean(diff ** 2)

    assert high_power_after < 0.05 * high_power_before


def test_butterworth_highpass_attenuation():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig_low = np.sin(2 * np.pi * 80 * t).astype(np.float32)
    sig_high = np.sin(2 * np.pi * 2000 * t).astype(np.float32)
    combined = sig_low + sig_high

    filtered, sos, info = apply_butterworth_filter(
        combined, sr, filter_type="highpass", cutoff_hz=500.0, order=5
    )

    diff = filtered - sig_high
    low_power_after = np.mean(diff ** 2)
    low_power_before = np.mean(sig_low ** 2)

    assert low_power_after < 0.05 * low_power_before


def test_butterworth_notch_filter():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    hum = np.sin(2 * np.pi * 50 * t).astype(np.float32)
    desired = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    combined = desired + hum

    filtered, sos, info = apply_butterworth_filter(
        combined, sr, filter_type="notch", cutoff_hz=50.0
    )

    # 440 Hz should be preserved
    corr = np.corrcoef(desired, filtered)[0, 1]
    assert corr > 0.90


def test_filter_frequency_response():
    sr = 16000
    _, sos, _ = apply_butterworth_filter(np.zeros(100), sr, "lowpass", 2000.0, order=4)
    w, mag_db, phase_deg = compute_filter_frequency_response(sos, sr, num_points=512)

    assert len(w) == 512
    assert len(mag_db) == 512
    assert len(phase_deg) == 512
    # At DC, gain should be ~ 0 dB
    assert abs(mag_db[0]) < 0.1
    # At Nyquist, gain should be strongly negative (< -20 dB)
    assert mag_db[-1] < -20.0
