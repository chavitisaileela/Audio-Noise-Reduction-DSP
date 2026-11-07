"""
Unit Tests for Performance Evaluation Metrics
"""

import pytest
import numpy as np

from modules.evaluation import (
    compute_reference_metrics,
    compute_blind_metrics,
    BenchmarkTimer,
)


def test_reference_metrics_calculation():
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    clean = np.sin(2 * np.pi * 440 * t).astype(np.float64)

    # Noisy with large noise
    noisy = clean + 0.5 * np.random.randn(len(t))
    # Enhanced with lower error
    enhanced = clean + 0.1 * np.random.randn(len(t))

    metrics = compute_reference_metrics(clean, noisy, enhanced)

    assert metrics["has_reference"] is True
    assert metrics["output_snr_db"] > metrics["input_snr_db"]
    assert metrics["snr_improvement_db"] > 0.0
    assert metrics["mse"] > 0.0
    assert metrics["psnr_db"] > 0.0
    assert metrics["correlation"] > 0.90


def test_blind_metrics_calculation():
    noisy = np.random.randn(16000) * 0.5
    # Attenuate noise in enhanced
    enhanced = noisy * 0.2

    metrics = compute_blind_metrics(noisy, enhanced, noise_start_sample=0, noise_end_sample=4000)

    assert metrics["has_reference"] is False
    assert metrics["estimated_noise_reduction_db"] > 10.0  # ~14 dB reduction
    assert metrics["enhanced_rms"] < metrics["noisy_rms"]
    assert metrics["rms_reduction_pct"] > 50.0


def test_benchmark_timer():
    with BenchmarkTimer() as timer:
        # Quick computation
        _ = np.sum(np.random.randn(100000))

    assert timer.elapsed_sec > 0.0
    assert timer.elapsed_ms > 0.0
