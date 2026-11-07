"""
Unit Tests for Audio Loader, Synthetic Signal Generator, and Preprocessing
"""

import os
import io
import pytest
import numpy as np
from pathlib import Path

from modules.audio_loader import (
    load_audio,
    save_audio,
    save_audio_to_bytes,
    get_audio_metadata,
    generate_synthetic_speech_signal,
    generate_synthetic_noise,
    create_noisy_mixture,
)
from modules.preprocessing import (
    validate_signal,
    remove_dc_offset,
    normalize_signal,
    frame_signal,
    estimate_noise_profile,
)


def test_synthetic_speech_generation():
    sr = 16000
    duration = 2.0
    sig = generate_synthetic_speech_signal(duration=duration, sample_rate=sr)

    assert len(sig) == int(sr * duration)
    assert np.max(np.abs(sig)) <= 1.0
    assert not np.isnan(sig).any()
    assert not np.isinf(sig).any()
    # Check that speech contains energy
    assert np.std(sig) > 0.01


def test_synthetic_noise_types():
    sr = 16000
    duration = 1.0
    for noise_type in ["white", "hum", "fan", "traffic", "mixed"]:
        noise = generate_synthetic_noise(noise_type, duration, sr)
        assert len(noise) == int(sr * duration)
        assert not np.isnan(noise).any()
        assert np.std(noise) > 0.5


def test_create_noisy_mixture_snr_accuracy():
    sr = 16000
    clean = generate_synthetic_speech_signal(duration=2.0, sample_rate=sr)
    target_snr = 10.0

    noisy, ref_clean, noise, actual_snr = create_noisy_mixture(
        clean, noise_type="white", target_snr_db=target_snr, sample_rate=sr
    )

    assert len(noisy) == len(clean)
    assert abs(actual_snr - target_snr) < 0.5


def test_audio_metadata():
    sr = 16000
    data = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32)
    meta = get_audio_metadata(data, sr)

    assert meta["sample_rate"] == 16000
    assert meta["duration_seconds"] == 1.0
    assert meta["num_samples"] == 16000
    assert meta["num_channels"] == 1
    assert abs(meta["peak_amplitude"] - 1.0) < 0.01


def test_audio_save_and_load(tmp_path):
    sr = 16000
    data = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32)
    test_file = tmp_path / "test.wav"

    save_audio(str(test_file), data, sr)
    assert test_file.exists()

    loaded, loaded_sr, meta = load_audio(str(test_file))
    assert loaded_sr == sr
    assert len(loaded) == len(data)
    assert np.allclose(loaded, data, atol=1e-3)


def test_audio_bytes_roundtrip():
    sr = 16000
    data = 0.4 * np.cos(2 * np.pi * 500 * np.linspace(0, 0.5, sr // 2, endpoint=False)).astype(np.float32)
    raw_bytes = save_audio_to_bytes(data, sr)

    assert len(raw_bytes) > 0
    buf = io.BytesIO(raw_bytes)
    loaded, loaded_sr, _ = load_audio(buf)

    assert loaded_sr == sr
    assert np.allclose(loaded, data, atol=1e-3)


def test_preprocessing_dc_removal():
    data = np.ones(1000, dtype=np.float32) * 0.5 + 0.1 * np.random.randn(1000).astype(np.float32)
    no_dc = remove_dc_offset(data)
    assert abs(np.mean(no_dc)) < 1e-6


def test_preprocessing_normalization():
    data = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    norm = normalize_signal(data, target_peak=0.95)
    assert abs(np.max(np.abs(norm)) - 0.95) < 1e-5


def test_preprocessing_framing():
    data = np.random.randn(1024).astype(np.float32)
    frames, win = frame_signal(data, frame_size=256, hop_size=128)
    assert frames.ndim == 2
    assert frames.shape[1] == 256
    assert len(win) == 256


def test_preprocessing_noise_estimation():
    sr = 16000
    data = np.random.randn(sr * 2).astype(np.float32)
    noise_seg, start, end = estimate_noise_profile(data, sr, noise_duration_sec=0.25)
    assert len(noise_seg) == int(0.25 * sr)
    assert start == 0
    assert end == int(0.25 * sr)
