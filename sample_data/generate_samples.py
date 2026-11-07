"""
Sample Audio Files Generator
============================
Generates standard ECE test signals for audio noise reduction benchmarking:
1. sample_clean_speech.wav (Ground-truth clean speech formants)
2. sample_noisy_white.wav (Clean speech + Gaussian white noise at 5 dB SNR)
3. sample_noisy_hum.wav (Clean speech + 50 Hz electrical hum at 0 dB SNR)
4. sample_noisy_fan.wav (Clean speech + fan noise at 5 dB SNR)
5. sample_noisy_traffic.wav (Clean speech + traffic rumble at 5 dB SNR)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.audio_loader import (
    generate_synthetic_speech_signal,
    create_noisy_mixture,
    save_audio,
)


def generate_all_sample_files(output_dir: Path = None):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    duration = 3.5

    print(f"Generating synthetic audio dataset in {output_dir}...")

    # 1. Clean speech
    clean = generate_synthetic_speech_signal(duration=duration, sample_rate=sample_rate)
    clean_path = output_dir / "sample_clean_speech.wav"
    save_audio(str(clean_path), clean, sample_rate)
    print(f"Saved: {clean_path.name}")

    # 2. White noise mixture
    noisy_white, _, _, snr_white = create_noisy_mixture(clean, "white", target_snr_db=5.0, sample_rate=sample_rate)
    white_path = output_dir / "sample_noisy_white.wav"
    save_audio(str(white_path), noisy_white, sample_rate)
    print(f"Saved: {white_path.name} (SNR: {snr_white} dB)")

    # 3. Hum mixture
    noisy_hum, _, _, snr_hum = create_noisy_mixture(clean, "hum", target_snr_db=0.0, sample_rate=sample_rate)
    hum_path = output_dir / "sample_noisy_hum.wav"
    save_audio(str(hum_path), noisy_hum, sample_rate)
    print(f"Saved: {hum_path.name} (SNR: {snr_hum} dB)")

    # 4. Fan noise mixture
    noisy_fan, _, _, snr_fan = create_noisy_mixture(clean, "fan", target_snr_db=5.0, sample_rate=sample_rate)
    fan_path = output_dir / "sample_noisy_fan.wav"
    save_audio(str(fan_path), noisy_fan, sample_rate)
    print(f"Saved: {fan_path.name} (SNR: {snr_fan} dB)")

    # 5. Traffic noise mixture
    noisy_traffic, _, _, snr_traffic = create_noisy_mixture(clean, "traffic", target_snr_db=5.0, sample_rate=sample_rate)
    traffic_path = output_dir / "sample_noisy_traffic.wav"
    save_audio(str(traffic_path), noisy_traffic, sample_rate)
    print(f"Saved: {traffic_path.name} (SNR: {snr_traffic} dB)")

    print("All sample audio files generated successfully.")


if __name__ == "__main__":
    generate_all_sample_files()
