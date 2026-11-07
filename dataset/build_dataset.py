"""
ECE Audio Noise Reduction Dataset Builder
=========================================
Generates a complete, structured acoustic dataset for signal processing research:
1. Clean Speech Recordings (Male, Female, Harvard Sentences, Digits)
2. Isolated Noise Sources (White, 50 Hz Hum, Fan, Traffic, Office Babble, Machinery)
3. Calibrated Noisy Mixtures across standardized SNR levels (-5 dB, 0 dB, 5 dB, 10 dB)
4. Comprehensive dataset index CSV
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.audio_loader import (
    save_audio,
    create_noisy_mixture,
)


def synthesize_male_speech(duration: float = 3.6, sr: int = 16000) -> np.ndarray:
    """
    Synthesize male voiced speech with fundamental pitch f0 ~ 115-130 Hz
    and acoustic formants F1=700 Hz, F2=1220 Hz, F3=2600 Hz.
    Includes initial 0.4s ambient silence lead-in.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)

    # 3 vowel syllables with silence pauses
    segments = [
        {"start": 0.40, "end": 1.25, "f0": 125.0, "formants": [(700, 0.45), (1220, 0.35), (2600, 0.15)]},
        {"start": 1.45, "end": 2.30, "f0": 118.0, "formants": [(300, 0.50), (2200, 0.30), (3000, 0.15)]},
        {"start": 2.50, "end": 3.35, "f0": 110.0, "formants": [(380, 0.40), (850, 0.35), (2300, 0.12)]},
    ]

    for seg in segments:
        idx = (t >= seg["start"]) & (t <= seg["end"])
        t_seg = t[idx] - seg["start"]
        N_seg = len(t_seg)
        if N_seg == 0:
            continue
        env = np.sin(np.pi * np.linspace(0, 1, N_seg)) ** 2
        seg_wave = np.zeros(N_seg)
        f0 = seg["f0"]
        for harm in range(1, 24):
            freq = harm * f0
            if freq >= sr / 2:
                break
            gain = 0.015
            for f_res, amp in seg["formants"]:
                bw = 85.0
                gain += amp * (bw**2) / ((freq - f_res)**2 + bw**2)
            seg_wave += gain * np.sin(2 * np.pi * freq * t_seg)
        sig[idx] += seg_wave * env

    peak = np.max(np.abs(sig))
    if peak > 1e-6:
        sig = 0.85 * (sig / peak)
    return sig.astype(np.float32)


def synthesize_female_speech(duration: float = 3.6, sr: int = 16000) -> np.ndarray:
    """
    Synthesize female voiced speech with higher fundamental pitch f0 ~ 210-230 Hz
    and elevated formants typical of female vocal tract geometry.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)

    segments = [
        {"start": 0.40, "end": 1.25, "f0": 225.0, "formants": [(850, 0.45), (1450, 0.35), (2900, 0.18)]},
        {"start": 1.45, "end": 2.30, "f0": 215.0, "formants": [(380, 0.50), (2450, 0.35), (3200, 0.15)]},
        {"start": 2.50, "end": 3.35, "f0": 200.0, "formants": [(420, 0.40), (980, 0.30), (2550, 0.15)]},
    ]

    for seg in segments:
        idx = (t >= seg["start"]) & (t <= seg["end"])
        t_seg = t[idx] - seg["start"]
        N_seg = len(t_seg)
        if N_seg == 0:
            continue
        env = np.sin(np.pi * np.linspace(0, 1, N_seg)) ** 2
        seg_wave = np.zeros(N_seg)
        f0 = seg["f0"]
        for harm in range(1, 16):
            freq = harm * f0
            if freq >= sr / 2:
                break
            gain = 0.02
            for f_res, amp in seg["formants"]:
                bw = 95.0
                gain += amp * (bw**2) / ((freq - f_res)**2 + bw**2)
            seg_wave += gain * np.sin(2 * np.pi * freq * t_seg)
        sig[idx] += seg_wave * env

    peak = np.max(np.abs(sig))
    if peak > 1e-6:
        sig = 0.85 * (sig / peak)
    return sig.astype(np.float32)


def synthesize_harvard_sentence(duration: float = 4.2, sr: int = 16000) -> np.ndarray:
    """
    Synthesize phonetically rich multi-syllabic sentence cadence:
    'The birch canoe slid on the smooth planks.'
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)

    # Word syllables with natural inflection cadence
    words = [
        {"start": 0.40, "dur": 0.30, "f0": 135.0, "formants": [(500, 0.3), (1500, 0.2)]},       # The
        {"start": 0.75, "dur": 0.45, "f0": 145.0, "formants": [(650, 0.4), (1800, 0.3)]},       # birch
        {"start": 1.25, "dur": 0.55, "f0": 130.0, "formants": [(450, 0.4), (1200, 0.3)]},       # canoe
        {"start": 1.90, "dur": 0.40, "f0": 140.0, "formants": [(400, 0.5), (2100, 0.2)]},       # slid
        {"start": 2.35, "dur": 0.25, "f0": 125.0, "formants": [(600, 0.3), (1100, 0.2)]},       # on
        {"start": 2.65, "dur": 0.25, "f0": 120.0, "formants": [(500, 0.3), (1500, 0.2)]},       # the
        {"start": 2.95, "dur": 0.45, "f0": 115.0, "formants": [(350, 0.5), (1000, 0.3)]},       # smooth
        {"start": 3.45, "dur": 0.55, "f0": 105.0, "formants": [(700, 0.4), (1700, 0.2)]},       # planks
    ]

    for w in words:
        idx = (t >= w["start"]) & (t <= (w["start"] + w["dur"]))
        t_w = t[idx] - w["start"]
        N_w = len(t_w)
        if N_w == 0:
            continue
        env = np.sin(np.pi * np.linspace(0, 1, N_w)) ** 2
        w_wave = np.zeros(N_w)
        f0 = w["f0"]
        for harm in range(1, 22):
            freq = harm * f0
            if freq >= sr / 2:
                break
            gain = 0.02
            for f_res, amp in w["formants"]:
                bw = 90.0
                gain += amp * (bw**2) / ((freq - f_res)**2 + bw**2)
            w_wave += gain * np.sin(2 * np.pi * freq * t_w)
        sig[idx] += w_wave * env

    peak = np.max(np.abs(sig))
    if peak > 1e-6:
        sig = 0.85 * (sig / peak)
    return sig.astype(np.float32)


def synthesize_digits(duration: float = 4.0, sr: int = 16000) -> np.ndarray:
    """
    Synthesize spoken counting numbers 'One, Two, Three, Four' with clean pauses.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)

    digits = [
        {"start": 0.40, "dur": 0.55, "f0": 140.0, "formants": [(450, 0.5), (1000, 0.35)]},      # 'One'
        {"start": 1.25, "dur": 0.55, "f0": 130.0, "formants": [(350, 0.5), (1200, 0.3)]},       # 'Two'
        {"start": 2.10, "dur": 0.55, "f0": 125.0, "formants": [(300, 0.4), (2200, 0.35)]},      # 'Three'
        {"start": 2.95, "dur": 0.65, "f0": 115.0, "formants": [(650, 0.45), (1100, 0.3)]},      # 'Four'
    ]

    for d in digits:
        idx = (t >= d["start"]) & (t <= (d["start"] + d["dur"]))
        t_d = t[idx] - d["start"]
        N_d = len(t_d)
        if N_d == 0:
            continue
        env = np.sin(np.pi * np.linspace(0, 1, N_d)) ** 2
        d_wave = np.zeros(N_d)
        f0 = d["f0"]
        for harm in range(1, 20):
            freq = harm * f0
            if freq >= sr / 2:
                break
            gain = 0.02
            for f_res, amp in d["formants"]:
                bw = 90.0
                gain += amp * (bw**2) / ((freq - f_res)**2 + bw**2)
            d_wave += gain * np.sin(2 * np.pi * freq * t_d)
        sig[idx] += d_wave * env

    peak = np.max(np.abs(sig))
    if peak > 1e-6:
        sig = 0.85 * (sig / peak)
    return sig.astype(np.float32)


def generate_office_babble(duration: float, sr: int) -> np.ndarray:
    """Multi-speaker background cocktail-party chatter."""
    N = int(duration * sr)
    t = np.linspace(0, duration, N, endpoint=False)
    rng = np.random.default_rng(seed=77)
    babble = np.zeros(N)

    # Overlap 4 distinct murmuring speech-like formants
    pitches = [110, 160, 210, 240]
    for p in pitches:
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 2.5) * t)
        noise = rng.normal(0, 0.2, N)
        for h in range(1, 8):
            babble += 0.08 * mod * np.sin(2 * np.pi * h * p * t + rng.uniform(0, 2*np.pi))
        babble += noise * 0.15

    std = np.std(babble)
    return (babble / (std + 1e-6)).astype(np.float32)


def generate_factory_machinery(duration: float, sr: int) -> np.ndarray:
    """Industrial drone with repetitive mechanical impact thuds."""
    N = int(duration * sr)
    t = np.linspace(0, duration, N, endpoint=False)
    rng = np.random.default_rng(seed=99)

    # 60 Hz motor hum + harmonics
    hum = (
        0.5 * np.sin(2 * np.pi * 60 * t)
        + 0.3 * np.sin(2 * np.pi * 120 * t)
        + 0.2 * np.sin(2 * np.pi * 180 * t)
    )

    # Periodic stamping impacts every 0.6 seconds
    impact = np.zeros(N)
    interval = int(0.6 * sr)
    for k in range(0, N, interval):
        decay_len = min(int(0.12 * sr), N - k)
        decay = np.exp(-np.linspace(0, 10, decay_len))
        impact[k : k + decay_len] += 0.8 * decay * rng.normal(0, 1.0, decay_len)

    machinery = hum + impact + rng.normal(0, 0.2, N)
    std = np.std(machinery)
    return (machinery / (std + 1e-6)).astype(np.float32)


def build_complete_dataset(base_dir: Path = None):
    if base_dir is None:
        base_dir = PROJECT_ROOT / "dataset"

    clean_dir = base_dir / "clean_speech"
    noise_dir = base_dir / "noise_sources"
    mixtures_dir = base_dir / "noisy_mixtures"

    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    mixtures_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    catalog = []

    print(f"Building ECE Audio Dataset inside: {base_dir}")

    # -------------------------------------------------------------
    # 1. Clean Speech Recordings
    # -------------------------------------------------------------
    clean_speech_catalog = {
        "male_speech.wav": (synthesize_male_speech(3.6, sr), "Male voiced speech formants (/a/, /i/, /u/)"),
        "female_speech.wav": (synthesize_female_speech(3.6, sr), "Female voiced speech formants (pitch 220 Hz)"),
        "harvard_sentence.wav": (synthesize_harvard_sentence(4.2, sr), "Phonetically rich Harvard sentence cadence"),
        "counting_digits.wav": (synthesize_digits(4.0, sr), "Spoken counting digits (1 to 4) with clear pauses"),
    }

    for fname, (audio, desc) in clean_speech_catalog.items():
        fpath = clean_dir / fname
        save_audio(str(fpath), audio, sr)
        dur = round(len(audio) / sr, 2)
        catalog.append({
            "filename": f"clean_speech/{fname}",
            "category": "Clean Speech",
            "description": desc,
            "noise_type": "None",
            "snr_db": "Clean (Ground Truth)",
            "clean_reference": fname,
            "duration_s": dur,
            "sample_rate": sr,
        })
        print(f"  [Clean Speech] Saved: {fname} ({dur}s)")

    # -------------------------------------------------------------
    # 2. Pure Noise Profiles
    # -------------------------------------------------------------
    from modules.audio_loader import generate_synthetic_noise

    noise_catalog = {
        "white_noise.wav": (generate_synthetic_noise("white", 4.5, sr), "Gaussian broadband white thermal noise"),
        "electrical_hum_50hz.wav": (generate_synthetic_noise("hum", 4.5, sr), "50 Hz electromagnetic power-line hum & harmonics"),
        "cooling_fan.wav": (generate_synthetic_noise("fan", 4.5, sr), "Low-frequency blower cooling fan colored noise"),
        "traffic_rumble.wav": (generate_synthetic_noise("traffic", 4.5, sr), "Brownian low-frequency vehicular engine rumble"),
        "office_babble.wav": (generate_office_babble(4.5, sr), "Multi-speaker background cafeteria/office cocktail-party babble"),
        "factory_machinery.wav": (generate_factory_machinery(4.5, sr), "Industrial machinery hum with repetitive impacts"),
    }

    for fname, (audio, desc) in noise_catalog.items():
        fpath = noise_dir / fname
        save_audio(str(fpath), audio, sr)
        dur = round(len(audio) / sr, 2)
        catalog.append({
            "filename": f"noise_sources/{fname}",
            "category": "Noise Source",
            "description": desc,
            "noise_type": fname.replace(".wav", ""),
            "snr_db": "Pure Noise (No Speech)",
            "clean_reference": "None",
            "duration_s": dur,
            "sample_rate": sr,
        })
        print(f"  [Noise Source] Saved: {fname} ({dur}s)")

    # -------------------------------------------------------------
    # 3. Calibrated Noisy Mixtures (Multi-SNR Benchmark)
    # -------------------------------------------------------------
    mixture_configs = [
        # Male speech with white noise across SNR range
        ("male_speech.wav", "white", -5.0, "male_white_minus5dB.wav", "Male speech + heavy white noise (-5 dB SNR)"),
        ("male_speech.wav", "white", 0.0, "male_white_0dB.wav", "Male speech + white noise (0 dB SNR)"),
        ("male_speech.wav", "white", 5.0, "male_white_5dB.wav", "Male speech + moderate white noise (5 dB SNR)"),
        ("male_speech.wav", "white", 10.0, "male_white_10dB.wav", "Male speech + mild white noise (10 dB SNR)"),

        # Hum interference (ideal for Notch and LMS Adaptive Filter)
        ("male_speech.wav", "hum", -5.0, "male_hum_minus5dB.wav", "Male speech + strong 50 Hz power-line hum (-5 dB SNR)"),
        ("male_speech.wav", "hum", 0.0, "male_hum_0dB.wav", "Male speech + 50 Hz power-line hum (0 dB SNR)"),
        ("male_speech.wav", "hum", 5.0, "male_hum_5dB.wav", "Male speech + 50 Hz power-line hum (5 dB SNR)"),

        # Fan noise (ideal for Butterworth LP and Wiener Filter)
        ("female_speech.wav", "fan", 0.0, "female_fan_0dB.wav", "Female speech + cooling fan noise (0 dB SNR)"),
        ("female_speech.wav", "fan", 5.0, "female_fan_5dB.wav", "Female speech + cooling fan noise (5 dB SNR)"),
        ("female_speech.wav", "fan", 10.0, "female_fan_10dB.wav", "Female speech + mild fan noise (10 dB SNR)"),

        # Traffic rumble (ideal for Butterworth HP and Spectral Subtraction)
        ("harvard_sentence.wav", "traffic", 0.0, "sentence_traffic_0dB.wav", "Harvard sentence + traffic rumble (0 dB SNR)"),
        ("harvard_sentence.wav", "traffic", 5.0, "sentence_traffic_5dB.wav", "Harvard sentence + traffic rumble (5 dB SNR)"),
        ("harvard_sentence.wav", "traffic", 10.0, "sentence_traffic_10dB.wav", "Harvard sentence + mild traffic rumble (10 dB SNR)"),

        # Mixed & Counting digits
        ("counting_digits.wav", "mixed", 5.0, "digits_mixed_5dB.wav", "Spoken digits + mixed interference (Hum + Fan + White, 5 dB SNR)"),
        ("counting_digits.wav", "white", 5.0, "digits_white_5dB.wav", "Spoken digits + Gaussian white noise (5 dB SNR)"),
    ]

    for clean_key, noise_t, snr_target, mix_fname, desc in mixture_configs:
        clean_audio = clean_speech_catalog[clean_key][0]
        noisy, ref_clean, noise_sig, actual_snr = create_noisy_mixture(
            clean_audio, noise_type=noise_t, target_snr_db=snr_target, sample_rate=sr
        )
        fpath = mixtures_dir / mix_fname
        save_audio(str(fpath), noisy, sr)
        dur = round(len(noisy) / sr, 2)
        catalog.append({
            "filename": f"noisy_mixtures/{mix_fname}",
            "category": "Noisy Mixture",
            "description": desc,
            "noise_type": noise_t,
            "snr_db": str(actual_snr),
            "clean_reference": clean_key,
            "duration_s": dur,
            "sample_rate": sr,
        })
        print(f"  [Mixture] Saved: {mix_fname} (Achieved SNR: {actual_snr} dB)")

    # Save Catalog CSV
    df = pd.DataFrame(catalog)
    csv_path = base_dir / "dataset_index.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved Dataset Catalog CSV to: {csv_path}")
    print(f"Total audio files generated: {len(df)} files.")
    return df


if __name__ == "__main__":
    build_complete_dataset()
