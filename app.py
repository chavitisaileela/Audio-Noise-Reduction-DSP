"""
Audio Noise Reduction Using Digital Signal Processing (DSP) Techniques
======================================================================
Interactive Streamlit Application for Electronics & Communication Engineering (ECE)
Students and Signal Processing Researchers.

Features:
- Home Dashboard with DSP background & algorithm overview
- Audio Upload & Realistic Signal/Noise Synthesizer
- Time & Spectral Domain Analysis (FFT, Spectrogram, Dominant Peaks)
- 6 Independent DSP Denoising Algorithms (Butterworth LP/HP/BP, Spectral Subtraction, Wiener, LMS/NLMS)
- Enhanced Signal Verification & Side-by-side Visual Comparisons
- Rigorous Reference & Non-Reference Performance Metrics (SNR, Delta SNR, MSE, PSNR)
- Multi-Algorithm Benchmarking
- Lossless Audio & Report Export
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure modules package can be imported
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from modules.audio_loader import (
    load_audio,
    save_audio_to_bytes,
    save_audio,
    get_audio_metadata,
    generate_synthetic_speech_signal,
    generate_synthetic_noise,
    create_noisy_mixture,
)
from modules.preprocessing import (
    validate_signal,
    remove_dc_offset,
    normalize_signal,
    estimate_noise_profile,
)
from modules.fft_analysis import (
    compute_fft,
    find_dominant_frequencies,
    compute_spectrogram,
    compute_spectral_features,
)
from modules.filters import (
    apply_butterworth_filter,
    compute_filter_frequency_response,
)
from modules.spectral_subtraction import spectral_subtraction
from modules.wiener_filter import wiener_filter
from modules.adaptive_filter import (
    lms_adaptive_filter,
    generate_reference_noise_if_none,
)
from modules.reconstruction import reconstruct_and_sanitize
from modules.evaluation import (
    compute_reference_metrics,
    compute_blind_metrics,
    BenchmarkTimer,
    METRIC_EXPLANATIONS,
)
from modules.visualization import (
    plot_waveform,
    plot_waveform_comparison,
    plot_fft_spectrum,
    plot_fft_comparison,
    plot_spectrogram,
    plot_spectrogram_comparison,
    plot_filter_bode,
    plot_lms_convergence,
    plot_metrics_comparison_chart,
)

# Page configuration
st.set_page_config(
    page_title="Audio Noise Reduction | DSP Suite",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .algo-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-ece {
        background-color: #e6f4ea;
        color: #137333;
        font-size: 0.8rem;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = 16000
if "audio_meta" not in st.session_state:
    st.session_state.audio_meta = {}
if "audio_name" not in st.session_state:
    st.session_state.audio_name = "None"
if "clean_reference" not in st.session_state:
    st.session_state.clean_reference = None
if "enhanced_data" not in st.session_state:
    st.session_state.enhanced_data = None
if "algo_info" not in st.session_state:
    st.session_state.algo_info = {}
if "perf_metrics" not in st.session_state:
    st.session_state.perf_metrics = {}
if "lms_convergence" not in st.session_state:
    st.session_state.lms_convergence = None
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []

# Sidebar Navigation
st.sidebar.image(
    "https://img.icons8.com/fluency/96/audio-wave.png",
    width=64,
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Upload Audio",
        "Signal Analysis",
        "Noise Reduction",
        "Results",
        "Performance Evaluation",
        "About Project",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **ECE DSP Mini-Project**  
    *Audio Noise Reduction Suite*  
    Pure Python & DSP Implementation  
    No hardware required.
    """
)

# Status pill in sidebar
if st.session_state.audio_data is not None:
    st.sidebar.success(f"Loaded: {st.session_state.audio_name}")
    st.sidebar.caption(
        f"Fs: {st.session_state.sample_rate} Hz | Len: {st.session_state.audio_meta.get('duration_seconds', 0)}s"
    )
else:
    st.sidebar.info("No audio currently loaded.")


# ==============================================================================
# 1. HOME DASHBOARD
# ==============================================================================
if page == "Home":
    st.markdown('<div class="main-title">Audio Noise Reduction Using Digital Signal Processing (DSP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Software-Only DSP Signal Enhancement Platform for Electronics and Communication Engineering</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Architecture", "Software-Only", help="Fully implemented in Python; no hardware/sensors needed.")
    with col2:
        st.metric("Algorithms", "6 Implemented", help="Butterworth LP/HP/BP, Spectral Subtraction, Wiener, Adaptive LMS")
    with col3:
        st.metric("Analysis Engine", "FFT & STFT", help="One-sided Fourier Transform & Short-Time Spectrogram")
    with col4:
        st.metric("Evaluation", "SNR & MSE", help="Objective ground-truth and blind signal-to-noise metrics")

    st.markdown("### 📌 Project Overview")
    st.write(
        r"""
        In real-world acoustic communication systems, audio signals captured by microphones are inevitably
        corrupted by background acoustic noise such as thermal receiver noise, room reverberation, air conditioner/fan hum,
        power-line interference, and vehicular traffic rumble.

        This software application provides a **comprehensive Digital Signal Processing (DSP) toolkit** to:
        1. **Analyze** raw audio signals in both continuous time and frequency domains (FFT, Phase, Spectrogram).
        2. **Filter & Suppress** additive noise using classic digital IIR filters, spectral domain estimators, and adaptive filters.
        3. **Evaluate** signal enhancement objectively using reference-based Signal-to-Noise Ratio (SNR), SNR improvement ($\Delta$SNR), Mean Squared Error (MSE), and processing latency.
        """
    )

    st.markdown("### 🎛️ Implemented DSP Algorithms")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="algo-card">
            <h4>1. Butterworth Filters</h4>
            <span class="badge-ece">IIR Digital Filter</span>
            <p>Maximally flat magnitude response in passband. Implemented as <b>Low-pass</b> (hiss/white noise), <b>High-pass</b> (DC/rumble), and <b>Band-pass</b> (speech telephony 300-3400 Hz) with zero-phase filtering.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="algo-card">
            <h4>2. Spectral Subtraction</h4>
            <span class="badge-ece">Frequency-Domain</span>
            <p>Berouti generalized power spectral subtraction in the STFT domain. Subtracts estimated noise power spectrum with oversubtraction factor &alpha; and spectral floor &beta; to eliminate residual musical noise.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="algo-card">
            <h4>3. Wiener Filtering</h4>
            <span class="badge-ece">MMSE Optimal Estimator</span>
            <p>Minimizes Mean Square Error (MMSE). Employs Ephraim-Malah Decision-Directed (DD) a priori SNR estimation to dynamically compute optimal spectral gains and eliminate background hiss.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            """
            <div class="algo-card">
            <h4>4. LMS Adaptive Filtering</h4>
            <span class="badge-ece">Adaptive FIR Filter</span>
            <p>Widrow-Hoff Least Mean Squares (LMS) and Normalized LMS (NLMS). Dynamically tracks and cancels correlated noise or 50/60 Hz power-line hum using adaptive weight updates.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
            <div class="algo-card">
            <h4>5. Spectrogram & FFT Suite</h4>
            <span class="badge-ece">Signal Diagnostics</span>
            <p>High-resolution radix-2 FFT, magnitude spectrum (dB), unwrapped phase, dominant harmonic peak search, and 2D Short-Time Fourier Transform spectrograms.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            """
            <div class="algo-card">
            <h4>6. Objective Evaluation</h4>
            <span class="badge-ece">ECE Metrics</span>
            <p>Calculates Input SNR, Output SNR, SNR Improvement (&Delta;SNR), Mean Squared Error (MSE), Peak SNR (PSNR), RMS, and real-time execution duration.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🚀 How to Use the Software")
    st.info(
        """
        1. Navigate to **Upload Audio** to upload a WAV/MP3 recording or select from pre-generated sample test signals.
        2. View the initial time and frequency characteristics in **Signal Analysis**.
        3. Proceed to **Noise Reduction**, pick your desired DSP algorithm, tune parameters, and click **Process Audio**.
        4. Inspect before-and-after waveforms and spectra in **Results**.
        5. Verify quantitative gains in **Performance Evaluation** and download the enhanced WAV audio!
        """
    )


# ==============================================================================
# 2. AUDIO UPLOAD & SIGNAL GENERATOR
# ==============================================================================
elif page == "Upload Audio":
    st.markdown("## 🎙️ Audio Upload & Signal Generator")
    st.write("Upload an audio recording or generate standardized synthetic ECE test mixtures.")

    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio File", "🧪 ECE Benchmark Dataset (25 Files)", "⚡ Custom Signal Generator"])

    with tab1:
        st.markdown("#### Upload WAV / MP3 Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "ogg"],
            help="Accepts standard single or multi-channel audio files.",
        )
        if uploaded_file is not None:
            try:
                with st.spinner("Decoding audio samples..."):
                    audio, sr, meta = load_audio(uploaded_file, mono=True)
                    st.session_state.audio_data = audio
                    st.session_state.sample_rate = sr
                    st.session_state.audio_meta = meta
                    st.session_state.audio_name = uploaded_file.name
                    st.session_state.clean_reference = None  # Uploaded audio has no ground truth
                    st.session_state.enhanced_data = None
                st.success(f"Successfully loaded '{uploaded_file.name}'!")
            except Exception as e:
                st.error(f"Error loading audio file: {e}")

    with tab2:
        st.markdown("#### Select from ECE Acoustic Dataset (25 Built-in Audio Files)")
        st.write(
            "Browse pre-generated clean speech, pure noise sources, and calibrated noisy mixtures across multiple SNR levels. "
            "Selecting a mixture automatically attaches its ground-truth reference speech for exact objective SNR evaluation!"
        )

        dataset_dir = APP_DIR / "dataset"
        csv_index = dataset_dir / "dataset_index.csv"

        if csv_index.exists():
            df_dataset = pd.read_csv(csv_index)

            # Category filter
            cat_filter = st.radio(
                "Filter by Category:",
                ["All", "Noisy Mixtures (for Denoising)", "Clean Speech (References)", "Noise Sources (Interference)"],
                horizontal=True,
            )

            if cat_filter == "Noisy Mixtures (for Denoising)":
                filtered_df = df_dataset[df_dataset["category"] == "Noisy Mixture"]
            elif cat_filter == "Clean Speech (References)":
                filtered_df = df_dataset[df_dataset["category"] == "Clean Speech"]
            elif cat_filter == "Noise Sources (Interference)":
                filtered_df = df_dataset[df_dataset["category"] == "Noise Source"]
            else:
                filtered_df = df_dataset

            # Select audio item
            options_dict = {}
            for _, row in filtered_df.iterrows():
                key = row["filename"]
                label = f"[{row['category']}] {Path(row['filename']).name} — {row['description']} (SNR: {row['snr_db']})"
                options_dict[key] = label

            selected_file_key = st.selectbox(
                "Choose an audio signal to load:",
                list(options_dict.keys()),
                format_func=lambda k: options_dict[k],
            )

            col_btn, col_table = st.columns([1, 2])
            with col_btn:
                if st.button("📥 Load Selected Signal into Pipeline", type="primary"):
                    target_audio_path = dataset_dir / selected_file_key
                    if target_audio_path.exists():
                        audio, sr, meta = load_audio(str(target_audio_path), mono=True)
                        st.session_state.audio_data = audio
                        st.session_state.sample_rate = sr
                        st.session_state.audio_meta = meta
                        st.session_state.audio_name = Path(selected_file_key).name
                        st.session_state.enhanced_data = None

                        # Check if clean reference is associated
                        row_info = df_dataset[df_dataset["filename"] == selected_file_key].iloc[0]
                        clean_ref_name = row_info.get("clean_reference", "None")

                        if clean_ref_name != "None" and row_info["category"] == "Noisy Mixture":
                            clean_ref_path = dataset_dir / "clean_speech" / clean_ref_name
                            if clean_ref_path.exists():
                                ref_audio, _, _ = load_audio(str(clean_ref_path), mono=True)
                                st.session_state.clean_reference = ref_audio[:len(audio)]
                            else:
                                st.session_state.clean_reference = None
                        else:
                            st.session_state.clean_reference = None

                        st.success(f"Successfully loaded '{Path(selected_file_key).name}'!")
                    else:
                        st.error(f"File not found at: {target_audio_path}")

            with col_table:
                with st.expander("📋 View Complete Dataset Catalog Table", expanded=False):
                    st.dataframe(filtered_df[["filename", "category", "noise_type", "snr_db", "duration_s"]], use_container_width=True)

        else:
            # Fallback to sample_data directory
            st.info("Dataset directory not found. Using sample_data/ fallback.")
            samples_dir = APP_DIR / "sample_data"
            sample_options = {
                "sample_noisy_white.wav": "Speech + Gaussian White Noise (5 dB SNR)",
                "sample_noisy_hum.wav": "Speech + 50 Hz Power-line Hum (0 dB SNR)",
                "sample_noisy_fan.wav": "Speech + Low-frequency Fan Noise (5 dB SNR)",
                "sample_noisy_traffic.wav": "Speech + Traffic Rumble Noise (5 dB SNR)",
                "sample_clean_speech.wav": "Ground-truth Clean Speech (Reference)",
            }
            selected_sample_file = st.selectbox(
                "Choose a sample file:",
                list(sample_options.keys()),
                format_func=lambda x: f"{x} - {sample_options[x]}",
            )
            if st.button("Load Selected Sample", type="primary"):
                s_path = samples_dir / selected_sample_file
                if s_path.exists():
                    audio, sr, meta = load_audio(str(s_path), mono=True)
                    st.session_state.audio_data = audio
                    st.session_state.sample_rate = sr
                    st.session_state.audio_meta = meta
                    st.session_state.audio_name = selected_sample_file
                    st.session_state.enhanced_data = None
                    clean_path = samples_dir / "sample_clean_speech.wav"
                    if clean_path.exists() and selected_sample_file != "sample_clean_speech.wav":
                        c_audio, _, _ = load_audio(str(clean_path), mono=True)
                        st.session_state.clean_reference = c_audio[:len(audio)]
                    else:
                        st.session_state.clean_reference = None
                    st.success(f"Loaded '{selected_sample_file}'!")

    with tab3:
        st.markdown("#### Synthesize Custom Speech + Noise Mixture")
        st.write("Create a customized acoustic mixture with calibrated mathematical Signal-to-Noise Ratio (SNR).")

        gen_col1, gen_col2, gen_col3 = st.columns(3)
        with gen_col1:
            gen_noise_type = st.selectbox(
                "Noise Interference Type",
                ["white", "hum", "fan", "traffic", "mixed"],
                format_func=lambda x: {
                    "white": "Gaussian White Noise (Broadband)",
                    "hum": "50 Hz Power-line Hum + Harmonics",
                    "fan": "Blower Fan (Low-pass colored)",
                    "traffic": "Traffic Rumble (Pink 1/f noise)",
                    "mixed": "Mixed Interference (Hum + Fan + White)",
                }[x],
            )
        with gen_col2:
            gen_snr = st.slider("Target Input SNR (dB)", min_value=-5.0, max_value=20.0, value=5.0, step=1.0)
        with gen_col3:
            gen_dur = st.slider("Signal Duration (seconds)", min_value=1.5, max_value=6.0, value=3.5, step=0.5)

        if st.button("Generate Synthetic Signal & Load", type="primary"):
            sr = 16000
            with st.spinner("Synthesizing speech formants and calibrated noise..."):
                clean = generate_synthetic_speech_signal(duration=gen_dur, sample_rate=sr)
                noisy, ref_clean, noise, actual_snr = create_noisy_mixture(
                    clean, noise_type=gen_noise_type, target_snr_db=gen_snr, sample_rate=sr
                )
                meta = get_audio_metadata(noisy, sr)

                st.session_state.audio_data = noisy
                st.session_state.sample_rate = sr
                st.session_state.audio_meta = meta
                st.session_state.clean_reference = ref_clean
                st.session_state.audio_name = f"synthetic_{gen_noise_type}_{gen_snr}dB.wav"
                st.session_state.enhanced_data = None

            st.success(f"Synthetic mixture generated! Achieved Input SNR: {actual_snr} dB")

    # Current Loaded Audio Details
    if st.session_state.audio_data is not None:
        st.markdown("---")
        st.markdown("### 📊 Active Audio Properties")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        meta = st.session_state.audio_meta
        with m1:
            st.metric("Filename", st.session_state.audio_name)
        with m2:
            st.metric("Sampling Rate", f"{meta.get('sample_rate', 0)} Hz")
        with m3:
            st.metric("Duration", f"{meta.get('duration_seconds', 0)} s")
        with m4:
            st.metric("Total Samples", f"{meta.get('num_samples', 0):,}")
        with m5:
            st.metric("Peak Amplitude", f"{meta.get('peak_amplitude', 0)}")
        with m6:
            st.metric("RMS Amplitude", f"{meta.get('rms_amplitude', 0)}")

        st.markdown("#### Audio Playback")
        audio_bytes = save_audio_to_bytes(st.session_state.audio_data, st.session_state.sample_rate)
        st.audio(audio_bytes, format="audio/wav")

        if st.session_state.clean_reference is not None:
            st.caption("✅ Clean ground-truth reference audio is attached. Reference-based SNR evaluation is active.")
        else:
            st.caption("ℹ️ No clean reference audio attached (real recording mode). Blind noise floor evaluation will be used.")


# ==============================================================================
# 3. ORIGINAL AUDIO SIGNAL ANALYSIS
# ==============================================================================
elif page == "Signal Analysis":
    st.markdown("## 📈 Original Audio Signal Analysis")
    st.write("Inspect time-domain waveform, Discrete Fourier Transform (FFT) spectrum, and Short-Time Spectrogram.")

    if st.session_state.audio_data is None:
        st.warning("⚠️ No audio currently loaded. Please go to the 'Upload Audio' tab first.")
    else:
        audio = st.session_state.audio_data
        sr = st.session_state.sample_rate

        tab_wave, tab_fft, tab_spec = st.tabs(["🌊 Time-Domain Waveform", "⚡ FFT Frequency Spectrum", "🌌 STFT Spectrogram"])

        with tab_wave:
            st.markdown("#### Continuous Time-Domain Representation: $x[n]$")
            fig_wave = plot_waveform(
                audio, sr, title=f"Original Audio Waveform - {st.session_state.audio_name}", color="#1f77b4"
            )
            st.pyplot(fig_wave)
            plt.close(fig_wave)

            # Time-domain summary statistics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sample Count", f"{len(audio):,}")
            c2.metric("Peak Amplitude", f"{np.max(np.abs(audio)):.4f}")
            c3.metric("RMS Amplitude", f"{np.sqrt(np.mean(audio**2)):.4f}")
            zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0
            c4.metric("Zero-Crossing Rate", f"{zcr:.4f}")

        with tab_fft:
            st.markdown("#### One-Sided Discrete Fourier Transform (FFT) Spectrum")
            st.write(
                "Decomposes the time-domain signal $x[n]$ into constituent sinusoidal frequency components: "
                "$X[k] = \\sum_{n=0}^{N-1} x[n] e^{-j 2\\pi k n / N}$"
            )

            freqs, mag_linear, mag_db, phase = compute_fft(audio, sr)

            fig_fft = plot_fft_spectrum(
                freqs, mag_db, title=f"FFT Magnitude Spectrum (dB) - {st.session_state.audio_name}", color="#1f77b4"
            )
            st.pyplot(fig_fft)
            plt.close(fig_fft)

            # Phase spectrum toggle
            with st.expander("Show Phase Spectrum"):
                fig_phase, ax_p = plt.subplots(figsize=(10, 2.8), dpi=100)
                ax_p.plot(freqs, phase, color="#ff7f0e", linewidth=0.8, alpha=0.8)
                ax_p.set_title("Phase Spectrum: $\\angle X[k]$", fontsize=11, fontweight="bold")
                ax_p.set_xlabel("Frequency (Hz)", fontsize=9)
                ax_p.set_ylabel("Phase (radians)", fontsize=9)
                ax_p.set_ylim(-np.pi, np.pi)
                ax_p.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig_phase)
                plt.close(fig_phase)

            # Dominant harmonic peaks
            peaks = find_dominant_frequencies(freqs, mag_linear, top_k=5)
            if peaks:
                st.markdown("##### Strongest Spectral Peaks:")
                peak_df = pd.DataFrame(peaks)
                peak_df.columns = ["Frequency (Hz)", "Relative Magnitude"]
                st.dataframe(peak_df, use_container_width=True)

            # Spectral Features
            features = compute_spectral_features(freqs, mag_linear)
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Spectral Centroid", f"{features['spectral_centroid_hz']} Hz")
            f2.metric("Spectral Spread", f"{features['spectral_spread_hz']} Hz")
            f3.metric("85% Rolloff Frequency", f"{features['spectral_rolloff_85_hz']} Hz")
            f4.metric("Spectral Flatness", f"{features['spectral_flatness']}")

        with tab_spec:
            st.markdown("#### Short-Time Fourier Transform (STFT) Spectrogram")
            st.write("Displays time-varying frequency content, vital for identifying non-stationary speech vs stationary noise.")

            col_win, col_hop, col_cmap = st.columns(3)
            with col_win:
                spec_nfft = st.selectbox("Window / FFT Size (N)", [256, 512, 1024, 2048], index=1)
            with col_hop:
                spec_hop = st.selectbox("Hop Size (Overlap)", [128, 256, 512], index=1)
            with col_cmap:
                spec_cmap = st.selectbox("Color Palette", ["magma", "viridis", "inferno", "plasma"], index=0)

            times, s_freqs, spec_db, _ = compute_spectrogram(
                audio, sr, n_fft=spec_nfft, hop_size=spec_hop
            )
            fig_spec = plot_spectrogram(
                times, s_freqs, spec_db,
                title=f"Spectrogram - {st.session_state.audio_name} (N_fft={spec_nfft})",
                cmap=spec_cmap
            )
            st.pyplot(fig_spec)
            plt.close(fig_spec)


# ==============================================================================
# 4. DSP NOISE REDUCTION (PROCESSING CENTER)
# ==============================================================================
elif page == "Noise Reduction":
    st.markdown("## ⚙️ DSP Noise Reduction Processing Center")
    st.write("Select and configure digital signal processing algorithms to attenuate background acoustic interference.")

    if st.session_state.audio_data is None:
        st.warning("⚠️ No audio currently loaded. Please go to the 'Upload Audio' tab first.")
    else:
        audio = st.session_state.audio_data
        sr = st.session_state.sample_rate

        algo_choice = st.selectbox(
            "Select DSP Noise Reduction Algorithm:",
            [
                "Butterworth Low-pass Filter",
                "Butterworth High-pass Filter",
                "Butterworth Band-pass Filter",
                "Spectral Subtraction (Berouti Generalized)",
                "Wiener Filter (MMSE Decision-Directed)",
                "LMS Adaptive Filter (ANC)",
            ],
            index=3,
        )

        st.markdown("---")

        # -------------------------------------------------------------
        # 1. BUTTERWORTH LOW-PASS
        # -------------------------------------------------------------
        if algo_choice == "Butterworth Low-pass Filter":
            st.markdown("### 🎛️ Butterworth Low-Pass Filter Configuration")
            st.info(
                "Attenuates high-frequency hissing, white noise, and out-of-band harmonics above the cutoff frequency. "
                "Maximally flat passband with no ripple."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                cutoff = st.slider("Cutoff Frequency (Hz)", min_value=200, max_value=int(sr // 2 - 200), value=3200, step=100)
            with col2:
                order = st.slider("Filter Order (N)", min_value=1, max_value=10, value=5, step=1)
            with col3:
                zero_phase = st.checkbox("Zero-Phase Filtering (sosfiltfilt)", value=True, help="Eliminates phase distortion and group delay.")

            # Live Bode Plot Preview
            with st.expander("Preview Filter Bode Frequency Response", expanded=True):
                _, test_sos, _ = apply_butterworth_filter(np.zeros(100), sr, "lowpass", cutoff, order, zero_phase)
                w, m_db, p_deg = compute_filter_frequency_response(test_sos, sr)
                fig_bode = plot_filter_bode(w, m_db, p_deg, cutoff, f"Butterworth Low-Pass Response (Order={order}, fc={cutoff} Hz)")
                st.pyplot(fig_bode)
                plt.close(fig_bode)

            if st.button("🚀 Process Audio with Butterworth Low-Pass", type="primary"):
                with st.spinner("Filtering audio samples..."):
                    with BenchmarkTimer() as timer:
                        cleaned_sig = remove_dc_offset(audio)
                        filtered, _, info = apply_butterworth_filter(cleaned_sig, sr, "lowpass", cutoff, order, zero_phase)
                        enhanced, rec_report = reconstruct_and_sanitize(filtered)

                    info["processing_time_ms"] = round(timer.elapsed_ms, 2)
                    st.session_state.enhanced_data = enhanced
                    st.session_state.algo_info = info
                    st.session_state.lms_convergence = None
                    st.success(f"Processing complete in {timer.elapsed_ms:.1f} ms! Navigate to 'Results' to inspect output.")

        # -------------------------------------------------------------
        # 2. BUTTERWORTH HIGH-PASS
        # -------------------------------------------------------------
        elif algo_choice == "Butterworth High-pass Filter":
            st.markdown("### 🎛️ Butterworth High-Pass Filter Configuration")
            st.info(
                "Attenuates low-frequency mechanical rumble, wind rumble, and 50/60 Hz electrical hum while preserving voice formants."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                cutoff = st.slider("Cutoff Frequency (Hz)", min_value=40, max_value=1500, value=300, step=20)
            with col2:
                order = st.slider("Filter Order (N)", min_value=1, max_value=10, value=5, step=1)
            with col3:
                zero_phase = st.checkbox("Zero-Phase Filtering (sosfiltfilt)", value=True)

            with st.expander("Preview Filter Bode Frequency Response", expanded=True):
                _, test_sos, _ = apply_butterworth_filter(np.zeros(100), sr, "highpass", cutoff, order, zero_phase)
                w, m_db, p_deg = compute_filter_frequency_response(test_sos, sr)
                fig_bode = plot_filter_bode(w, m_db, p_deg, cutoff, f"Butterworth High-Pass Response (Order={order}, fc={cutoff} Hz)")
                st.pyplot(fig_bode)
                plt.close(fig_bode)

            if st.button("🚀 Process Audio with Butterworth High-Pass", type="primary"):
                with st.spinner("Filtering audio samples..."):
                    with BenchmarkTimer() as timer:
                        cleaned_sig = remove_dc_offset(audio)
                        filtered, _, info = apply_butterworth_filter(cleaned_sig, sr, "highpass", cutoff, order, zero_phase)
                        enhanced, rec_report = reconstruct_and_sanitize(filtered)

                    info["processing_time_ms"] = round(timer.elapsed_ms, 2)
                    st.session_state.enhanced_data = enhanced
                    st.session_state.algo_info = info
                    st.session_state.lms_convergence = None
                    st.success(f"Processing complete in {timer.elapsed_ms:.1f} ms!")

        # -------------------------------------------------------------
        # 3. BUTTERWORTH BAND-PASS
        # -------------------------------------------------------------
        elif algo_choice == "Butterworth Band-pass Filter":
            st.markdown("### 🎛️ Butterworth Band-Pass Filter Configuration")
            st.info(
                "Isolates the intelligibility speech passband (e.g. Standard Telephony 300 - 3400 Hz or Wideband 100 - 7000 Hz) "
                "and sharply rejects both low-frequency rumbles and high-frequency noise."
            )

            col1, col2 = st.columns(2)
            with col1:
                f_low = st.slider("Lower Cutoff Frequency (Hz)", min_value=50, max_value=1000, value=300, step=25)
            with col2:
                f_high = st.slider("Upper Cutoff Frequency (Hz)", min_value=1200, max_value=int(sr // 2 - 200), value=3400, step=100)

            col3, col4 = st.columns(2)
            with col3:
                order = st.slider("Filter Order (N)", min_value=1, max_value=8, value=4, step=1)
            with col4:
                zero_phase = st.checkbox("Zero-Phase Filtering", value=True)

            with st.expander("Preview Filter Bode Frequency Response", expanded=True):
                _, test_sos, _ = apply_butterworth_filter(np.zeros(100), sr, "bandpass", (f_low, f_high), order, zero_phase)
                w, m_db, p_deg = compute_filter_frequency_response(test_sos, sr)
                fig_bode = plot_filter_bode(w, m_db, p_deg, (f_low, f_high), f"Butterworth Band-Pass Response ({f_low}-{f_high} Hz)")
                st.pyplot(fig_bode)
                plt.close(fig_bode)

            if st.button("🚀 Process Audio with Butterworth Band-Pass", type="primary"):
                with st.spinner("Filtering audio samples..."):
                    with BenchmarkTimer() as timer:
                        cleaned_sig = remove_dc_offset(audio)
                        filtered, _, info = apply_butterworth_filter(cleaned_sig, sr, "bandpass", (f_low, f_high), order, zero_phase)
                        enhanced, rec_report = reconstruct_and_sanitize(filtered)

                    info["processing_time_ms"] = round(timer.elapsed_ms, 2)
                    st.session_state.enhanced_data = enhanced
                    st.session_state.algo_info = info
                    st.session_state.lms_convergence = None
                    st.success(f"Processing complete in {timer.elapsed_ms:.1f} ms!")

        # -------------------------------------------------------------
        # 4. SPECTRAL SUBTRACTION
        # -------------------------------------------------------------
        elif algo_choice == "Spectral Subtraction (Berouti Generalized)":
            st.markdown("### 🎛️ Spectral Subtraction Configuration")
            st.info(
                "Subtracts the estimated additive noise power spectrum from the noisy speech spectrum in the STFT domain. "
                "Uses oversubtraction (\\alpha) and spectral floor (\\beta) to eliminate metallic 'musical noise'."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.slider("Oversubtraction Factor (α)", min_value=1.0, max_value=4.5, value=2.0, step=0.1, help="Higher values attenuate more noise but may slightly reduce speech energy.")
            with col2:
                beta = st.slider("Spectral Floor (β)", min_value=0.001, max_value=0.10, value=0.02, step=0.005, format="%.3f", help="Prevents spectral bins from dropping to zero, avoiding musical tones.")
            with col3:
                noise_dur = st.slider("Noise Profile Lead-in (s)", min_value=0.1, max_value=1.0, value=0.35, step=0.05, help="Duration of initial ambient silence to estimate noise spectrum.")

            col4, col5, col6 = st.columns(3)
            with col4:
                domain = st.selectbox("Subtraction Domain", ["power", "magnitude"], format_func=lambda x: "Power Spectrum (γ=2)" if x == "power" else "Magnitude Spectrum (γ=1)")
            with col5:
                ss_nfft = st.selectbox("STFT Frame Size (N_fft)", [256, 512, 1024, 2048], index=1)
            with col6:
                ss_hop = st.selectbox("Hop Size (Overlap)", [128, 256, 512], index=1)

            if st.button("🚀 Process Audio with Spectral Subtraction", type="primary"):
                with st.spinner("Computing STFT, estimating noise spectrum, and synthesizing enhanced speech..."):
                    with BenchmarkTimer() as timer:
                        cleaned_sig = remove_dc_offset(audio)
                        enhanced_raw, info = spectral_subtraction(
                            cleaned_sig, sr,
                            noise_duration_sec=noise_dur,
                            alpha=alpha,
                            beta=beta,
                            n_fft=ss_nfft,
                            hop_size=ss_hop,
                            subtraction_domain=domain
                        )
                        enhanced, rec_report = reconstruct_and_sanitize(enhanced_raw)

                    info["processing_time_ms"] = round(timer.elapsed_ms, 2)
                    st.session_state.enhanced_data = enhanced
                    st.session_state.algo_info = info
                    st.session_state.lms_convergence = None
                    st.success(f"Spectral Subtraction complete in {timer.elapsed_ms:.1f} ms!")

        # -------------------------------------------------------------
        # 5. WIENER FILTERING
        # -------------------------------------------------------------
        elif algo_choice == "Wiener Filter (MMSE Decision-Directed)":
            st.markdown("### 🎛️ Wiener Filter Configuration")
            st.info(
                "Optimal Minimum Mean Square Error (MMSE) filter. Uses Ephraim-Malah Decision-Directed (DD) "
                "prior SNR tracking to dynamically assign optimal spectral gains: H(k, m) = [ ξ(k, m) / (1 + ξ(k, m)) ]^μ."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                alpha_dd = st.slider("Decision-Directed Smoothing (α_dd)", min_value=0.80, max_value=0.99, value=0.95, step=0.01, help="Frame-to-frame SNR smoothing coefficient.")
            with col2:
                exponent = st.slider("Wiener Exponent (μ)", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="Exponent tuning filter aggressiveness.")
            with col3:
                gain_floor = st.slider("Gain Floor (dB)", min_value=-45.0, max_value=-15.0, value=-30.0, step=2.5, help="Minimum attenuation floor to prevent dead silence.")

            col4, col5 = st.columns(2)
            with col4:
                w_nfft = st.selectbox("STFT Frame Size", [256, 512, 1024], index=1)
            with col5:
                noise_dur_w = st.slider("Noise Profile Lead-in (s)", min_value=0.1, max_value=1.0, value=0.35, step=0.05)

            if st.button("🚀 Process Audio with Wiener Filter", type="primary"):
                with st.spinner("Computing Decision-Directed Wiener Filter gains..."):
                    with BenchmarkTimer() as timer:
                        cleaned_sig = remove_dc_offset(audio)
                        enhanced_raw, info = wiener_filter(
                            cleaned_sig, sr,
                            noise_duration_sec=noise_dur_w,
                            alpha_dd=alpha_dd,
                            exponent=exponent,
                            gain_floor_db=gain_floor,
                            n_fft=w_nfft,
                            hop_size=w_nfft // 2
                        )
                        enhanced, rec_report = reconstruct_and_sanitize(enhanced_raw)

                    info["processing_time_ms"] = round(timer.elapsed_ms, 2)
                    st.session_state.enhanced_data = enhanced
                    st.session_state.algo_info = info
                    st.session_state.lms_convergence = None
                    st.success(f"Wiener filtering complete in {timer.elapsed_ms:.1f} ms!")

        # -------------------------------------------------------------
        # 6. LMS ADAPTIVE FILTERING
        # -------------------------------------------------------------
        elif algo_choice == "LMS Adaptive Filter (ANC)":
            st.markdown("### 🎛️ LMS Adaptive Filter Configuration")
            st.info(
                "Adaptive Noise Cancellation (ANC). Dynamically updates FIR filter weights to cancel "
                "correlated background interference or 50/60 Hz electrical hum based on the instantaneous error signal: e[n] = d[n] - y[n]."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                use_nlms = st.checkbox("Use Normalized LMS (NLMS)", value=True, help="Guarantees numerical stability and prevents divergence.")
            with col2:
                order = st.slider("Filter Taps Length (M)", min_value=8, max_value=128, value=32, step=8)
            with col3:
                mu = st.slider("Step Size (μ)", min_value=0.001, max_value=0.15, value=0.02, step=0.002, format="%.3f")

            col4, col5 = st.columns(2)
            with col4:
                ref_mode = st.selectbox(
                    "Reference Noise Source Mode",
                    ["hum_reference", "delay_line", "correlated_noise"],
                    format_func=lambda x: {
                        "hum_reference": "Synthetic 50/60 Hz Hum Reference Oscillator",
                        "delay_line": "Adaptive Line Enhancer (Delay Line decorrelation)",
                        "correlated_noise": "Synthetic Background Noise Model",
                    }[x]
                )
            with col5:
                hum_freq = st.number_input("Hum Interference Frequency (Hz)", min_value=40.0, max_value=120.0, value=50.0, step=10.0)

            if st.button("🚀 Process Audio with Adaptive Filter", type="primary"):
                with st.spinner("Adapting filter taps across samples..."):
                    with BenchmarkTimer() as timer:
                        cleaned_sig = remove_dc_offset(audio)
                        enhanced_raw, est_noise, err_hist, info = lms_adaptive_filter(
                            cleaned_sig,
                            reference_signal=None,
                            sample_rate=sr,
                            filter_order=order,
                            mu=mu,
                            normalized=use_nlms,
                            reference_mode=ref_mode,
                            hum_freq_hz=hum_freq
                        )
                        enhanced, rec_report = reconstruct_and_sanitize(enhanced_raw)

                    info["processing_time_ms"] = round(timer.elapsed_ms, 2)
                    st.session_state.enhanced_data = enhanced
                    st.session_state.algo_info = info
                    st.session_state.lms_convergence = err_hist
                    st.success(f"Adaptive filtering complete in {timer.elapsed_ms:.1f} ms!")


# ==============================================================================
# 5. RESULTS & VISUAL COMPARISONS
# ==============================================================================
elif page == "Results":
    st.markdown("## 📊 Enhanced Audio Results & Diagnostic Comparisons")
    st.write("Inspect enhanced output, listen to playback, and compare before-and-after waveforms and spectra.")

    if st.session_state.audio_data is None:
        st.warning("⚠️ No audio currently loaded.")
    elif st.session_state.enhanced_data is None:
        st.info("ℹ️ Audio has not yet been processed. Please go to 'Noise Reduction' and click 'Process Audio'.")
    else:
        orig = st.session_state.audio_data
        enh = st.session_state.enhanced_data
        sr = st.session_state.sample_rate
        info = st.session_state.algo_info

        st.markdown(f"### Applied Algorithm: **{info.get('algorithm', 'DSP Algorithm')}**")

        # Audio Playback Controls
        c_play1, c_play2, c_play3 = st.columns(3)
        with c_play1:
            st.markdown("##### 🔊 Original Audio (Noisy)")
            orig_bytes = save_audio_to_bytes(orig, sr)
            st.audio(orig_bytes, format="audio/wav")
        with c_play2:
            st.markdown("##### 🎧 Enhanced Audio (Denoised)")
            enh_bytes = save_audio_to_bytes(enh, sr)
            st.audio(enh_bytes, format="audio/wav")
        with c_play3:
            st.markdown("##### 🔇 Suppressed Noise (Difference)")
            min_l = min(len(orig), len(enh))
            diff_sig = orig[:min_l] - enh[:min_l]
            diff_bytes = save_audio_to_bytes(diff_sig, sr)
            st.audio(diff_bytes, format="audio/wav")

        st.markdown("---")

        tab_comp_wave, tab_comp_fft, tab_comp_spec, tab_lms = st.tabs([
            "🌊 Waveform Comparison",
            "⚡ Spectral FFT Comparison",
            "🌌 Spectrogram Comparison",
            "📈 Adaptive Convergence",
        ])

        with tab_comp_wave:
            st.markdown("#### Time-Domain Comparison: Original vs Enhanced vs Suppressed Noise")
            fig_w_comp = plot_waveform_comparison(orig, enh, sr, title=f"{info.get('algorithm', 'DSP')}")
            st.pyplot(fig_w_comp)
            plt.close(fig_w_comp)

        with tab_comp_fft:
            st.markdown("#### Discrete Fourier Transform (FFT) Overlay Comparison")
            st.write("Notice the direct suppression of background noise floor in the enhanced spectrum.")
            f1, m1, db1, _ = compute_fft(orig, sr)
            f2, m2, db2, _ = compute_fft(enh, sr)
            fig_f_comp = plot_fft_comparison(f1, db1, f2, db2, title=f"Spectrum: Original Noisy vs {info.get('algorithm', 'Enhanced')}")
            st.pyplot(fig_f_comp)
            plt.close(fig_f_comp)

        with tab_comp_spec:
            st.markdown("#### Dual Spectrogram Comparison (STFT)")
            t1, f_s1, s_db1, _ = compute_spectrogram(orig, sr, n_fft=512, hop_size=256)
            t2, f_s2, s_db2, _ = compute_spectrogram(enh, sr, n_fft=512, hop_size=256)
            fig_s_comp = plot_spectrogram_comparison(t1, f_s1, s_db1, t2, f_s2, s_db2)
            st.pyplot(fig_s_comp)
            plt.close(fig_s_comp)

        with tab_lms:
            if st.session_state.lms_convergence is not None:
                st.markdown("#### LMS Adaptive Filter Error Energy Convergence Trajectory")
                fig_lms = plot_lms_convergence(st.session_state.lms_convergence, sr)
                st.pyplot(fig_lms)
                plt.close(fig_lms)
            else:
                st.info("Adaptive filter error convergence is only available when 'LMS Adaptive Filter' has been executed.")

        # Quick Download Button
        st.markdown("---")
        st.download_button(
            label="⬇️ Download Enhanced Audio (WAV)",
            data=enh_bytes,
            file_name=f"enhanced_{st.session_state.audio_name}.wav",
            mime="audio/wav",
            type="primary",
        )


# ==============================================================================
# 6. PERFORMANCE EVALUATION
# ==============================================================================
elif page == "Performance Evaluation":
    st.markdown("## 🎯 Objective Performance Evaluation")
    st.write("Quantify denoising efficiency using ECE signal processing benchmarks.")

    if st.session_state.audio_data is None:
        st.warning("⚠️ No audio currently loaded.")
    elif st.session_state.enhanced_data is None:
        st.info("ℹ️ Please process the audio in 'Noise Reduction' first.")
    else:
        orig = st.session_state.audio_data
        enh = st.session_state.enhanced_data
        clean_ref = st.session_state.clean_reference
        info = st.session_state.algo_info

        st.markdown(f"### Evaluation for: **{info.get('algorithm', 'Current Algorithm')}**")

        # -------------------------------------------------------------
        # REFERENCE-BASED OR BLIND METRICS
        # -------------------------------------------------------------
        if clean_ref is not None:
            st.success("🎯 **Ground-Truth Reference Mode Active**: Exact input and output SNR calculated against reference speech.")
            metrics = compute_reference_metrics(clean_ref, orig, enh)
            st.session_state.perf_metrics = metrics

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Input SNR", f"{metrics['input_snr_db']} dB")
            m2.metric("Output SNR", f"{metrics['output_snr_db']} dB")
            delta_color = "normal" if metrics['snr_improvement_db'] >= 0 else "inverse"
            m3.metric("SNR Improvement (ΔSNR)", f"+{metrics['snr_improvement_db']} dB", delta=f"{metrics['snr_improvement_db']} dB")
            m4.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.6f}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Peak SNR (PSNR)", f"{metrics['psnr_db']} dB")
            m6.metric("Correlation Coefficient", f"{metrics['correlation']:.4f}")
            m7.metric("Processing Latency", f"{info.get('processing_time_ms', 0)} ms")
            m8.metric("Enhanced RMS", f"{metrics['enhanced_rms']:.4f}")

        else:
            st.info("🔍 **Non-Reference (Blind) Mode**: Ground-truth clean audio is unavailable. Real noise floor attenuation is measured.")
            metrics = compute_blind_metrics(orig, enh)
            st.session_state.perf_metrics = metrics

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Noise Floor Attenuation", f"{metrics['estimated_noise_reduction_db']} dB")
            m2.metric("Noisy RMS Amplitude", f"{metrics['noisy_rms']:.4f}")
            m3.metric("Enhanced RMS Amplitude", f"{metrics['enhanced_rms']:.4f}")
            m4.metric("Processing Latency", f"{info.get('processing_time_ms', 0)} ms")

        st.markdown("---")

        # -------------------------------------------------------------
        # ALL-ALGORITHM BENCHMARK SUITE
        # -------------------------------------------------------------
        st.markdown("### ⚡ Benchmark All Algorithms on Current Audio")
        st.write("Run all 6 DSP techniques sequentially on the active audio to directly compare their performance.")

        if st.button("🏁 Run Comprehensive Multi-Algorithm Benchmark", type="secondary"):
            sr = st.session_state.sample_rate
            bench_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            test_suite = [
                ("Butterworth Low-pass (3.2 kHz)", lambda x: apply_butterworth_filter(x, sr, "lowpass", 3200, 5)[0]),
                ("Butterworth High-pass (300 Hz)", lambda x: apply_butterworth_filter(x, sr, "highpass", 300, 5)[0]),
                ("Butterworth Band-pass (300-3400 Hz)", lambda x: apply_butterworth_filter(x, sr, "bandpass", (300, 3400), 4)[0]),
                ("Spectral Subtraction (α=2.0, β=0.02)", lambda x: spectral_subtraction(x, sr, 0.35, 2.0, 0.02)[0]),
                ("Wiener Filter (α=0.95, DD)", lambda x: wiener_filter(x, sr, 0.35, 0.95, 1.0)[0]),
                ("NLMS Adaptive Filter", lambda x: lms_adaptive_filter(x, None, sr, 32, 0.02, True, reference_mode="hum_reference")[0]),
            ]

            for idx, (name, fn) in enumerate(test_suite):
                status_text.text(f"Benchmarking: {name}...")
                with BenchmarkTimer() as timer:
                    raw_out = fn(remove_dc_offset(orig))
                    clean_out, _ = reconstruct_and_sanitize(raw_out)

                if clean_ref is not None:
                    m = compute_reference_metrics(clean_ref, orig, clean_out)
                    bench_results.append({
                        "algorithm": name,
                        "has_reference": True,
                        "input_snr_db": m["input_snr_db"],
                        "output_snr_db": m["output_snr_db"],
                        "snr_improvement_db": m["snr_improvement_db"],
                        "mse": m["mse"],
                        "psnr_db": m["psnr_db"],
                        "time_ms": round(timer.elapsed_ms, 1),
                    })
                else:
                    m = compute_blind_metrics(orig, clean_out)
                    bench_results.append({
                        "algorithm": name,
                        "has_reference": False,
                        "estimated_noise_reduction_db": m["estimated_noise_reduction_db"],
                        "rms_reduction_pct": m["rms_reduction_pct"],
                        "time_ms": round(timer.elapsed_ms, 1),
                    })
                progress_bar.progress((idx + 1) / len(test_suite))

            status_text.text("Benchmark complete!")
            st.session_state.benchmark_results = bench_results

        if st.session_state.benchmark_results:
            st.markdown("#### Benchmark Results Table:")
            df_bench = pd.DataFrame(st.session_state.benchmark_results)
            st.dataframe(df_bench, use_container_width=True)

            # Benchmark Chart
            fig_bench = plot_metrics_comparison_chart(st.session_state.benchmark_results)
            st.pyplot(fig_bench)
            plt.close(fig_bench)

        st.markdown("---")

        # -------------------------------------------------------------
        # ECE METRIC EXPLANATIONS ACCORDION
        # -------------------------------------------------------------
        st.markdown("### 📚 ECE Signal Processing Metric Guide")
        for metric_name, explanation in METRIC_EXPLANATIONS.items():
            with st.expander(f"📌 What is {metric_name}?"):
                st.write(explanation)


# ==============================================================================
# 7. ABOUT PROJECT & DSP THEORY
# ==============================================================================
elif page == "About Project":
    st.markdown("## 📖 About Project & DSP Theoretical Foundations")
    st.write("Detailed mathematical formulations and viva-voce reference guide for ECE students.")

    st.markdown(
        """
        ### 🎓 Academic Context
        - **Project Title:** Audio Noise Reduction Using Digital Signal Processing Techniques
        - **Discipline:** Electronics & Communication Engineering (ECE)
        - **Sub-discipline:** Digital Signal Processing (DSP) / Speech Enhancement
        - **Implementation Type:** Pure Software Application (No external microcontrollers/DSP chips)
        - **Core Stack:** Python 3, NumPy, SciPy, Librosa, SoundFile, Streamlit, Matplotlib, Pandas
        """
    )

    st.markdown("---")

    st.markdown("### 🧮 Mathematical Foundations of Implemented DSP Algorithms")

    with st.expander("1. Additive Noise Model Formulation", expanded=True):
        st.latex(r"x[n] = s[n] + d[n]")
        st.write(
            """
            Where:
            - $s[n]$: Desired clean speech/audio signal.
            - $d[n]$: Uncorrelated additive acoustic noise (white, thermal, hum, or fan).
            - $x[n]$: Observed noisy signal captured by microphone transducer.
            - The objective is to estimate $\\hat{s}[n]$ such that the error $e[n] = s[n] - \\hat{s}[n]$ is minimized.
            """
        )

    with st.expander("2. Butterworth Digital IIR Filter Transfer Function"):
        st.latex(r"|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2N}}")
        st.write(
            """
            - $N$: Filter order (controls steepness of stopband roll-off at $-20N$ dB/decade).
            - $\\omega_c$: 3 dB cutoff angular frequency.
            - Maximally flat magnitude response in the passband (zero passband ripple).
            - Implemented via Second-Order Sections (SOS) for numerical stability and zero-phase forward-backward filtering to prevent group delay distortion.
            """
        )

    with st.expander("3. Spectral Subtraction (Berouti Generalized Formulation)"):
        st.latex(
            r"|\hat{S}(k, m)|^\gamma = \begin{cases} |X(k, m)|^\gamma - \alpha \cdot P_d(k), & \text{if } |X(k, m)|^\gamma > (\alpha + \beta) P_d(k) \\ \beta \cdot P_d(k), & \text{otherwise} \end{cases}"
        )
        st.write(
            r"""
            - $|X(k, m)|$: Short-Time Fourier Transform (STFT) magnitude at frequency bin $k$ and time frame $m$.
            - $P_d(k)$: Noise power spectrum estimated from non-speech prefix frames.
            - $\alpha \ge 1$: Oversubtraction factor to eliminate residual noise peaks.
            - $\beta > 0$: Spectral floor to suppress artificial metallic 'musical noise'.
            - $\gamma$: Domain exponent ($\gamma = 2$ for power spectral subtraction, $\gamma = 1$ for magnitude spectral subtraction).
            - Synthesized via Inverse STFT (ISTFT) with original phase $\angle X(k, m)$.
            """
        )

    with st.expander("4. Wiener Filter & Ephraim-Malah Decision-Directed (DD) Prior SNR"):
        st.latex(r"H(k, m) = \left[ \frac{\xi(k, m)}{1 + \xi(k, m)} \right]^\mu")
        st.latex(r"\xi(k, m) = \alpha_{dd} \frac{|\hat{S}(k, m-1)|^2}{P_d(k)} + (1 - \alpha_{dd}) \max\left(\frac{|X(k, m)|^2}{P_d(k)} - 1, 0\right)")
        st.write(
            """
            - Minimum Mean Square Error (MMSE) optimal linear spectral estimator.
            - $\\xi(k, m)$: A priori Signal-to-Noise Ratio.
            - $\\gamma(k, m) = \\frac{|X(k, m)|^2}{P_d(k)}$: A posteriori Signal-to-Noise Ratio.
            - $\\alpha_{dd} \\approx 0.95$: Decision-directed smoothing factor preventing sudden gain fluctuations.
            """
        )

    with st.expander("5. LMS & Normalized LMS (NLMS) Adaptive Noise Cancellation"):
        st.latex(r"y[n] = \mathbf{w}^T[n] \mathbf{x}[n] = \sum_{k=0}^{M-1} w_k[n] x[n-k]")
        st.latex(r"e[n] = d[n] - y[n]")
        st.latex(r"\mathbf{w}[n+1] = \mathbf{w}[n] + \frac{\mu}{\epsilon + \|\mathbf{x}[n]\|^2} e[n] \mathbf{x}[n]")
        st.write(
            """
            - $d[n]$: Primary input signal (speech + noise).
            - $\\mathbf{x}[n]$: Reference noise vector (correlated with the noise interference).
            - $y[n]$: Adaptive filter replica of the interference.
            - $e[n]$: Error signal representing the enhanced output speech.
            - $\\mu$: Adaptation step size controlling convergence rate and steady-state misadjustment.
            - The Normalized LMS (NLMS) formulation prevents divergence when reference power fluctuates.
            """
        )

    st.markdown("---")
    st.caption("Developed for ECE Mini-Project Curriculum | Digital Signal Processing Laboratory.")
