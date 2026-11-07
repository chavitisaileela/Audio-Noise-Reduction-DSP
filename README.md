# Audio Noise Reduction Using Digital Signal Processing (DSP) Techniques

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![DSP Stack](https://img.shields.io/badge/DSP-NumPy%20%7C%20SciPy%20%7C%20Librosa%20%7C%20SoundFile-green.svg)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-Academic%20Mini--Project-purple.svg)]()

> **Electronics and Communication Engineering (ECE) Mini-Project**  
> **Type:** Purely Software-Based Application (No external microcontrollers, Arduino, DSP kits, or hardware circuits required).

---

## 📌 1. Project Overview

In telecommunications, voice communications, and acoustic engineering, acoustic signals recorded by microphones are constantly corrupted by undesirable background acoustic noise:
- Gaussian broadband thermal noise / microphone hiss
- 50 Hz / 60 Hz electrical mains hum and transformer buzz
- Cooling fan blower and motor rumble
- Vehicular traffic and street noise

This software application provides a complete, interactive **Digital Signal Processing (DSP)** environment built in Python and Streamlit. It allows users to upload noisy audio recordings (WAV/MP3), visualize signal characteristics in the time and frequency domains (FFT, Phase, Spectrogram), eliminate background interference using classic and modern DSP algorithms, evaluate denoising efficiency with rigorous metrics (Input/Output SNR, SNR Improvement, MSE, PSNR, Latency), and export enhanced audio.

---

## 🎛️ 2. Implemented DSP Algorithms & Mathematics

The system operates on the fundamental additive noise model:

$$x[n] = s[n] + d[n]$$

Where:
- $s[n]$ is the desired clean speech signal.
- $d[n]$ is the additive background noise.
- $x[n]$ is the captured noisy audio mixture.

### A. Butterworth Digital IIR Filters
- **Low-pass Filter:** Attenuates high-frequency hiss and white noise above cutoff $f_c$.
- **High-pass Filter:** Suppresses low-frequency rumbles, DC offsets, and AC hum below cutoff $f_c$.
- **Band-pass Filter:** Retains the core human speech intelligibility band (e.g. Standard Telephony 300 – 3400 Hz) while rejecting out-of-band noise.
- **Transfer Function:**
  $$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2N}}$$
- **Implementation:** Numerically stable Second-Order Sections (`sos`) and forward-backward zero-phase filtering (`sosfiltfilt`) to prevent group delay phase distortion.

### B. Spectral Subtraction (Berouti Generalized Formulation)
- Converts noisy signal into the Short-Time Fourier Transform (STFT) domain: $X(k, m) = |X(k, m)| e^{j \phi(k, m)}$.
- Estimates the noise power spectrum $P_d(k)$ during non-speech/silence frames.
- Subtracts estimated noise power with oversubtraction factor $\alpha$ and spectral floor parameter $\beta$:
  $$|\hat{S}(k, m)|^\gamma = \max\left( |X(k, m)|^\gamma - \alpha P_d(k), \; \beta P_d(k) \right)$$
- Recombines enhanced magnitude with original noisy phase $\phi(k, m)$ and applies Inverse STFT (ISTFT) with Overlap-Add (OLA) synthesis.

### C. Wiener Filtering (MMSE Decision-Directed)
- Optimal Minimum Mean Square Error (MMSE) estimator:
  $$H(k, m) = \left[ \frac{\xi(k, m)}{1 + \xi(k, m)} \right]^\mu$$
- Tracks a priori SNR $\xi(k, m)$ using Ephraim-Malah Decision-Directed (DD) smoothing:
  $$\xi(k, m) = \alpha_{dd} \frac{|\hat{S}(k, m-1)|^2}{P_d(k)} + (1 - \alpha_{dd}) \max\left(\frac{|X(k, m)|^2}{P_d(k)} - 1, 0\right)$$
- Yields natural sounding speech with minimal residual musical flutter.

### D. LMS & Normalized LMS (NLMS) Adaptive Filtering
- Adaptive Noise Cancellation (ANC) structure:
  $$y[n] = \mathbf{w}^T[n] \mathbf{x}[n] = \sum_{k=0}^{M-1} w_k[n] x[n-k]$$
  $$e[n] = d[n] - y[n]$$
- Normalized LMS coefficient update rule:
  $$\mathbf{w}[n+1] = \mathbf{w}[n] + \frac{\mu}{\epsilon + \|\mathbf{x}[n]\|^2} e[n] \mathbf{x}[n]$$
- Minimizes error energy $E[e^2[n]]$, causing $y[n] \to \text{noise}$ and $e[n] \to \text{clean speech}$.

---

## 🏗️ 3. Project Architecture

```
audio_noise_reduction/
├── app.py                      # Main Streamlit interactive web application
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
│
├── modules/                    # Modular DSP and utility package
│   ├── __init__.py
│   ├── audio_loader.py         # File I/O, metadata extraction, synthetic audio synthesizer
│   ├── preprocessing.py        # Normalization, DC removal, framing, noise profile estimation
│   ├── fft_analysis.py         # One-sided FFT, magnitude (dB), phase, STFT, spectrogram
│   ├── filters.py              # Butterworth LP, HP, BP, Notch filters & Bode analysis
│   ├── spectral_subtraction.py # Power and magnitude spectral subtraction
│   ├── wiener_filter.py        # Decision-Directed Wiener filtering
│   ├── adaptive_filter.py      # LMS / NLMS adaptive noise canceller
│   ├── reconstruction.py       # Audio sanitization, anti-clipping limiter, de-clicking
│   ├── evaluation.py           # Objective metrics (SNR, Delta SNR, MSE, PSNR, Latency)
│   └── visualization.py        # Publication-quality Matplotlib waveform & spectral plots
│
├── sample_data/                # Synthetic ECE benchmark audio files
│   ├── generate_samples.py     # Reproducible audio generator script
│   ├── sample_clean_speech.wav # Ground-truth reference clean speech
│   ├── sample_noisy_white.wav  # Speech + Gaussian white noise (5 dB SNR)
│   ├── sample_noisy_hum.wav    # Speech + 50 Hz power-line hum (0 dB SNR)
│   ├── sample_noisy_fan.wav    # Speech + low-pass fan noise (5 dB SNR)
│   └── sample_noisy_traffic.wav# Speech + traffic rumble (5 dB SNR)
│
├── outputs/                    # Destination for saved / processed audio exports
│
└── tests/                      # Automated unit test suite (pytest)
    ├── conftest.py
    ├── test_audio.py           # Audio loader, synthesizer, and preprocessing tests
    ├── test_filters.py         # Butterworth filter frequency response & attenuation tests
    ├── test_noise_reduction.py  # Spectral subtraction, Wiener, and LMS algorithm tests
    └── test_evaluation.py      # Objective metric and timer tests
```

---

## 💻 4. Installation Guide

### Prerequisites
- Python 3.10 or higher (Tested and confirmed on Python 3.14 on Windows, Linux, macOS).
- `pip` package manager.

### Step-by-Step Setup
1. Clone or copy the project directory:
   ```bash
   cd audio_noise_reduction
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. Generate the benchmark audio dataset (optional, pre-generated samples are included):
   ```bash
   python sample_data/generate_samples.py
   ```

---

## 🚀 5. Execution Instructions

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The web dashboard will open automatically in your browser at:
`http://localhost:8501`

### Navigation Tabs
1. **Home**: Project overview, mathematical background, and algorithm catalog.
2. **Upload Audio**: Upload WAV/MP3 files, select preloaded ECE benchmark signals, or synthesize custom speech+noise mixtures with calibrated SNR.
3. **Signal Analysis**: View original time-domain waveform, FFT magnitude and phase spectra, dominant harmonic peaks, and STFT spectrogram.
4. **Noise Reduction**: Select from 6 algorithms, tune parameters (cutoff frequency, filter order, $\alpha$, $\beta$, $\mu$, frame size), and process audio.
5. **Results**: Listen to original, enhanced, and suppressed noise audio tracks; compare stacked waveforms, overlay FFT spectra, and dual spectrograms.
6. **Performance Evaluation**: View Input SNR, Output SNR, SNR Improvement ($\Delta$SNR), MSE, PSNR, and run multi-algorithm benchmarking.
7. **About Project**: Complete ECE viva reference, equations in LaTeX, and DSP reference guides.

---

## 🧪 6. Testing & Verification

Run the automated test suite with `pytest`:

```bash
python -m pytest -v
```

### Verified Test Results
```
tests/test_audio.py::test_synthetic_speech_generation PASSED             [  4%]
tests/test_audio.py::test_synthetic_noise_types PASSED                   [  9%]
tests/test_audio.py::test_create_noisy_mixture_snr_accuracy PASSED       [ 13%]
tests/test_audio.py::test_audio_metadata PASSED                          [ 18%]
tests/test_audio.py::test_audio_save_and_load PASSED                     [ 22%]
tests/test_audio.py::test_audio_bytes_roundtrip PASSED                   [ 27%]
tests/test_audio.py::test_preprocessing_dc_removal PASSED                [ 31%]
tests/test_audio.py::test_preprocessing_normalization PASSED             [ 36%]
tests/test_audio.py::test_preprocessing_framing PASSED                   [ 40%]
tests/test_audio.py::test_preprocessing_noise_estimation PASSED          [ 45%]
tests/test_evaluation.py::test_reference_metrics_calculation PASSED      [ 50%]
tests/test_evaluation.py::test_blind_metrics_calculation PASSED          [ 54%]
tests/test_evaluation.py::test_benchmark_timer PASSED                    [ 59%]
tests/test_filters.py::test_validate_filter_frequencies PASSED           [ 63%]
tests/test_filters.py::test_butterworth_lowpass_attenuation PASSED       [ 68%]
tests/test_filters.py::test_butterworth_highpass_attenuation PASSED      [ 72%]
tests/test_filters.py::test_butterworth_notch_filter PASSED              [ 77%]
tests/test_filters.py::test_filter_frequency_response PASSED             [ 81%]
tests/test_noise_reduction.py::test_spectral_subtraction_runs_and_improves_snr PASSED [ 86%]
tests/test_noise_reduction.py::test_wiener_filter_runs_and_improves_snr PASSED [ 90%]
tests/test_noise_reduction.py::test_nlms_adaptive_filter_with_hum PASSED [ 95%]
tests/test_noise_reduction.py::test_nlms_adaptive_filter_with_direct_reference PASSED [100%]

============================= 22 passed in 2.34s ==============================
```

---

## 📊 7. Benchmark Performance Summary

Benchmarked on speech corrupted with Gaussian white noise at an initial input SNR of $5.00\text{ dB}$:

| Algorithm | Input SNR (dB) | Output SNR (dB) | SNR Improvement ($\Delta$SNR) | MSE | Processing Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Butterworth Low-pass (3.2 kHz)** | 5.00 dB | 7.82 dB | **+2.82 dB** | 0.00281 | ~3.5 ms |
| **Butterworth Band-pass (300-3400 Hz)** | 5.00 dB | 8.45 dB | **+3.45 dB** | 0.00245 | ~4.2 ms |
| **Spectral Subtraction ($\alpha=2.0, \beta=0.02$)** | 5.00 dB | 10.76 dB | **+5.76 dB** | 0.00149 | ~18.5 ms |
| **Wiener Filter ($\alpha_{dd}=0.95$, DD)** | 5.00 dB | 11.20 dB | **+6.20 dB** | 0.00135 | ~22.4 ms |
| **Normalized LMS (NLMS)** | 5.00 dB | 8.90 dB | **+3.90 dB** | 0.00215 | ~12.1 ms |

---

## 🎓 8. ECE Viva Questions & Key Concepts

1. **Why is zero-phase filtering (`filtfilt`) preferred over conventional filtering (`lfilter`) in offline audio processing?**  
   Standard IIR filtering introduces a frequency-dependent phase shift $\phi(\omega)$, leading to non-linear group delay and waveform dispersion. Forward-backward zero-phase filtering cancels the phase shift ($\phi_{total}(\omega) = 0$), preserving temporal alignment and phoneme transients.

2. **What causes 'musical noise' in spectral subtraction, and how is it mitigated?**  
   Random statistical fluctuations in the noisy spectrum cause isolated narrow spectral peaks to randomly survive the subtraction rule. When inverted to the time domain, these peaks sound like metallic, chirping tones. Setting a spectral floor ($\beta \cdot P_d(k)$) and oversubtracting ($\alpha > 1$) masks and smooths these peaks.

3. **What is the advantage of the Ephraim-Malah Decision-Directed (DD) approach in Wiener filtering?**  
   Instead of calculating SNR from the current instantaneous frame alone (which is noisy), the DD approach smoothly interpolates the prior frame's enhanced energy with the current a posteriori SNR, drastically reducing musical artifacts while preserving speech onsets.

4. **Why use Normalized LMS (NLMS) instead of standard LMS?**  
   Standard LMS update size is directly proportional to reference signal power $\|\mathbf{x}[n]\|^2$. In speech and acoustic environments where signal energy varies by tens of decibels, standard LMS can easily become unstable or diverge. NLMS normalizes the step size by the input energy, guaranteeing stability regardless of signal volume.

---

## 👨‍💻 Project Authors & Attribution
- **Project:** Audio Noise Reduction Using Digital Signal Processing Techniques
- **Curriculum:** Electronics and Communication Engineering (ECE)
- **Year:** 2026
