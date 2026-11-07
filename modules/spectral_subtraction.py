"""
Spectral Subtraction Noise Reduction Module
===========================================
Implements Classical and Generalized (Berouti) Spectral Subtraction in the STFT domain.
Includes:
- Power Spectral Subtraction (gamma=2) and Magnitude Spectral Subtraction (gamma=1)
- Oversubtraction factor (alpha) to eliminate noise residual peaks
- Spectral floor parameter (beta) to suppress metallic 'musical noise'
- Phase reconstruction and synthesis via Inverse STFT
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.signal


def spectral_subtraction(
    audio: np.ndarray,
    sample_rate: int,
    noise_duration_sec: float = 0.35,
    alpha: float = 2.0,
    beta: float = 0.02,
    n_fft: int = 512,
    hop_size: int = 256,
    subtraction_domain: str = "power"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Spectral Subtraction to attenuate additive background noise.

    DSP Mathematical Foundation:
    ----------------------------
    Given noisy speech: x[n] = s[n] + d[n]
    In STFT domain: X(k, m) = |X(k, m)| * exp(j * phi_X(k, m))

    Noise Power Estimation (over initial M_noise frames):
        P_d(k) = (1 / M_noise) * sum_{m=1}^{M_noise} |X(k, m)|^gamma

    Berouti Oversubtraction Rule:
        |S_hat(k, m)|^gamma = max( |X(k, m)|^gamma - alpha * P_d(k),  beta * P_d(k) )

    Complex Spectrum Synthesis:
        S_hat(k, m) = |S_hat(k, m)| * exp(j * angle(X(k, m)))

    Reconstructed Time Signal:
        s_hat[n] = ISTFT{ S_hat(k, m) }

    Parameters:
    -----------
    audio : np.ndarray
        1D input noisy audio signal.
    sample_rate : int
        Audio sampling rate in Hz.
    noise_duration_sec : float, default 0.35
        Duration (seconds) of initial silence used to estimate background noise spectrum.
    alpha : float, default 2.0
        Oversubtraction factor (typically 1.0 to 4.0). Higher values remove more noise.
    beta : float, default 0.02
        Spectral floor parameter (typically 0.005 to 0.05). Prevents musical noise.
    n_fft : int, default 512
        FFT analysis window size.
    hop_size : int, default 256
        STFT frame hop step size.
    subtraction_domain : str, default 'power'
        'power' (gamma=2) or 'magnitude' (gamma=1).

    Returns:
    --------
    enhanced_audio : np.ndarray
        Denoised time-domain audio array.
    info : dict
        Execution metrics and parameters.
    """
    signal = np.asarray(audio, dtype=np.float32)
    sig_len = len(signal)

    if sig_len < n_fft:
        return signal, {"status": "signal too short for STFT"}

    gamma = 2.0 if subtraction_domain.lower() == "power" else 1.0

    # Step 1: Compute STFT of the noisy signal
    f, t, Zxx = scipy.signal.stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        nfft=n_fft,
    )

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Step 2: Determine noise frames from initial segment
    noise_frames_count = int(np.ceil((noise_duration_sec * sample_rate) / hop_size))
    noise_frames_count = max(2, min(noise_frames_count, Zxx.shape[1] // 2))

    noise_mag_frames = magnitude[:, :noise_frames_count]
    # Average noise spectrum across the noise frames
    noise_spectrum_est = np.mean(noise_mag_frames ** gamma, axis=1, keepdims=True)

    # Step 3: Apply Berouti Spectral Subtraction
    noisy_spectrum = magnitude ** gamma
    subtracted = noisy_spectrum - alpha * noise_spectrum_est
    floor = beta * noise_spectrum_est

    enhanced_spectrum = np.maximum(subtracted, floor)
    enhanced_magnitude = enhanced_spectrum ** (1.0 / gamma)

    # Step 4: Recombine with original phase
    enhanced_Zxx = enhanced_magnitude * np.exp(1j * phase)

    # Step 5: Inverse STFT reconstruction
    _, enhanced_time = scipy.signal.istft(
        enhanced_Zxx,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        nfft=n_fft,
    )

    # Align length with original
    if len(enhanced_time) > sig_len:
        enhanced_time = enhanced_time[:sig_len]
    elif len(enhanced_time) < sig_len:
        enhanced_time = np.pad(enhanced_time, (0, sig_len - len(enhanced_time)))

    # Prevent digital clipping
    peak = np.max(np.abs(enhanced_time))
    if peak > 0.99:
        enhanced_time = 0.95 * (enhanced_time / peak)

    info = {
        "algorithm": "Spectral Subtraction",
        "domain": subtraction_domain,
        "alpha_oversubtraction": alpha,
        "beta_spectral_floor": beta,
        "noise_frames_used": noise_frames_count,
        "n_fft": n_fft,
        "hop_size": hop_size,
    }

    return enhanced_time.astype(np.float32), info
