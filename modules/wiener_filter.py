"""
Wiener Filter Noise Reduction Module
====================================
Implements Minimum Mean Square Error (MMSE) Wiener Filtering in the STFT domain.
Includes:
- A posteriori SNR and A priori SNR calculation
- Ephraim-Malah Decision-Directed (DD) smoothing approach
- Parametric Wiener filter transfer function H(k, m) = (xi / (1 + xi))^mu
- Gain floor suppression to eliminate musical noise
- ISTFT reconstruction
"""

from typing import Tuple, Dict, Any
import numpy as np
import scipy.signal


def wiener_filter(
    audio: np.ndarray,
    sample_rate: int,
    noise_duration_sec: float = 0.35,
    alpha_dd: float = 0.95,
    exponent: float = 1.0,
    gain_floor_db: float = -30.0,
    n_fft: int = 512,
    hop_size: int = 256
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Wiener Filtering to reduce stationary or quasi-stationary background noise.

    DSP Mathematical Foundation:
    ----------------------------
    Wiener gain function:
        H(k, m) = [ xi(k, m) / (1 + xi(k, m)) ]^mu

    where:
    - A posteriori SNR:
        gamma(k, m) = |X(k, m)|^2 / P_d(k)
    - Ephraim-Malah Decision-Directed A priori SNR:
        xi(k, m) = alpha_dd * [ |S_hat(k, m-1)|^2 / P_d(k) ] + (1 - alpha_dd) * max(gamma(k, m) - 1, 0)
    - Gain floor limiter:
        H(k, m) = max( H(k, m), 10^(gain_floor_db / 20) )

    Filtered STFT spectrum:
        S_hat(k, m) = H(k, m) * X(k, m)

    Parameters:
    -----------
    audio : np.ndarray
        1D input noisy signal.
    sample_rate : int
        Audio sampling rate in Hz.
    noise_duration_sec : float, default 0.35
        Duration of initial silence segment to estimate noise PSD.
    alpha_dd : float, default 0.95
        Decision-Directed smoothing factor (0.85 to 0.99).
    exponent : float, default 1.0
        Parametric Wiener filter exponent mu (typically 1.0 to 2.0).
    gain_floor_db : float, default -30.0
        Minimum attenuation floor in dB (e.g., -30 dB prevents dead silence artifacts).
    n_fft : int, default 512
        Analysis window / FFT length.
    hop_size : int, default 256
        Frame shift size.

    Returns:
    --------
    enhanced_audio : np.ndarray
        Wiener-filtered time-domain audio array.
    info : dict
        Algorithm execution metadata and parameters.
    """
    signal = np.asarray(audio, dtype=np.float32)
    sig_len = len(signal)

    if sig_len < n_fft:
        return signal, {"status": "signal too short for STFT"}

    gain_floor = 10.0 ** (gain_floor_db / 20.0)

    # Step 1: STFT
    f, t, Zxx = scipy.signal.stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        nfft=n_fft,
    )

    num_bins, num_frames = Zxx.shape
    magnitude_sq = np.abs(Zxx) ** 2

    # Step 2: Estimate noise PSD P_d(k) from initial frames
    noise_frames_count = int(np.ceil((noise_duration_sec * sample_rate) / hop_size))
    noise_frames_count = max(2, min(noise_frames_count, num_frames // 2))

    noise_psd = np.mean(magnitude_sq[:, :noise_frames_count], axis=1, keepdims=True)
    noise_psd = np.maximum(noise_psd, 1e-12)

    # Step 3: Frame-by-frame Decision-Directed Wiener Gain computation
    enhanced_Zxx = np.zeros_like(Zxx, dtype=np.complex64)
    prev_s_hat_mag_sq = np.copy(magnitude_sq[:, 0:1])

    gains_history = np.zeros((num_bins, num_frames), dtype=np.float32)

    for m in range(num_frames):
        mag_sq_m = magnitude_sq[:, m:m+1]

        # A posteriori SNR: gamma = |X|^2 / P_d
        gamma_m = mag_sq_m / noise_psd

        # A priori SNR: xi = alpha_dd * (prev_|S_hat|^2 / P_d) + (1 - alpha_dd) * max(gamma - 1, 0)
        if m == 0:
            xi_m = np.maximum(gamma_m - 1.0, 0.0)
        else:
            prior_term = prev_s_hat_mag_sq / noise_psd
            direct_term = np.maximum(gamma_m - 1.0, 0.0)
            xi_m = alpha_dd * prior_term + (1.0 - alpha_dd) * direct_term

        # Wiener Gain: H = [ xi / (1 + xi) ]^mu
        wiener_gain = (xi_m / (1.0 + xi_m)) ** exponent
        # Apply gain floor
        wiener_gain = np.maximum(wiener_gain, gain_floor)
        wiener_gain = np.minimum(wiener_gain, 1.0)

        gains_history[:, m:m+1] = wiener_gain

        # Enhanced complex bin
        s_hat_m = wiener_gain * Zxx[:, m:m+1]
        enhanced_Zxx[:, m:m+1] = s_hat_m
        prev_s_hat_mag_sq = np.abs(s_hat_m) ** 2

    # Step 4: Inverse STFT synthesis
    _, enhanced_time = scipy.signal.istft(
        enhanced_Zxx,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        nfft=n_fft,
    )

    # Align length
    if len(enhanced_time) > sig_len:
        enhanced_time = enhanced_time[:sig_len]
    elif len(enhanced_time) < sig_len:
        enhanced_time = np.pad(enhanced_time, (0, sig_len - len(enhanced_time)))

    # Prevent digital clipping
    peak = np.max(np.abs(enhanced_time))
    if peak > 0.99:
        enhanced_time = 0.95 * (enhanced_time / peak)

    info = {
        "algorithm": "Wiener Filter",
        "alpha_dd": alpha_dd,
        "exponent": exponent,
        "gain_floor_db": gain_floor_db,
        "average_gain": round(float(np.mean(gains_history)), 4),
        "n_fft": n_fft,
        "hop_size": hop_size,
    }

    return enhanced_time.astype(np.float32), info
