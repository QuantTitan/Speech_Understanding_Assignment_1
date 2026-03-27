"""
mfcc_manual.py
==============
Handcrafted MFCC / Cepstrum engine.

Pipeline:
  1. Pre-emphasis
  2. Framing + Windowing (Hamming / Hanning / Rectangular)
  3. FFT
  4. Mel-Filterbank application
  5. Log compression
  6. DCT  →  MFCCs
  7. Full real cepstrum (for voiced/unvoiced boundary work)

No librosa.feature.mfcc (or equivalent high-level call) is used.
Only numpy, scipy.fft and scipy.io.wavfile are allowed helpers.
"""

import numpy as np
from scipy.fft import fft, dct
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os


# ──────────────────────────────────────────────
# 1. Pre-emphasis
# ──────────────────────────────────────────────
def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    High-pass filter that amplifies high-frequency content.
    y[n] = x[n] - coeff * x[n-1]
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


# ──────────────────────────────────────────────
# 2. Framing
# ──────────────────────────────────────────────
def frame_signal(signal: np.ndarray,
                 frame_length: int,
                 frame_step: int) -> np.ndarray:
    """
    Split signal into overlapping frames.
    Returns array of shape (num_frames, frame_length).
    """
    signal_length = len(signal)
    num_frames = 1 + (signal_length - frame_length) // frame_step
    # Zero-pad so every frame is full
    pad_length = (num_frames - 1) * frame_step + frame_length - signal_length
    padded = np.append(signal, np.zeros(pad_length))

    indices = (np.tile(np.arange(frame_length), (num_frames, 1))
               + np.tile(np.arange(0, num_frames * frame_step, frame_step),
                         (frame_length, 1)).T)
    return padded[indices]          # shape: (num_frames, frame_length)


# ──────────────────────────────────────────────
# 3. Windows
# ──────────────────────────────────────────────
def get_window(window_type: str, length: int) -> np.ndarray:
    """Return a 1-D window function of the requested type."""
    wt = window_type.lower()
    if wt == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(length) / (length - 1))
    elif wt in ("hanning", "hann"):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))
    elif wt == "rectangular":
        return np.ones(length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def apply_window(frames: np.ndarray, window_type: str = "hamming") -> np.ndarray:
    """Multiply every frame by the chosen window."""
    win = get_window(window_type, frames.shape[1])
    return frames * win


# ──────────────────────────────────────────────
# 4. FFT power spectrum
# ──────────────────────────────────────────────
def power_spectrum(windowed_frames: np.ndarray,
                   n_fft: int = 512) -> np.ndarray:
    """
    Compute one-sided power spectrum for each frame.
    Returns shape: (num_frames, n_fft // 2 + 1)
    """
    magnitude = np.abs(fft(windowed_frames, n=n_fft, axis=1))
    return (1.0 / n_fft) * (magnitude ** 2)[:, : n_fft // 2 + 1]


# ──────────────────────────────────────────────
# 5. Mel filterbank
# ──────────────────────────────────────────────
def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_filters: int,
                   n_fft: int,
                   sample_rate: int,
                   f_min: float = 0.0,
                   f_max: float = None) -> np.ndarray:
    """
    Build a (n_filters, n_fft//2+1) triangular mel-filterbank matrix.
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # n_filters + 2 equally spaced points in mel space
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points  = mel_to_hz(mel_points)

    # Map to nearest FFT bin
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        f_left   = bin_points[m - 1]
        f_center = bin_points[m]
        f_right  = bin_points[m + 1]

        for k in range(f_left, f_center):
            if f_center != f_left:
                filters[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filters[m - 1, k] = (f_right - k) / (f_right - f_center)

    return filters


# ──────────────────────────────────────────────
# 6. Log compression
# ──────────────────────────────────────────────
def log_compress(filterbank_energies: np.ndarray,
                 floor: float = 1e-10) -> np.ndarray:
    return np.log(np.maximum(filterbank_energies, floor))


# ──────────────────────────────────────────────
# 7. DCT  →  MFCC coefficients
# ──────────────────────────────────────────────
def apply_dct(log_filterbank: np.ndarray, n_ceps: int = 13) -> np.ndarray:
    """
    Type-II DCT (same as librosa convention).
    Returns shape: (num_frames, n_ceps)
    """
    return dct(log_filterbank, type=2, axis=1, norm="ortho")[:, :n_ceps]


# ──────────────────────────────────────────────
# 8. Full real cepstrum (for boundary detection)
# ──────────────────────────────────────────────
def real_cepstrum(frame: np.ndarray, n_fft: int = 512) -> np.ndarray:
    """
    Real cepstrum = IFFT{ log |FFT{x}| }
    Returns the full cepstrum (length n_fft).
    """
    spectrum = np.abs(fft(frame, n=n_fft))
    log_spectrum = np.log(np.maximum(spectrum, 1e-10))
    cepstrum = np.real(np.fft.ifft(log_spectrum))
    return cepstrum


# ──────────────────────────────────────────────
# 9. High-level MFCC pipeline
# ──────────────────────────────────────────────
def compute_mfcc(signal: np.ndarray,
                 sample_rate: int,
                 n_mfcc: int = 13,
                 n_filters: int = 26,
                 n_fft: int = 512,
                 frame_length_ms: float = 25.0,
                 frame_step_ms: float = 10.0,
                 window_type: str = "hamming",
                 pre_emph_coeff: float = 0.97,
                 f_min: float = 0.0,
                 f_max: float = None) -> dict:
    """
    End-to-end manual MFCC computation.

    Returns a dict with keys:
      mfcc          : (T, n_mfcc) array
      log_filterbank: (T, n_filters) array
      power_spec    : (T, n_fft//2+1) array
      frames        : (T, frame_length) array
      filterbank    : (n_filters, n_fft//2+1) filter matrix
      frame_length  : int (samples)
      frame_step    : int (samples)
    """
    # Convert ms → samples
    fl = int(sample_rate * frame_length_ms / 1000)
    fs = int(sample_rate * frame_step_ms  / 1000)

    # Step 1: Pre-emphasis
    emph_signal = pre_emphasis(signal.astype(np.float64), pre_emph_coeff)

    # Step 2: Framing
    frames = frame_signal(emph_signal, fl, fs)

    # Step 3: Windowing
    windowed = apply_window(frames, window_type)

    # Step 4: Power spectrum
    pow_spec = power_spectrum(windowed, n_fft)

    # Step 5: Mel filterbank
    fb = mel_filterbank(n_filters, n_fft, sample_rate, f_min,
                        f_max or sample_rate / 2)

    # Step 6: Apply filterbank  →  energies
    fb_energies = pow_spec @ fb.T          # (T, n_filters)

    # Step 7: Log compression
    log_fb = log_compress(fb_energies)

    # Step 8: DCT
    mfcc = apply_dct(log_fb, n_mfcc)

    return dict(
        mfcc=mfcc,
        log_filterbank=log_fb,
        power_spec=pow_spec,
        frames=frames,
        filterbank=fb,
        frame_length=fl,
        frame_step=fs,
    )


# ──────────────────────────────────────────────
# 10. Visualisation helpers
# ──────────────────────────────────────────────
def plot_mfcc(mfcc: np.ndarray,
              sample_rate: int,
              frame_step: int,
              title: str = "MFCCs",
              save_path: str = None):
    T, C = mfcc.shape
    times = np.arange(T) * frame_step / sample_rate
    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(mfcc.T, aspect="auto", origin="lower",
                    extent=[times[0], times[-1], 0, C],
                    cmap="inferno")
    plt.colorbar(img, ax=ax, label="Coefficient value")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC index")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_filterbank(fb: np.ndarray, sample_rate: int, n_fft: int,
                    save_path: str = None):
    freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for f in fb:
        ax.plot(freqs, f, alpha=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Mel Filterbank")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# Main demo
# ──────────────────────────────────────────────
def _load_or_synthesise(path: str = None, duration: float = 2.0,
                         sr: int = 16000) -> tuple:
    """
    Load a WAV file, or synthesise a test tone if no path supplied.
    Returns (signal_int16_array, sample_rate).
    """
    if path and os.path.isfile(path):
        sr, sig = wav.read(path)
        if sig.ndim == 2:
            sig = sig[:, 0]       # mono
        return sig, sr

    # Synthesise: voiced chirp + silence + unvoiced noise
    t = np.linspace(0, duration, int(sr * duration))
    voiced   = (np.sin(2 * np.pi * 120 * t) * 0.5
                + np.sin(2 * np.pi * 240 * t) * 0.3
                + np.sin(2 * np.pi * 480 * t) * 0.2)
    unvoiced = np.random.randn(len(t)) * 0.1
    mask     = t > 1.0          # second half is unvoiced
    signal   = np.where(mask, unvoiced, voiced)
    signal   = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)
    return signal, sr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual MFCC engine demo")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to a WAV file (synthesised if omitted)")
    parser.add_argument("--window", type=str, default="hamming",
                        choices=["hamming", "hanning", "rectangular"])
    parser.add_argument("--n_mfcc", type=int, default=13)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    signal, sr = _load_or_synthesise(args.audio)
    print(f"Audio: {len(signal)/sr:.2f}s  |  SR: {sr} Hz")

    result = compute_mfcc(signal, sr,
                          n_mfcc=args.n_mfcc,
                          window_type=args.window)

    print(f"MFCC shape : {result['mfcc'].shape}")
    print(f"MFCC[0]    : {result['mfcc'][0].round(4)}")

    plot_mfcc(result["mfcc"], sr, result["frame_step"],
              title=f"MFCCs ({args.window} window)",
              save_path=os.path.join(args.outdir, f"mfcc_{args.window}.png"))

    plot_filterbank(result["filterbank"], sr, n_fft=512,
                    save_path=os.path.join(args.outdir, "mel_filterbank.png"))

    print("Done. Plots saved to", args.outdir)
