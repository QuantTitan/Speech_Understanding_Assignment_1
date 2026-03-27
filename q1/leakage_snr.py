"""
leakage_snr.py
==============
Spectral Leakage & SNR Analysis
Compares Rectangular, Hamming, and Hanning windows on a speech segment.

Metrics
-------
* Spectral Leakage  – energy that "bleeds" outside the main lobe of a pure
  tone, measured as the ratio of out-of-band energy to total energy (dB).
* SNR (Signal-to-Noise Ratio) – estimated from the power spectrum as the
  ratio of the signal peak power to the noise floor power (dB).

Usage
-----
    python leakage_snr.py [--audio path/to/file.wav] [--outdir outputs]
"""

import numpy as np
from scipy.fft import fft
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from mfcc_manual import get_window, pre_emphasis, frame_signal

# ──────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────

def spectral_leakage_db(frame: np.ndarray,
                         window_type: str,
                         n_fft: int = 512) -> float:
    """
    Apply window → FFT → measure leakage.

    Strategy:
      1. Find the bin with peak energy (assumed to be the main signal tone).
      2. Define the main lobe as ±main_lobe_half_width bins around the peak.
      3. Leakage = out-of-main-lobe energy / total energy  (converted to dB).

    A smaller value (more negative dB) means LESS leakage.
    """
    win   = get_window(window_type, len(frame))
    wframe = frame * win

    spectrum = np.abs(fft(wframe, n=n_fft))[: n_fft // 2]
    power    = spectrum ** 2
    total    = power.sum()

    # Main lobe: peak bin ±4 bins (empirical; covers the 3-dB BW for these windows)
    peak_bin        = int(np.argmax(power))
    half_width      = 4
    lobe_start      = max(0, peak_bin - half_width)
    lobe_end        = min(n_fft // 2, peak_bin + half_width + 1)
    main_lobe_power = power[lobe_start:lobe_end].sum()

    leakage_power   = total - main_lobe_power
    # Guard against total == 0
    ratio = leakage_power / (total + 1e-30)
    return 10 * np.log10(ratio + 1e-30)


def snr_db(frame: np.ndarray,
           window_type: str,
           n_fft: int = 512,
           noise_percentile: float = 10.0) -> float:
    """
    Estimate SNR from the windowed spectrum.

    Signal power  = peak bin power.
    Noise floor   = median of the lower `noise_percentile`% of bins (in power).

    SNR = 10 log10(signal_power / noise_power)
    """
    win    = get_window(window_type, len(frame))
    wframe = frame * win

    spectrum = np.abs(fft(wframe, n=n_fft))[: n_fft // 2]
    power    = spectrum ** 2

    signal_power = power.max()

    # Noise floor: lowest noise_percentile % of bins
    threshold  = np.percentile(power, noise_percentile)
    noise_bins = power[power <= threshold]
    noise_power = noise_bins.mean() if len(noise_bins) > 0 else 1e-30

    return 10 * np.log10(signal_power / (noise_power + 1e-30))


# ──────────────────────────────────────────────
# Analysis over an entire utterance
# ──────────────────────────────────────────────

WINDOWS = ["rectangular", "hamming", "hanning"]


def analyse_windows(signal: np.ndarray,
                    sample_rate: int,
                    frame_length_ms: float = 25.0,
                    frame_step_ms: float = 10.0,
                    n_fft: int = 512) -> dict:
    """
    Compute per-frame leakage and SNR for each window type.
    Returns dict  window_type → {"leakage": array, "snr": array}
    """
    fl = int(sample_rate * frame_length_ms / 1000)
    fs = int(sample_rate * frame_step_ms  / 1000)

    sig_f   = pre_emphasis(signal.astype(np.float64))
    frames  = frame_signal(sig_f, fl, fs)

    results = {}
    for wt in WINDOWS:
        leakages, snrs = [], []
        for frame in frames:
            leakages.append(spectral_leakage_db(frame, wt, n_fft))
            snrs.append(snr_db(frame, wt, n_fft))
        results[wt] = {
            "leakage": np.array(leakages),
            "snr"    : np.array(snrs),
        }
    return results


# ──────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────

def print_summary_table(results: dict):
    header = f"{'Window':<14} {'Mean Leakage (dB)':>20} {'Std Leakage':>12} {'Mean SNR (dB)':>15} {'Std SNR':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for wt, v in results.items():
        print(f"{wt:<14} "
              f"{v['leakage'].mean():>20.3f} "
              f"{v['leakage'].std():>12.3f} "
              f"{v['snr'].mean():>15.3f} "
              f"{v['snr'].std():>10.3f}")
    print("=" * len(header) + "\n")


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

def plot_window_spectra(frame: np.ndarray, n_fft: int = 512,
                        save_path: str = None):
    """
    Side-by-side FFT magnitude spectra for the three windows on one frame.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colors = {"rectangular": "#e05c5c",
               "hamming"    : "#5c9fe0",
               "hanning"    : "#5ce07c"}

    for ax, wt in zip(axes, WINDOWS):
        win     = get_window(wt, len(frame))
        wframe  = frame * win
        mag     = np.abs(fft(wframe, n=n_fft))[: n_fft // 2]
        mag_db  = 20 * np.log10(np.maximum(mag, 1e-10))
        ax.plot(mag_db, color=colors[wt], linewidth=0.9)
        ax.set_title(wt.capitalize(), fontsize=12)
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_ylim([-80, 5])
        ax.grid(alpha=0.3)

    fig.suptitle("Spectral leakage comparison – single frame", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_metrics_over_time(results: dict,
                           sample_rate: int,
                           frame_step: int,
                           save_path: str = None):
    """
    Time-series plots of leakage and SNR for all three windows.
    """
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.4)
    colors = {"rectangular": "#e05c5c",
               "hamming"    : "#5c9fe0",
               "hanning"    : "#5ce07c"}

    for row, metric in enumerate(["leakage", "snr"]):
        ax = fig.add_subplot(gs[row])
        for wt, v in results.items():
            times = np.arange(len(v[metric])) * frame_step / sample_rate
            ax.plot(times, v[metric], label=wt.capitalize(),
                    color=colors[wt], linewidth=0.9, alpha=0.85)
        ylabel = ("Leakage (dB)" if metric == "leakage" else "SNR (dB)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " over time")
        ax.legend()
        ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_bar_comparison(results: dict, save_path: str = None):
    """
    Bar chart of mean leakage and mean SNR for each window.
    """
    means_leakage = [results[w]["leakage"].mean() for w in WINDOWS]
    means_snr     = [results[w]["snr"].mean()     for w in WINDOWS]

    x   = np.arange(len(WINDOWS))
    w   = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#e05c5c", "#5c9fe0", "#5ce07c"]
    ax1.bar(x, means_leakage, width=w, color=colors, edgecolor="black")
    ax1.set_xticks(x); ax1.set_xticklabels([w.capitalize() for w in WINDOWS])
    ax1.set_ylabel("Mean Leakage (dB)"); ax1.set_title("Spectral Leakage")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, means_snr, width=w, color=colors, edgecolor="black")
    ax2.set_xticks(x); ax2.set_xticklabels([w.capitalize() for w in WINDOWS])
    ax2.set_ylabel("Mean SNR (dB)"); ax2.set_title("Estimated SNR")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Window Function Comparison", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def _load_or_synthesise(path=None, sr=16000, duration=2.0):
    if path and os.path.isfile(path):
        sr, sig = wav.read(path)
        if sig.ndim == 2:
            sig = sig[:, 0]
        return sig, sr

    t  = np.linspace(0, duration, int(sr * duration))
    # Mix of tones: voiced-like + broadband noise
    sig = (0.6 * np.sin(2 * np.pi * 150 * t)
           + 0.3 * np.sin(2 * np.pi * 300 * t)
           + 0.1 * np.random.randn(len(t)))
    sig = (sig / np.max(np.abs(sig)) * 32767).astype(np.int16)
    return sig, sr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spectral leakage & SNR analysis")
    parser.add_argument("--audio",  default=None)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    signal, sr = _load_or_synthesise(args.audio)
    print(f"Audio loaded: {len(signal)/sr:.2f}s @ {sr} Hz")

    fl = int(sr * 25 / 1000)
    fs = int(sr * 10 / 1000)

    results = analyse_windows(signal, sr, n_fft=512)
    print_summary_table(results)

    # Use the middle frame for single-frame plots
    from mfcc_manual import frame_signal, pre_emphasis
    frames = frame_signal(pre_emphasis(signal.astype(np.float64)), fl, fs)
    mid_frame = frames[len(frames) // 2]

    plot_window_spectra(mid_frame, n_fft=512,
                        save_path=os.path.join(args.outdir, "window_spectra.png"))
    plot_metrics_over_time(results, sr, fs,
                           save_path=os.path.join(args.outdir, "leakage_snr_time.png"))
    plot_bar_comparison(results,
                        save_path=os.path.join(args.outdir, "leakage_snr_bar.png"))

    print("All plots saved to:", args.outdir)
