"""
voiced_unvoiced.py
==================
Automatic voiced / unvoiced boundary detection using the real cepstrum.

Algorithm
---------
Low-Quefrency region  (quefrencies < low_q_thresh)
  → represents the vocal-tract spectral envelope
High-Quefrency region (quefrencies ≥ low_q_thresh)
  → represents periodic (pitch) structures

Decision rule per frame
  A frame is "voiced" if the normalised energy in the high-quefrency region
  exceeds a threshold τ.  The threshold is chosen adaptively as the Otsu
  threshold of the high-quefrency energy distribution across all frames.

Boundary detection
  Voiced/unvoiced labels are median-filtered (smoothing), then boundaries
  are reported as the frames where the label changes.

Usage
-----
    python voiced_unvoiced.py [--audio path/to/file.wav] [--outdir outputs]
"""

import numpy as np
from scipy.fft import fft
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from mfcc_manual import (pre_emphasis, frame_signal, apply_window,
                          real_cepstrum)


# ──────────────────────────────────────────────
# Quefrency energy features
# ──────────────────────────────────────────────

def cepstral_features(signal: np.ndarray,
                       sample_rate: int,
                       frame_length_ms: float = 25.0,
                       frame_step_ms: float   = 10.0,
                       n_fft: int             = 512,
                       window_type: str       = "hamming") -> dict:
    """
    Compute per-frame cepstrum and split into low / high quefrency energy.

    Returns
    -------
    dict with keys:
      cepstra       : (T, n_fft) array – full real cepstrum per frame
      low_q_energy  : (T,) array
      high_q_energy : (T,) array
      frames        : (T, frame_length)
      frame_step    : int (samples)
      frame_length  : int (samples)
    """
    fl = int(sample_rate * frame_length_ms / 1000)
    fs = int(sample_rate * frame_step_ms  / 1000)

    emph   = pre_emphasis(signal.astype(np.float64))
    frames = frame_signal(emph, fl, fs)
    win_fr = apply_window(frames, window_type)

    # Low quefrency boundary: ~ 1 ms corresponds to pitch > 1000 Hz
    # We use a fixed 1 ms boundary (≈ sample_rate * 0.001 samples)
    low_q_boundary = int(sample_rate * 0.001)   # ~ 16 samples at 16 kHz

    cepstra     = np.array([real_cepstrum(fr, n_fft) for fr in win_fr])
    low_q_eng   = (cepstra[:, :low_q_boundary] ** 2).mean(axis=1)
    high_q_eng  = (cepstra[:, low_q_boundary:n_fft // 2] ** 2).mean(axis=1)

    return dict(cepstra=cepstra,
                low_q_energy=low_q_eng,
                high_q_energy=high_q_eng,
                frames=frames,
                frame_step=fs,
                frame_length=fl,
                low_q_boundary=low_q_boundary)


# ──────────────────────────────────────────────
# Otsu threshold (for binarisation)
# ──────────────────────────────────────────────

def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    1-D Otsu threshold from a distribution of values.
    """
    hist, bin_edges = np.histogram(values, bins=n_bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return values.mean()

    probs      = hist / total
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    best_thresh = bin_centers[0]
    best_var    = 0.0
    w0, mu0     = 0.0, 0.0

    for i, (p, c) in enumerate(zip(probs, bin_centers)):
        w1 = 1 - w0 - p
        if w0 + p == 0 or w1 <= 0:
            w0 += p
            mu0 = ((mu0 * (w0 - p) + c * p) / w0) if w0 > 0 else 0
            continue

        mu0_new = (mu0 * w0 + c * p) / (w0 + p) if (w0 + p) > 0 else 0
        mu1     = ((values.sum() - (mu0 * w0 + c * p) * total / total)
                   / (w1 * total / total + 1e-30))
        mu1     = (bin_centers[i+1:] * probs[i+1:]).sum() / (probs[i+1:].sum() + 1e-30)

        between_var = (w0 + p) * w1 * (mu0_new - mu1) ** 2
        if between_var > best_var:
            best_var    = between_var
            best_thresh = c

        w0  += p
        mu0  = (mu0 * (w0 - p) + c * p) / w0 if w0 > 0 else 0

    return best_thresh


# ──────────────────────────────────────────────
# Median filter (without scipy.signal dependency)
# ──────────────────────────────────────────────

def median_filter_1d(arr: np.ndarray, kernel: int = 5) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.array([np.median(padded[i:i+kernel])
                     for i in range(len(arr))])


# ──────────────────────────────────────────────
# Main boundary detector
# ──────────────────────────────────────────────

def detect_boundaries(signal: np.ndarray,
                       sample_rate: int,
                       frame_length_ms: float = 25.0,
                       frame_step_ms:   float = 10.0,
                       smooth_kernel:   int   = 5,
                       n_fft:           int   = 512) -> dict:
    """
    Returns
    -------
    dict with:
      labels        : (T,) int array  1 = voiced, 0 = unvoiced
      boundaries    : list of (start_sample, end_sample, label) tuples
      high_q_energy : (T,) float array
      threshold     : float
      frame_step    : int (samples)
      feature_dict  : full cepstral_features output
    """
    feats = cepstral_features(signal, sample_rate,
                               frame_length_ms, frame_step_ms, n_fft)
    hqe   = feats["high_q_energy"]

    # Normalise
    hqe_norm = (hqe - hqe.min()) / (hqe.max() - hqe.min() + 1e-30)

    # Otsu threshold on normalised energy
    threshold = otsu_threshold(hqe_norm)

    # Raw labels
    raw_labels = (hqe_norm > threshold).astype(int)

    # Smooth
    smooth_labels = (median_filter_1d(raw_labels.astype(float),
                                       smooth_kernel) > 0.5).astype(int)

    # Find boundaries (frame indices where label changes)
    changes = np.where(np.diff(smooth_labels) != 0)[0] + 1
    seg_starts = np.concatenate([[0], changes])
    seg_ends   = np.concatenate([changes, [len(smooth_labels)]])

    fs = feats["frame_step"]
    boundaries = []
    for s, e in zip(seg_starts, seg_ends):
        label   = smooth_labels[s]
        s_samp  = s * fs
        e_samp  = e * fs
        boundaries.append((s_samp, e_samp, int(label)))

    return dict(labels=smooth_labels,
                boundaries=boundaries,
                high_q_energy=hqe_norm,
                threshold=threshold,
                frame_step=fs,
                feature_dict=feats)


# ──────────────────────────────────────────────
# Convenience: boundary times in seconds
# ──────────────────────────────────────────────

def boundary_times(boundaries: list, sample_rate: int) -> list:
    """Convert sample-based boundaries → (start_s, end_s, label) in seconds."""
    return [(s / sample_rate, e / sample_rate, lbl)
            for s, e, lbl in boundaries]


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

def plot_voiced_unvoiced(signal: np.ndarray,
                          sample_rate: int,
                          result: dict,
                          save_path: str = None):
    """
    Four-panel figure:
      1. Waveform with voiced/unvoiced colour overlay
      2. High-quefrency energy with Otsu threshold line
      3. Smoothed V/UV label sequence
      4. First 3 cepstra (low vs high quefrency)
    """
    labels     = result["labels"]
    hqe        = result["high_q_energy"]
    threshold  = result["threshold"]
    fs         = result["frame_step"]
    boundaries = result["boundaries"]
    cepstra    = result["feature_dict"]["cepstra"]
    lqb        = result["feature_dict"]["low_q_boundary"]

    t_sig   = np.arange(len(signal)) / sample_rate
    t_frame = np.arange(len(labels)) * fs / sample_rate

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # ── Panel 1: waveform ──
    ax = axes[0]
    ax.plot(t_sig, signal, color="#444", linewidth=0.5, alpha=0.7)
    colors = {1: "#5c9fe0", 0: "#e05c5c"}
    for s_samp, e_samp, lbl in boundaries:
        ax.axvspan(s_samp / sample_rate, e_samp / sample_rate,
                   alpha=0.25, color=colors[lbl])
    voiced_patch   = mpatches.Patch(color="#5c9fe0", alpha=0.5, label="Voiced")
    unvoiced_patch = mpatches.Patch(color="#e05c5c", alpha=0.5, label="Unvoiced")
    ax.legend(handles=[voiced_patch, unvoiced_patch], loc="upper right")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.set_title("Waveform with Voiced/Unvoiced Overlay")
    ax.grid(alpha=0.3)

    # ── Panel 2: high-quefrency energy ──
    ax = axes[1]
    ax.plot(t_frame, hqe, color="#5ce07c", linewidth=1.0)
    ax.axhline(threshold, color="red", linestyle="--",
               label=f"Otsu threshold = {threshold:.3f}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalised HQ energy")
    ax.set_title("High-Quefrency Cepstral Energy")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Panel 3: label sequence ──
    ax = axes[2]
    ax.step(t_frame, labels, where="mid", color="#9b59b6", linewidth=1.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Unvoiced", "Voiced"])
    ax.set_xlabel("Time (s)"); ax.set_title("Voiced / Unvoiced Labels")
    ax.grid(alpha=0.3)

    # ── Panel 4: a few cepstra (low vs high region) ──
    ax = axes[3]
    n_show = min(5, len(cepstra))
    sample_idx = np.linspace(0, len(cepstra) - 1, n_show, dtype=int)
    quefrency = np.arange(cepstra.shape[1] // 2)
    for i, idx in enumerate(sample_idx):
        c_half = cepstra[idx, : len(quefrency)]
        lbl    = "V" if labels[idx] else "UV"
        ax.plot(quefrency, np.abs(c_half), alpha=0.7,
                label=f"Frame {idx} ({lbl})")
    ax.axvline(lqb, color="red", linestyle="--",
               label=f"Low/High boundary (q={lqb})")
    ax.set_xlabel("Quefrency (samples)")
    ax.set_ylabel("|Cepstrum|")
    ax.set_title("Sample Cepstra – Low vs High Quefrency Regions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

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

def _load_or_synthesise(path=None, sr=16000, duration=3.0):
    if path and os.path.isfile(path):
        sr, sig = wav.read(path)
        if sig.ndim == 2:
            sig = sig[:, 0]
        return sig, sr

    t   = np.linspace(0, duration, int(sr * duration))
    # Voiced: harmonic signal (first 1 s)
    voiced = (np.sin(2*np.pi*120*t)
              + 0.5*np.sin(2*np.pi*240*t)
              + 0.3*np.sin(2*np.pi*480*t))
    # Unvoiced: broadband noise (1–2 s)
    unvoiced = np.random.randn(len(t)) * 0.15
    # Silence (2–3 s)
    silence  = np.zeros(len(t))

    mask = np.zeros(len(t))
    mask[t < 1.0]  = 1    # voiced
    mask[(t >= 1.0) & (t < 2.0)] = 2  # unvoiced
    # rest stays 0 (silence → treated as unvoiced)

    sig = np.where(mask == 1, voiced,
           np.where(mask == 2, unvoiced, silence))
    sig = (sig / np.max(np.abs(sig) + 1e-10) * 32767).astype(np.int16)
    return sig, sr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voiced/unvoiced boundary detector")
    parser.add_argument("--audio",  default=None)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    signal, sr = _load_or_synthesise(args.audio)
    print(f"Audio: {len(signal)/sr:.2f}s @ {sr} Hz")

    result = detect_boundaries(signal, sr)
    btimes = boundary_times(result["boundaries"], sr)

    print("\nDetected segments:")
    print(f"  {'Start (s)':>10} {'End (s)':>10} {'Label':>8}")
    for s, e, lbl in btimes:
        print(f"  {s:>10.3f} {e:>10.3f} {'Voiced' if lbl else 'Unvoiced':>8}")

    plot_voiced_unvoiced(signal, sr, result,
                          save_path=os.path.join(args.outdir,
                                                  "voiced_unvoiced.png"))
    print(f"\nOtsu threshold: {result['threshold']:.4f}")
    print("Done.")
