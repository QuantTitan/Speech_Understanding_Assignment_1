"""
evaluation_scripts/fad_eval.py — Fréchet Audio Distance (FAD) Evaluation
==========================================================================
FAD measures perceptual quality degradation introduced by voice conversion.
Lower FAD = less distortion, better audio quality.

Pipeline:
  1. Extract VGGish embeddings from reference and test audio files
  2. Fit multivariate Gaussians (μ, Σ) to each embedding set
  3. Compute Fréchet distance: FAD = ||μ₁-μ₂||² + Tr(Σ₁+Σ₂ - 2·√(Σ₁Σ₂))

Usage:
    python evaluation_scripts/fad_eval.py \
        --reference examples/original.wav \
        --test      examples/converted_female_young.wav

    python evaluation_scripts/fad_eval.py \
        --reference_dir examples/ \
        --test_dir      examples/ \
        --pattern       converted_*.wav
"""

import argparse
import glob
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy import linalg

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# VGGISH-LIKE EMBEDDING NETWORK (proxy — full VGGish needs large checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class LogMelEmbedder(torch.nn.Module):
    """
    Lightweight VGGish-style CNN for extracting 128-D audio embeddings.
    Architecture mirrors the VGGish mel pathway but uses fewer parameters
    to avoid requiring the full pretrained checkpoint.

    For production, replace with:
        torch.hub.load('harritaylor/torchvggish', 'vggish')
    """
    def __init__(self, n_mels: int = 64, emb_dim: int = 128):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=16_000, n_fft=512, hop_length=160,
            win_length=400, n_mels=n_mels, f_min=125.0, f_max=7500.0,
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=80.0)

        self.cnn = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(1, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            # Block 2
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            # Block 3
            torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, emb_dim),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (1, T) → embedding: (emb_dim,)"""
        mel = self.amp_to_db(self.mel(wav))      # (1, n_mels, T_frames)
        mel = (mel + 40.0) / 40.0
        mel = mel.unsqueeze(0)                   # (1, 1, n_mels, T_frames)
        features = self.cnn(mel)
        return self.fc(features).squeeze(0)      # (emb_dim,)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_16k(path: str) -> Optional[torch.Tensor]:
    """Load audio → mono 16 kHz tensor (1, T)."""
    try:
        wav, sr = torchaudio.load(path)
        if sr != 16_000:
            wav = T.Resample(sr, 16_000)(wav)
        wav = wav.mean(dim=0, keepdim=True)     # stereo → mono
        return wav
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def chunk_audio(wav: torch.Tensor, chunk_sec: float = 1.0,
                sr: int = 16_000) -> List[torch.Tensor]:
    """Split waveform into fixed-length chunks for embedding extraction."""
    chunk_len = int(sr * chunk_sec)
    chunks = []
    total = wav.shape[1]
    for start in range(0, total - chunk_len + 1, chunk_len):
        chunks.append(wav[:, start: start + chunk_len])
    if not chunks and total > 0:
        # Audio shorter than one chunk — pad
        padded = torch.nn.functional.pad(wav, (0, chunk_len - total))
        chunks = [padded]
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(paths: List[str], embedder: LogMelEmbedder,
                       chunk_sec: float = 1.0) -> np.ndarray:
    """
    Extract embedding matrix from a list of audio files.

    Returns:
        embeddings: (N_chunks, emb_dim) array
    """
    all_embs = []
    embedder.eval()
    for path in paths:
        wav = load_audio_16k(path)
        if wav is None:
            continue
        chunks = chunk_audio(wav, chunk_sec)
        for chunk in chunks:
            emb = embedder(chunk)
            all_embs.append(emb.cpu().numpy())

    if not all_embs:
        raise ValueError("No valid audio embeddings extracted.")
    return np.vstack(all_embs)   # (N, emb_dim)


# ─────────────────────────────────────────────────────────────────────────────
# FRÉCHET DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of embedding distribution."""
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """
    Compute the Fréchet distance between two Gaussian distributions.
    FD = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·√(Σ₁Σ₂))
    """
    diff = mu1 - mu2
    mean_diff_sq = np.dot(diff, diff)

    # Numerically stable matrix square root
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m:.4g} too large")
        covmean = covmean.real

    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(mean_diff_sq + trace_term)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def fad_from_files(reference_paths: List[str],
                   test_paths: List[str]) -> dict:
    """
    Compute FAD between reference and test audio sets.

    Returns:
        result dict with 'fad', 'interpretation', and statistics
    """
    embedder = LogMelEmbedder()

    print(f"  Extracting reference embeddings ({len(reference_paths)} file(s)) …")
    ref_embs = extract_embeddings(reference_paths, embedder)

    print(f"  Extracting test embeddings ({len(test_paths)} file(s)) …")
    test_embs = extract_embeddings(test_paths, embedder)

    mu_ref, sigma_ref = compute_statistics(ref_embs)
    mu_test, sigma_test = compute_statistics(test_embs)

    fad = frechet_distance(mu_ref, sigma_ref, mu_test, sigma_test)

    # Interpretation thresholds (empirical from literature)
    if fad < 2.0:
        quality = "EXCELLENT — minimal perceptual distortion"
    elif fad < 5.0:
        quality = "GOOD — acceptable quality, minor artefacts"
    elif fad < 15.0:
        quality = "MODERATE — noticeable degradation"
    else:
        quality = "POOR — significant audio artefacts"

    return {
        "fad": round(fad, 4),
        "quality": quality,
        "ref_n_chunks": len(ref_embs),
        "test_n_chunks": len(test_embs),
        "ref_mean_norm": float(np.linalg.norm(mu_ref)),
        "test_mean_norm": float(np.linalg.norm(mu_test)),
    }


def fad_single_pair(reference: str, test: str) -> dict:
    """Single reference vs single test file."""
    return fad_from_files([reference], [test])


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def batch_evaluate(reference: str, test_dir: str,
                   pattern: str = "converted_*.wav") -> None:
    """
    Evaluate FAD between a reference file and all converted files in a directory.
    """
    test_paths = sorted(glob.glob(os.path.join(test_dir, pattern)))
    if not test_paths:
        print(f"[WARN] No files matching '{pattern}' in {test_dir}")
        return

    print("\n" + "═" * 65)
    print("  BATCH FAD EVALUATION")
    print("═" * 65)
    print(f"  Reference: {reference}")
    print(f"  Test dir:  {test_dir}  ({len(test_paths)} files)")
    print("─" * 65)

    results = []
    for test_path in test_paths:
        label = Path(test_path).stem
        try:
            result = fad_single_pair(reference, test_path)
            fad_val = result["fad"]
            quality = result["quality"].split("—")[0].strip()
            results.append((label, fad_val, quality))
            print(f"  {label:<35} FAD={fad_val:8.4f}  [{quality}]")
        except Exception as e:
            print(f"  {label:<35} ERROR: {e}")

    if results:
        fads = [r[1] for r in results]
        print("─" * 65)
        print(f"  {'Average FAD':<35} {np.mean(fads):.4f}")
        print(f"  {'Min FAD':<35} {np.min(fads):.4f}")
        print(f"  {'Max FAD':<35} {np.max(fads):.4f}")
        print("═" * 65)

        # Save results CSV
        import csv
        csv_path = os.path.join(test_dir, "fad_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["conversion", "fad", "quality"])
            writer.writerows(results)
        print(f"\n  Results saved: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fréchet Audio Distance Evaluation")
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference audio file")
    parser.add_argument("--test", type=str, default=None,
                        help="Test/converted audio file")
    parser.add_argument("--reference_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default="examples")
    parser.add_argument("--pattern", type=str, default="converted_*.wav")
    args = parser.parse_args()

    if args.reference and args.test:
        print(f"\n[FAD] Single pair evaluation")
        print(f"  Reference: {args.reference}")
        print(f"  Test:      {args.test}")
        result = fad_single_pair(args.reference, args.test)
        print(f"\n  FAD = {result['fad']}")
        print(f"  Quality: {result['quality']}")

    elif args.reference:
        batch_evaluate(args.reference, args.test_dir, args.pattern)

    else:
        # Demo with synthetic audio
        print("[INFO] No input specified. Generating synthetic audio for demo …")
        import torchaudio

        def _synth(freq: float, duration: float = 2.0, sr: int = 16_000) -> torch.Tensor:
            t = torch.linspace(0, duration, int(sr * duration))
            wav = 0.5 * torch.sin(2 * 3.14159 * freq * t)
            return wav.unsqueeze(0)

        ref_path = "/tmp/fad_ref.wav"
        test_path = "/tmp/fad_test.wav"
        torchaudio.save(ref_path, _synth(220.0), 16_000)
        torchaudio.save(test_path, _synth(440.0), 16_000)

        result = fad_single_pair(ref_path, test_path)
        print(f"\n[DEMO RESULT]")
        print(f"  FAD between 220 Hz and 440 Hz tones: {result['fad']:.4f}")
        print(f"  Quality:                              {result['quality']}")

        print("\n[USAGE]")
        print("  python evaluation_scripts/fad_eval.py \\")
        print("      --reference examples/original.wav \\")
        print("      --test      examples/converted_female_young.wav")


if __name__ == "__main__":
    main()
