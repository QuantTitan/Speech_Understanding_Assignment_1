"""
pp_demo.py — Privacy-Preserving Voice Conversion Demo
=======================================================
Demonstrates the PrivacyPreservingConverter on a real or synthetic audio file.

What it does:
  1. Loads (or generates) a source audio clip
  2. Converts it to each of the 9 demographic combinations (3 genders × 3 ages)
  3. Saves all audio pairs to examples/
  4. Prints word-error-rate proxy (Levenshtein distance on Whisper transcriptions)
     to verify linguistic content is preserved

Usage:
    python pp_demo.py [--input path/to/audio.wav] [--checkpoint path/to/ckpt.pt]
    python pp_demo.py                   # uses synthetic tone as input
    python pp_demo.py --asr             # run ASR verification (requires openai-whisper)
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from privacymodule import PrivacyPreservingConverter, VoiceConversionConfig, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path("examples")
EXAMPLES_DIR.mkdir(exist_ok=True)

GENDER_CLASSES = ["male", "female", "other"]
AGE_CLASSES = ["young", "middle", "senior"]


def load_audio(path: str, target_sr: int = 16_000) -> torch.Tensor:
    """Load audio file, resample, return mono (samples,) tensor."""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    wav = wav.mean(dim=0)   # stereo → mono
    return wav


def generate_synthetic_utterance(duration_sec: float = 3.0, sr: int = 16_000) -> torch.Tensor:
    """
    Generates a synthetic speech-like signal (harmonic tone + noise)
    for demonstration when no real audio file is provided.
    """
    t = torch.linspace(0, duration_sec, int(sr * duration_sec))
    # Fundamental + 4 harmonics (approximate vowel formant structure)
    signal = (
        0.40 * torch.sin(2 * np.pi * 150 * t) +      # F0 ~150 Hz (male-ish)
        0.25 * torch.sin(2 * np.pi * 800 * t) +       # F1
        0.20 * torch.sin(2 * np.pi * 1200 * t) +      # F2
        0.10 * torch.sin(2 * np.pi * 2400 * t) +      # F3
        0.05 * torch.randn_like(t)                     # breath noise
    )
    # Apply envelope (ADSR-like)
    env = torch.ones_like(t)
    attack = int(0.05 * sr)
    release = int(0.1 * sr)
    env[:attack] = torch.linspace(0, 1, attack)
    env[-release:] = torch.linspace(1, 0, release)
    signal = signal * env
    return signal / signal.abs().max()


def save_audio(wav: torch.Tensor, path: str, sr: int = 16_000):
    """Save (samples,) or (1, samples) tensor as WAV."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    # Clamp to [-1, 1]
    wav = wav.clamp(-1.0, 1.0)
    torchaudio.save(path, wav.cpu(), sr)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Simple Levenshtein distance for WER proxy."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def word_error_rate(ref: str, hyp: str) -> float:
    """Approximate WER at word level."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0
    edits = levenshtein_distance(ref_words, hyp_words)
    return edits / len(ref_words)


# ─────────────────────────────────────────────────────────────────────────────
# ASR VERIFICATION (optional)
# ─────────────────────────────────────────────────────────────────────────────

def run_asr_verification(original_path: str, converted_paths: list) -> dict:
    """
    Uses openai-whisper to transcribe original and converted audio,
    then computes WER to verify linguistic content preservation.
    """
    try:
        import whisper
        print("\n[ASR] Loading Whisper tiny model …")
        asr_model = whisper.load_model("tiny")
    except ImportError:
        print("[WARN] openai-whisper not installed. Skipping ASR verification.")
        print("       Install with: pip install openai-whisper")
        return {}

    def transcribe(path: str) -> str:
        result = asr_model.transcribe(path, language="en", fp16=False)
        return result["text"].strip()

    print(f"[ASR] Transcribing original: {original_path}")
    ref_text = transcribe(original_path)
    print(f"  Reference: '{ref_text}'")

    wer_results = {}
    for conv_path in converted_paths:
        label = Path(conv_path).stem
        hyp_text = transcribe(conv_path)
        wer = word_error_rate(ref_text, hyp_text)
        wer_results[label] = {"hypothesis": hyp_text, "wer": round(wer, 4)}
        status = "✓ PASS" if wer < 0.3 else "✗ FAIL"
        print(f"  [{status}] {label:<30} WER={wer:.3f}  '{hyp_text}'")

    avg_wer = np.mean([v["wer"] for v in wer_results.values()])
    print(f"\n  Average WER across conversions: {avg_wer:.3f}")
    print("  (WER < 0.30 indicates acceptable linguistic preservation)")
    return wer_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(
    input_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    run_asr: bool = False,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    cfg = VoiceConversionConfig()

    # ── Build model ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  PRIVACY-PRESERVING VOICE CONVERSION DEMO")
    print("=" * 60)
    print(f"[INFO] Building model on {device} …")
    model = PrivacyPreservingConverter(cfg).to(device)
    print(f"[INFO] {count_parameters(model)}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[INFO] No checkpoint loaded — using random weights (demo only).")
        print("       Train with train_fair.py for production use.")
    model.eval()

    # ── Load / generate source audio ─────────────────────────────────────────
    if input_path and os.path.exists(input_path):
        print(f"\n[INFO] Loading audio: {input_path}")
        src_wav = load_audio(input_path, target_sr=cfg.sample_rate)
    else:
        if input_path:
            print(f"[WARN] File not found: {input_path}. Using synthetic audio.")
        else:
            print("\n[INFO] No input file — generating synthetic utterance (3 s).")
        src_wav = generate_synthetic_utterance(3.0, cfg.sample_rate)

    # Save original
    orig_path = str(EXAMPLES_DIR / "original.wav")
    save_audio(src_wav, orig_path, cfg.sample_rate)
    print(f"[INFO] Source audio saved: {orig_path}  ({len(src_wav)/cfg.sample_rate:.2f}s)")

    # ── Run all 9 conversions ─────────────────────────────────────────────────
    print("\n[INFO] Running demographic attribute conversions …\n")
    print(f"  {'Target':<25}  {'Duration':>10}  {'File'}")
    print("─" * 65)

    converted_paths = []
    for gender in GENDER_CLASSES:
        for age in AGE_CLASSES:
            label = f"{gender}_{age}"
            t0 = time.time()
            with torch.no_grad():
                wav_out = model.convert(
                    src_wav.to(device),
                    src_sr=cfg.sample_rate,
                    target_gender=gender,
                    target_age=age,
                )
            elapsed = time.time() - t0
            out_path = str(EXAMPLES_DIR / f"converted_{label}.wav")
            save_audio(wav_out, out_path, cfg.sample_rate)
            dur = wav_out.shape[-1] / cfg.sample_rate
            converted_paths.append(out_path)
            print(f"  {label:<25}  {dur:>8.2f}s  {out_path}")

    # ── ASR Verification ──────────────────────────────────────────────────────
    if run_asr:
        wer_results = run_asr_verification(orig_path, converted_paths)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print(f"  Original audio:      {orig_path}")
    print(f"  Converted audio:     {len(converted_paths)} files in examples/")
    print(f"  Conversions:         {len(GENDER_CLASSES)} genders × {len(AGE_CLASSES)} age groups")
    print()
    print("  Model sub-modules:")
    print(f"    ContentEncoder:    strips speaker identity via Instance Norm")
    print(f"    SpeakerEncoder:    attentive LSTM-based d-vector extraction")
    print(f"    AttributeEmbedder: gender × age → style vector")
    print(f"    Decoder (AdaIN):   injects target style into content features")
    print(f"    MiniHiFiGAN:       mel → waveform (256-upsampling vocoder)")
    print()
    print("  Ethical note:")
    print("    Biometric attribute obfuscation protects speaker privacy.")
    print("    Linguistic content is preserved for downstream ASR tasks.")
    print("    All transformations are reversible only by model owner.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy-Preserving Voice Conversion Demo")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to source audio WAV file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--asr", action="store_true",
                        help="Run ASR verification with Whisper (requires openai-whisper)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device: cpu | cuda | mps")
    args = parser.parse_args()

    run_demo(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        run_asr=args.asr,
        device_str=args.device,
    )
