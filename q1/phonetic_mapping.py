"""
phonetic_mapping.py
===================
Forced-alignment phoneme mapping using a pre-trained Wav2Vec2 model
(facebook/wav2vec2-base-960h) from Hugging Face, plus RMSE calculation
between manual cepstrum boundaries and the model's phoneme boundaries.

Pipeline
--------
1. Load audio  →  run voiced/unvoiced.detect_boundaries() for "manual" boundaries
2. Run Wav2Vec2 CTC decoding to get emission probabilities
3. Viterbi forced alignment on the emission logits → frame-level phone sequence
4. Convert frame-level phones to phone boundaries
5. Match manual V/UV boundaries to nearest phone boundaries
6. Compute RMSE (in seconds)

Forced Alignment approach
--------------------------
We use the `torchaudio.pipelines.MMS_FA` / WAV2VEC2_ASR_BASE_960H pipeline for
forced alignment, or fall back to a manual CTC Viterbi decoder when torchaudio
forced alignment is unavailable.

Usage
-----
    python phonetic_mapping.py [--audio path.wav] [--outdir outputs]
"""

import os
import numpy as np
import torch
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from voiced_unvoiced import detect_boundaries, boundary_times


# ──────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────

MODEL_NAME = "facebook/wav2vec2-base-960h"

# Character-level vocab that WAV2VEC2_BASE outputs
# (all uppercase; we keep them for phone label display)
SILENCE_TOKEN = "|"


def load_model(model_name: str = MODEL_NAME, device: str = "cpu"):
    """Download / cache and return processor + model."""
    print(f"Loading model: {model_name} …")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


# ──────────────────────────────────────────────
# Audio loading / resampling
# ──────────────────────────────────────────────

def load_audio_16k(path: str) -> np.ndarray:
    """
    Load a WAV file, convert to mono float32, resample to 16 kHz if needed.
    Returns (signal_float32, sample_rate=16000).
    """
    sr, sig = wav.read(path)
    if sig.ndim == 2:
        sig = sig[:, 0]
    sig = sig.astype(np.float32)

    if sr != 16000:
        # Simple linear resampling (good enough for alignment)
        target_len = int(len(sig) * 16000 / sr)
        sig = np.interp(np.linspace(0, len(sig) - 1, target_len),
                        np.arange(len(sig)), sig)
        sr = 16000

    # Normalise to [-1, 1]
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val
    return sig, sr


# ──────────────────────────────────────────────
# CTC emission + Viterbi forced alignment
# ──────────────────────────────────────────────

def get_emissions(processor, model, signal: np.ndarray,
                  device: str = "cpu") -> torch.Tensor:
    """
    Run the Wav2Vec2 model and return log-softmax emission probabilities.
    Shape: (T_frames, vocab_size)
    """
    inputs = processor(signal, sampling_rate=16000,
                       return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits[0]   # (T, V)
    return torch.log_softmax(logits, dim=-1).cpu()


def greedy_decode(emissions: torch.Tensor, processor) -> list:
    """
    Greedy CTC decode: argmax per frame, collapse repeated + remove blank.
    Returns list of (char, start_frame, end_frame).
    """
    blank_id = processor.tokenizer.pad_token_id
    ids      = emissions.argmax(dim=-1).tolist()

    tokens = []
    prev   = None
    for f, tok in enumerate(ids):
        if tok != blank_id and tok != prev:
            tokens.append((processor.tokenizer.convert_ids_to_tokens([tok])[0], f))
        prev = tok

    # Build (char, start, end) triples
    result = []
    for i, (ch, sf) in enumerate(tokens):
        ef = tokens[i+1][1] if i+1 < len(tokens) else len(ids)
        result.append((ch, sf, ef))
    return result


def viterbi_align(emissions: torch.Tensor,
                  token_ids: list,
                  blank_id:  int) -> list:
    """
    Simple Viterbi forced alignment for a short token sequence.
    Returns list of (token_id, start_frame, end_frame).

    For long utterances (>200 tokens) we chunk into segments; here
    we assume short demo files and run the full DP in memory.
    """
    T, V   = emissions.shape
    n_tok  = len(token_ids)
    # CTC: interleave blank  →  2*n_tok+1 states
    states = []
    for tid in token_ids:
        states.append(blank_id)   # blank before each token
        states.append(tid)
    states.append(blank_id)       # trailing blank
    S = len(states)

    # DP table: shape (T, S)
    NEG_INF = float("-inf")
    dp      = torch.full((T, S), NEG_INF)
    back    = torch.full((T, S), -1, dtype=torch.long)

    # Initialise
    dp[0, 0] = emissions[0, states[0]]
    if S > 1:
        dp[0, 1] = emissions[0, states[1]]

    for t in range(1, T):
        for s in range(S):
            best  = NEG_INF
            best_s = s
            # Can come from same state
            if dp[t-1, s] > best:
                best, best_s = dp[t-1, s], s
            # Can come from previous state
            if s > 0 and dp[t-1, s-1] > best:
                best, best_s = dp[t-1, s-1], s-1
            # Can skip blank if states[s] != blank and states[s-2] != states[s]
            if (s > 1
                    and states[s] != blank_id
                    and states[s] != states[s-2]
                    and dp[t-1, s-2] > best):
                best, best_s = dp[t-1, s-2], s-2

            dp[t, s]   = best + emissions[t, states[s]]
            back[t, s] = best_s

    # Traceback
    # End in last or second-to-last state
    best_final = max(S-1, 0)
    if dp[T-1, S-2] > dp[T-1, S-1]:
        best_final = S-2

    path = []
    cur  = best_final
    for t in range(T-1, -1, -1):
        path.append((t, states[cur]))
        cur = back[t, cur].item()
    path.reverse()

    # Convert path to segments (consecutive same state)
    segments = []
    cur_tok  = path[0][1]
    cur_start = 0
    for t, tok in path[1:]:
        if tok != cur_tok:
            segments.append((cur_tok, cur_start, t))
            cur_tok   = tok
            cur_start = t
    segments.append((cur_tok, cur_start, T))

    # Filter blanks, return token segments
    token_segments = [(tid, s, e) for tid, s, e in segments
                       if tid != blank_id]
    return token_segments


def forced_align(processor, model, signal: np.ndarray,
                 transcript: str = None, device: str = "cpu") -> list:
    """
    Full forced alignment pipeline.

    1. Get emissions.
    2. If no transcript supplied, use greedy decode.
    3. Run Viterbi alignment.
    4. Convert to (phone_str, start_s, end_s) using the model's frame shift
       (wav2vec2-base has a stride of 320 samples at 16 kHz → 20 ms/frame).

    Returns list of dicts:
      {phone, start_frame, end_frame, start_s, end_s}
    """
    FRAME_STRIDE = 320   # samples per frame for wav2vec2-base-960h at 16 kHz
    SAMPLE_RATE  = 16000

    emissions = get_emissions(processor, model, signal, device)
    blank_id  = processor.tokenizer.pad_token_id

    if transcript:
        # Tokenise transcript to ids
        tokens    = processor.tokenizer.tokenize(transcript.upper())
        token_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
    else:
        # Greedy decode to get approximate token sequence
        decoded   = greedy_decode(emissions, processor)
        token_ids = [processor.tokenizer.convert_tokens_to_ids([ch])[0]
                     for ch, _, _ in decoded]
        tokens    = [ch for ch, _, _ in decoded]

    if not token_ids:
        return []

    aligned = viterbi_align(emissions, token_ids, blank_id)

    results = []
    id2tok  = {processor.tokenizer.convert_tokens_to_ids([t])[0]: t
               for t in processor.tokenizer.get_vocab().keys()}

    for tok_id, sf, ef in aligned:
        phone   = id2tok.get(tok_id, f"<{tok_id}>")
        start_s = sf * FRAME_STRIDE / SAMPLE_RATE
        end_s   = ef * FRAME_STRIDE / SAMPLE_RATE
        results.append(dict(phone=phone, start_frame=sf, end_frame=ef,
                             start_s=start_s, end_s=end_s))

    return results


# ──────────────────────────────────────────────
# RMSE calculation
# ──────────────────────────────────────────────

def compute_rmse(manual_boundaries: list,
                 model_boundaries: list,
                 sample_rate: int,
                 match_radius_s: float = 0.1) -> dict:
    """
    Match each manual boundary to the nearest model boundary within
    match_radius_s and compute RMSE (in seconds).

    Parameters
    ----------
    manual_boundaries : output of boundary_times() → list of (start_s, end_s, label)
    model_boundaries  : output of forced_align()  → list of dicts with start_s, end_s
    match_radius_s    : max distance (s) to consider a match

    Returns
    -------
    dict with: rmse, matched_pairs, unmatched_manual, unmatched_model
    """
    # Collect all boundary positions (start & end) from each source
    manual_pts = []
    for s, e, lbl in manual_boundaries:
        if s > 0:
            manual_pts.append(s)
        if e < 1e9:
            manual_pts.append(e)
    manual_pts = sorted(set(manual_pts))

    model_pts = []
    for seg in model_boundaries:
        model_pts.append(seg["start_s"])
        model_pts.append(seg["end_s"])
    model_pts = sorted(set(model_pts))

    if not manual_pts or not model_pts:
        return dict(rmse=float("nan"), matched_pairs=[], errors=[],
                    unmatched_manual=manual_pts, unmatched_model=model_pts)

    errors          = []
    matched_pairs   = []
    used_model      = set()

    for mp in manual_pts:
        # Find nearest unused model boundary
        dists = [(abs(mp - op), i) for i, op in enumerate(model_pts)
                  if i not in used_model]
        if not dists:
            continue
        dist, best_i = min(dists, key=lambda x: x[0])
        if dist <= match_radius_s:
            errors.append(dist)
            matched_pairs.append((mp, model_pts[best_i]))
            used_model.add(best_i)

    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else float("nan")
    unmatched_manual = [mp for mp in manual_pts
                         if mp not in [p[0] for p in matched_pairs]]
    unmatched_model  = [op for i, op in enumerate(model_pts)
                         if i not in used_model]

    return dict(rmse=rmse,
                matched_pairs=matched_pairs,
                errors=errors,
                unmatched_manual=unmatched_manual,
                unmatched_model=unmatched_model)


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

def plot_alignment(signal: np.ndarray,
                   sample_rate: int,
                   manual_boundaries: list,
                   model_segments: list,
                   rmse_result: dict,
                   save_path: str = None):
    """
    Two-panel visualisation:
      Top:    Waveform with manual V/UV boundaries
      Bottom: Waveform with model phone boundaries
    """
    t = np.arange(len(signal)) / sample_rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ── Top: manual ──
    ax1.plot(t, signal, color="#555", linewidth=0.5, alpha=0.6)
    colors = {1: "#5c9fe0", 0: "#e05c5c"}
    for s_s, e_s, lbl in manual_boundaries:
        ax1.axvspan(s_s, e_s, alpha=0.2, color=colors[lbl])
        ax1.axvline(s_s, color=colors[lbl], linewidth=1.2, alpha=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Manual Cepstrum Boundaries (Voiced=Blue / Unvoiced=Red)")
    voiced_p   = mpatches.Patch(color="#5c9fe0", alpha=0.5, label="Voiced")
    unvoiced_p = mpatches.Patch(color="#e05c5c", alpha=0.5, label="Unvoiced")
    ax1.legend(handles=[voiced_p, unvoiced_p])
    ax1.grid(alpha=0.3)

    # ── Bottom: model ──
    ax2.plot(t, signal, color="#555", linewidth=0.5, alpha=0.6)
    cmap   = plt.cm.tab20
    phones = sorted(set(s["phone"] for s in model_segments))
    p2c    = {p: cmap(i / max(len(phones), 1)) for i, p in enumerate(phones)}

    for seg in model_segments:
        ax2.axvspan(seg["start_s"], seg["end_s"],
                    alpha=0.3, color=p2c[seg["phone"]])
        ax2.axvline(seg["start_s"], color=p2c[seg["phone"]],
                    linewidth=1.0, alpha=0.9)
        mid = (seg["start_s"] + seg["end_s"]) / 2
        ax2.text(mid, 0, seg["phone"], ha="center", va="center",
                 fontsize=7, color="black", rotation=90)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    rmse_str = (f"{rmse_result['rmse']*1000:.1f} ms"
                if not np.isnan(rmse_result["rmse"]) else "N/A")
    ax2.set_title(f"Wav2Vec2 Forced Alignment  |  RMSE vs manual = {rmse_str}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def print_rmse_table(rmse_result: dict):
    print("\n" + "=" * 55)
    print("RMSE between manual and model phone boundaries")
    print("=" * 55)
    print(f"  Matched pairs  : {len(rmse_result['matched_pairs'])}")
    print(f"  RMSE           : {rmse_result['rmse']*1000:.2f} ms"
          if not np.isnan(rmse_result["rmse"])
          else "  RMSE           : N/A (no matches)")
    print(f"  Unmatched manual : {len(rmse_result['unmatched_manual'])}")
    print(f"  Unmatched model  : {len(rmse_result['unmatched_model'])}")
    print("=" * 55)
    if rmse_result["errors"]:
        errs_ms = [e * 1000 for e in rmse_result["errors"]]
        print(f"  Individual errors (ms): "
              f"min={min(errs_ms):.1f} "
              f"max={max(errs_ms):.1f} "
              f"mean={np.mean(errs_ms):.1f}")
    print()


# ──────────────────────────────────────────────
# Synthesis fallback
# ──────────────────────────────────────────────

def _synthesise(sr=16000, duration=3.0):
    t   = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    sig = (0.5 * np.sin(2*np.pi*120*t)
           + 0.3 * np.sin(2*np.pi*240*t)
           + 0.1 * np.random.randn(len(t)).astype(np.float32))
    sig = sig / np.max(np.abs(sig))
    return sig, sr


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wav2Vec2 forced alignment + RMSE vs manual boundaries")
    parser.add_argument("--audio",      default=None,
                        help="Path to 16-kHz WAV file")
    parser.add_argument("--transcript", default=None,
                        help="Optional transcript for forced alignment")
    parser.add_argument("--outdir",     default="outputs")
    parser.add_argument("--device",     default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load audio ──
    if args.audio and os.path.isfile(args.audio):
        signal_f32, sr = load_audio_16k(args.audio)
        signal_int16   = (signal_f32 * 32767).astype(np.int16)
    else:
        print("No audio supplied — using synthesised signal.")
        signal_f32, sr = _synthesise()
        signal_int16   = (signal_f32 * 32767).astype(np.int16)

    print(f"Audio: {len(signal_f32)/sr:.2f}s @ {sr} Hz")

    # ── Step 1: Manual boundaries ──
    manual_result = detect_boundaries(signal_int16, sr)
    manual_bt     = boundary_times(manual_result["boundaries"], sr)
    print(f"\nManual boundary count: {len(manual_bt)}")

    # ── Step 2: Load HuggingFace model ──
    processor, model = load_model(MODEL_NAME, args.device)

    # ── Step 3: Forced alignment ──
    print("Running forced alignment …")
    model_segs = forced_align(processor, model, signal_f32,
                               transcript=args.transcript,
                               device=args.device)
    print(f"Phone segments found: {len(model_segs)}")
    for seg in model_segs[:10]:
        print(f"  [{seg['phone']:>4}]  {seg['start_s']:.3f}s – {seg['end_s']:.3f}s")
    if len(model_segs) > 10:
        print(f"  … ({len(model_segs) - 10} more)")

    # ── Step 4: RMSE ──
    rmse_res = compute_rmse(manual_bt, model_segs, sr, match_radius_s=0.15)
    print_rmse_table(rmse_res)

    # ── Step 5: Plot ──
    plot_alignment(
        signal_f32 * 32767, sr,
        manual_bt, model_segs, rmse_res,
        save_path=os.path.join(args.outdir, "phonetic_alignment.png")
    )
    print("Done.")
