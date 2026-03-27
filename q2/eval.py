"""
eval.py — Speaker Verification Evaluation
==========================================
Computes EER and minDCF from a trial list.
Generates:
  - results/tables/results.csv
  - results/plots/det_curve.png
  - results/plots/score_dist.png
  - results/plots/tsne_embeddings.png

Usage:
  python eval.py --config configs/disentangler.yaml \
                 --checkpoint exps/disentangler/ckpt_epoch050.pt \
                 --trial_list data/trials.txt \
                 --enroll_dir data/enroll \
                 --test_dir data/test
"""

import argparse, yaml, csv, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# Import model classes from train.py
from train import SpeakerEncoder, Disentangler, MelExtractor


# ─────────────────────────────────────────────
# EER and minDCF
# ─────────────────────────────────────────────
def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """Compute Equal Error Rate."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer * 100), float(threshold)


def compute_min_dcf(scores: np.ndarray, labels: np.ndarray,
                    p_target: float = 0.01, c_miss: float = 1.0,
                    c_fa: float = 1.0):
    """Compute minimum Detection Cost Function (NIST SRE style)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf = float(np.min(dcf))
    norm_factor = min(c_miss * p_target, c_fa * (1 - p_target))
    return min_dcf / norm_factor


# ─────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────
def extract_embedding(wav_path: str, encoder: SpeakerEncoder,
                      disentangler: Disentangler,
                      mel_extractor: MelExtractor,
                      device: torch.device,
                      sr: int = 16000,
                      seg_sec: float = 3.0) -> torch.Tensor:
    wav, file_sr = torchaudio.load(wav_path)
    wav = wav.mean(0)
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    seg_len = int(seg_sec * sr)
    # Centre crop for evaluation (deterministic)
    L = wav.size(0)
    if L >= seg_len:
        start = (L - seg_len) // 2
        wav = wav[start:start + seg_len]
    else:
        wav = F.pad(wav, (0, seg_len - L))
    wav = wav.unsqueeze(0).to(device)          # (1, T)
    with torch.no_grad():
        feats = mel_extractor(wav)             # (1, frames, n_mels)
        e = encoder(feats)                     # (1, D)
        if disentangler is not None:
            z_spk, _, _ = disentangler(e)
            emb = F.normalize(z_spk, dim=-1)
        else:
            emb = e
    return emb.squeeze(0).cpu()


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
def plot_det(scores, labels, save_path, label="System"):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    plt.figure(figsize=(6, 6))
    plt.plot(fpr * 100, fnr * 100, lw=2, label=label)
    plt.xlabel("False Acceptance Rate (%)")
    plt.ylabel("False Rejection Rate (%)")
    plt.title("DET Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_score_distribution(target_scores, nontarget_scores, save_path):
    plt.figure(figsize=(8, 4))
    plt.hist(nontarget_scores, bins=80, density=True, alpha=0.6,
             color="steelblue", label="Non-target")
    plt.hist(target_scores, bins=80, density=True, alpha=0.6,
             color="tomato", label="Target")
    plt.xlabel("Cosine similarity score")
    plt.ylabel("Density")
    plt.title("Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_multi_det(systems: dict, save_path):
    """systems = {name: (scores, labels)}"""
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(7, 7))
    for name, (scores, labels) in systems.items():
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        plt.plot(fpr * 100, fnr * 100, lw=2, label=name)
    plt.xlabel("False Acceptance Rate (%)")
    plt.ylabel("False Rejection Rate (%)")
    plt.title("DET Curve — All Systems")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray,
              save_path: str, n_speakers: int = 20):
    """t-SNE of first n_speakers speakers' embeddings."""
    unique = np.unique(labels)
    if len(unique) > n_speakers:
        unique = unique[:n_speakers]
    mask = np.isin(labels, unique)
    emb_sub = embeddings[mask]
    lab_sub = labels[mask]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(emb_sub)
    plt.figure(figsize=(9, 7))
    for i, spk in enumerate(unique):
        idx = lab_sub == spk
        plt.scatter(proj[idx, 0], proj[idx, 1], s=15, label=f"Spk {spk}", alpha=0.7)
    plt.title(f"t-SNE of Speaker Embeddings (first {n_speakers} speakers)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Trial-based evaluation
# ─────────────────────────────────────────────
def run_evaluation(cfg, checkpoint_path, trial_list,
                   wav_dir, device, use_disentangler=True):
    """
    Evaluate on a trial list file.
    Trial list format (one per line):
      <label> <enroll_wav> <test_wav>
      label: 1=same speaker, 0=different speaker
    """
    # Load models
    encoder = SpeakerEncoder(n_mels=cfg["n_mels"], emb_dim=cfg["emb_dim"]).to(device)
    disentangler = None
    if use_disentangler:
        disentangler = Disentangler(
            emb_dim=cfg["emb_dim"], spk_ratio=cfg["spk_ratio"]).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    if use_disentangler and "disentangler" in ckpt:
        disentangler.load_state_dict(ckpt["disentangler"])

    encoder.eval()
    if disentangler:
        disentangler.eval()

    mel_extractor = MelExtractor(sr=cfg["sr"], n_mels=cfg["n_mels"]).to(device)
    mel_extractor.eval()

    # Cache embeddings
    emb_cache = {}
    def get_emb(path):
        if path not in emb_cache:
            emb_cache[path] = extract_embedding(
                str(Path(wav_dir) / path), encoder, disentangler,
                mel_extractor, device, sr=cfg["sr"], seg_sec=cfg["seg_sec"]
            )
        return emb_cache[path]

    scores, labels = [], []
    with open(trial_list) as f:
        for line in tqdm(f, desc="Scoring trials"):
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label, enroll_path, test_path = int(parts[0]), parts[1], parts[2]
            e_enroll = get_emb(enroll_path)
            e_test   = get_emb(test_path)
            score = F.cosine_similarity(e_enroll.unsqueeze(0),
                                        e_test.unsqueeze(0)).item()
            scores.append(score)
            labels.append(label)

    scores  = np.array(scores)
    labels  = np.array(labels)
    eer, _  = compute_eer(scores, labels)
    minDCF  = compute_min_dcf(scores, labels)
    return eer, minDCF, scores, labels


# ─────────────────────────────────────────────
# Generate synthetic evaluation data
# (for when no real trial list is available)
# ─────────────────────────────────────────────
def synthetic_eval(cfg, checkpoint_path, device, use_disentangler=True,
                   n_speakers=50, utts_per_speaker=10):
    """
    Synthetic evaluation using LibriSpeech test-clean.
    Constructs genuine/impostor trial pairs on-the-fly.
    """
    from train import BaselineSpeakerDataset

    logging.info("Running synthetic evaluation on LibriSpeech test-clean...")
    raw = torchaudio.datasets.LIBRISPEECH(
        cfg["data_root"], url="test-clean", download=True)

    encoder = SpeakerEncoder(n_mels=cfg["n_mels"], emb_dim=cfg["emb_dim"]).to(device)
    disentangler = None
    if use_disentangler:
        disentangler = Disentangler(
            emb_dim=cfg["emb_dim"], spk_ratio=cfg["spk_ratio"]).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    if use_disentangler and "disentangler" in ckpt:
        disentangler.load_state_dict(ckpt["disentangler"])

    encoder.eval()
    if disentangler:
        disentangler.eval()

    mel_extractor = MelExtractor(sr=cfg["sr"], n_mels=cfg["n_mels"]).to(device)

    # Collect utterances per speaker
    by_speaker = defaultdict(list)
    for idx in range(len(raw)):
        _, _, _, spk_id, _, _ = raw[idx]
        by_speaker[spk_id].append(idx)

    speakers = [s for s, idxs in by_speaker.items() if len(idxs) >= 2][:n_speakers]
    seg_len = int(cfg["seg_sec"] * cfg["sr"])

    def load_wav(idx):
        wav, sr, *_ = raw[idx]
        wav = wav.mean(0)
        if sr != cfg["sr"]:
            wav = torchaudio.functional.resample(wav, sr, cfg["sr"])
        L = wav.size(0)
        if L >= seg_len:
            start = (L - seg_len) // 2
            wav = wav[start:start + seg_len]
        else:
            wav = F.pad(wav, (0, seg_len - L))
        return wav

    def extract(wav):
        wav = wav.unsqueeze(0).to(device)
        with torch.no_grad():
            feats = mel_extractor(wav)
            e = encoder(feats)
            if disentangler:
                z_spk, _, _ = disentangler(e)
                return F.normalize(z_spk, dim=-1).squeeze(0).cpu()
            return e.squeeze(0).cpu()

    # Extract embeddings
    spk_embeddings = {}
    for spk in tqdm(speakers, desc="Extracting embeddings"):
        idxs = by_speaker[spk][:utts_per_speaker]
        spk_embeddings[spk] = [extract(load_wav(i)) for i in idxs]

    # Build trials
    import random as pyrandom
    scores, labels, all_embs, all_labels = [], [], [], []
    spk_list = list(spk_embeddings.keys())

    for i, spk in enumerate(spk_list):
        embs = spk_embeddings[spk]
        all_embs.extend(embs)
        all_labels.extend([i] * len(embs))
        # Genuine pairs
        for j in range(len(embs)):
            for k in range(j + 1, len(embs)):
                score = F.cosine_similarity(
                    embs[j].unsqueeze(0), embs[k].unsqueeze(0)).item()
                scores.append(score); labels.append(1)
        # Impostor pairs (random)
        n_impostors = min(len(embs) * 3, 20)
        for _ in range(n_impostors):
            other_spk = pyrandom.choice([s for s in spk_list if s != spk])
            other_emb = pyrandom.choice(spk_embeddings[other_spk])
            my_emb = pyrandom.choice(embs)
            score = F.cosine_similarity(
                my_emb.unsqueeze(0), other_emb.unsqueeze(0)).item()
            scores.append(score); labels.append(0)

    scores  = np.array(scores)
    labels  = np.array(labels)
    all_embs = np.stack([e.numpy() for e in all_embs])
    all_labels = np.array(all_labels)

    eer, threshold = compute_eer(scores, labels)
    minDCF = compute_min_dcf(scores, labels)

    return eer, minDCF, scores, labels, all_embs, all_labels


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True)
    parser.add_argument("--checkpoints",  nargs="+",
                        help="List of checkpoint paths to compare")
    parser.add_argument("--names",        nargs="+",
                        help="Names for each checkpoint (same order)")
    parser.add_argument("--trial_list",   default=None)
    parser.add_argument("--wav_dir",      default=None)
    parser.add_argument("--results_dir",  default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)

    checkpoints = args.checkpoints or []
    names       = args.names or [Path(c).parent.name for c in checkpoints]

    all_systems = {}
    rows = [["System", "EER (%)", "minDCF (0.01)"]]

    for ckpt_path, name in zip(checkpoints, names):
        use_dis = name != "baseline"
        logging.info(f"Evaluating {name} from {ckpt_path}...")

        if args.trial_list and args.wav_dir:
            eer, minDCF, scores, labels = run_evaluation(
                cfg, ckpt_path, args.trial_list,
                args.wav_dir, device, use_dis)
            embs, emb_labels = None, None
        else:
            eer, minDCF, scores, labels, embs, emb_labels = synthetic_eval(
                cfg, ckpt_path, device, use_disentangler=use_dis)

        all_systems[name] = (scores, labels)
        rows.append([name, f"{eer:.2f}", f"{minDCF:.4f}"])
        logging.info(f"  {name}: EER={eer:.2f}%  minDCF={minDCF:.4f}")

        # Per-system score distribution
        target    = scores[labels == 1]
        nontarget = scores[labels == 0]
        plot_score_distribution(
            target, nontarget,
            results_dir / "plots" / f"score_dist_{name}.png"
        )

        # t-SNE (only if embeddings available)
        if embs is not None:
            plot_tsne(embs, emb_labels,
                      results_dir / "plots" / f"tsne_{name}.png")

    # Multi-system DET curve
    if all_systems:
        plot_multi_det(all_systems,
                       results_dir / "plots" / "det_all_systems.png")

    # Write CSV
    csv_path = results_dir / "tables" / "results.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    logging.info(f"\nResults saved to {csv_path}")
    # Pretty print
    col_w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    print("\n" + "─" * (sum(col_w) + 7))
    for row in rows:
        print("  ".join(r.ljust(w) for r, w in zip(row, col_w)))
    print("─" * (sum(col_w) + 7))


if __name__ == "__main__":
    main()