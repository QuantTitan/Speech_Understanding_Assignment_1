"""
train.py — Speaker Disentanglement Training
==========================================
Implements three modes via --mode flag:
  baseline     : TDNN encoder + AAM-Softmax only
  disentangler : + Disentangler (paper: Nam et al. Interspeech 2024)
  improved     : + InfoNCE contrastive loss on spk subspace

Dataset: LibriSpeech train-clean-100 (auto-downloaded via torchaudio)
         MUSAN noise (download separately, see README)

Usage:
  python train.py --config configs/disentangler.yaml
"""

import os, sys, argparse, yaml, random, logging, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# 2. Gradient Reversal Layer
# ─────────────────────────────────────────────
class GradReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReversalFunction.apply(x, alpha)


# ─────────────────────────────────────────────
# 3. Model Components
# ─────────────────────────────────────────────
class TDNNLayer(nn.Module):
    """Time-Delay Neural Network layer."""
    def __init__(self, in_dim: int, out_dim: int, context_size: int, dilation: int = 1):
        super().__init__()
        padding = (context_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_dim, out_dim, context_size,
                              dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):  # x: (B, C, T)
        return F.relu(self.bn(self.conv(x)))


class SpeakerEncoder(nn.Module):
    """
    Lightweight x-vector style TDNN encoder.
    Input : (B, T, n_mels) mel-spectrogram frames
    Output: (B, emb_dim) L2-normalised embedding
    """
    def __init__(self, n_mels: int = 80, emb_dim: int = 256):
        super().__init__()
        self.frame_net = nn.Sequential(
            TDNNLayer(n_mels, 512, 5, dilation=1),
            TDNNLayer(512, 512, 3, dilation=2),
            TDNNLayer(512, 512, 3, dilation=3),
            TDNNLayer(512, 512, 1),
            TDNNLayer(512, 1500, 1),
        )
        # Stats pooling → 3000-dim
        self.fc1 = nn.Linear(3000, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, emb_dim)

    def forward(self, x):  # x: (B, T, F)
        h = self.frame_net(x.transpose(1, 2))      # (B, 1500, T)
        mean = h.mean(dim=-1)
        std  = h.std(dim=-1).clamp(min=1e-9)
        h = torch.cat([mean, std], dim=1)           # (B, 3000)
        h = F.relu(self.bn1(self.fc1(h)))
        h = self.fc2(h)
        return F.normalize(h, p=2, dim=-1)


class Disentangler(nn.Module):
    """
    Auto-encoder disentangler (Section 3.2).
    Splits bottleneck z into [z_spk | z_env].
    """
    def __init__(self, emb_dim: int = 256, spk_ratio: float = 0.5):
        super().__init__()
        self.spk_dim = int(emb_dim * spk_ratio)
        self.env_dim = emb_dim - self.spk_dim

        # Encoder: e → z
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        # Decoder: z → ê
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def encode(self, e):
        z = self.encoder(e)
        return z[:, :self.spk_dim], z[:, self.spk_dim:]  # z_spk, z_env

    def decode(self, z_spk, z_env):
        z = torch.cat([z_spk, z_env], dim=-1)
        return self.decoder(z)

    def forward(self, e):
        z_spk, z_env = self.encode(e)
        e_hat = self.decode(z_spk, z_env)
        return z_spk, z_env, e_hat


class EnvironmentDiscriminator(nn.Module):
    """Classifies environment from z_env (or tries to from z_spk via GRL)."""
    def __init__(self, in_dim: int, num_env_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_env_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 4. Loss Functions
# ─────────────────────────────────────────────
class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax."""
    def __init__(self, emb_dim: int, num_classes: int,
                 margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.m = margin
        self.s = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        cosine = F.linear(F.normalize(x, dim=-1),
                          F.normalize(self.weight, dim=-1))
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        cos_m = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.s * (one_hot * cos_m + (1 - one_hot) * cosine)
        return self.ce(logits, labels)


class InfoNCELoss(nn.Module):
    """
    NT-Xent / InfoNCE contrastive loss.
    Positive pairs: (z_spk from env_1, z_spk from env_2) of same speaker.
    Proposed improvement over paper's pure AAM-Softmax on z_spk.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z_a, z_p):
        """
        z_a: (B, d) anchor embeddings  (x_{i,1} speaker codes)
        z_p: (B, d) positive embeddings (x_{i,3} speaker codes, same spk, diff env)
        """
        z_a = F.normalize(z_a, dim=-1)
        z_p = F.normalize(z_p, dim=-1)
        B = z_a.size(0)
        # Build full 2B×2B similarity matrix (SimCLR style)
        z_all = torch.cat([z_a, z_p], dim=0)          # (2B, d)
        sim = torch.mm(z_all, z_all.t()) / self.tau   # (2B, 2B)
        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z_a.device).bool()
        sim.masked_fill_(mask, -1e9)
        # Labels: for index i, positive is at i+B (and vice versa)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z_a.device),
            torch.arange(0, B, device=z_a.device),
        ])
        return F.cross_entropy(sim, labels)


def correlation_loss(z_spk, z_env):
    """Penalise linear correlation between spk and env subspaces (L_corr)."""
    z_spk = z_spk - z_spk.mean(0, keepdim=True)
    z_env = z_env - z_env.mean(0, keepdim=True)
    # Frobenius norm of cross-covariance matrix
    cov = torch.mm(z_spk.t(), z_env) / (z_spk.size(0) - 1)
    return (cov ** 2).sum()


# ─────────────────────────────────────────────
# 5. Dataset
# ─────────────────────────────────────────────
class MelExtractor(nn.Module):
    """Converts raw waveform to log-mel spectrogram on-the-fly."""
    def __init__(self, sr=16000, n_mels=80, n_fft=512, hop=160, win=400):
        super().__init__()
        self.mel = T.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop,
                                    win_length=win, n_mels=n_mels,
                                    f_min=20, f_max=7600)
        self.amp2db = T.AmplitudeToDB()

    @torch.no_grad()
    def forward(self, wav):  # wav: (B, T) or (T,)
        m = self.amp2db(self.mel(wav))
        return m.transpose(-1, -2)  # (B, frames, n_mels)


class NoiseAugment:
    """Additive noise augmentation from a noise file directory."""
    def __init__(self, noise_dir: str = None, snr_range=(5, 20)):
        self.noise_files = []
        if noise_dir and Path(noise_dir).exists():
            self.noise_files = list(Path(noise_dir).rglob("*.wav"))
        self.snr_range = snr_range

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.noise_files:
            # Gaussian fallback when no noise dir provided
            snr = random.uniform(*self.snr_range)
            noise = torch.randn_like(wav)
            signal_rms = wav.pow(2).mean().sqrt().clamp(1e-9)
            noise_rms  = noise.pow(2).mean().sqrt().clamp(1e-9)
            scale = signal_rms / (noise_rms * 10 ** (snr / 20))
            return (wav + scale * noise).clamp(-1, 1)
        noise_path = random.choice(self.noise_files)
        noise_wav, sr = torchaudio.load(noise_path)
        noise_wav = noise_wav.mean(0)
        # Crop/repeat noise to match signal length
        L = wav.size(-1)
        if noise_wav.size(-1) < L:
            reps = L // noise_wav.size(-1) + 1
            noise_wav = noise_wav.repeat(reps)
        start = random.randint(0, noise_wav.size(-1) - L)
        noise_wav = noise_wav[start:start + L]
        snr = random.uniform(*self.snr_range)
        signal_rms = wav.pow(2).mean().sqrt().clamp(1e-9)
        noise_rms  = noise_wav.pow(2).mean().sqrt().clamp(1e-9)
        scale = signal_rms / (noise_rms * 10 ** (snr / 20))
        return (wav + scale * noise_wav).clamp(-1, 1)


class TripletSpeakerDataset(Dataset):
    """
    Triplet dataset for disentanglement training.

    Each item returns three waveforms from the same speaker:
      x1, x2 — same "session" (same chapter in LibriSpeech), same augmentation
      x3     — different session (different chapter), different augmentation

    This approximates the VoxCeleb same-video / different-video split using
    LibriSpeech chapter IDs as session proxies.
    """
    def __init__(self, root: str, split: str = "train-clean-100",
                 seg_sec: float = 3.0, sr: int = 16000,
                 noise_dir: str = None, min_utts_per_session: int = 2):
        self.sr = sr
        self.seg_len = int(seg_sec * sr)
        self.noise_aug = NoiseAugment(noise_dir)

        # Download / use cached LibriSpeech
        logging.info(f"Loading LibriSpeech {split} from {root}...")
        raw = torchaudio.datasets.LIBRISPEECH(root, url=split, download=True)

        # Organise: speaker_id → chapter_id → [file_indices]
        by_spk_chapter = defaultdict(lambda: defaultdict(list))
        for idx in range(len(raw)):
            _, _, _, spk_id, chapter_id, _ = raw[idx]
            by_spk_chapter[spk_id][chapter_id].append(idx)

        # Keep only speakers with ≥2 chapters and each chapter with ≥2 utts
        self.triplets = []          # list of (idx1, idx2, idx3, spk_label)
        self.spk_to_label = {}
        label_counter = 0
        self.raw = raw

        for spk_id, chapters in by_spk_chapter.items():
            valid_chapters = {c: idxs for c, idxs in chapters.items()
                              if len(idxs) >= min_utts_per_session}
            if len(valid_chapters) < 2:
                continue
            if spk_id not in self.spk_to_label:
                self.spk_to_label[spk_id] = label_counter
                label_counter += 1
            chapter_list = list(valid_chapters.keys())
            for _ in range(10):   # 10 triplets per speaker
                c_same = random.choice(chapter_list)
                c_diff = random.choice([c for c in chapter_list if c != c_same])
                i1, i2 = random.sample(valid_chapters[c_same], 2)
                i3 = random.choice(valid_chapters[c_diff])
                self.triplets.append((i1, i2, i3, self.spk_to_label[spk_id]))

        self.num_speakers = label_counter
        logging.info(f"Dataset: {len(self.triplets)} triplets, "
                     f"{self.num_speakers} speakers")

    def _load_segment(self, idx: int) -> torch.Tensor:
        wav, sr, *_ = self.raw[idx]
        wav = wav.mean(0)                   # mono
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        # Random crop / pad to fixed length
        L = wav.size(0)
        if L >= self.seg_len:
            start = random.randint(0, L - self.seg_len)
            wav = wav[start:start + self.seg_len]
        else:
            wav = F.pad(wav, (0, self.seg_len - L))
        return wav

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        i1, i2, i3, label = self.triplets[idx]
        w1 = self._load_segment(i1)
        w2 = self._load_segment(i2)
        w3 = self._load_segment(i3)

        # Same augmentation type for w1 and w2 (same env)
        # Different augmentation for w3 (different env)
        aug_type_12 = random.choice(["gaussian", "noise"])
        aug_type_3  = random.choice(["gaussian", "noise"])
        while aug_type_3 == aug_type_12:
            aug_type_3 = random.choice(["gaussian", "noise"])

        w1 = self.noise_aug(w1)
        w2 = self.noise_aug(w2)
        # For w3 apply slightly different SNR range
        old_range = self.noise_aug.snr_range
        self.noise_aug.snr_range = (0, 10)     # harsher noise → more distinct env
        w3 = self.noise_aug(w3)
        self.noise_aug.snr_range = old_range

        return w1, w2, w3, torch.tensor(label, dtype=torch.long)


class BaselineSpeakerDataset(Dataset):
    """
    Simple single-utterance dataset for baseline training.
    No triplets needed — just (waveform, speaker_label).
    """
    def __init__(self, root: str, split: str = "train-clean-100",
                 seg_sec: float = 3.0, sr: int = 16000, noise_dir: str = None):
        self.sr = sr
        self.seg_len = int(seg_sec * sr)
        self.noise_aug = NoiseAugment(noise_dir)
        raw = torchaudio.datasets.LIBRISPEECH(root, url=split, download=True)
        spk_to_label = {}
        label_counter = 0
        self.items = []
        self.raw = raw
        for idx in range(len(raw)):
            _, _, _, spk_id, _, _ = raw[idx]
            if spk_id not in spk_to_label:
                spk_to_label[spk_id] = label_counter
                label_counter += 1
            self.items.append((idx, spk_to_label[spk_id]))
        self.num_speakers = label_counter
        logging.info(f"Baseline dataset: {len(self.items)} utts, "
                     f"{self.num_speakers} speakers")

    def _load_segment(self, idx):
        wav, sr, *_ = self.raw[idx]
        wav = wav.mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        L = wav.size(0)
        if L >= self.seg_len:
            start = random.randint(0, L - self.seg_len)
            wav = wav[start:start + self.seg_len]
        else:
            wav = F.pad(wav, (0, self.seg_len - L))
        return wav

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        raw_idx, label = self.items[idx]
        wav = self._load_segment(raw_idx)
        wav = self.noise_aug(wav)
        return wav, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────
# 6. Training Loops
# ─────────────────────────────────────────────
def train_baseline(cfg, device):
    """Train TDNN encoder + AAM-Softmax (no disentangler)."""
    mel_extractor = MelExtractor(sr=cfg["sr"], n_mels=cfg["n_mels"]).to(device)

    dataset = BaselineSpeakerDataset(
        root=cfg["data_root"], split=cfg["train_split"],
        seg_sec=cfg["seg_sec"], sr=cfg["sr"],
        noise_dir=cfg.get("noise_dir", None),
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=cfg["num_workers"],
                        pin_memory=True, drop_last=True)

    encoder = SpeakerEncoder(n_mels=cfg["n_mels"], emb_dim=cfg["emb_dim"]).to(device)
    classifier = AAMSoftmax(cfg["emb_dim"], dataset.num_speakers,
                            cfg["aam_margin"], cfg["aam_scale"]).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    save_dir = Path(cfg["save_dir"]) / "baseline"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        encoder.train(); classifier.train()
        total_loss = 0.0
        for wav, labels in tqdm(loader, desc=f"Baseline Epoch {epoch}"):
            wav, labels = wav.to(device), labels.to(device)
            with torch.no_grad():
                feats = mel_extractor(wav)          # (B, T, F)
            emb = encoder(feats)
            loss = classifier(emb, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()), 5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        logging.info(f"[Baseline] Epoch {epoch:3d} | Loss {avg:.4f}")

        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_dir / f"ckpt_epoch{epoch:03d}.pt")
            logging.info(f"  → Saved checkpoint to {save_dir}/ckpt_epoch{epoch:03d}.pt")


def train_disentangler(cfg, device, mode="disentangler"):
    """
    Train full disentanglement pipeline.
    mode='disentangler' : paper's method (AAM-Softmax on z_spk)
    mode='improved'     : replace L_spk with InfoNCE contrastive loss
    """
    mel_extractor = MelExtractor(sr=cfg["sr"], n_mels=cfg["n_mels"]).to(device)

    dataset = TripletSpeakerDataset(
        root=cfg["data_root"], split=cfg["train_split"],
        seg_sec=cfg["seg_sec"], sr=cfg["sr"],
        noise_dir=cfg.get("noise_dir", None),
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=cfg["num_workers"],
                        pin_memory=True, drop_last=True)

    # Models
    encoder    = SpeakerEncoder(n_mels=cfg["n_mels"], emb_dim=cfg["emb_dim"]).to(device)
    disentangler = Disentangler(emb_dim=cfg["emb_dim"],
                                spk_ratio=cfg["spk_ratio"]).to(device)
    spk_dim = disentangler.spk_dim
    env_dim = disentangler.env_dim

    # Discriminators
    env_disc_on_env = EnvironmentDiscriminator(env_dim, cfg["num_env_classes"]).to(device)
    env_disc_on_spk = EnvironmentDiscriminator(spk_dim, cfg["num_env_classes"]).to(device)

    # Losses
    spk_classifier = AAMSoftmax(spk_dim, dataset.num_speakers,
                                cfg["aam_margin"], cfg["aam_scale"]).to(device)
    infonce = InfoNCELoss(temperature=cfg.get("infonce_temp", 0.07))

    all_params = (list(encoder.parameters()) +
                  list(disentangler.parameters()) +
                  list(spk_classifier.parameters()) +
                  list(env_disc_on_env.parameters()) +
                  list(env_disc_on_spk.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    # Loss weights
    w_recons   = cfg.get("w_recons",   1.0)
    w_spk      = cfg.get("w_spk",      1.0)
    w_env_env  = cfg.get("w_env_env",  0.5)
    w_env_spk  = cfg.get("w_env_spk",  0.5)
    w_corr     = cfg.get("w_corr",     0.1)
    w_infonce  = cfg.get("w_infonce",  1.0)

    # Environment labels: define 2 classes (same-env=0, diff-env=1) as proxy
    # In each triplet: x1,x2 → env class 0; x3 → env class 1
    env_labels_same = torch.zeros(cfg["batch_size"], dtype=torch.long, device=device)
    env_labels_diff = torch.ones(cfg["batch_size"], dtype=torch.long, device=device)

    save_dir = Path(cfg["save_dir"]) / mode
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        encoder.train(); disentangler.train()
        env_disc_on_env.train(); env_disc_on_spk.train()
        spk_classifier.train()

        metrics = defaultdict(float)

        for w1, w2, w3, spk_labels in tqdm(loader, desc=f"[{mode}] Epoch {epoch}"):
            w1 = w1.to(device); w2 = w2.to(device)
            w3 = w3.to(device); spk_labels = spk_labels.to(device)

            with torch.no_grad():
                f1 = mel_extractor(w1)
                f2 = mel_extractor(w2)
                f3 = mel_extractor(w3)

            # --- Extract base embeddings ---
            e1 = encoder(f1)   # (B, D) — same env as e2
            e2 = encoder(f2)   # (B, D)
            e3 = encoder(f3)   # (B, D) — different env

            # --- Disentangle ---
            z_spk1, z_env1, e1_hat = disentangler(e1)
            z_spk2, z_env2, e2_hat = disentangler(e2)
            z_spk3, z_env3, e3_hat = disentangler(e3)

            # ── L_recons (Eq.1) ──────────────────────────
            L_recons = (
                F.l1_loss(e1_hat, e1) +
                F.l1_loss(e2_hat, e2) +
                F.l1_loss(e3_hat, e3)
            )

            # ── Code swapping (Section 3.3) ──────────────
            # Swap z_spk of x2 and x3, re-decode, and add another recon term
            e2_swap_hat = disentangler.decode(z_spk3.detach(), z_env2)
            e3_swap_hat = disentangler.decode(z_spk2.detach(), z_env3)
            L_recons_swap = (
                F.l1_loss(e2_swap_hat, e2) +
                F.l1_loss(e3_swap_hat, e3)
            )
            L_recons = L_recons + L_recons_swap

            # ── L_env_env (env discriminator on z_env) ───
            env_logits1 = env_disc_on_env(z_env1)
            env_logits2 = env_disc_on_env(z_env2)
            env_logits3 = env_disc_on_env(z_env3)
            L_env_env = (
                F.cross_entropy(env_logits1, env_labels_same[:w1.size(0)]) +
                F.cross_entropy(env_logits2, env_labels_same[:w1.size(0)]) +
                F.cross_entropy(env_logits3, env_labels_diff[:w1.size(0)])
            ) / 3.0

            # ── L_env_spk (adversarial via GRL on z_spk) ─
            z_spk1_rev = grad_reverse(z_spk1, alpha=cfg.get("grl_alpha", 1.0))
            z_spk3_rev = grad_reverse(z_spk3, alpha=cfg.get("grl_alpha", 1.0))
            adv_logits1 = env_disc_on_spk(z_spk1_rev)
            adv_logits3 = env_disc_on_spk(z_spk3_rev)
            L_env_spk = (
                F.cross_entropy(adv_logits1, env_labels_same[:w1.size(0)]) +
                F.cross_entropy(adv_logits3, env_labels_diff[:w1.size(0)])
            ) / 2.0

            # ── L_corr (correlation penalty) ─────────────
            L_corr = (
                correlation_loss(z_spk1, z_env1) +
                correlation_loss(z_spk3, z_env3)
            ) / 2.0

            # ── L_spk (speaker classification or InfoNCE) ─
            if mode == "improved":
                # InfoNCE: positive pair = (z_spk1, z_spk3) same speaker, diff env
                L_spk = infonce(z_spk1, z_spk3)
            else:
                # AAM-Softmax classification (paper's original)
                L_spk = (
                    spk_classifier(z_spk1, spk_labels) +
                    spk_classifier(z_spk3, spk_labels)
                ) / 2.0

            # ── Total loss ───────────────────────────────
            loss = (w_recons  * L_recons  +
                    w_spk     * L_spk     +
                    w_env_env * L_env_env +
                    w_env_spk * L_env_spk +
                    w_corr    * L_corr)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()

            metrics["total"]    += loss.item()
            metrics["recons"]   += L_recons.item()
            metrics["spk"]      += L_spk.item()
            metrics["env_env"]  += L_env_env.item()
            metrics["env_spk"]  += L_env_spk.item()
            metrics["corr"]     += L_corr.item()

        scheduler.step()
        n = len(loader)
        logging.info(
            f"[{mode}] Epoch {epoch:3d} | "
            f"Total {metrics['total']/n:.4f} | "
            f"Recons {metrics['recons']/n:.4f} | "
            f"Spk {metrics['spk']/n:.4f} | "
            f"Env_Env {metrics['env_env']/n:.4f} | "
            f"Env_Spk {metrics['env_spk']/n:.4f} | "
            f"Corr {metrics['corr']/n:.4f}"
        )

        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            ckpt = {
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "disentangler": disentangler.state_dict(),
                "spk_classifier": spk_classifier.state_dict(),
                "env_disc_env": env_disc_on_env.state_dict(),
                "env_disc_spk": env_disc_on_spk.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            ckpt_path = save_dir / f"ckpt_epoch{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            logging.info(f"  → Saved {ckpt_path}")


# ─────────────────────────────────────────────
# 7. Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(cfg["save_dir"]) / f"{cfg['mode']}_train.log"
            )
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device} | Mode: {cfg['mode']}")

    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    if cfg["mode"] == "baseline":
        train_baseline(cfg, device)
    elif cfg["mode"] in ("disentangler", "improved"):
        train_disentangler(cfg, device, mode=cfg["mode"])
    else:
        raise ValueError(f"Unknown mode: {cfg['mode']}")


if __name__ == "__main__":
    main()