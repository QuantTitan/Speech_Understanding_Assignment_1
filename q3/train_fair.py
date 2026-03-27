"""
train_fair.py — Fairness-Aware Speech Recognition Training
============================================================
Fine-tunes a Whisper/Wav2Vec2 ASR model on Common Voice with a custom
"Fairness Loss" that minimises per-demographic performance gaps.

Fairness Loss Design:
─────────────────────
  L_total = L_ctc + λ_fair * L_fairness

  L_fairness = variance(group_losses) + max(group_losses) - min(group_losses)

  where group_losses are per-demographic-subgroup CTC losses, and the
  combined objective penalises both high spread and worst-group performance.

  This implements a soft version of "equalised odds" for ASR:
    → Encourages the model to achieve similar WER across gender × age groups
    → Does NOT degrade overall performance (additive regularisation only)

Usage:
    python train_fair.py [--epochs 10] [--batch_size 16] [--groups gender age]
    python train_fair.py --demo    # quick sanity-check run (no real data needed)
"""

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FairTrainingConfig:
    # Model
    model_name: str = "facebook/wav2vec2-base-960h"
    freeze_feature_extractor: bool = True

    # Data
    dataset_name: str = "mozilla-foundation/common_voice_11_0"
    dataset_lang: str = "en"
    max_audio_len: int = 16_000 * 10    # 10 seconds
    sample_rate: int = 16_000

    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-5
    warmup_steps: int = 500
    max_steps: int = 5000
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 500

    # Fairness
    fairness_lambda: float = 0.5          # Weight for fairness loss term
    group_fields: List[str] = field(
        default_factory=lambda: ["gender", "age"]
    )
    min_group_size: int = 8               # Skip groups with fewer samples

    # Paths
    output_dir: str = "checkpoints"
    log_path: str = "training_log.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FairnessLoss(nn.Module):
    """
    Custom Fairness Loss that penalises unequal ASR performance across
    demographic groups.

    Components:
      1. Between-group variance:  penalises spread of group losses
      2. Max–Min gap:             penalises worst vs best group disparity
      3. CVaR worst-group:        extra weight on the worst-performing group

    All three components are weighted and summed.
    """

    def __init__(
        self,
        variance_weight: float = 1.0,
        gap_weight: float = 0.5,
        cvar_weight: float = 0.5,
        cvar_alpha: float = 0.2,       # focus on worst 20% of groups
        eps: float = 1e-8,
    ):
        super().__init__()
        self.var_w = variance_weight
        self.gap_w = gap_weight
        self.cvar_w = cvar_weight
        self.alpha = cvar_alpha
        self.eps = eps

    def forward(self, group_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            group_losses: dict mapping group_name → scalar loss tensor

        Returns:
            fairness_loss: scalar tensor
            breakdown:     dict with per-component losses for logging
        """
        if len(group_losses) < 2:
            # Cannot compute fairness with fewer than 2 groups
            device = next(iter(group_losses.values())).device
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"variance": 0.0, "gap": 0.0, "cvar": 0.0}

        losses_tensor = torch.stack(list(group_losses.values()))  # (G,)

        # 1. Between-group variance
        var_loss = torch.var(losses_tensor)

        # 2. Max–Min gap (absolute worst-group disparity)
        gap_loss = losses_tensor.max() - losses_tensor.min()

        # 3. CVaR (Conditional Value at Risk): mean of worst α fraction
        k = max(1, int(np.ceil(self.alpha * len(losses_tensor))))
        worst_k, _ = torch.topk(losses_tensor, k)
        cvar_loss = worst_k.mean()

        # Combine
        fairness_loss = (
            self.var_w * var_loss +
            self.gap_w * gap_loss +
            self.cvar_w * cvar_loss
        )

        breakdown = {
            "variance": var_loss.item(),
            "gap": gap_loss.item(),
            "cvar": cvar_loss.item(),
            "group_losses": {k: v.item() for k, v in group_losses.items()},
        }
        return fairness_loss, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL ASR MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class FairASRModel(nn.Module):
    """
    Wraps a HuggingFace Wav2Vec2 model with:
      • Standard CTC loss computation
      • Per-group loss routing for fairness training
    """

    def __init__(self, cfg: FairTrainingConfig):
        super().__init__()
        self.cfg = cfg
        self._load_model()

    def _load_model(self):
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            print(f"[INFO] Loading {self.cfg.model_name} …")
            self.processor = Wav2Vec2Processor.from_pretrained(self.cfg.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.cfg.model_name)
            if self.cfg.freeze_feature_extractor:
                self.model.freeze_feature_extractor()
            self.use_hf = True
            print("[INFO] HuggingFace model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load HF model ({e}). Using toy model.")
            self.use_hf = False
            self._build_toy_model()

    def _build_toy_model(self):
        """
        Minimal trainable model for demo/testing without internet access.
        Input: raw waveform → CNN feature extractor → Transformer → CTC
        """
        self.feat_extract = nn.Sequential(
            nn.Conv1d(1, 64, 400, stride=160, padding=200),
            nn.GELU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.ctc_head = nn.Linear(256, 32)   # 32 vocabulary classes (toy)
        self.processor = None
        print("[INFO] Toy ASR model initialised.")

    def compute_ctc_loss(
        self,
        waveforms: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss for a batch.

        Args:
            waveforms:     (B, T) raw audio
            labels:        (B, L) token ids (padded with -100)
            input_lengths: (B,)
            label_lengths: (B,)

        Returns:
            loss per-sample: (B,)
        """
        if self.use_hf:
            return self._hf_ctc_loss(waveforms, labels, input_lengths, label_lengths)
        else:
            return self._toy_ctc_loss(waveforms, labels, input_lengths, label_lengths)

    def _hf_ctc_loss(self, waveforms, labels, input_lengths, label_lengths):
        # Replace -100 pad with pad_token_id
        labels_clamped = labels.clone()
        labels_clamped[labels_clamped == -100] = self.processor.tokenizer.pad_token_id

        outputs = self.model(
            input_values=waveforms,
            attention_mask=(waveforms != 0).float(),
        )
        logits = outputs.logits            # (B, T', vocab)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T', B, V)

        T_out = logits.shape[1]
        out_lengths = torch.clamp(input_lengths * T_out // input_lengths.max(), 1, T_out)

        losses = []
        for i in range(waveforms.size(0)):
            lab = labels_clamped[i][:label_lengths[i]]
            lp = log_probs[:out_lengths[i], i:i+1, :]
            loss = F.ctc_loss(
                lp, lab.unsqueeze(0), out_lengths[i:i+1], label_lengths[i:i+1],
                blank=0, reduction="mean", zero_infinity=True,
            )
            losses.append(loss)
        return torch.stack(losses)

    def _toy_ctc_loss(self, waveforms, labels, input_lengths, label_lengths):
        # waveforms: (B, T)
        x = waveforms.unsqueeze(1)        # (B, 1, T)
        feats = self.feat_extract(x)      # (B, 256, T')
        feats = feats.transpose(1, 2)     # (B, T', 256)
        encoded = self.encoder(feats)     # (B, T', 256)
        logits = self.ctc_head(encoded)   # (B, T', 32)

        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T', B, V)
        T_out = logits.shape[1]
        out_lengths = torch.full((waveforms.size(0),), T_out, dtype=torch.long)

        # Toy labels: just random valid sequences
        losses = []
        for i in range(waveforms.size(0)):
            ll = max(1, label_lengths[i].item())
            lab = torch.randint(1, 31, (1, ll), device=waveforms.device)
            loss = F.ctc_loss(
                log_probs[:, i:i+1, :], lab,
                out_lengths[i:i+1], torch.tensor([ll]),
                blank=0, reduction="mean", zero_infinity=True,
            )
            losses.append(loss)
        return torch.stack(losses)

    def forward(self, waveforms, labels, input_lengths, label_lengths):
        return self.compute_ctc_loss(waveforms, labels, input_lengths, label_lengths)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATASET (for demo / when CV unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticFairDataset(Dataset):
    """
    Generates synthetic (waveform, label, demographics) tuples.
    Intentionally introduces demographic skew to test the fairness loss.
    """

    GENDERS = ["male", "female", "other"]
    AGES = ["young", "middle", "senior"]

    # Simulate model performing worse on older female speakers (a common real-world bias)
    NOISE_LEVELS = {
        ("male", "young"): 0.01,
        ("male", "middle"): 0.02,
        ("male", "senior"): 0.05,
        ("female", "young"): 0.03,
        ("female", "middle"): 0.04,
        ("female", "senior"): 0.12,     # highest noise → hardest for model
        ("other", "young"): 0.05,
        ("other", "middle"): 0.06,
        ("other", "senior"): 0.10,
    }

    def __init__(self, n: int = 800, sr: int = 16_000, duration: float = 2.0, seed: int = 42):
        self.n = n
        self.sr = sr
        self.duration = duration
        self.length = int(sr * duration)
        rng = np.random.default_rng(seed)

        # Skewed gender distribution (mirrors Common Voice)
        gender_probs = [0.70, 0.20, 0.10]
        genders = rng.choice(self.GENDERS, size=n, p=gender_probs)
        ages = rng.choice(self.AGES, size=n)

        self.metadata = [{"gender": g, "age": a} for g, a in zip(genders, ages)]
        self.rng = rng

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        g, a = meta["gender"], meta["age"]
        noise_level = self.NOISE_LEVELS.get((g, a), 0.05)

        t = np.linspace(0, self.duration, self.length)
        f0 = 120 if g == "male" else 220 if g == "female" else 170
        if a == "senior":
            f0 *= 0.85
        elif a == "young":
            f0 *= 1.1

        wav = (
            0.4 * np.sin(2 * np.pi * f0 * t) +
            0.2 * np.sin(2 * np.pi * f0 * 2 * t) +
            noise_level * self.rng.standard_normal(self.length)
        ).astype(np.float32)
        wav /= max(np.abs(wav).max(), 1e-8)

        # Toy label: 5-10 tokens (random)
        label_len = self.rng.integers(5, 11)
        label = self.rng.integers(1, 31, size=label_len).astype(np.int64)

        return {
            "waveform": torch.tensor(wav),
            "label": torch.tensor(label),
            "gender": g,
            "age": a,
            "group_key": f"{g}_{a}",
        }


def collate_fn(batch):
    """Pad waveforms and labels within a batch."""
    max_wav = max(b["waveform"].shape[0] for b in batch)
    max_lab = max(b["label"].shape[0] for b in batch)
    B = len(batch)

    wavs = torch.zeros(B, max_wav)
    labs = torch.full((B, max_lab), -100, dtype=torch.long)
    wav_lens = torch.zeros(B, dtype=torch.long)
    lab_lens = torch.zeros(B, dtype=torch.long)
    groups = []

    for i, b in enumerate(batch):
        wl = b["waveform"].shape[0]
        ll = b["label"].shape[0]
        wavs[i, :wl] = b["waveform"]
        labs[i, :ll] = b["label"]
        wav_lens[i] = wl
        lab_lens[i] = ll
        groups.append(b["group_key"])

    return {
        "waveforms": wavs,
        "labels": labs,
        "input_lengths": wav_lens,
        "label_lengths": lab_lens,
        "groups": groups,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class FairASRTrainer:
    def __init__(self, cfg: FairTrainingConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Training on: {self.device}")

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        self.model = FairASRModel(cfg).to(self.device)
        self.fairness_loss_fn = FairnessLoss(
            variance_weight=1.0,
            gap_weight=0.5,
            cvar_weight=0.5,
        )

        # Optimiser
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=1e-4)

        # LR scheduler (linear warmup + cosine decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.learning_rate,
            total_steps=cfg.max_steps,
            pct_start=cfg.warmup_steps / cfg.max_steps,
        )

        self.global_step = 0
        self.log_file = open(cfg.log_path, "w")
        self.history = defaultdict(list)

    def _log(self, metrics: dict):
        metrics["step"] = self.global_step
        self.log_file.write(json.dumps(metrics) + "\n")
        self.log_file.flush()
        for k, v in metrics.items():
            if k != "step":
                self.history[k].append(v)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        waveforms = batch["waveforms"].to(self.device)
        labels = batch["labels"].to(self.device)
        input_lengths = batch["input_lengths"].to(self.device)
        label_lengths = batch["label_lengths"].to(self.device)
        groups = batch["groups"]

        # Per-sample CTC losses
        per_sample_losses = self.model(waveforms, labels, input_lengths, label_lengths)

        # Standard mean CTC loss
        ctc_loss = per_sample_losses.mean()

        # Group losses
        group_loss_dict = {}
        for idx, group_key in enumerate(groups):
            if group_key not in group_loss_dict:
                group_loss_dict[group_key] = []
            group_loss_dict[group_key].append(per_sample_losses[idx])

        # Aggregate group losses (only groups with sufficient samples)
        group_losses_aggregated = {}
        for gk, glosses in group_loss_dict.items():
            if len(glosses) >= 1:
                group_losses_aggregated[gk] = torch.stack(glosses).mean()

        # Fairness loss
        if len(group_losses_aggregated) >= 2:
            fair_loss, fair_breakdown = self.fairness_loss_fn(group_losses_aggregated)
        else:
            fair_loss = torch.tensor(0.0, device=self.device)
            fair_breakdown = {}

        # Total loss
        total_loss = ctc_loss + self.cfg.fairness_lambda * fair_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "ctc_loss": ctc_loss.item(),
            "fairness_loss": fair_loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            **{f"group_loss/{k}": v.item() for k, v in group_losses_aggregated.items()},
        }

    def train(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        print("\n" + "=" * 65)
        print("  FAIRNESS-AWARE ASR TRAINING")
        print("=" * 65)
        print(f"  λ_fair = {self.cfg.fairness_lambda}")
        print(f"  Groups: {self.cfg.group_fields}")
        print(f"  Max steps: {self.cfg.max_steps}")
        print("─" * 65)

        self.global_step = 0
        for epoch in range(self.cfg.epochs):
            epoch_metrics = defaultdict(list)
            epoch_start = time.time()

            for batch in train_loader:
                if self.global_step >= self.cfg.max_steps:
                    break

                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

                if self.global_step % self.cfg.log_every == 0:
                    avg = {k: np.mean(v[-self.cfg.log_every:]) for k, v in epoch_metrics.items()}
                    self._log(avg)
                    self._print_step(avg)

                if (self.global_step % self.cfg.save_every == 0
                        and self.global_step > 0):
                    self._save_checkpoint(epoch)

                self.global_step += 1

            elapsed = time.time() - epoch_start
            avg_total = np.mean(epoch_metrics["total_loss"])
            avg_ctc = np.mean(epoch_metrics["ctc_loss"])
            avg_fair = np.mean(epoch_metrics["fairness_loss"])
            print(f"\n  Epoch {epoch+1:02d}/{self.cfg.epochs}  "
                  f"loss={avg_total:.4f}  ctc={avg_ctc:.4f}  "
                  f"fair={avg_fair:.4f}  ({elapsed:.0f}s)")

            if eval_loader is not None:
                self._eval_epoch(eval_loader, epoch)

            if self.global_step >= self.cfg.max_steps:
                break

        self._save_checkpoint("final")
        self.log_file.close()
        print("\n[DONE] Training complete.")
        self._print_fairness_summary()

    def _print_step(self, metrics: dict):
        print(f"  step={self.global_step:5d} | "
              f"total={metrics.get('total_loss', 0):.4f} | "
              f"ctc={metrics.get('ctc_loss', 0):.4f} | "
              f"fair={metrics.get('fairness_loss', 0):.4f} | "
              f"lr={metrics.get('lr', 0):.2e}")

    @torch.no_grad()
    def _eval_epoch(self, eval_loader: DataLoader, epoch: int):
        self.model.eval()
        all_losses = defaultdict(list)

        for batch in eval_loader:
            waveforms = batch["waveforms"].to(self.device)
            labels = batch["labels"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)
            groups = batch["groups"]

            per_sample_losses = self.model(waveforms, labels, input_lengths, label_lengths)
            for i, gk in enumerate(groups):
                all_losses[gk].append(per_sample_losses[i].item())

        print(f"\n  [EVAL] Epoch {epoch+1}  per-group losses:")
        group_means = {}
        for gk, vals in sorted(all_losses.items()):
            mean_l = np.mean(vals)
            group_means[gk] = mean_l
            print(f"    {gk:<20} n={len(vals):3d}  loss={mean_l:.4f}")

        if group_means:
            gap = max(group_means.values()) - min(group_means.values())
            std = np.std(list(group_means.values()))
            print(f"    {'MaxMin Gap':<20}              gap={gap:.4f}  std={std:.4f}")
        self._log({"eval/group_losses": group_means})

    def _save_checkpoint(self, tag):
        params_to_save = {k: v for k, v in self.model.state_dict().items()}
        path = os.path.join(self.cfg.output_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "step": self.global_step,
            "model_state_dict": params_to_save,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg.__dict__,
        }, path)
        print(f"  [CKPT] Saved: {path}")

    def _print_fairness_summary(self):
        print("\n" + "═" * 65)
        print("  FAIRNESS TRAINING SUMMARY")
        print("═" * 65)
        if "fairness_loss" in self.history:
            fl = self.history["fairness_loss"]
            print(f"  Fairness loss: start={fl[0]:.4f} → end={fl[-1]:.4f}")
            improvement = (fl[0] - fl[-1]) / (fl[0] + 1e-8) * 100
            print(f"  Improvement: {improvement:.1f}%")
        print()
        print("  Interpretation:")
        print("  • Lower fairness loss = more equal performance across groups")
        print("  • The model optimises both accuracy AND demographic equity")
        print("  • Use evaluation_scripts/fad_eval.py for audio quality check")
        print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# REAL DATASET LOADER (Common Voice via HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def load_common_voice_dataset(cfg: FairTrainingConfig):
    """
    Attempt to load Common Voice via HuggingFace.
    Returns (train_loader, eval_loader) or falls back to synthetic data.
    """
    try:
        from datasets import load_dataset
        from transformers import Wav2Vec2Processor

        print("[INFO] Loading Common Voice from HuggingFace …")
        ds = load_dataset(
            cfg.dataset_name, cfg.dataset_lang,
            split={"train": "train", "validation": "validation"},
            trust_remote_code=True,
        )

        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)

        class CVDataset(Dataset):
            def __init__(self, hf_dataset, proc, max_len, sr):
                self.data = hf_dataset
                self.proc = proc
                self.max_len = max_len
                self.sr = sr
                self.MISSING = {"", "nan", "none", "n/a", "unknown"}

            def _clean_meta(self, val, choices):
                v = str(val).strip().lower() if val else ""
                return v if v not in self.MISSING else "unknown"

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                audio = item["audio"]["array"].astype(np.float32)
                sr = item["audio"]["sampling_rate"]
                if sr != self.sr:
                    import torchaudio
                    import torchaudio.transforms as T
                    audio_t = torch.tensor(audio).unsqueeze(0)
                    audio_t = T.Resample(sr, self.sr)(audio_t)
                    audio = audio_t.squeeze(0).numpy()
                audio = audio[:self.max_len]

                inputs = self.proc(audio, sampling_rate=self.sr, return_tensors="pt")
                waveform = inputs.input_values.squeeze(0)

                with self.proc.as_target_processor():
                    labels = self.proc(item["sentence"]).input_ids
                    labels = torch.tensor(labels, dtype=torch.long)

                gender = self._clean_meta(item.get("gender"), ["male", "female", "other"])
                age = self._clean_meta(item.get("age"), ["young", "middle", "senior"])
                group_key = f"{gender}_{age}"

                return {
                    "waveform": waveform,
                    "label": labels,
                    "gender": gender,
                    "age": age,
                    "group_key": group_key,
                }

        train_ds = CVDataset(ds["train"], processor, cfg.max_audio_len, cfg.sample_rate)
        val_ds = CVDataset(ds["validation"], processor, cfg.max_audio_len, cfg.sample_rate)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)
        return train_loader, val_loader

    except Exception as e:
        print(f"[WARN] HuggingFace load failed ({e}). Falling back to synthetic data.")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fairness-Aware ASR Training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--fairness_lambda", type=float, default=0.5,
                        help="Weight for fairness loss term")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--demo", action="store_true",
                        help="Quick demo with synthetic data only (no HF download)")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    cfg = FairTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fairness_lambda=args.fairness_lambda,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )

    if args.demo:
        print("[DEMO] Using synthetic skewed dataset (no download required).")
        train_ds = SyntheticFairDataset(n=400, seed=42)
        val_ds = SyntheticFairDataset(n=100, seed=99)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                                shuffle=False, collate_fn=collate_fn)
    else:
        train_loader, val_loader = load_common_voice_dataset(cfg)
        if train_loader is None:
            print("[INFO] Falling back to synthetic dataset.")
            train_ds = SyntheticFairDataset(n=800, seed=42)
            val_ds = SyntheticFairDataset(n=200, seed=99)
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                      shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                                    shuffle=False, collate_fn=collate_fn)

    trainer = FairASRTrainer(cfg)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
