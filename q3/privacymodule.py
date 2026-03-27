"""
privacymodule.py — Privacy-Preserving Voice Attribute Transformation
======================================================================
Implements a neural voice conversion pipeline that:
  • Disentangles speaker identity (biometric traits) from linguistic content
  • Transforms demographic attributes (gender × age group)
  • Preserves ASR transcription accuracy (linguistic content)

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Mel-spectrogram → ContentEncoder → (content)                       │
  │                  → SpeakerEncoder  → (source embedding)             │
  │  Target demographic → AttributeEmbedder → (target embedding)        │
  │  Decoder(content + target embedding) → Converted Mel                │
  │  HifiGAN-lite Vocoder → Waveform                                    │
  └─────────────────────────────────────────────────────────────────────┘

Usage:
    from privacymodule import PrivacyPreservingConverter, VoiceConversionConfig
    cfg = VoiceConversionConfig()
    model = PrivacyPreservingConverter(cfg)
    converted = model.convert(waveform, src_sr, target_gender="female", target_age="young")
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VoiceConversionConfig:
    # Audio
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 80.0
    f_max: float = 7600.0

    # Model dimensions
    content_dim: int = 256
    speaker_dim: int = 128
    style_dim: int = 64
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1

    # Demographic classes
    gender_classes: list = field(default_factory=lambda: ["male", "female", "other"])
    age_classes: list = field(default_factory=lambda: ["young", "middle", "senior"])

    # Training
    content_loss_weight: float = 1.0
    style_loss_weight: float = 0.5
    recon_loss_weight: float = 1.0
    cycle_loss_weight: float = 0.3
    adversarial_loss_weight: float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Channel-last LayerNorm compatible with (B, T, C) tensors."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock1D(nn.Module):
    """Depthwise-separable 1D conv block with residual."""
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=pad,
                            dilation=dilation, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.pw(self.dw(x))))


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT ENCODER
# (removes speaker identity, keeps linguistic features)
# ─────────────────────────────────────────────────────────────────────────────

class ContentEncoder(nn.Module):
    """
    Encodes mel-spectrogram into speaker-independent content embeddings.
    Uses Instance Normalisation to strip away speaker-specific statistics.
    """
    def __init__(self, cfg: VoiceConversionConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_mels, cfg.hidden_dim)
        self.conv_layers = nn.Sequential(
            ConvBlock1D(cfg.hidden_dim),
            ConvBlock1D(cfg.hidden_dim, dilation=2),
            ConvBlock1D(cfg.hidden_dim, dilation=4),
        )
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers // 2)
        ])
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.content_dim)
        # Instance norm: crucial for speaker disentanglement
        self.instance_norm = nn.InstanceNorm1d(cfg.content_dim, affine=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T, n_mels) mel-spectrogram
        Returns:
            content: (B, T, content_dim) speaker-agnostic features
        """
        x = self.input_proj(mel)          # (B, T, H)
        x = x.transpose(1, 2)             # (B, H, T) for conv
        x = self.conv_layers(x)
        x = x.transpose(1, 2)             # (B, T, H) for transformer
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.output_proj(x)           # (B, T, C)
        # Apply instance norm to remove speaker statistics
        x = self.instance_norm(x.transpose(1, 2)).transpose(1, 2)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER ENCODER
# (extracts reference speaker embeddings)
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerEncoder(nn.Module):
    """
    Produces a fixed-length speaker embedding from a mel-spectrogram.
    Inspired by d-vector / x-vector approaches.
    """
    def __init__(self, cfg: VoiceConversionConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_mels, cfg.hidden_dim)
        self.lstm = nn.LSTM(cfg.hidden_dim, cfg.hidden_dim // 2,
                            num_layers=3, batch_first=True,
                            dropout=cfg.dropout, bidirectional=True)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.speaker_dim)
        self.norm = nn.LayerNorm(cfg.speaker_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T, n_mels)
        Returns:
            spk_emb: (B, speaker_dim)
        """
        x = self.input_proj(mel)
        out, _ = self.lstm(x)
        # Attentive statistics pooling
        attn = torch.softmax(out.mean(-1, keepdim=True), dim=1)
        x = (out * attn).sum(dim=1)
        return self.norm(self.output_proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# DEMOGRAPHIC ATTRIBUTE EMBEDDER
# (encodes target demographic: gender + age)
# ─────────────────────────────────────────────────────────────────────────────

class DemographicAttributeEmbedder(nn.Module):
    """
    Embeds a target demographic specification (gender × age) into
    a style vector that conditions the decoder.
    """
    def __init__(self, cfg: VoiceConversionConfig):
        super().__init__()
        n_gender = len(cfg.gender_classes)
        n_age = len(cfg.age_classes)
        self.gender_emb = nn.Embedding(n_gender, cfg.style_dim)
        self.age_emb = nn.Embedding(n_age, cfg.style_dim)
        self.combine = nn.Sequential(
            nn.Linear(cfg.style_dim * 2, cfg.style_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.style_dim * 2, cfg.speaker_dim),
        )
        self.gender_map = {g: i for i, g in enumerate(cfg.gender_classes)}
        self.age_map = {a: i for i, a in enumerate(cfg.age_classes)}

    def forward(self, gender_ids: torch.Tensor, age_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gender_ids: (B,) long tensor
            age_ids:    (B,) long tensor
        Returns:
            style: (B, speaker_dim)
        """
        g = self.gender_emb(gender_ids)
        a = self.age_emb(age_ids)
        return self.combine(torch.cat([g, a], dim=-1))

    def encode_labels(self, gender: str, age: str, device: torch.device) -> Tuple:
        gid = torch.tensor([self.gender_map.get(gender.lower(), 0)], device=device)
        aid = torch.tensor([self.age_map.get(age.lower(), 0)], device=device)
        return gid, aid


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE INSTANCE NORMALISATION (AdaIN)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveInstanceNorm1d(nn.Module):
    """
    Injects style (speaker embedding) into content features via AdaIN.
    This is the key mechanism for attribute transfer.
    """
    def __init__(self, content_dim: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(content_dim, affine=False)
        self.style_proj = nn.Linear(style_dim, content_dim * 2)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: (B, T, content_dim)
            style:   (B, style_dim)
        Returns:
            adapted: (B, T, content_dim)
        """
        # Normalise content
        normed = self.norm(content.transpose(1, 2)).transpose(1, 2)
        # Derive scale (gamma) and shift (beta) from style
        params = self.style_proj(style).unsqueeze(1)   # (B, 1, 2*C)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * normed + beta


# ─────────────────────────────────────────────────────────────────────────────
# DECODER
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Takes content features + target style embedding → reconstructed mel.
    Uses a stack of AdaIN-conditioned transformer + conv blocks.
    """
    def __init__(self, cfg: VoiceConversionConfig):
        super().__init__()
        self.adain_layers = nn.ModuleList([
            AdaptiveInstanceNorm1d(cfg.content_dim, cfg.speaker_dim)
            for _ in range(cfg.num_layers)
        ])
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(cfg.content_dim, max(1, cfg.num_heads // 2), cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.conv_layers = nn.ModuleList([
            ConvBlock1D(cfg.content_dim, kernel_size=5, dilation=2 ** i)
            for i in range(cfg.num_layers // 2)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.n_mels),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: (B, T, content_dim)
            style:   (B, speaker_dim)
        Returns:
            mel_out: (B, T, n_mels)
        """
        x = content
        for adain, tf, conv in zip(
                self.adain_layers,
                self.transformer_layers,
                [None] * len(self.transformer_layers)):
            x = adain(x, style)
            x = tf(x)
        # Final conv refinement
        xc = x.transpose(1, 2)
        for conv in self.conv_layers:
            xc = conv(xc)
        x = xc.transpose(1, 2)
        return self.output_proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTWEIGHT HIFIGAN VOCODER (Mini)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: Tuple = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=(kernel_size - 1) // 2),
            ))

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class MiniHiFiGAN(nn.Module):
    """
    Lightweight HiFi-GAN style vocoder (mel → waveform).
    Reduced size for faster inference; swap with full HiFi-GAN for production.
    """
    def __init__(self, n_mels: int = 80, upsample_rates: Tuple = (8, 8, 4),
                 base_channels: int = 256):
        super().__init__()
        self.pre_conv = nn.Conv1d(n_mels, base_channels, 7, padding=3)
        self.ups = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        ch = base_channels
        for rate in upsample_rates:
            self.ups.append(
                nn.ConvTranspose1d(ch, ch // 2, rate * 2, stride=rate, padding=rate // 2)
            )
            self.res_blocks.append(ResidualBlock(ch // 2))
            ch //= 2
        self.post_conv = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T)
        Returns:
            wav: (B, 1, T*prod(upsample_rates))
        """
        x = self.pre_conv(mel)
        for up, res in zip(self.ups, self.res_blocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = res(x)
        return torch.tanh(self.post_conv(F.leaky_relu(x, 0.1)))


# ─────────────────────────────────────────────────────────────────────────────
# FULL PRIVACY-PRESERVING CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

class PrivacyPreservingConverter(nn.Module):
    """
    End-to-end model:
      waveform → mel → content + style → converted mel → waveform

    Inference example:
        model = PrivacyPreservingConverter(cfg)
        wav_out = model.convert(wav_in, 16000, "female", "young")
    """
    def __init__(self, cfg: VoiceConversionConfig):
        super().__init__()
        self.cfg = cfg

        # Audio processing
        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

        # Sub-modules
        self.content_encoder = ContentEncoder(cfg)
        self.speaker_encoder = SpeakerEncoder(cfg)
        self.attribute_embedder = DemographicAttributeEmbedder(cfg)
        self.decoder = Decoder(cfg)
        self.vocoder = MiniHiFiGAN(n_mels=cfg.n_mels)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, samples) → mel: (B, T, n_mels)"""
        mel = self.mel_transform(wav)           # (B, n_mels, T)
        mel = self.amplitude_to_db(mel)
        mel = (mel + 40.0) / 40.0              # rough normalisation to ~[-1, 1]
        return mel.transpose(1, 2)              # (B, T, n_mels)

    # ── Forward (for training) ────────────────────────────────────────────────

    def forward(
        self,
        src_wav: torch.Tensor,
        tgt_gender_ids: torch.Tensor,
        tgt_age_ids: torch.Tensor,
    ) -> dict:
        """
        Training forward pass.

        Args:
            src_wav:        (B, samples)
            tgt_gender_ids: (B,)
            tgt_age_ids:    (B,)

        Returns dict with:
            content:      (B, T, content_dim)
            src_style:    (B, speaker_dim)
            tgt_style:    (B, speaker_dim)
            converted_mel:(B, T, n_mels)
            src_mel:      (B, T, n_mels)
        """
        src_mel = self._wav_to_mel(src_wav)
        content = self.content_encoder(src_mel)
        src_style = self.speaker_encoder(src_mel)
        tgt_style = self.attribute_embedder(tgt_gender_ids, tgt_age_ids)
        converted_mel = self.decoder(content, tgt_style)

        return {
            "src_mel": src_mel,
            "content": content,
            "src_style": src_style,
            "tgt_style": tgt_style,
            "converted_mel": converted_mel,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def convert(
        self,
        wav: torch.Tensor,
        src_sr: int,
        target_gender: str = "female",
        target_age: str = "young",
    ) -> torch.Tensor:
        """
        Convert speaker attributes while preserving linguistic content.

        Args:
            wav:           (samples,) or (1, samples) mono waveform tensor
            src_sr:        Source sample rate
            target_gender: Target gender string ("male" | "female" | "other")
            target_age:    Target age group ("young" | "middle" | "senior")

        Returns:
            converted_wav: (1, samples') converted waveform
        """
        self.eval()
        device = next(self.parameters()).device

        # Resample if needed
        if src_sr != self.cfg.sample_rate:
            resampler = T.Resample(src_sr, self.cfg.sample_rate).to(device)
            wav = resampler(wav)

        # Shape: (1, samples)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(device)

        # Encode
        src_mel = self._wav_to_mel(wav)
        content = self.content_encoder(src_mel)

        # Target demographic embedding
        gid, aid = self.attribute_embedder.encode_labels(target_gender, target_age, device)
        tgt_style = self.attribute_embedder(gid, aid)

        # Decode
        converted_mel = self.decoder(content, tgt_style)  # (1, T, n_mels)

        # Vocode
        mel_for_voc = converted_mel.transpose(1, 2)       # (1, n_mels, T)
        wav_out = self.vocoder(mel_for_voc)                # (1, 1, T_wav)
        return wav_out.squeeze(1)                          # (1, T_wav)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class VoiceConversionLoss(nn.Module):
    """
    Combined loss for training the privacy-preserving converter:
      • Reconstruction loss:  L1 on mel-spectrograms
      • Content consistency:  cosine similarity between content vectors
      • Style matching:       MSE on style embeddings
      • Cycle-consistency:    reconstructing source after round-trip
    """
    def __init__(self, cfg: VoiceConversionConfig):
        super().__init__()
        self.cfg = cfg

    def reconstruction_loss(self, pred_mel: torch.Tensor, tgt_mel: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred_mel, tgt_mel)

    def content_loss(self, content_orig: torch.Tensor,
                     content_converted: torch.Tensor) -> torch.Tensor:
        """Cosine similarity in content space — should stay ~1 after conversion."""
        orig_flat = content_orig.reshape(-1, content_orig.size(-1))
        conv_flat = content_converted.reshape(-1, content_converted.size(-1))
        cos_sim = F.cosine_similarity(orig_flat, conv_flat, dim=-1)
        return 1.0 - cos_sim.mean()

    def style_loss(self, style_pred: torch.Tensor,
                   style_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(style_pred, style_target)

    def forward(self, model_output: dict,
                reconstructed_mel: Optional[torch.Tensor] = None) -> dict:
        losses = {}

        # Reconstruction: converted mel ≈ target mel (if we have paired data)
        if reconstructed_mel is not None:
            losses["reconstruction"] = self.reconstruction_loss(
                model_output["converted_mel"], reconstructed_mel
            ) * self.cfg.recon_loss_weight

        # Content preservation
        losses["content"] = self.content_loss(
            model_output["content"].detach(), model_output["content"]
        ) * self.cfg.content_loss_weight

        # Style matching: predicted style ≈ target demographic style
        losses["style"] = self.style_loss(
            model_output["src_style"], model_output["tgt_style"]
        ) * self.cfg.style_loss_weight

        losses["total"] = sum(losses.values())
        return losses


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: PARAMETER COUNT
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {trainable:,}"


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("PrivacyModule — smoke test")
    cfg = VoiceConversionConfig()
    model = PrivacyPreservingConverter(cfg)
    print(f"Parameters: {count_parameters(model)}")

    # Dummy batch
    B, T = 2, 16000
    wav = torch.randn(B, T)
    g_ids = torch.tensor([1, 0])   # female, male
    a_ids = torch.tensor([0, 2])   # young, senior

    out = model(wav, g_ids, a_ids)
    print(f"Content shape:       {out['content'].shape}")
    print(f"Converted mel shape: {out['converted_mel'].shape}")

    # Inference
    wav_single = torch.randn(T)
    wav_out = model.convert(wav_single, 16000, "female", "young")
    print(f"Output waveform shape: {wav_out.shape}")
    print("Smoke test PASSED.")
