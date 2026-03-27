# Q3 — Ethical Auditing & "Documentation Debt" Mitigation

## Overview

This project performs a full **Sound Check** ethical audit on the Mozilla Common Voice dataset, implements a **Privacy-Preserving Voice Conversion** system, trains a **Fairness-aware ASR model**, and validates audio quality using FAD and DNSMOS-proxy metrics.

---

## Project Structure

```
q3/
├── audit.py                        # Bias audit: documentation debt + representation bias
├── privacymodule.py                # PyTorch privacy-preserving voice converter
├── pp_demo.py                      # Demo: run voice conversion on audio files
├── train_fair.py                   # ASR training with Fairness Loss
├── evaluation_scripts/
│   ├── fad_eval.py                 # Fréchet Audio Distance evaluation
│   └── dnsmos_proxy.py             # DNSMOS / MOS proxy evaluation
├── examples/
│   └── README.md                   # Populated by pp_demo.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Bias Audit

```bash
python audit.py --sample 5000 --output audit_plots.pdf
```

**What it does:**
- Downloads Mozilla Common Voice (en) via HuggingFace (or uses a built-in synthetic surrogate if offline)
- Analyses gender, age, and accent/dialect distributions
- Computes Gini coefficients per demographic axis
- Produces documentation debt report (missing metadata %)
- Saves `audit_plots.pdf` with 3 pages of visualisations

### 3. Run the Privacy-Preserving Demo

```bash
# With your own audio file:
python pp_demo.py --input path/to/speech.wav

# Synthetic demo (no audio file needed):
python pp_demo.py

# With ASR verification (needs openai-whisper):
pip install openai-whisper
python pp_demo.py --asr
```

**What it does:**
- Converts a source voice to all 9 demographic combinations (3 genders × 3 ages)
- Saves all audio pairs to `examples/`
- Optionally computes WER to verify linguistic content preservation

### 4. Train with Fairness Loss

```bash
# Quick demo (synthetic data, no download):
python train_fair.py --demo --epochs 5 --max_steps 200

# Full Common Voice training:
python train_fair.py --epochs 10 --fairness_lambda 0.5
```

**What it does:**
- Fine-tunes Wav2Vec2 (or uses a toy model in demo mode)
- Adds a `FairnessLoss` term: `L = L_CTC + λ · L_fair`
- `L_fair` = inter-group variance + max–min gap + CVaR worst group
- Logs per-group WER at each evaluation step

### 5. Evaluate Audio Quality

```bash
# FAD evaluation
python evaluation_scripts/fad_eval.py \
    --reference examples/original.wav \
    --test_dir  examples/

# DNSMOS proxy evaluation
python evaluation_scripts/dnsmos_proxy.py \
    --dir examples/

# Side-by-side comparison
python evaluation_scripts/dnsmos_proxy.py \
    --compare examples/original.wav examples/converted_female_young.wav
```

---

## Architecture: Privacy-Preserving Converter

```
Input WAV
    │
    ▼
MelSpectrogram (80-mel, 16 kHz)
    │
    ├──► ContentEncoder (Transformer + InstanceNorm)
    │       → speaker-agnostic linguistic features
    │
    └──► SpeakerEncoder (BiLSTM + attentive pooling)
            → source speaker embedding (not used in conversion)

Target Demographic ──► DemographicAttributeEmbedder
 (gender × age label)   (gender_emb + age_emb → style vector)

ContentEncoder output + Style vector
    │
    ▼
Decoder (AdaIN-conditioned Transformer)
    → Converted Mel-spectrogram
    │
    ▼
MiniHiFiGAN Vocoder
    → Converted Waveform
```

**Key technique:** Instance Normalisation in the content encoder removes speaker-specific statistics (mean and variance), stripping biometric identity while preserving phoneme sequences. Adaptive Instance Normalisation (AdaIN) in the decoder re-injects the target demographic style.

---

## Fairness Loss Function

```python
L_total = L_CTC + λ · L_fairness

L_fairness = w₁·Var({L_g}) + w₂·(max(L_g) - min(L_g)) + w₃·CVaR₀.₂({L_g})
```

Where:
- `L_g` = CTC loss averaged over all samples from demographic group g
- `Var(·)` = inter-group variance (penalises spread)
- `max − min` = worst-case disparity (penalises outlier groups)
- `CVaR_α` = mean of worst α fraction of groups (DRO-inspired)

This implements a soft form of **equalised odds**: the model is penalised whenever any demographic group performs significantly worse than others.

---

## Datasets

| Dataset | Used In | Access |
|---------|---------|--------|
| Mozilla Common Voice 11.0 (en) | `audit.py`, `train_fair.py` | HuggingFace: `mozilla-foundation/common_voice_11_0` |
| Synthetic surrogate | All scripts (fallback) | Built-in — no download required |

---

## Ethical Considerations

1. **Consent**: Voice conversion must only be applied with explicit speaker consent.
2. **Transparency**: All transformations should be logged and disclosed.
3. **Dual-use risk**: The same technology that protects privacy could be misused for voice spoofing. Model access should be restricted.
4. **Bias amplification**: Demographic embeddings may encode stereotypes. Regular re-auditing is essential.
5. **Fairness trade-offs**: Reducing the worst-group performance gap may slightly lower average accuracy — this trade-off should be explicitly communicated.

---

## Files Generated

| Output | Script | Description |
|--------|--------|-------------|
| `audit_plots.pdf` | `audit.py` | 3-page demographic audit visualisation |
| `audit_demographics.csv` | `audit.py` | Raw demographic counts |
| `audit_debt.csv` | `audit.py` | Documentation debt per field |
| `examples/original.wav` | `pp_demo.py` | Source audio |
| `examples/converted_*.wav` | `pp_demo.py` | 9 converted audio files |
| `examples/fad_results.csv` | `fad_eval.py` | FAD scores per conversion |
| `examples/dnsmos_results.csv` | `dnsmos_proxy.py` | MOS scores per conversion |
| `checkpoints/checkpoint_*.pt` | `train_fair.py` | Model checkpoints |
| `training_log.jsonl` | `train_fair.py` | Per-step training metrics |

---

## Citation / References

- Mozilla Common Voice: https://commonvoice.mozilla.org
- Wav2Vec 2.0: Baevski et al. (2020), NeurIPS
- HiFi-GAN: Kong et al. (2020), NeurIPS
- AdaIN: Huang & Belongie (2017), ICCV
- DNSMOS: Reddy et al. (2022), ICASSP
- Fréchet Audio Distance: Kilgour et al. (2019), Interspeech
- Fairness in ML: Hardt et al. (2016), NeurIPS
