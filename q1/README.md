# Question 1 – Multi-Stage Cepstral Feature Extraction & Phoneme Boundary Detection

## Repository Layout

```
q1_submission/
├── mfcc_manual.py          # Handcrafted MFCC / cepstrum engine (no librosa.feature.mfcc)
├── leakage_snr.py          # Spectral leakage & SNR analysis for 3 window types
├── voiced_unvoiced.py      # Cepstrum-based voiced/unvoiced boundary detector
├── phonetic_mapping.py     # Wav2Vec2 forced alignment + RMSE vs manual boundaries
├── requirements.txt        # Python package dependencies
├── data/
│   └── manifest.csv        # List of audio files used (download instructions inside)
└── outputs/                # All plots & results land here (auto-created at runtime)
```

---

## Quick-start

### 1 · Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # recommended
pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch note** – if you want GPU support replace the `torch` line in
> `requirements.txt` with the appropriate CUDA wheel from
> https://pytorch.org/get-started/locally/

---

### 2 · Prepare audio (optional)

Every script has a built-in synthetic signal fallback.
If you want to use real audio, download at least one file as described in
`data/manifest.csv` and pass its path via `--audio`.

---

### 3 · Run each deliverable

#### `mfcc_manual.py` — handcrafted MFCC engine

```bash
# With synthesised audio (default)
python mfcc_manual.py

# With a real WAV file
python mfcc_manual.py --audio data/librispeech_84-121550-0000.flac \
                      --window hamming --n_mfcc 13 --outdir outputs
```

**Outputs** (in `outputs/`):
- `mfcc_hamming.png`    – MFCC heat-map
- `mel_filterbank.png`  – triangular mel-filterbank visualisation

---

#### `leakage_snr.py` — spectral leakage & SNR analysis

```bash
python leakage_snr.py --audio data/librispeech_84-121550-0000.flac \
                      --outdir outputs
```

**Outputs** (in `outputs/`):
- `window_spectra.png`    – side-by-side FFT spectra for Rectangular / Hamming / Hanning
- `leakage_snr_time.png`  – per-frame leakage & SNR time series
- `leakage_snr_bar.png`   – mean leakage & SNR bar chart
- Console: summary table (mean ± std for each metric / window)

---

#### `voiced_unvoiced.py` — boundary detection

```bash
python voiced_unvoiced.py --audio data/librispeech_84-121550-0000.flac \
                           --outdir outputs
```

**Outputs** (in `outputs/`):
- `voiced_unvoiced.png` – 4-panel figure:
  waveform overlay · high-quefrency energy · label sequence · sample cepstra

---

#### `phonetic_mapping.py` — forced alignment + RMSE

```bash
# CPU (default); add --device cuda for GPU
python phonetic_mapping.py \
    --audio data/librispeech_84-121550-0000.flac \
    --transcript "HE SAID THE QUICK BROWN FOX" \
    --outdir outputs

# Without transcript (greedy decode used instead)
python phonetic_mapping.py --audio data/librispeech_84-121550-0000.flac
```

> The script downloads `facebook/wav2vec2-base-960h` from Hugging Face on first
> run (~380 MB). It is cached in `~/.cache/huggingface/`.

**Outputs** (in `outputs/`):
- `phonetic_alignment.png` – two-panel waveform: manual V/UV bounds vs model phone bounds
- Console: phone segment table, RMSE table (ms)

---

## Key Hyperparameters

| Parameter | Default | Where |
|-----------|---------|-------|
| Pre-emphasis coeff | 0.97 | `mfcc_manual.compute_mfcc()` |
| Frame length | 25 ms | all scripts |
| Frame step | 10 ms | all scripts |
| Window type | Hamming | `mfcc_manual.compute_mfcc()` |
| FFT size | 512 | all scripts |
| Mel filters | 26 | `mfcc_manual.compute_mfcc()` |
| MFCC coefficients | 13 | `mfcc_manual.compute_mfcc()` |
| Low quefrency boundary | ~1 ms (≈16 samples) | `voiced_unvoiced.cepstral_features()` |
| Otsu smoothing kernel | 5 frames | `voiced_unvoiced.detect_boundaries()` |
| RMSE match radius | 150 ms | `phonetic_mapping.compute_rmse()` |
| HF model | `facebook/wav2vec2-base-960h` | `phonetic_mapping.py` |

---

## Design Notes

### MFCC engine (no librosa.feature.mfcc)
All DSP steps are implemented from scratch using only `numpy` and `scipy.fft`:
pre-emphasis → framing → windowing → FFT → mel filterbank → log → DCT.
The mel filterbank uses the standard 2595·log₁₀(1 + f/700) mel scale.

### Spectral leakage metric
Leakage is measured as the ratio of out-of-main-lobe power to total power
(dB). The main lobe is defined as ±4 bins around the dominant spectral peak.

### Voiced/unvoiced detector
High-quefrency energy (quefrency > 1 ms, capturing pitch harmonics) is
normalised and thresholded using Otsu's method.  A 5-frame median filter
smooths the binary label sequence before boundary extraction.

### Forced alignment
Wav2Vec2 log-softmax emissions are decoded with a CTC Viterbi DP that
enforces the blank-interleaved state sequence.  Frame stride is 320 samples
(20 ms) for `wav2vec2-base-960h`.  RMSE is computed by matching each manual
boundary to its nearest model boundary within a 150 ms radius.

---

## Datasets Used
See `data/manifest.csv` for the full list with download URLs.

- **LibriSpeech test-clean** (OpenSLR) — real English read speech
- **Mozilla Common Voice v15** — crowd-sourced English
- **ESC-50** — environmental noise reference
- Synthesised signals — generated at runtime (no download needed)

---

## Report
`q1_report.pdf` (submitted separately) contains methods, hyperparameter
justification, representative plots, and the RMSE table.
