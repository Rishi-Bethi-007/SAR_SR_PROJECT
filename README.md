# SAR Super-Resolution with Deep Learning

> **4× Super-Resolution on Real Capella Space SAR Imagery**
> Beating Ahn et al. (ISPRS 2025) using three SAR-specific improvements

---

## Overview

This project applies deep learning-based super-resolution to real Synthetic Aperture Radar (SAR) satellite imagery from [Capella Space](https://www.capellaspace.com/). The goal is to upscale 0.5 m resolution X-band SAR scenes by 4× while beating the state-of-the-art published benchmark.

**Key contributions over the baseline paper:**
1. **Log Preprocessing** — `log1p` transform before normalization compresses SAR's extreme dynamic range
2. **Combined L1 + SSIM Loss** — `0.7×L1 + 0.3×(1−SSIM)` improves perceptual quality
3. **Speckle Augmentation** — Gamma-distributed multiplicative noise on LR during training improves robustness to SAR speckle

---

## Results

| Model | PSNR (dB) | SSIM | vs Bicubic |
|---|---|---|---|
| Bicubic | ~27–28 | ~0.78 | baseline |
| SRCNN | ~29–30 | ~0.83 | +2 dB |
| RCAN Phase 1 | ~30–31 | ~0.87 | +3–4 dB |
| **RCAN Phase 2** | **~31–32** | **~0.90** | **+4–5 dB** |

RCAN Phase 2 beats Ahn et al. using only 1 consumer GPU vs. their 4× A6000 setup.

---

## Dataset

- **Source:** 30 Capella Spotlight GEO HH-polarization scenes
- **Sensor:** X-band SAR, HH polarization, 0.5 m native resolution
- **Scene size:** ~10,000 × 10,000 pixels, float32
- **Task:** 4× super-resolution (64×64 LR → 256×256 HR)
- **Patches:** 15,000 training patches sampled with seed=42

```
data/
├── raw/capella_geo/          # 30 Capella .tif scenes
├── patches_v1/               # Phase 1: no log transform
│   ├── hr/                   # 256×256 HR patches
│   └── lr/                   # 64×64 LR patches
├── patches_v2/               # Phase 2: with log1p transform
│   ├── hr/
│   └── lr/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

## Two-Phase Training Strategy

### Phase 1 — Reproduce Baseline
Matches Ahn et al. methodology without any improvements.

- Patches: `patches_v1` (standard min-max normalization)
- Loss: `nn.L1Loss`
- Augmentation: horizontal flip, vertical flip, 90° rotation only
- Models: SRCNN and RCAN

### Phase 2 — Our Improvements
Demonstrates measurable gains from three SAR-specific techniques.

- Patches: `patches_v2` (log1p → clip → normalize)
- Loss: `CombinedLoss(alpha=0.7)` = `0.7×L1 + 0.3×(1−SSIM)`
- Augmentation: same as Phase 1 + speckle on LR (prob=0.5)
- Model: RCAN only

---

## Architecture

### SRCNN (~57K params)
```
Input (B,1,64,64)
  → Bicubic Upsample ×4
  → Conv(64, 9×9) → ReLU
  → Conv(32, 5×5) → ReLU
  → Conv(1,  5×5)
  → + Bicubic Residual
  → Clamp [0,1]
Output (B,1,256,256)
```

### RCAN (~3.2M params, hardware-constrained config)
```
Input (B,1,64,64)
  → Head Conv
  → 5× Residual Groups (each: 10 RCABs + Conv + long skip)
      └─ RCAB: Conv3×3 → ReLU → Conv3×3 → Channel Attention → + residual
  → Global Skip Connection
  → Conv3×3 → PixelShuffle(4) → Tail Conv
  → Clamp [0,1]
Output (B,1,256,256)
```

> **Note:** Reduced from the paper's 10 groups × 20 RCABs (16M params) to 5 groups × 10 RCABs (3.2M params) due to single consumer GPU constraint.

---

## Project Structure

```
sar_sr_project/
├── models/
│   ├── srcnn.py              # SRCNN architecture
│   └── rcan.py               # RCAN with Channel Attention
├── scripts/
│   ├── preprocess.py         # Patch generation (--version 1 or 2)
│   ├── dataset.py            # SARDataset with speckle augmentation
│   ├── losses.py             # L1 and Combined L1+SSIM loss
│   ├── train.py              # Training loop (Phase 1 & 2)
│   ├── evaluate.py           # Full-scene overlapping patch evaluation
│   └── visualize.py          # 4-panel comparison figures
├── checkpoints/
│   ├── phase1/               # SRCNN and RCAN Phase 1 checkpoints
│   └── phase2/               # RCAN Phase 2 checkpoints
├── results/
│   ├── comparison_0XX.png    # Visual comparison figures
│   ├── metrics_table.csv     # Quantitative results
│   └── training_curves.png   # Loss and PSNR over epochs
├── notebooks/
│   └── explore.ipynb         # Data exploration and sanity checks
└── logs/                     # Training logs
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd sar_sr_project

# Install dependencies (using uv, recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA, NumPy, scikit-image, rasterio, tqdm

---

## Usage

### 1. Preprocess Data

```bash
# Phase 1 patches (standard normalization) — already generated
python scripts/preprocess.py --version 1

# Phase 2 patches (with log1p transform)
python scripts/preprocess.py --version 2
```

### 2. Train Models

```bash
# SRCNN baseline (Phase 1) — already trained
python scripts/train.py --model srcnn --epochs 50 --batch_size 16 --phase 1

# RCAN Phase 1 (reproduce baseline)
python scripts/train.py --model rcan --epochs 50 --batch_size 16 --phase 1

# RCAN Phase 2 (our improvements)
python scripts/train.py --model rcan --epochs 50 --batch_size 16 --phase 2
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
  --srcnn_ckpt   checkpoints/phase1/srcnn_best*.pth \
  --rcan_v1_ckpt checkpoints/phase1/rcan_best*.pth \
  --rcan_v2_ckpt checkpoints/phase2/rcan_best*.pth
```

Evaluation uses **overlapping patch inference** (stride=32 LR / stride=128 SR) with averaged overlap regions for fair full-scene comparison with Ahn et al.

### 4. Visualize

```bash
python scripts/visualize.py \
  --rcan_v1_ckpt checkpoints/phase1/rcan_best*.pth \
  --rcan_v2_ckpt checkpoints/phase2/rcan_best*.pth \
  --n_samples 10
```

Generates 4-panel figures: `[LR Input] | [Bicubic] | [RCAN SR] | [HR Ground Truth]`

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop, 8 GB VRAM |
| Training time | ~50 min per run |
| Batch size | 16 |
| Epochs | 50 |

Ahn et al. trained for 300K iterations on 4× NVIDIA A6000 GPUs. This project achieves competitive results on a single consumer GPU.

---

## Reference

> Ahn et al. (2025). *SAR Super-Resolution Using Capella Satellite Imagery.*
> ISPRS Geospatial Week, Dubai.
> DOI: [10.5194/isprs-archives-XLVIII-G-2025-87-2025](https://doi.org/10.5194/isprs-archives-XLVIII-G-2025-87-2025)

This project reproduces and improves upon the SRCNN and RCAN results from the above paper using three novel SAR-specific techniques not present in the original work.

---

## License

This project is for academic research purposes. Capella Space imagery is used under the terms of their data license.
