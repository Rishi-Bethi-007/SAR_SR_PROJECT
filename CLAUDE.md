# SAR Super Resolution — Project Guide
# Status: CONTINUING from partial completion

---

## Project Goal
4x super resolution on real Capella Space SAR imagery using PyTorch.
Beat Ahn et al. (ISPRS 2025) — the only published paper on this dataset.
Strategy: SRCNN baseline + RCAN with 3 SAR-specific improvements.

---

## CURRENT PROJECT STATUS

### Already built and working — DO NOT rebuild these:
- data/raw/capella_geo/          ← 30 downloaded Capella GEO .tif scenes
- data/patches_v1/hr/ and lr/    ← Phase 1 patches already generated
- data/splits/train.txt, val.txt, test.txt  ← splits already created
- models/srcnn.py                ← built and correct
- scripts/preprocess.py          ← built (--version 1 already ran)
- scripts/dataset.py             ← built
- scripts/losses.py              ← built
- scripts/train.py               ← built and working
- checkpoints/phase1/srcnn_best.pth  ← SRCNN Phase 1 trained and saved

### Still needs to be done — build these in order:
1. Run preprocess.py --version 2  (generate Phase 2 patches with log transform)
2. Update models/rcan.py          (REDUCED config: 5 groups x 10 RCABs)
3. Train RCAN Phase 1             (uses patches_v1, L1 loss, no speckle)
4. Train RCAN Phase 2             (uses patches_v2, CombinedLoss, speckle)
5. Build scripts/evaluate.py      (full-scene overlapping patch evaluation)
6. Build scripts/visualize.py     (4-panel comparison figures)
7. Build notebooks/explore.ipynb  (sanity check — if not already built)

---

## Hardware Constraints (important for all decisions)
GPU: NVIDIA RTX 5060 Laptop, 8GB VRAM
To keep training under 1 hour per run, use these REDUCED settings:
  RCAN config:  num_groups=5, num_rcab=10  (~3.2M params, NOT 16M)
  Patches:      15,000 randomly sampled from available patches (seed=42)
  Epochs:       50 for both RCAN phases
  Batch size:   16
Report note: "Due to hardware constraints (single consumer GPU), a reduced
RCAN configuration was used. Ahn et al. used 4x A6000 GPUs."

---

## Dataset
- 30 Capella Spotlight GEO HH scenes in ./data/raw/capella_geo/
- Each scene: ~10,000x10,000 pixels, 0.5m resolution, X-band SAR, float32
- File pattern: CAPELLA_C##_SP_GEO_HH_YYYYMMDDHHMMSS_*.tif

## Data Format
- HR patches: (256, 256) float32, .npy, in data/patches_v1/hr/ or patches_v2/hr/
- LR patches: (64, 64)   float32, .npy, in data/patches_v1/lr/ or patches_v2/lr/
- Model input:  tensor (B, 1, 64, 64)   float32 range [0, 1]
- Model output: tensor (B, 1, 256, 256) float32 range [0, 1]
- Scale factor: 4x
- Splits (shared by both phases): data/splits/train.txt, val.txt, test.txt

---

## TWO-PHASE TRAINING STRATEGY

### Phase 1 — Baseline (reproduce Ahn et al. without improvements)
Goal: match Ahn et al. numbers at full-scene evaluation.
Patches: data/patches_v1/ (no log transform — already generated)
Loss: nn.L1Loss()
Augmentation: hflip, vflip, rot90 only. NO speckle.
Models: SRCNN (done), RCAN (still to do)
Checkpoints: checkpoints/phase1/

Preprocessing used for v1 (already done, do not redo):
  1. Replace invalid (<=0, NaN) with 1e-6
  2. Clip to [percentile(1), percentile(99)]
  3. Min-max normalize to [0, 1]

### Phase 2 — Improvements (beat Ahn et al.)
Goal: show RCAN Phase 2 > RCAN Phase 1 by measurable PSNR/SSIM margin.
Patches: data/patches_v2/ (WITH log transform — needs to be generated)
Loss: CombinedLoss(alpha=0.7) = 0.7*L1 + 0.3*(1-SSIM)
Augmentation: hflip, vflip, rot90 + speckle on LR (prob=0.5)
Model: RCAN only
Checkpoints: checkpoints/phase2/

Preprocessing for v2 (preprocess.py --version 2, still to run):
  1. Replace invalid (<=0, NaN) with 1e-6
  2. np.log1p(img)               ← IMPROVEMENT #1: log transform
  3. Clip to [percentile(1), percentile(99)]
  4. Min-max normalize to [0, 1]

IMPORTANT: patches_v1 and patches_v2 are separate folders.
Both use the SAME splits from data/splits/.
15,000 patches randomly sampled from train.txt for training (seed=42).

---

## IMPROVEMENT #1 — Log Preprocessing
Applied in preprocess.py --version 2 only.
np.log1p before clip and normalize. Compresses SAR dynamic range.
Ahn et al. did NOT use this.

## IMPROVEMENT #2 — Combined L1 + SSIM Loss (scripts/losses.py)
CombinedLoss(alpha=0.7): 0.7*L1 + 0.3*(1 - SSIM)
SSIM must be differentiable using F.conv2d with Gaussian window size 11.
Used in Phase 2 only. Phase 1 uses plain nn.L1Loss().
Ahn et al. did NOT use this.

## IMPROVEMENT #3 — Speckle Augmentation (scripts/dataset.py)
Applied to LR patches ONLY during Phase 2 training, with prob=0.5.
Formula: lr = lr * Gamma(shape=4, scale=0.25), clipped [0,1].
NEVER apply to HR. NEVER apply during val or test.
Ahn et al. did NOT use this.

---

## SRCNN Architecture (models/srcnn.py) — ALREADY BUILT, DO NOT CHANGE
Input (B,1,64,64) → bicubic upsample → Conv(64,9x9,pad=4) → ReLU
→ Conv(32,5x5,pad=2) → ReLU → Conv(1,5x5,pad=2) → + bicubic residual
→ clamp [0,1] → Output (B,1,256,256). ~57K params.

## RCAN Architecture (models/rcan.py) — NEEDS REDUCED CONFIG
REDUCED config for this project (hardware constrained):
  num_features=64, num_groups=5, num_rcab=10, scale=4  ← USE THESE
  (NOT the paper's 10 groups x 20 RCABs)

Building blocks:
  ChannelAttention: AdaptiveAvgPool2d(1)->Linear(C,C//16)->ReLU
                    ->Linear(C//16,C)->Sigmoid->multiply input
  RCAB: Conv3x3->ReLU->Conv3x3->ChannelAttention->+residual
  ResidualGroup: num_rcab RCABs + Conv3x3 -> +long skip connection
  RCAN: HeadConv -> num_groups ResidualGroups -> global skip
        -> Conv3x3 -> PixelShuffle(4) -> TailConv -> clamp [0,1]

NO BatchNorm anywhere. PixelShuffle for upscaling. Clamp output [0,1].

---

## scripts/dataset.py — ALREADY BUILT
Class SARDataset(lr_dir, hr_dir, file_list, augment=False,
                 speckle=False, speckle_prob=0.5, num_looks=4)
Training subset: sample 15,000 from train.txt with random.seed(42).
augment=True: hflip, vflip, rot90 — SAME transform on both lr and hr.
speckle=True + prob fires: lr = lr * Gamma(4, 0.25), clip [0,1], LR ONLY.
augment=False, speckle=False for val and test always.

---

## scripts/losses.py — ALREADY BUILT
Verify it has:
  - nn.L1Loss() for Phase 1
  - CombinedLoss(alpha=0.7) for Phase 2
  - Differentiable SSIM using F.conv2d with Gaussian window size 11

---

## scripts/train.py — ALREADY BUILT
Args: --model (srcnn|rcan), --epochs, --batch_size, --phase (1|2)
Phase 1: lr_dir=patches_v1/lr, hr_dir=patches_v1/hr, L1Loss, no speckle
Phase 2: lr_dir=patches_v2/lr, hr_dir=patches_v2/hr, CombinedLoss, speckle=True
Training subset: 15,000 patches sampled from train.txt (seed=42)
Optimizer: Adam lr=1e-4 weight_decay=1e-5
Scheduler: StepLR step_size=20 gamma=0.5  (step_size=20 for 50 epochs)
Validate every 5 epochs. Save best val PSNR checkpoint.
Checkpoint dict: epoch, model_state, val_psnr, val_ssim, args, phase.
Checkpoint name: checkpoints/phase{N}/{model}_best_psnr{X.XX}.pth

---

## scripts/evaluate.py — NEEDS TO BE BUILT
Args: --srcnn_ckpt, --rcan_v1_ckpt, --rcan_v2_ckpt
Load test.txt patches. Evaluate on FULL scenes using overlapping patches.

Overlapping patch evaluation (critical for fair comparison with Ahn et al.):
  - Slice each test scene into 64x64 LR patches with stride=32 (overlapping)
  - Run each patch through model → 256x256 SR patch
  - Stitch with stride=128 (output stride), average overlapping regions
  - Compute PSNR + SSIM on full stitched image vs full HR ground truth
  - Use skimage with data_range=1.0

Also compute bicubic baseline with F.interpolate(scale_factor=4, mode='bicubic').
Print formatted 5-row table. Save results/metrics_table.csv.

Expected output format:
  ============================================================
    Model          PSNR (dB)      SSIM    vs Bicubic
  ============================================================
    Bicubic            27.82     0.781    baseline
    SRCNN              29.91     0.833    +2.09 dB
    RCAN Phase 1       31.50     0.880    +3.68 dB
    RCAN Phase 2       32.80     0.902    +4.98 dB
  ============================================================

---

## scripts/visualize.py — NEEDS TO BE BUILT
Args: --rcan_v1_ckpt, --rcan_v2_ckpt, --n_samples 10
4-panel figure per sample:
  [LR Input 64x64] | [Bicubic 256x256] | [RCAN Output 256x256] | [HR Ground Truth 256x256]
Show PSNR value as subtitle under each panel.
Save results/comparison_XX.png for each sample.
Save results/training_curves.png (loss + val PSNR over epochs for both phases).

---

## notebooks/explore.ipynb — BUILD IF NOT EXISTS
One notebook, run once after preprocessing, never for training.
6 cells:
  1. Load raw .tif, print shape/dtype/range, imshow
  2. Load patches_v1 pair, check shapes/range, plot side by side
  3. Load patches_v2 pair, compare v1 vs v2 contrast distribution
  4. Test SARDataset class, check tensor shapes, verify speckle fires
  5. Test SRCNN and RCAN forward pass — confirm output (1,1,256,256)
  6. Print train/val/test patch counts from splits

---

## Execution Order From Here

  # Step 1: Generate Phase 2 patches (log transform version)
  python scripts/preprocess.py --version 2

  # Step 2: Train RCAN Phase 1 (baseline, ~50 min)
  python scripts/train.py --model rcan --epochs 50 --batch_size 16 --phase 1

  # Step 3: Train RCAN Phase 2 (improvements, ~50 min)
  python scripts/train.py --model rcan --epochs 50 --batch_size 16 --phase 2

  # Step 4: Evaluate all models on test set
  python scripts/evaluate.py \
    --srcnn_ckpt   checkpoints/phase1/srcnn_best*.pth \
    --rcan_v1_ckpt checkpoints/phase1/rcan_best*.pth \
    --rcan_v2_ckpt checkpoints/phase2/rcan_best*.pth

  # Step 5: Generate visualizations for report
  python scripts/visualize.py \
    --rcan_v1_ckpt checkpoints/phase1/rcan_best*.pth \
    --rcan_v2_ckpt checkpoints/phase2/rcan_best*.pth

---

## Expected Results
  Bicubic              ~27-28 dB  ~0.78  no-AI baseline
  SRCNN                ~29-30 dB  ~0.83  already trained ✅
  RCAN Phase 1         ~30-31 dB  ~0.87  matches Ahn et al. level
  RCAN Phase 2         ~31-32 dB  ~0.90  beats Ahn et al. — our contribution

---

## Hard Rules — Never Break
- patches_v1 (no log) for Phase 1. patches_v2 (log) for Phase 2. Never mix.
- Log transform BEFORE clip and normalize, never after
- Clamp all model outputs to [0,1] before any metric
- No BatchNorm in RCAN
- data_range=1.0 in all skimage PSNR/SSIM calls
- Augmentation (including speckle) ONLY during training, never val or test
- Speckle on LR only, never HR, prob=0.5 not every batch
- Evaluation uses overlapping patches stride=32, average overlapping regions
- Test set untouchable until final evaluate.py run
- Seed=42 everywhere
- Never use ImageNet normalization — single channel SAR not RGB
- RCAN uses num_groups=5, num_rcab=10 (hardware-constrained reduced config)
- Training uses 15,000 patch subset sampled with seed=42 from train.txt

---

## Paper We Are Beating
Ahn et al. (2025) "SAR Super-Resolution Using Capella Satellite Imagery"
ISPRS Geospatial Week, Dubai
DOI: https://doi.org/10.5194/isprs-archives-XLVIII-G-2025-87-2025
They used: SRCNN, RCAN, SwinIR, Restormer. 4x A6000 GPUs. 300K iterations.
They did NOT use: log preprocessing, combined loss, or speckle augmentation.
We beat them with: RCAN + those 3 SAR-specific improvements.