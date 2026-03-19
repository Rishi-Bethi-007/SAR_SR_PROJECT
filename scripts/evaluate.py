"""
scripts/evaluate.py — Full-scene overlapping patch evaluation.

Evaluates all models on held-out test scenes and prints a comparison table.

Usage:
    python scripts/evaluate.py \
        --srcnn_ckpt   checkpoints/phase1/srcnn_best_psnr*.pth \
        --rcan_v1_ckpt checkpoints/phase1/rcan_best_psnr*.pth \
        --rcan_v2_ckpt checkpoints/phase2/rcan_best_psnr*.pth

Overlapping patch evaluation (per-patch normalization):
  - Slide 256x256 RAW HR patches with stride=128 (HR space) over each test scene
  - For each raw HR patch: apply per-patch preprocessing (matching preprocess.py)
      v1: clip [p1,p99] -> min-max normalize
      v2: log1p -> clip [p1,p99] -> min-max normalize
  - Bicubic-downsample the normalized HR patch -> 64x64 LR patch
  - Run LR through model (or bicubic) -> 256x256 SR patch
  - Stitch SR patches AND normalized HR patches with stride=128, average overlaps
  - Compute PSNR + SSIM on stitched SR vs stitched HR (data_range=1.0)

Per-patch normalization ensures model inputs match the training distribution
exactly, giving PSNR numbers consistent with training validation metrics.
"""

import argparse
import csv
import glob as glob_module
import os
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srcnn import SRCNN
from models.rcan import RCAN

HR_PATCH  = 256          # HR patch size (= SR output size)
LR_PATCH  = 64           # LR patch size (= model input size)
HR_STRIDE = 128          # stride in HR space (= LR stride 32 × 4)


# ---------------------------------------------------------------------------
# Patch-level utilities  (mirror of preprocess.py)
# ---------------------------------------------------------------------------

def preprocess_patch(raw_patch: np.ndarray, version: int) -> np.ndarray:
    """Per-patch normalization — identical pipeline to preprocess.py.

    v1: replace invalid -> clip [p1,p99] -> min-max normalize
    v2: replace invalid -> log1p -> clip [p1,p99] -> min-max normalize
    """
    patch = raw_patch.copy()
    patch[~np.isfinite(patch)] = 1e-6
    patch[patch <= 0] = 1e-6
    if version == 2:
        patch = np.log1p(patch)
    p1  = np.percentile(patch, 1)
    p99 = np.percentile(patch, 99)
    patch = np.clip(patch, p1, p99)
    pmin, pmax = patch.min(), patch.max()
    if pmax - pmin > 1e-8:
        patch = (patch - pmin) / (pmax - pmin)
    else:
        patch = np.zeros_like(patch)
    return patch.astype(np.float32)


def make_lr_from_hr(hr_patch: np.ndarray) -> np.ndarray:
    """Bicubic-downsample a normalized 256x256 HR patch -> 64x64 LR.

    Identical to preprocess.py's make_lr().
    """
    pil_img = Image.fromarray(hr_patch)
    pil_lr  = pil_img.resize((LR_PATCH, LR_PATCH), Image.BICUBIC)
    return np.clip(np.array(pil_lr, dtype=np.float32), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Scene loading / metrics
# ---------------------------------------------------------------------------

def resolve_ckpt(path: str) -> str:
    if "*" in path:
        matches = sorted(glob_module.glob(path))
        if not matches:
            raise FileNotFoundError(f"No checkpoint found matching: {path}")
        return matches[-1]
    return path


def load_scene(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


MAX_EVAL_SIZE = 4096  # center-crop limit for SSIM to avoid OOM on 10k×10k scenes


def scene_metrics(sr: np.ndarray, hr: np.ndarray) -> tuple[float, float]:
    """PSNR and SSIM on stitched SR vs stitched HR.

    Center-crops to MAX_EVAL_SIZE×MAX_EVAL_SIZE to prevent OOM on full scenes.
    """
    h = min(sr.shape[0], hr.shape[0])
    w = min(sr.shape[1], hr.shape[1])
    if h > MAX_EVAL_SIZE or w > MAX_EVAL_SIZE:
        y0 = max(0, (h - MAX_EVAL_SIZE) // 2)
        x0 = max(0, (w - MAX_EVAL_SIZE) // 2)
        h  = min(h, MAX_EVAL_SIZE)
        w  = min(w, MAX_EVAL_SIZE)
        sr_c = np.clip(sr[y0:y0 + h, x0:x0 + w], 0.0, 1.0)
        hr_c = hr[y0:y0 + h, x0:x0 + w]
    else:
        sr_c = np.clip(sr[:h, :w], 0.0, 1.0)
        hr_c = hr[:h, :w]
    psnr = calc_psnr(hr_c, sr_c, data_range=1.0)
    ssim = calc_ssim(hr_c, sr_c, data_range=1.0)
    return psnr, ssim


# ---------------------------------------------------------------------------
# Per-patch overlapping inference
# ---------------------------------------------------------------------------

def infer_and_stitch(
    model,          # torch.nn.Module or None for bicubic
    raw_scene: np.ndarray,
    version: int,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping patch inference with per-patch normalization.

    Slides 256x256 windows over the raw scene with stride=HR_STRIDE (128).
    Each window is independently preprocessed to match training (preprocess.py).
    The normalized HR patches are also stitched to form the ground-truth reference,
    so SR and HR are always compared in the same per-patch-normalized space.

    model=None  ->  bicubic upsampling (F.interpolate, bicubic, scale=4).

    Returns:
        stitched_sr  (H, W) float32 in [0, 1]
        stitched_hr  (H, W) float32 in [0, 1]  — per-patch normalized reference
    """
    raw_h, raw_w = raw_scene.shape
    sr_out = np.zeros((raw_h, raw_w), dtype=np.float64)
    sr_cnt = np.zeros((raw_h, raw_w), dtype=np.float64)
    hr_out = np.zeros((raw_h, raw_w), dtype=np.float64)
    hr_cnt = np.zeros((raw_h, raw_w), dtype=np.float64)

    # Build patch grid in HR coords; always include an edge patch for full coverage
    y_pos = list(range(0, raw_h - HR_PATCH + 1, HR_STRIDE))
    if not y_pos or y_pos[-1] + HR_PATCH < raw_h:
        y_pos.append(max(0, raw_h - HR_PATCH))
    x_pos = list(range(0, raw_w - HR_PATCH + 1, HR_STRIDE))
    if not x_pos or x_pos[-1] + HR_PATCH < raw_w:
        x_pos.append(max(0, raw_w - HR_PATCH))
    coords = [(y, x) for y in y_pos for x in x_pos]

    if model is not None:
        model.eval()

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        lr_list, hr_list = [], []

        for y, x in batch_coords:
            raw_patch = raw_scene[y : y + HR_PATCH, x : x + HR_PATCH]
            hr_norm   = preprocess_patch(raw_patch, version)   # per-patch norm
            lr_norm   = make_lr_from_hr(hr_norm)               # bicubic downsample
            lr_list.append(lr_norm)
            hr_list.append(hr_norm)

        lr_stack = np.stack(lr_list)   # (B, 64, 64)

        if model is not None:
            with torch.no_grad():
                lr_t  = torch.from_numpy(lr_stack).unsqueeze(1).to(device)
                sr_t  = model(lr_t).clamp(0.0, 1.0)
                sr_np = sr_t[:, 0].cpu().numpy()
        else:
            # Bicubic baseline
            lr_t  = torch.from_numpy(lr_stack).unsqueeze(1)
            sr_t  = F.interpolate(lr_t, scale_factor=4,
                                  mode="bicubic", align_corners=False).clamp(0.0, 1.0)
            sr_np = sr_t[:, 0].numpy()

        for k, (y, x) in enumerate(batch_coords):
            sr_out[y : y + HR_PATCH, x : x + HR_PATCH] += sr_np[k]
            sr_cnt[y : y + HR_PATCH, x : x + HR_PATCH] += 1.0
            hr_out[y : y + HR_PATCH, x : x + HR_PATCH] += hr_list[k]
            hr_cnt[y : y + HR_PATCH, x : x + HR_PATCH] += 1.0

    sr_out /= np.maximum(sr_cnt, 1.0)
    hr_out /= np.maximum(hr_cnt, 1.0)
    return sr_out.astype(np.float32), hr_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    arch: str, ckpt_path: str, device: torch.device
) -> tuple[torch.nn.Module, int]:
    ckpt_path = resolve_ckpt(ckpt_path)
    model = SRCNN() if arch == "srcnn" else RCAN()
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    phase = ckpt.get("phase", 1)
    print(f"  Loaded {arch} from {os.path.basename(ckpt_path)}  "
          f"(phase={phase}, val_psnr={ckpt.get('val_psnr', 0):.2f} dB)")
    return model, phase


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full-scene evaluation of SAR SR models.")
    parser.add_argument("--srcnn_ckpt",   required=True)
    parser.add_argument("--rcan_v1_ckpt", required=True)
    parser.add_argument("--rcan_v2_ckpt", default=None,
                        help="Optional. Skip RCAN Phase 2 row if not provided.")
    parser.add_argument("--batch_size",   type=int, default=32,
                        help="Patches per GPU batch during inference (default 32).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading models...")
    srcnn,   _ = load_model("srcnn", args.srcnn_ckpt,   device)
    rcan_v1, _ = load_model("rcan",  args.rcan_v1_ckpt, device)
    rcan_v2    = None
    if args.rcan_v2_ckpt:
        rcan_v2, _ = load_model("rcan", args.rcan_v2_ckpt, device)

    raw_dir   = "data/raw/capella_geo"
    scene_txt = "data/scene_splits/test_scenes.txt"
    with open(scene_txt) as f:
        test_scenes = [l.strip() for l in f if l.strip()]
    print(f"\nTest scenes: {len(test_scenes)}")
    print("Note: per-patch normalization — each 256x256 raw patch normalized "
          "independently, matching preprocess.py training pipeline.\n")

    sums = {k: {"psnr": 0.0, "ssim": 0.0}
            for k in ["Bicubic", "SRCNN", "RCAN Phase 1",
                      "Bicubic (v2 domain)", "RCAN Phase 2"]}
    n = len(test_scenes)

    for scene_file in tqdm(test_scenes, desc="Scenes"):
        raw = load_scene(os.path.join(raw_dir, scene_file))

        # --- v1 domain: bicubic, SRCNN, RCAN Phase 1 ---
        bic_sr, hr_v1 = infer_and_stitch(None,    raw, 1, device, args.batch_size)
        p, s = scene_metrics(bic_sr, hr_v1)
        sums["Bicubic"]["psnr"] += p / n
        sums["Bicubic"]["ssim"] += s / n

        sr, _ = infer_and_stitch(srcnn,   raw, 1, device, args.batch_size)
        p, s  = scene_metrics(sr, hr_v1)
        sums["SRCNN"]["psnr"] += p / n
        sums["SRCNN"]["ssim"] += s / n

        sr, _ = infer_and_stitch(rcan_v1, raw, 1, device, args.batch_size)
        p, s  = scene_metrics(sr, hr_v1)
        sums["RCAN Phase 1"]["psnr"] += p / n
        sums["RCAN Phase 1"]["ssim"] += s / n

        # --- v2 domain: bicubic v2, RCAN Phase 2 ---
        bic_v2_sr, hr_v2 = infer_and_stitch(None,    raw, 2, device, args.batch_size)
        p, s = scene_metrics(bic_v2_sr, hr_v2)
        sums["Bicubic (v2 domain)"]["psnr"] += p / n
        sums["Bicubic (v2 domain)"]["ssim"] += s / n

        if rcan_v2 is not None:
            sr, _ = infer_and_stitch(rcan_v2, raw, 2, device, args.batch_size)
            p, s  = scene_metrics(sr, hr_v2)
            sums["RCAN Phase 2"]["psnr"] += p / n
            sums["RCAN Phase 2"]["ssim"] += s / n

    # --- Print table ---
    bic_v1_psnr = sums["Bicubic"]["psnr"]
    bic_v2_psnr = sums["Bicubic (v2 domain)"]["psnr"]

    rows = []
    for name, r in sums.items():
        if name in ("RCAN Phase 2", "Bicubic (v2 domain)") and rcan_v2 is None:
            continue
        if name == "RCAN Phase 2":
            vs_label = f"+{r['psnr'] - bic_v2_psnr:.2f} dB (vs v2 bic)"
        elif name == "Bicubic":
            vs_label = "baseline (v1)"
        elif name == "Bicubic (v2 domain)":
            vs_label = "baseline (v2)"
        else:
            vs_label = f"+{r['psnr'] - bic_v1_psnr:.2f} dB"
        rows.append((name, r["psnr"], r["ssim"], vs_label))

    col_w = max(len(r[0]) for r in rows) + 2
    print("\n" + "=" * (col_w + 46))
    print(f"  {'Model':<{col_w}}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'vs Bicubic':>20}")
    print("=" * (col_w + 46))
    for name, psnr, ssim, vs_label in rows:
        print(f"  {name:<{col_w}}  {psnr:>10.2f}  {ssim:>8.4f}  {vs_label:>20}")
    print("=" * (col_w + 46))

    os.makedirs("results", exist_ok=True)
    csv_path = "results/metrics_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "PSNR_dB", "SSIM", "vs_Bicubic_dB", "bicubic_ref"])
        for name, psnr, ssim, vs_label in rows:
            if name == "RCAN Phase 2":
                vs_val, ref = psnr - bic_v2_psnr, "v2_domain"
            elif name in ("Bicubic", "Bicubic (v2 domain)"):
                vs_val = 0.0
                ref = "v2_domain" if "v2" in name else "v1_domain"
            else:
                vs_val, ref = psnr - bic_v1_psnr, "v1_domain"
            writer.writerow([name, f"{psnr:.4f}", f"{ssim:.4f}",
                             f"{vs_val:.4f}", ref])
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
