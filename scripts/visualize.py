"""
scripts/visualize.py — Comparison figures for SAR SR report.

Produces:
  - results/comparison_NNN.png  (5-panel: LR | Bicubic | RCAN v1 | RCAN v2 | HR)
  - results/training_curves.png (loss + val PSNR for both RCAN phases)

Usage:
    python scripts/visualize.py \
        --rcan_v1_ckpt checkpoints/phase1/rcan_best_psnr*.pth \
        --rcan_v2_ckpt checkpoints/phase2/rcan_best_psnr*.pth \
        --n_samples 10
"""

import argparse
import glob as glob_module
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as calc_psnr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rcan import RCAN
from scripts.dataset import SARDataset

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_ckpt(path: str) -> str:
    if "*" in path:
        matches = sorted(glob_module.glob(path))
        if not matches:
            raise FileNotFoundError(f"No checkpoint found matching: {path}")
        return matches[-1]
    return path


def load_rcan(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt_path = resolve_ckpt(ckpt_path)
    model = RCAN()
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  Loaded RCAN from {os.path.basename(ckpt_path)}")
    return model


def psnr_val(pred: np.ndarray, gt: np.ndarray) -> float:
    return calc_psnr(gt, np.clip(pred, 0.0, 1.0), data_range=1.0)


def show_img(ax: "plt.Axes", img: np.ndarray, title: str) -> None:
    """Display image with per-panel min-max stretch so every panel looks natural."""
    vmin, vmax = img.min(), img.max()
    disp = (img - vmin) / (vmax - vmin) if vmax - vmin > 1e-8 else np.zeros_like(img)
    ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Comparison figures
# ---------------------------------------------------------------------------

def make_comparison_figures(
    rcan_v1: torch.nn.Module,
    rcan_v2: torch.nn.Module,
    n_samples: int,
    device: torch.device,
) -> None:
    with open("data/splits/test.txt") as f:
        test_names = [l.strip() for l in f if l.strip()][:n_samples]

    # Two datasets over the SAME patch names — same scenes, different preprocessing.
    # v1: linear domain (no log)  — for Bicubic and RCAN Phase 1
    # v2: log domain              — for RCAN Phase 2
    ds_v1 = SARDataset(
        lr_dir="data/patches_v1/lr",
        hr_dir="data/patches_v1/hr",
        file_list=test_names,
        augment=False,
        speckle=False,
    )
    ds_v2 = SARDataset(
        lr_dir="data/patches_v2/lr",
        hr_dir="data/patches_v2/hr",
        file_list=test_names,
        augment=False,
        speckle=False,
    )

    for i, ((lr_v1_t, hr_v1_t), (lr_v2_t, hr_v2_t)) in enumerate(zip(ds_v1, ds_v2)):
        # v1 tensors → Phase 1 models
        lr_v1_in = lr_v1_t.unsqueeze(0).to(device)   # (1,1,64,64)
        hr_v1_np = hr_v1_t[0].numpy()                 # (256,256)
        lr_v1_np = lr_v1_t[0].numpy()                 # (64,64)

        # v2 tensors → Phase 2 model
        lr_v2_in = lr_v2_t.unsqueeze(0).to(device)
        hr_v2_np = hr_v2_t[0].numpy()

        # Bicubic (v1 domain)
        bic_t  = F.interpolate(lr_v1_in, scale_factor=4, mode="bicubic", align_corners=False)
        bic_np = bic_t[0, 0].cpu().numpy().clip(0.0, 1.0)

        # RCAN Phase 1 (v1 domain)
        with torch.no_grad():
            sr_v1_np = rcan_v1(lr_v1_in)[0, 0].cpu().numpy()

        # RCAN Phase 2 (v2 domain — correct inputs)
        with torch.no_grad():
            sr_v2_np = rcan_v2(lr_v2_in)[0, 0].cpu().numpy()

        # PSNR computed in each model's own domain
        psnr_bic  = psnr_val(bic_np,   hr_v1_np)
        psnr_v1   = psnr_val(sr_v1_np, hr_v1_np)
        psnr_v2   = psnr_val(sr_v2_np, hr_v2_np)

        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        show_img(axes[0], lr_v1_np,  "LR Input\n(64×64, v1)")
        show_img(axes[1], bic_np,    f"Bicubic\n{psnr_bic:.2f} dB (v1)")
        show_img(axes[2], sr_v1_np,  f"RCAN Phase 1\n{psnr_v1:.2f} dB (v1)")
        show_img(axes[3], sr_v2_np,  f"RCAN Phase 2\n{psnr_v2:.2f} dB (v2)")
        show_img(axes[4], hr_v1_np,  "HR Ground Truth\n(256×256, v1)")

        plt.suptitle(f"Sample {i+1:03d}  —  {test_names[i]}", fontsize=10)
        plt.tight_layout()
        out_path = f"results/comparison_{i+1:03d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for phase in (1, 2):
        log_path = f"logs/phase{phase}_rcan.log"
        if not os.path.exists(log_path):
            print(f"  [training_curves] {log_path} not found, skipping phase {phase}.")
            continue

        epochs, losses, val_epochs, val_psnrs = [], [], [], []
        with open(log_path) as f:
            for line in f:
                # Val epoch line: "Epoch   N | loss X.XXXX | PSNR XX.XX dB | ..."
                m = re.search(r"Epoch\s+(\d+) \| loss ([\d.]+) \| PSNR ([\d.]+) dB", line)
                if m:
                    ep = int(m.group(1))
                    epochs.append(ep)
                    losses.append(float(m.group(2)))
                    val_epochs.append(ep)
                    val_psnrs.append(float(m.group(3)))
                    continue
                # Non-val epoch line: "Epoch   N | loss X.XXXX | lr ..."
                m = re.search(r"Epoch\s+(\d+) \| loss ([\d.]+) \| lr", line)
                if m:
                    epochs.append(int(m.group(1)))
                    losses.append(float(m.group(2)))

        label = f"RCAN Phase {phase}"
        if epochs:
            ax1.plot(epochs, losses, label=label)
        if val_epochs:
            ax2.plot(val_epochs, val_psnrs, marker="o", markersize=4, label=label)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val PSNR (dB)")
    ax2.set_title("Validation PSNR")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "results/training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SR comparison figures.")
    parser.add_argument("--rcan_v1_ckpt", required=True)
    parser.add_argument("--rcan_v2_ckpt", required=True)
    parser.add_argument("--n_samples",    type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading models...")
    rcan_v1 = load_rcan(args.rcan_v1_ckpt, device)
    rcan_v2 = load_rcan(args.rcan_v2_ckpt, device)

    print(f"\nGenerating {args.n_samples} comparison figures...")
    make_comparison_figures(rcan_v1, rcan_v2, args.n_samples, device)

    print("\nGenerating training curves...")
    plot_training_curves()

    print("\nDone. All outputs saved to results/")


if __name__ == "__main__":
    main()
