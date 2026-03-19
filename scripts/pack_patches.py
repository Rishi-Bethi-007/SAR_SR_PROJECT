"""
scripts/pack_patches.py — Pack individual .npy patch files into single arrays.

Reads all the individual patch_XXXXXX.npy files and concatenates them into
one array per split per type, saved as:
    data/patches_v{N}/train_lr.npy   (n_train, 64, 64)
    data/patches_v{N}/train_hr.npy   (n_train, 256, 256)
    data/patches_v{N}/val_lr.npy     (n_val, 64, 64)
    data/patches_v{N}/val_hr.npy     (n_val, 256, 256)

This eliminates per-file open overhead during training (~77K file opens/epoch
reduced to 0). SARDataset auto-detects and uses packed files when present.

Usage:
    python scripts/pack_patches.py --version 1
    python scripts/pack_patches.py --version 2
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLITS_DIR = "data/splits"
HR_SIZE = 256
LR_SIZE = 64


def load_names(split: str) -> list[str]:
    with open(os.path.join(SPLITS_DIR, f"{split}.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def pack_split(
    split: str,
    lr_dir: str,
    hr_dir: str,
    out_dir: str,
) -> None:
    names = load_names(split)
    n = len(names)

    lr_out = os.path.join(out_dir, f"{split}_lr.npy")
    hr_out = os.path.join(out_dir, f"{split}_hr.npy")

    lr_mb = n * LR_SIZE * LR_SIZE * 4 / 1e6
    hr_mb = n * HR_SIZE * HR_SIZE * 4 / 1e6
    print(f"\n  {split}: {n} patches  |  LR {lr_mb:.0f} MB  |  HR {hr_mb:.0f} MB")

    # Allocate output arrays
    lr_arr = np.empty((n, LR_SIZE, LR_SIZE), dtype=np.float32)
    hr_arr = np.empty((n, HR_SIZE, HR_SIZE), dtype=np.float32)

    for i, name in enumerate(tqdm(names, desc=f"  Packing {split}")):
        lr_arr[i] = np.load(os.path.join(lr_dir, name + ".npy"))
        hr_arr[i] = np.load(os.path.join(hr_dir, name + ".npy"))

    print(f"  Saving {lr_out} ...")
    np.save(lr_out, lr_arr)
    print(f"  Saving {hr_out} ...")
    np.save(hr_out, hr_arr)
    print(f"  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack patches into split-level arrays.")
    parser.add_argument("--version", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()

    patch_root = f"data/patches_v{args.version}"
    lr_dir = os.path.join(patch_root, "lr")
    hr_dir = os.path.join(patch_root, "hr")

    print(f"Packing patches_v{args.version} ...")

    for split in ("train", "val"):
        pack_split(split, lr_dir, hr_dir, patch_root)

    print(f"\nAll packed files saved to {patch_root}/")
    print("SARDataset will automatically detect and use them.")


if __name__ == "__main__":
    main()
