"""
scripts/preprocess.py — SAR patch extraction for Phase 1 and Phase 2.

Usage:
    python scripts/preprocess.py --version 1   # simple normalize -> data/patches_v1/
    python scripts/preprocess.py --version 2   # log + normalize  -> data/patches_v2/

Both versions:
  - Stride-256 non-overlapping 256x256 HR patches
  - Skip patch if >50% pixels are near-zero (<=1e-3 on raw data)
  - PIL bicubic downsample HR -> 64x64 LR
  - 80/10/10 train/val/test split, seed=42
  - Saves data/splits/train.txt, val.txt, test.txt (shared between versions)
  - Saves data/scene_splits/{train,val,test}_scenes.txt (scene-level splits for evaluate.py)
  - Saves data/patch_manifest.json: {patch_name: scene_basename}

Phase 1 preprocessing:
  1. Replace invalid (<=0 or NaN) with 1e-6
  2. Clip to [percentile(1), percentile(99)]
  3. Min-max normalize to [0, 1]

Phase 2 preprocessing:
  1. Replace invalid (<=0 or NaN) with 1e-6
  2. np.log1p(img)
  3. Clip to [percentile(1), percentile(99)]
  4. Min-max normalize to [0, 1]
"""

import argparse
import json
import os
import random

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm


PATCH_SIZE = 256
LR_SIZE = 64
STRIDE = 256
NEAR_ZERO_THRESH = 1e-3
NEAR_ZERO_MAX_FRAC = 0.5
SEED = 42

RAW_DIR = os.path.join("data", "raw", "capella_geo")
SPLITS_DIR = os.path.join("data", "splits")
SCENE_SPLITS_DIR = os.path.join("data", "scene_splits")
MANIFEST_PATH = os.path.join("data", "patch_manifest.json")


def get_tif_files(raw_dir: str) -> list[str]:
    files = [
        os.path.join(raw_dir, f)
        for f in sorted(os.listdir(raw_dir))
        if f.endswith(".tif") and "_preview" not in f
    ]
    return files


def load_scene(path: str) -> np.ndarray:
    """Load a GeoTIFF scene as float32 2D array."""
    with rasterio.open(path) as src:
        img = src.read(1).astype(np.float32)
    return img


def preprocess_patch(patch: np.ndarray, version: int) -> np.ndarray:
    """Apply phase-specific preprocessing to a raw patch."""
    # Step 1: replace invalid values
    patch = patch.copy()
    patch[~np.isfinite(patch)] = 1e-6
    patch[patch <= 0] = 1e-6

    # Step 2 (Phase 2 only): log transform
    if version == 2:
        patch = np.log1p(patch)

    # Step 3: clip to [p1, p99]
    p1 = np.percentile(patch, 1)
    p99 = np.percentile(patch, 99)
    patch = np.clip(patch, p1, p99)

    # Step 4: min-max normalize to [0, 1]
    pmin, pmax = patch.min(), patch.max()
    if pmax - pmin > 1e-8:
        patch = (patch - pmin) / (pmax - pmin)
    else:
        patch = np.zeros_like(patch)

    return patch.astype(np.float32)


def make_lr(hr_patch: np.ndarray) -> np.ndarray:
    """Bicubic downsample a (256,256) HR patch to (64,64) LR using PIL."""
    pil_img = Image.fromarray(hr_patch)
    pil_lr = pil_img.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)
    lr = np.array(pil_lr, dtype=np.float32)
    return np.clip(lr, 0.0, 1.0)


def extract_patches(scene: np.ndarray, version: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract stride-256 HR/LR patch pairs from a scene."""
    h, w = scene.shape
    patches = []

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            raw_patch = scene[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            # Skip patches with >50% near-zero pixels (on raw data)
            near_zero_frac = np.mean(raw_patch <= NEAR_ZERO_THRESH)
            if near_zero_frac > NEAR_ZERO_MAX_FRAC:
                continue

            hr = preprocess_patch(raw_patch, version)
            lr = make_lr(hr)
            patches.append((hr, lr))

    return patches


def save_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    hr_dir: str,
    lr_dir: str,
    offset: int,
) -> list[str]:
    """Save patches to disk. Returns list of patch filenames (e.g. patch_000001)."""
    names = []
    for i, (hr, lr) in enumerate(patches):
        idx = offset + i + 1
        name = f"patch_{idx:06d}"
        np.save(os.path.join(hr_dir, name + ".npy"), hr)
        np.save(os.path.join(lr_dir, name + ".npy"), lr)
        names.append(name)
    return names


def write_splits(all_names: list[str], splits_dir: str) -> None:
    """Write 80/10/10 train/val/test patch-level split files (shared between versions)."""
    rng = random.Random(SEED)
    shuffled = all_names[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    os.makedirs(splits_dir, exist_ok=True)
    for split_name, split_list in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(splits_dir, f"{split_name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(split_list) + "\n")
        print(f"  {split_name}: {len(split_list)} patches -> {path}")


def write_scene_splits(tif_files: list[str], scene_splits_dir: str) -> None:
    """Write 80/10/10 scene-level split files for full-scene evaluation."""
    basenames = [os.path.basename(p) for p in tif_files]
    rng = random.Random(SEED)
    shuffled = basenames[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    os.makedirs(scene_splits_dir, exist_ok=True)
    for split_name, split_list in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(scene_splits_dir, f"{split_name}_scenes.txt")
        with open(path, "w") as f:
            f.write("\n".join(split_list) + "\n")
        print(f"  {split_name} scenes: {len(split_list)} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SAR scenes into patches.")
    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2],
        required=True,
        help="1 = simple normalize (Phase 1), 2 = log+normalize (Phase 2)",
    )
    args = parser.parse_args()

    patch_root = f"data/patches_v{args.version}"
    hr_dir = os.path.join(patch_root, "hr")
    lr_dir = os.path.join(patch_root, "lr")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    tif_files = get_tif_files(RAW_DIR)
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {RAW_DIR}")

    print(f"Phase {args.version} preprocessing — {len(tif_files)} scenes found")
    print(f"Output: {patch_root}")

    all_names: list[str] = []
    manifest: dict[str, str] = {}  # patch_name -> scene_basename
    total_patches = 0

    for scene_path in tqdm(tif_files, desc="Scenes"):
        scene_base = os.path.basename(scene_path)
        scene = load_scene(scene_path)
        patches = extract_patches(scene, args.version)
        names = save_patches(patches, hr_dir, lr_dir, offset=total_patches)
        for name in names:
            manifest[name] = scene_base
        all_names.extend(names)
        total_patches += len(patches)
        tqdm.write(f"  {scene_base}: {len(patches)} patches")

    print(f"\nTotal patches saved: {total_patches}")

    # Write patch-level splits only once (shared between v1 and v2)
    splits_exist = all(
        os.path.exists(os.path.join(SPLITS_DIR, f"{s}.txt"))
        for s in ("train", "val", "test")
    )
    if splits_exist:
        print(f"\nPatch splits already exist in {SPLITS_DIR}, skipping.")
    else:
        print(f"\nGenerating 80/10/10 patch splits in {SPLITS_DIR} ...")
        write_splits(all_names, SPLITS_DIR)

    # Write scene-level splits only once
    scene_splits_exist = all(
        os.path.exists(os.path.join(SCENE_SPLITS_DIR, f"{s}_scenes.txt"))
        for s in ("train", "val", "test")
    )
    if scene_splits_exist:
        print(f"Scene splits already exist in {SCENE_SPLITS_DIR}, skipping.")
    else:
        print(f"\nGenerating scene-level splits in {SCENE_SPLITS_DIR} ...")
        write_scene_splits(tif_files, SCENE_SPLITS_DIR)

    # Write manifest (overwrite each run — same content for v1 and v2)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {MANIFEST_PATH} ({len(manifest)} entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
