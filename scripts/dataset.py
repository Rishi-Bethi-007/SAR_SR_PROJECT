"""
scripts/dataset.py — PyTorch Dataset for SAR super-resolution patches.

SARDataset(lr_dir, hr_dir, file_list, augment=False, speckle=False,
           speckle_prob=0.5, num_looks=4)

  lr_dir       : directory containing LR .npy files (64x64)
  hr_dir       : directory containing HR .npy files (256x256)
  file_list    : list of patch stem names (e.g. ['patch_000001', ...])
                 OR path to a .txt file with one name per line
  augment      : if True, apply random hflip / vflip / rot90 (same transform to LR and HR)
  speckle      : if True, apply multiplicative speckle to LR with probability speckle_prob
  speckle_prob : probability of applying speckle per sample (default 0.5)
  num_looks    : Gamma shape parameter for speckle noise (scale = 1/num_looks)

Fast mode (automatic):
  If pack_patches.py has been run, packed split files are detected automatically.
  LR is loaded entirely into RAM; HR is memory-mapped for fast random access.
"""

import os
import random
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class SARDataset(Dataset):
    def __init__(
        self,
        lr_dir:       str,
        hr_dir:       str,
        file_list:    Union[list[str], str],
        augment:      bool  = False,
        speckle:      bool  = False,
        speckle_prob: float = 0.5,
        num_looks:    int   = 4,
    ) -> None:
        super().__init__()
        self.lr_dir       = lr_dir
        self.hr_dir       = hr_dir
        self.augment      = augment
        self.speckle      = speckle
        self.speckle_prob = speckle_prob
        self.num_looks    = num_looks

        if isinstance(file_list, str):
            with open(file_list) as f:
                self.names = [line.strip() for line in f if line.strip()]
        else:
            self.names = list(file_list)

        # --- Fast mode: use packed split arrays if available ----------------
        self._lr_packed: np.ndarray | None = None
        self._hr_packed: np.ndarray | None = None

        patch_root = os.path.dirname(lr_dir)
        split_name = self._infer_split(patch_root)
        if split_name is not None:
            lr_pack = os.path.join(patch_root, f"{split_name}_lr.npy")
            hr_pack = os.path.join(patch_root, f"{split_name}_hr.npy")
            if os.path.exists(lr_pack) and os.path.exists(hr_pack):
                splits_dir = os.path.join(os.path.dirname(patch_root), "splits")
                split_txt  = os.path.join(splits_dir, f"{split_name}.txt")
                with open(split_txt) as f:
                    ordered = [line.strip() for line in f if line.strip()]
                name_to_idx = {n: i for i, n in enumerate(ordered)}

                indices = [name_to_idx.get(n, -1) for n in self.names]
                if all(i >= 0 for i in indices):
                    self._packed_indices = np.array(indices, dtype=np.int64)
                    lr_full = np.load(lr_pack, mmap_mode='r')
                    self._lr_packed = lr_full[self._packed_indices].copy()
                    self._hr_packed = np.load(hr_pack, mmap_mode='r')
                    print(
                        f"[SARDataset] Packed mode: {split_name} split, "
                        f"{len(self.names)} patches, "
                        f"LR in RAM ({self._lr_packed.nbytes/1e6:.0f} MB), "
                        f"HR memory-mapped."
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._lr_packed is not None:
            lr = self._lr_packed[idx].copy()
            hr_row = self._packed_indices[idx]
            hr = self._hr_packed[hr_row].copy().astype(np.float32)
        else:
            name = self.names[idx]
            lr = np.load(os.path.join(self.lr_dir, name + ".npy")).astype(np.float32)
            hr = np.load(os.path.join(self.hr_dir, name + ".npy")).astype(np.float32)

        if self.augment:
            lr, hr = self._augment(lr, hr)

        if self.speckle and random.random() < self.speckle_prob:
            lr = self._add_speckle(lr, self.num_looks)

        lr_t = torch.from_numpy(lr).unsqueeze(0)
        hr_t = torch.from_numpy(hr).unsqueeze(0)
        return lr_t, hr_t

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_split(self, patch_root: str) -> str | None:
        """Infer 'train'/'val'/'test' from the first patch name in our list."""
        if not self.names:
            return None
        splits_dir = os.path.join(os.path.dirname(patch_root), "splits")
        for split in ("train", "val", "test"):
            txt = os.path.join(splits_dir, f"{split}.txt")
            if not os.path.exists(txt):
                continue
            with open(txt) as f:
                first = next((l.strip() for l in f if l.strip()), None)
            if first and first == self.names[0]:
                return split
            with open(txt) as f:
                names_in_split = {l.strip() for l in f if l.strip()}
            if self.names[0] in names_in_split:
                return split
        return None

    @staticmethod
    def _augment(
        lr: np.ndarray, hr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            lr = np.fliplr(lr)
            hr = np.fliplr(hr)
        if random.random() < 0.5:
            lr = np.flipud(lr)
            hr = np.flipud(hr)
        k = random.randint(0, 3)
        if k:
            lr = np.rot90(lr, k)
            hr = np.rot90(hr, k)
        return np.ascontiguousarray(lr), np.ascontiguousarray(hr)

    @staticmethod
    def _add_speckle(lr: np.ndarray, num_looks: int = 4) -> np.ndarray:
        noise = np.random.gamma(
            shape=num_looks, scale=1.0 / num_looks, size=lr.shape
        ).astype(np.float32)
        return np.clip(lr * noise, 0.0, 1.0)
