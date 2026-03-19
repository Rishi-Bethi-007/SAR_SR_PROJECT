"""
scripts/train.py — Training script for Phase 1 and Phase 2.

Usage:
    # Phase 1 — SRCNN baseline
    python scripts/train.py --model srcnn --epochs 50 --batch_size 16 --phase 1

    # Phase 1 — RCAN baseline
    python scripts/train.py --model rcan  --epochs 50 --batch_size 16 --phase 1

    # Phase 2 — RCAN improved
    python scripts/train.py --model rcan  --epochs 50 --batch_size 16 --phase 2

Phase config:
    Phase 1: patches_v1, L1Loss, no speckle, checkpoints/phase1/
    Phase 2: patches_v2, CombinedLoss(0.7), speckle=True, checkpoints/phase2/

Training config:
    15,000 patches sampled from train.txt with seed=42
    Adam lr=1e-4, weight_decay=1e-5
    StepLR step_size=20, gamma=0.5
    Validate every 5 epochs, save best val PSNR checkpoint.
    Early stopping: --patience N val checks (default 10 = 50 epochs).

Logs are written to logs/phase{N}_{model}.log (appended each run).
"""

import argparse
import glob as glob_module
import logging
import os
import random
import sys
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.losses import L1Loss, CombinedLoss
from scripts.dataset import SARDataset
from models.srcnn import SRCNN
from models.rcan import RCAN

SEED          = 42
TRAIN_SUBSET  = 15_000   # patches sampled from train.txt


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(model_name: str, phase: int) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/phase{phase}_{model_name}.log"

    logger = logging.getLogger(f"phase{phase}_{model_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if model_name == "srcnn":
        return SRCNN().to(device)
    if model_name == "rcan":
        return RCAN().to(device)
    raise ValueError(f"Unknown model: {model_name}")


def build_loss(phase: int, device: torch.device) -> torch.nn.Module:
    if phase == 1:
        return L1Loss().to(device)
    return CombinedLoss(alpha=0.7).to(device)


def build_loaders(phase: int, batch_size: int) -> tuple[DataLoader, DataLoader]:
    patch_root = f"data/patches_v{phase}"
    lr_dir     = os.path.join(patch_root, "lr")
    hr_dir     = os.path.join(patch_root, "hr")
    splits_dir = "data/splits"

    def _load_names(split: str) -> list[str]:
        path = os.path.join(splits_dir, f"{split}.txt")
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    all_train = _load_names("train")
    val_names = _load_names("val")

    # Sample 15,000 patches from train.txt (reproducible with seed=42)
    rng = random.Random(SEED)
    train_names = rng.sample(all_train, min(TRAIN_SUBSET, len(all_train)))

    use_speckle = (phase == 2)

    train_ds = SARDataset(lr_dir, hr_dir, train_names, augment=True,  speckle=use_speckle)
    val_ds   = SARDataset(lr_dir, hr_dir, val_names,   augment=False, speckle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    psnr_sum = ssim_sum = 0.0
    n = 0
    for lr_t, hr_t in loader:
        lr_t, hr_t = lr_t.to(device), hr_t.to(device)
        pred = model(lr_t).clamp(0.0, 1.0)
        for i in range(pred.shape[0]):
            p_np = pred[i, 0].cpu().numpy()
            h_np = hr_t[i, 0].cpu().numpy()
            psnr_sum += calc_psnr(h_np, p_np, data_range=1.0)
            ssim_sum += calc_ssim(h_np, p_np, data_range=1.0)
            n += 1
    return psnr_sum / n, ssim_sum / n


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    val_psnr: float,
    val_ssim: float,
    args: argparse.Namespace,
    ckpt_dir: str,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    # Remove previous best for this model before saving new one
    for old in glob_module.glob(os.path.join(ckpt_dir, f"{args.model}_best_psnr*.pth")):
        os.remove(old)
    path = os.path.join(ckpt_dir, f"{args.model}_best_psnr{val_psnr:.2f}.pth")
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_psnr":        val_psnr,
            "val_ssim":        val_ssim,
            "args":            vars(args),
            "phase":           args.phase,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAR SR model.")
    parser.add_argument("--model",      choices=["srcnn", "rcan"], required=True)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--phase",      type=int, choices=[1, 2], required=True)
    parser.add_argument("--patience",   type=int, default=10,
                        help="Early stopping: val checks with no improvement before stopping. "
                             "Each check = 5 epochs. Default 10 = 50 epochs. 0 = disabled.")
    parser.add_argument("--resume",     default=None,
                        help="Path to checkpoint to resume from. Restores model weights, "
                             "optimizer state, scheduler state, and epoch counter.")
    parser.add_argument("--step_size",  type=int, default=20,
                        help="StepLR decay period in epochs (default 20). Use 15 for extended runs.")
    args = parser.parse_args()

    logger = setup_logger(args.model, args.phase)
    set_seed(SEED)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    logger.info(f"{'='*60}")
    logger.info(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {device} ({gpu_name})")
    logger.info(f"Model: {args.model}  |  Phase: {args.phase}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  StepLR step_size: {args.step_size}")

    model     = build_model(args.model, device)
    criterion = build_loss(args.phase, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_loader, val_loader = build_loaders(args.phase, args.batch_size)
    ckpt_dir = f"checkpoints/phase{args.phase}"

    # --- Resume (must happen before scheduler is built) ---
    start_epoch      = 1
    best_psnr        = -1.0
    resume_ckpt_data = None
    if args.resume:
        resume_ckpt_data = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt_data["model_state"])
        if "optimizer_state" in resume_ckpt_data:
            optimizer.load_state_dict(resume_ckpt_data["optimizer_state"])
        start_epoch = resume_ckpt_data["epoch"] + 1
        best_psnr   = resume_ckpt_data.get("val_psnr", -1.0)

    # --- Scheduler (built after resume so state can be patched if needed) ---
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    if resume_ckpt_data is not None and "scheduler_state" in resume_ckpt_data:
        # New-style checkpoint: full scheduler state available.
        scheduler.load_state_dict(resume_ckpt_data["scheduler_state"])
        logger.info(f"Resumed from {args.resume}  (epoch {resume_ckpt_data['epoch']}, "
                    f"best_psnr={best_psnr:.2f} dB) -> starting at epoch {start_epoch}")
        logger.info(f"  Scheduler state restored. lr={scheduler.get_last_lr()[0]:.6f}")
    elif resume_ckpt_data is not None:
        # Old-style checkpoint: no scheduler_state saved.
        # Manually compute the correct LR based on how many step_size=20 decays
        # should have occurred by the resumed epoch, then patch scheduler state.
        resumed_epoch = resume_ckpt_data["epoch"]
        correct_lr = 1e-4 * (0.5 ** (resumed_epoch // 20))  # original step_size=20
        for g in optimizer.param_groups:
            g["lr"] = correct_lr
        scheduler.last_epoch = resumed_epoch   # StepLR.get_lr() is multiplicative:
        scheduler._last_lr   = [correct_lr]   # next decay at resumed_epoch + step_size
        logger.info(f"Resumed from {args.resume}  (epoch {resumed_epoch}, "
                    f"best_psnr={best_psnr:.2f} dB) -> starting at epoch {start_epoch}")
        logger.info(f"  No scheduler_state in checkpoint. "
                    f"Manually set lr={correct_lr:.6f} "
                    f"(next decay at epoch {resumed_epoch + args.step_size}).")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")
    logger.info(f"Train patches: {TRAIN_SUBSET}  |  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    logger.info(f"Early stopping patience: {args.patience} val checks ({args.patience * 5} epochs)  [0=disabled]")
    logger.info(f"Log: logs/phase{args.phase}_{args.model}.log")
    logger.info(f"{'='*60}")

    no_improve_count = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} (start={start_epoch})", leave=False)
        for lr_t, hr_t in pbar:
            lr_t, hr_t = lr_t.to(device), hr_t.to(device)
            optimizer.zero_grad()
            pred = model(lr_t)
            loss = criterion(pred, hr_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == args.epochs:
            val_psnr, val_ssim = validate(model, val_loader, device)
            msg = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f} | "
                f"PSNR {val_psnr:.2f} dB | SSIM {val_ssim:.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.6f}"
            )
            logger.info(msg)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                no_improve_count = 0
                ckpt_path = save_checkpoint(model, optimizer, scheduler, epoch, val_psnr, val_ssim, args, ckpt_dir)
                logger.info(f"  -> NEW BEST ({val_psnr:.2f} dB) -> {ckpt_path}")
            else:
                no_improve_count += 1
                logger.info(f"  -> No improvement ({no_improve_count}/{args.patience} patience)")
                if args.patience > 0 and no_improve_count >= args.patience:
                    logger.info(f"  -> Early stopping after {args.patience} checks without gain.")
                    break
        else:
            logger.info(f"Epoch {epoch:4d} | loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.6f}")

    logger.info(f"{'='*60}")
    logger.info(f"Training complete. Best val PSNR: {best_psnr:.2f} dB")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
