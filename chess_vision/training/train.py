"""
Training Script – Chess Piece Classifier
=========================================

Trains a MobileNetV3-Small classifier on synthetically-rendered squares.

Two-phase training strategy:
  1. **Head-only** (backbone frozen)  – fast convergence of the new head.
  2. **Full fine-tune** (backbone unfrozen, lower LR) – adapts low-level
     features to chess-piece textures.

Usage
-----
::

    python -m chess_vision.training.train \\
        --pieces-root dataset/pieces \\
        --epochs 20 \\
        --batch-size 128 \\
        --output-dir checkpoints

Run ``python -m chess_vision.training.train --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from chess_vision.models.classifier import (
    CLASS_NAMES,
    NUM_CLASSES,
    ChessPieceClassifier,
)
from chess_vision.training.dataset import (
    ChessSquareDataset,
    get_train_transform,
    get_val_transform,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── Training loop ──────────────────────────────────────────────────────

def train_one_epoch(
    model: ChessPieceClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch; return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: ChessPieceClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate on a validation set; return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


# ── Main training procedure ───────────────────────────────────────────

def train(
    pieces_root: str,
    epochs_phase1: int = 10,
    epochs_phase2: int = 10,
    batch_size: int = 128,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    samples_per_epoch: int = 20_000,
    val_fraction: float = 0.1,
    output_dir: str = "checkpoints",
    fen_file: Optional[str] = None,
    num_workers: int = 0,
    device: str = "auto",
    resume: Optional[str] = None,
) -> Path:
    """Run the full two-phase training procedure.

    Parameters
    ----------
    pieces_root : str
        Path to the piece asset directory (e.g. ``dataset/pieces``).
    epochs_phase1 : int
        Epochs for head-only training.
    epochs_phase2 : int
        Epochs for full fine-tuning.
    batch_size : int
        Mini-batch size.
    lr_phase1, lr_phase2 : float
        Learning rates for each phase.
    samples_per_epoch : int
        Number of synthetic squares generated per epoch.
    val_fraction : float
        Fraction of samples used for validation.
    output_dir : str
        Directory to save checkpoints.
    fen_file : str, optional
        Path to a JSON file containing FEN positions (list of dicts with
        ``"fen"`` key).
    num_workers : int
        DataLoader workers (0 = main process).
    device : str
        Device to use: 'auto' (default, auto-detect), 'cpu', or 'cuda'.
    resume : str, optional
        Path to a checkpoint file to resume training from.  The checkpoint
        must have been saved by this training script (contains model state,
        optimizer state, scheduler state, epoch, phase, and best accuracy).

    Returns
    -------
    Path
        Path to the saved best checkpoint.
    """
    # Resolve device
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    log.info("Device: %s", device_obj)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load FEN list -------------------------------------------------
    fen_list = None
    if fen_file and Path(fen_file).exists():
        with open(fen_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            fen_list = [
                item["fen"].split()[0] if isinstance(item, dict) else item.split()[0]
                for item in data
            ]
        log.info("Loaded %d FEN positions from %s", len(fen_list), fen_file)

    # ---- Build dataset -------------------------------------------------
    full_dataset = ChessSquareDataset(
        pieces_root=pieces_root,
        fen_list=fen_list,
        samples_per_epoch=samples_per_epoch,
        img_size=64,
        transform=get_train_transform(64),
    )

    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Override transform for validation split
    # (random_split shares the dataset object, so we create a separate
    #  wrapper for clean val evaluation)
    val_dataset_clean = ChessSquareDataset(
        pieces_root=pieces_root,
        fen_list=fen_list,
        samples_per_epoch=val_size,
        img_size=64,
        transform=get_val_transform(64),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset_clean, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    # ---- Model ---------------------------------------------------------
    model = ChessPieceClassifier(
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        dropout=0.3,
    ).to(device_obj)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = out / "best_classifier.pt"
    resume_path = out / "last_checkpoint.pt"

    # Bookkeeping for resume: which phase and epoch to start from
    start_phase = 1
    start_epoch = 1

    # ---- Resume from checkpoint ----------------------------------------
    ckpt = None
    if resume and Path(resume).exists():
        log.info("Resuming from checkpoint: %s", resume)
        ckpt = torch.load(resume, map_location=device_obj, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            best_acc = ckpt.get("best_acc", 0.0)
            start_phase = ckpt.get("phase", 1)
            start_epoch = ckpt.get("epoch", 0) + 1  # next epoch
            # If we were in phase 2 the backbone was unfrozen
            if start_phase == 2:
                model.unfreeze_backbone()
            log.info(
                "  Restored: phase=%d  epoch=%d  best_acc=%.4f",
                start_phase, start_epoch, best_acc,
            )
        else:
            # Resume from a plain model state_dict (e.g. best_classifier.pt)
            model.load_state_dict(ckpt)
            start_phase = 1
            start_epoch = 1
            log.info("  Restored model weights only (no optimizer/scheduler state).")
    elif resume:
        log.warning("Resume path not found (%s), starting fresh.", resume)

    def _save_resume_checkpoint(
        phase: int, epoch: int, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> None:
        """Save a full training-state checkpoint for resumption."""
        torch.save({
            "phase": phase,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
        }, resume_path)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1 – train head only (backbone frozen)
    # ═══════════════════════════════════════════════════════════════════
    if start_phase <= 1:
        p1_start = start_epoch if start_phase == 1 else 1
        remaining = epochs_phase1 - p1_start + 1

        if remaining > 0:
            log.info(
                "═══ Phase 1: Head-only training (epochs %d–%d) ═══",
                p1_start, epochs_phase1,
            )
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_phase1,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_phase1,
            )

            # Restore optimizer/scheduler state when resuming inside P1
            if resume and start_phase == 1 and ckpt and isinstance(ckpt, dict):
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            for epoch in range(p1_start, epochs_phase1 + 1):
                t0 = time.time()
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device_obj)
                val_metrics = evaluate(model, val_loader, criterion, device_obj)
                scheduler.step()

                elapsed = time.time() - t0
                log.info(
                    "P1 Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
                    epoch, epochs_phase1,
                    train_loss, val_metrics["loss"], val_metrics["accuracy"],
                    elapsed,
                )

                if val_metrics["accuracy"] > best_acc:
                    best_acc = val_metrics["accuracy"]
                    torch.save(model.state_dict(), best_path)
                    log.info("  ↳ Saved best model (acc=%.4f)", best_acc)

                _save_resume_checkpoint(1, epoch, optimizer, scheduler)
        else:
            log.info("Phase 1 already complete, skipping.")

        # Reset start_epoch for phase 2
        start_epoch = 1

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2 – full fine-tuning (backbone unfrozen)
    # ═══════════════════════════════════════════════════════════════════
    p2_start = start_epoch if start_phase == 2 else 1
    remaining = epochs_phase2 - p2_start + 1

    if remaining > 0:
        log.info(
            "═══ Phase 2: Full fine-tuning (epochs %d–%d) ═══",
            p2_start, epochs_phase2,
        )
        model.unfreeze_backbone()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_phase2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_phase2,
        )

        # Restore optimizer/scheduler state when resuming inside P2
        if resume and start_phase == 2 and ckpt and isinstance(ckpt, dict):
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        for epoch in range(p2_start, epochs_phase2 + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device_obj)
            val_metrics = evaluate(model, val_loader, criterion, device_obj)
            scheduler.step()

            elapsed = time.time() - t0
            log.info(
                "P2 Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
                epoch, epochs_phase2,
                train_loss, val_metrics["loss"], val_metrics["accuracy"],
                elapsed,
            )

            if val_metrics["accuracy"] > best_acc:
                best_acc = val_metrics["accuracy"]
                torch.save(model.state_dict(), best_path)
                log.info("  ↳ Saved best model (acc=%.4f)", best_acc)

            _save_resume_checkpoint(2, epoch, optimizer, scheduler)
    else:
        log.info("Phase 2 already complete, skipping.")

    log.info("Training complete.  Best val accuracy: %.4f", best_acc)
    log.info("Checkpoint saved to: %s", best_path.resolve())
    log.info("Resume checkpoint:   %s", resume_path.resolve())
    return best_path


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train chess piece classifier on synthetic data.",
    )
    parser.add_argument(
        "--pieces-root", type=str, required=True,
        help="Path to piece image root (e.g. dataset/pieces).",
    )
    parser.add_argument("--epochs-phase1", type=int, default=10)
    parser.add_argument("--epochs-phase2", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-phase1", type=float, default=1e-3)
    parser.add_argument("--lr-phase2", type=float, default=1e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=20_000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--fen-file", type=str, default=None,
        help="JSON file with FEN positions (e.g. dataset/fen.json).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device: auto (default), cpu, or cuda.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to last_checkpoint.pt to resume training from.",
    )

    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
