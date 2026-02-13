"""
Chess Piece Dataset – Synthetic Board Generator
================================================

Design philosophy:
  We do **not** have labelled images; we have individual piece images
  organised by style.  This module builds a training dataset by:

    1. **Compositing** – For each training sample we pick a random board
       style (background colour pair) and a random piece style, then
       render a random FEN position onto an 8×8 board.
    2. **Square extraction** – Each of the 64 squares becomes one labelled
       sample (piece class 0–12).
    3. **On-the-fly augmentation** – Heavy augmentations simulate the
       noise encountered in real images (compression, blur, jitter,
       slight rotation / perspective, random crop).

  This lets us train the classifier purely from the asset library without
  needing manually-labelled images.

Classes:
  0  empty
  1  white_pawn    2  white_knight  3  white_bishop
  4  white_rook    5  white_queen   6  white_king
  7  black_pawn    8  black_knight  9  black_bishop
  10 black_rook    11 black_queen   12 black_king
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from chess_vision.models.classifier import (
    CLASS_NAMES,
    FEN_CHAR_TO_CLASS_IDX,
    FILE_PREFIX_TO_CLASS,
    NUM_CLASSES,
)

# ── ImageNet normalisation (used everywhere) ──────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Board background colours (light, dark) tuples in RGB ──────────────
# A small palette that roughly covers common chess.com / lichess boards.

BOARD_COLORS: list[Tuple[Tuple[int, ...], Tuple[int, ...]]] = [
    ((240, 217, 181), (181, 136, 99)),    # chess.com brown
    ((235, 236, 208), (119, 149, 86)),    # chess.com green
    ((222, 227, 230), (140, 162, 173)),   # chess.com blue/grey
    ((255, 255, 255), (86, 133, 67)),     # lichess green
    ((212, 202, 190), (100, 92, 89)),     # newspaper-ish
    ((255, 255, 230), (118, 150, 86)),    # light green
    ((240, 240, 240), (200, 200, 200)),   # greyscale
    ((255, 206, 158), (209, 139, 71)),    # warm orange/brown
]


# ── Training augmentation pipeline ────────────────────────────────────

def get_train_transform(img_size: int = 64) -> transforms.Compose:
    """Heavy augmentation pipeline for training.

    Includes:
      • RandomResizedCrop – scale/translation robustness
      • ColorJitter       – lighting / white-balance robustness
      • RandomRotation    – slight tilt (±5°)
      • RandomPerspective – perspective distortion
      • GaussianBlur      – defocus simulation
      • JPEG compression  – artifact simulation (custom)
      • ImageNet normalisation
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(degrees=5),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomApply([JPEGCompression(quality_range=(40, 95))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size: int = 64) -> transforms.Compose:
    """Deterministic transform for validation / inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Custom augmentation: JPEG compression artifacts ───────────────────

class JPEGCompression:
    """Simulate lossy JPEG compression at a random quality level."""

    def __init__(self, quality_range: Tuple[int, int] = (40, 95)) -> None:
        self.lo, self.hi = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.lo, self.hi)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# ── Piece image cache ─────────────────────────────────────────────────

class PieceImageCache:
    """Load and cache piece images from the asset directory.

    Each style folder is expected to contain files named
    ``{color}{piece}.png`` – e.g. ``wp.png``, ``bk.png``.
    """

    def __init__(self, pieces_root: str | Path) -> None:
        self.root = Path(pieces_root)
        self._cache: Dict[str, Dict[str, Image.Image]] = {}
        self._styles: List[str] = [
            d.name for d in self.root.iterdir() if d.is_dir()
        ]
        if not self._styles:
            raise FileNotFoundError(f"No piece style dirs found in {self.root}")

    @property
    def styles(self) -> List[str]:
        return self._styles

    def get(self, style: str, piece_code: str, size: int = 64) -> Image.Image:
        """Return a piece image resized to ``(size, size)``."""
        key = f"{style}/{piece_code}"
        if key not in self._cache:
            path = self.root / style / f"{piece_code}.png"
            if not path.exists():
                raise FileNotFoundError(path)
            img = Image.open(path).convert("RGBA")
            self._cache[key] = img
        return self._cache[key].resize((size, size), Image.LANCZOS)


# ── FEN helpers ────────────────────────────────────────────────────────

def fen_to_board(fen: str) -> List[List[Optional[str]]]:
    """Parse a FEN position string into an 8×8 list.

    Returns a list of 8 rows (rank 8 first), each containing 8 entries.
    Each entry is a FEN character (``'P'``, ``'k'``, …) or ``None`` for
    empty squares.
    """
    rows = fen.split()[0].split("/")
    board: List[List[Optional[str]]] = []
    for row_str in rows:
        row: List[Optional[str]] = []
        for ch in row_str:
            if ch.isdigit():
                row.extend([None] * int(ch))
            else:
                row.append(ch)
        board.append(row)
    return board


# ── Synthetic dataset ─────────────────────────────────────────────────

class ChessSquareDataset(Dataset):
    """Generates labelled square images on-the-fly from piece assets.

    Each ``__getitem__`` call:
      1. Picks a random FEN position (from a provided list or generates one).
      2. Picks a random piece style and board colour scheme.
      3. Renders the full 8×8 board as a composite image.
      4. Extracts a random square and returns (image, label).

    Parameters
    ----------
    pieces_root : str | Path
        Root directory of piece style folders.
    fen_list : list[str], optional
        List of FEN position strings to sample from.  If ``None`` a small
        default set is used.
    samples_per_epoch : int
        Virtual epoch length (number of squares returned per epoch).
    img_size : int
        Output image size (default 64).
    transform : optional
        Torchvision transform applied to each PIL square image.
    """

    # A small default set of FEN positions for diversity
    _DEFAULT_FENS: List[str] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
        "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
        "4K3/2k1P3/8/8/8/8/5r2/6R1",
        "8/8/8/4p1K1/2k1P3/8/8/8",
        "rnbq1rk1/ppp2ppp/3bpn2/3p4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1",
        "r2q1rk1/pp2ppbp/2np1np1/8/2BNP1b1/2N1BP2/PPPQ2PP/R3K2R",
        "2r3k1/pp3pp1/4p2p/3pP3/1P1P4/P4N2/5PPP/R5K1",
        "8/8/4k3/8/8/4K3/4P3/8",
    ]

    def __init__(
        self,
        pieces_root: str | Path,
        fen_list: Optional[List[str]] = None,
        samples_per_epoch: int = 10_000,
        img_size: int = 64,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.cache = PieceImageCache(pieces_root)
        self.fens = fen_list or self._DEFAULT_FENS
        self.samples_per_epoch = samples_per_epoch
        self.img_size = img_size
        self.transform = transform or get_train_transform(img_size)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return ``(image_tensor, class_index)``."""
        # 1. Random FEN → board
        fen = random.choice(self.fens)
        board = fen_to_board(fen)

        # 2. Random style & colours
        style = random.choice(self.cache.styles)
        light, dark = random.choice(BOARD_COLORS)

        # 3. Random square position
        row = random.randint(0, 7)
        col = random.randint(0, 7)
        piece_char = board[row][col]

        # 4. Determine label
        if piece_char is None:
            label = 0  # empty
        else:
            label = FEN_CHAR_TO_CLASS_IDX.get(piece_char, 0)

        # 5. Render the square
        sq_img = self._render_square(style, piece_char, row, col, light, dark)

        # 6. Apply transforms
        tensor = self.transform(sq_img)
        return tensor, label

    def _render_square(
        self,
        style: str,
        piece_char: Optional[str],
        row: int,
        col: int,
        light: Tuple[int, ...],
        dark: Tuple[int, ...],
    ) -> Image.Image:
        """Render a single board square with optional piece overlay."""
        # Background colour (checkerboard pattern)
        is_light = (row + col) % 2 == 0
        bg_color = light if is_light else dark

        sq = Image.new("RGB", (self.img_size, self.img_size), bg_color)

        if piece_char is not None:
            # Map FEN char to file prefix
            color_prefix = "w" if piece_char.isupper() else "b"
            piece_prefix = piece_char.lower()
            code = f"{color_prefix}{piece_prefix}"

            try:
                piece_img = self.cache.get(style, code, self.img_size)
                # Composite piece onto background
                sq.paste(piece_img, (0, 0), piece_img)  # RGBA mask
            except FileNotFoundError:
                pass  # style missing this piece → treat as empty

        return sq


class ChessSquareDatasetFromImages(Dataset):
    """Dataset that loads pre-rendered square images from a directory tree.

    Expected layout::

        root/
          class_name/      # e.g. "white_pawn", "empty"
            image_001.png
            ...

    This is useful when you have a pre-generated dataset on disk rather
    than synthesising on the fly.
    """

    def __init__(
        self,
        root: str | Path,
        img_size: int = 64,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.transform = transform or get_val_transform(img_size)
        self.samples: List[Tuple[Path, int]] = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise FileNotFoundError(f"No samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label
