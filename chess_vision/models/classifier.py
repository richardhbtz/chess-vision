"""
Chess Piece Classifier – MobileNetV3-Small Transfer Learning
=============================================================

Architectural decisions:
  • MobileNetV3-Small chosen for CPU-friendliness (~2.5 M params) while
    retaining strong accuracy via squeeze-and-excite blocks and h-swish.
  • The ImageNet-pretrained backbone is kept frozen during initial training
    and optionally fine-tuned later (``freeze_backbone`` flag).
  • The classifier head is replaced with a lightweight MLP
    (dropout → 256-d → ReLU → dropout → 13-class softmax).
  • ``forward`` returns raw logits; softmax is applied separately so that
    ``CrossEntropyLoss`` can consume the logits directly during training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


# ── Canonical class list (index ↔ label mapping) ──────────────────────

CLASS_NAMES: list[str] = [
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]

NUM_CLASSES: int = len(CLASS_NAMES)

# Mapping from dataset file prefixes (e.g. "wp") → class label
FILE_PREFIX_TO_CLASS: dict[str, str] = {
    "wp": "white_pawn",
    "wn": "white_knight",
    "wb": "white_bishop",
    "wr": "white_rook",
    "wq": "white_queen",
    "wk": "white_king",
    "bp": "black_pawn",
    "bn": "black_knight",
    "bb": "black_bishop",
    "br": "black_rook",
    "bq": "black_queen",
    "bk": "black_king",
}

# Reverse: class label → FEN character
CLASS_TO_FEN: dict[str, str] = {
    "empty": "",
    "white_pawn": "P",
    "white_knight": "N",
    "white_bishop": "B",
    "white_rook": "R",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_knight": "n",
    "black_bishop": "b",
    "black_rook": "r",
    "black_queen": "q",
    "black_king": "k",
}

# FEN char → class index (for reverse lookups / dataset generation)
FEN_CHAR_TO_CLASS_IDX: dict[str, int] = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}


# ── Model ──────────────────────────────────────────────────────────────

class ChessPieceClassifier(nn.Module):
    """MobileNetV3-Small with a custom 13-class head for chess squares.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 13).
    freeze_backbone : bool
        If *True* the MobileNet feature-extractor layers are frozen;
        only the classifier head trains.  Set to *False* for full
        fine-tuning after an initial warm-up phase.
    dropout : float
        Dropout probability used in the classifier head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Load pretrained MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.backbone = models.mobilenet_v3_small(weights=weights)

        # Freeze feature extractor if requested
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace the classifier head.
        # Original: Linear(576, 1024) → Hardswish → Dropout → Linear(1024, 1000)
        in_features: int = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return **logits** of shape ``(B, num_classes)``.

        Apply ``torch.softmax(output, dim=1)`` externally if you need
        probabilities.
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities of shape ``(B, num_classes)``."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "ChessPieceClassifier":
        """Convenience loader that handles map_location automatically."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(**kwargs)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
