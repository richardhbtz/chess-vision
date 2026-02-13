"""
Inference Pipeline – End-to-End Image → FEN
=================================================

This is the single-call entry point for production inference.

Pipeline stages:
  1. Board detection     – YOLO or classical CV
  2. Perspective warp    – normalise to 512×512
  3. Square extraction   – 8×8 grid → 64 images (64×64)
  4. Batch classification – MobileNetV3-Small forward pass on all 64
  5. FEN reconstruction  – with legality validation & correction

Optional extras:
  • Test-time augmentation (TTA) – horizontal flip + average
  • Debug visualisation overlay
  • ONNX export helper
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from chess_vision.inference.fen_utils import (
    FENResult,
    SquarePrediction,
    correct_fen,
    fen_to_full,
    predictions_to_fen,
    validate_fen,
)
from chess_vision.models.classifier import (
    CLASS_NAMES,
    CLASS_TO_FEN,
    NUM_CLASSES,
    ChessPieceClassifier,
)
from chess_vision.models.yolo_detector import (
    WARP_SIZE,
    BoardDetection,
    detect_board,
    extract_squares,
)
from chess_vision.training.dataset import IMAGENET_MEAN, IMAGENET_STD

log = logging.getLogger(__name__)


# ── Inference transform (deterministic) ────────────────────────────────

def _inference_transform(img_size: int = 64) -> transforms.Compose:
    """Normalisation-only transform for inference squares."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Result dataclass ──────────────────────────────────────────────────

@dataclass
class RecognitionResult:
    """Full output of the recognition pipeline."""
    fen: str                                       # Position FEN
    full_fen: str                                  # FEN with move / castling fields
    confidence: float                              # Average softmax confidence
    is_valid: bool                                 # Legality check passed
    was_corrected: bool                            # Legality corrections applied
    violations: List[str]                          # Remaining violations (if any)
    board_image: np.ndarray                        # Warped 512×512 board
    square_predictions: List[Dict]                 # Per-square dicts
    detection_method: str                          # "yolo" | "classical"
    detection_confidence: float                    # Board detection confidence


# ── Pipeline class ─────────────────────────────────────────────────────

class ChessRecognitionPipeline:
    """End-to-end chess image → FEN pipeline.

    Parameters
    ----------
    classifier_weights : str | Path
        Path to the trained classifier ``.pt`` checkpoint.
    yolo_model_path : str | Path, optional
        Path to a YOLOv8 ``.pt`` for board detection.  If ``None`` the
        classical-CV fallback is used.
    confidence_threshold : float
        Minimum softmax confidence to accept a piece prediction.
    device : str
        ``"cpu"`` or ``"cuda"``.
    use_tta : bool
        Enable test-time augmentation (horizontal flip averaging).
    """

    def __init__(
        self,
        classifier_weights: str | Path,
        yolo_model_path: Optional[str | Path] = None,
        confidence_threshold: float = 0.6,
        device: str = "cpu",
        use_tta: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.yolo_model_path = str(yolo_model_path) if yolo_model_path else None
        self.use_tta = use_tta

        # Load classifier
        self.model = ChessPieceClassifier.load_from_checkpoint(
            str(classifier_weights), device=self.device,
        )
        self.model.eval()

        # Inference transform
        self.transform = _inference_transform(64)

        log.info(
            "Pipeline ready  classifier=%s  yolo=%s  device=%s  tta=%s",
            classifier_weights,
            self.yolo_model_path or "classical-fallback",
            self.device,
            self.use_tta,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Run the full pipeline on a BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image (OpenCV convention).

        Returns
        -------
        RecognitionResult
        """
        # 1. Board detection + perspective warp
        detection: BoardDetection = detect_board(
            image,
            yolo_model_path=self.yolo_model_path,
        )
        board_img = detection.warped  # 512×512

        # 2. Extract 64 squares (64×64 each)
        squares = extract_squares(board_img, square_size=64)

        # 3. Batch classify all 64 squares
        predictions = self._classify_squares(squares)

        # 4. FEN reconstruction with validation & correction
        fen_result: FENResult = correct_fen(
            predictions, confidence_threshold=self.confidence_threshold,
        )

        # 5. Build result
        sq_dicts = [
            {
                "index": p.index,
                "rank": p.rank,
                "file": p.file,
                "class_name": p.class_name,
                "class_index": p.class_index,
                "confidence": round(p.confidence, 4),
            }
            for p in fen_result.square_predictions
        ]

        return RecognitionResult(
            fen=fen_result.fen,
            full_fen=fen_to_full(fen_result.fen),
            confidence=round(fen_result.confidence, 4),
            is_valid=fen_result.is_valid,
            was_corrected=fen_result.was_corrected,
            violations=fen_result.violations,
            board_image=board_img,
            square_predictions=sq_dicts,
            detection_method=detection.method,
            detection_confidence=round(detection.confidence, 4),
        )

    # ── Square classification ──────────────────────────────────────────

    @torch.no_grad()
    def _classify_squares(
        self, squares: List[np.ndarray],
    ) -> List[SquarePrediction]:
        """Batch-classify 64 square images.

        Converts OpenCV BGR images → RGB tensors, stacks into a single
        batch, runs a single forward pass, and optionally applies TTA.
        """
        tensors: List[torch.Tensor] = []
        for sq in squares:
            # BGR → RGB, resize to 64×64
            rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (64, 64))
            t = self.transform(rgb)
            tensors.append(t)

        batch = torch.stack(tensors).to(self.device)  # (64, 3, 64, 64)

        # Forward pass
        probs = self.model.predict_proba(batch)  # (64, 13)

        # Optional TTA: horizontal flip
        if self.use_tta:
            batch_flip = torch.flip(batch, dims=[3])  # flip width
            probs_flip = self.model.predict_proba(batch_flip)
            probs = (probs + probs_flip) / 2.0

        # Build predictions
        predictions: List[SquarePrediction] = []
        for i in range(64):
            conf, cls_idx = probs[i].max(dim=0)
            cls_idx = int(cls_idx)
            predictions.append(SquarePrediction(
                index=i,
                class_name=CLASS_NAMES[cls_idx],
                class_index=cls_idx,
                confidence=float(conf),
            ))

        return predictions

    # ── Debug visualisation ────────────────────────────────────────────

    def visualize(
        self,
        result: RecognitionResult,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw piece labels and confidence on the warped board image.

        Parameters
        ----------
        result : RecognitionResult
            Output of ``recognize()``.
        show : bool
            Display with ``cv2.imshow`` (blocks until key press).
        save_path : str, optional
            Save the annotated image to disk.

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        vis = result.board_image.copy()
        h, w = vis.shape[:2]
        cell_h, cell_w = h // 8, w // 8

        for sq in result.square_predictions:
            row = 7 - (sq["rank"] - 1)  # rank 8 → row 0
            col = sq["file"]
            x = col * cell_w
            y = row * cell_h

            label = CLASS_TO_FEN.get(sq["class_name"], "·")
            conf = sq["confidence"]

            # Colour: green if confident, yellow if marginal, red if low
            if conf >= 0.8:
                color = (0, 200, 0)
            elif conf >= 0.5:
                color = (0, 200, 255)
            else:
                color = (0, 0, 255)

            # Draw label
            cv2.putText(
                vis, label,
                (x + 4, y + cell_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
            # Draw confidence
            cv2.putText(
                vis, f"{conf:.0%}",
                (x + 4, y + cell_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )
            # Grid line
            cv2.rectangle(vis, (x, y), (x + cell_w, y + cell_h), (80, 80, 80), 1)

        # FEN annotation at the bottom
        cv2.putText(
            vis, f"FEN: {result.fen}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

        if save_path:
            cv2.imwrite(save_path, vis)
            log.info("Saved debug image to %s", save_path)

        if show:
            cv2.imshow("Chess Recognition", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis


# ── ONNX Export (Bonus) ────────────────────────────────────────────────

def export_to_onnx(
    model: ChessPieceClassifier,
    output_path: str = "chess_classifier.onnx",
    img_size: int = 64,
) -> None:
    """Export the classifier to ONNX format.

    Parameters
    ----------
    model : ChessPieceClassifier
        Trained model (will be set to eval mode).
    output_path : str
        Destination ``.onnx`` file.
    img_size : int
        Input image size (default 64).
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=13,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    log.info("Exported ONNX model to %s", output_path)
