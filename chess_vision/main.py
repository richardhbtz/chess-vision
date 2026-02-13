"""
Chess Recognition System – Main Entry Point
=============================================

Commands:

  1. **Train**      – Train the piece classifier from scratch using the
                      chess.com piece asset library.
  2. **Recognize**  – Run the full pipeline on an image image and
                      print the FEN result.
  3. **Export**      – Export the trained model to ONNX format.

Usage examples
--------------

**Training**::

    python chess_vision.py train \\
        --pieces-root dataset/pieces \\
        --fen-file dataset/fen.json \\
        --epochs 10 \\
        --batch-size 128

**Inference**::

    python chess_vision.py recognize \\
        --image game.png \\
        --weights checkpoints/best_classifier.pt \\
        --visualize

**ONNX Export**::

    python chess_vision.py export \\
        --weights checkpoints/best_classifier.pt \\
        --output chess_classifier.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("chess_vision")


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def cmd_train(args: argparse.Namespace) -> None:
    """Train the piece classifier."""
    from chess_vision.training.train import train

    checkpoint = train(
        pieces_root=args.pieces_root,
        epochs_phase1=args.epochs,
        epochs_phase2=args.epochs,
        batch_size=args.batch_size,
        lr_phase1=args.lr,
        lr_phase2=args.lr * 0.1,
        samples_per_epoch=args.samples_per_epoch,
        output_dir=args.output_dir,
        fen_file=args.fen_file,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
    )
    log.info("Training complete. Best model saved to: %s", checkpoint)


# ═══════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════

def cmd_recognize(args: argparse.Namespace) -> None:
    """Run the recognition pipeline on an image."""
    from chess_vision.inference.pipeline import ChessRecognitionPipeline

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        log.error("Could not read image: %s", args.image)
        sys.exit(1)

    # Build pipeline
    pipeline = ChessRecognitionPipeline(
        classifier_weights=args.weights,
        yolo_model_path=args.yolo_model,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        use_tta=args.tta,
    )

    # Run
    result = pipeline.recognize(image)

    # Output
    print("\n" + "=" * 60)
    print("  CHESS RECOGNITION RESULT")
    print("=" * 60)
    print(f"  FEN (position) : {result.fen}")
    print(f"  FEN (full)     : {result.full_fen}")
    print(f"  Confidence     : {result.confidence:.2%}")
    print(f"  Valid position : {result.is_valid}")
    print(f"  Corrected      : {result.was_corrected}")
    print(f"  Detection      : {result.detection_method} "
          f"(conf={result.detection_confidence:.2%})")
    if result.violations:
        print(f"  Violations     : {result.violations}")
    print("=" * 60 + "\n")

    # JSON output
    # output = {
    #    "fen": result.fen,
    #    "confidence": result.confidence,
    #    "board_image": "<ndarray>",
    #    "square_predictions": result.square_predictions,
    # }
    # print(json.dumps(output, indent=2, default=str))

    # Visualise
    if args.visualize:
        save_path = args.save_debug or None
        pipeline.visualize(result, show=True, save_path=save_path)


# ═══════════════════════════════════════════════════════════════════════
# ONNX Export
# ═══════════════════════════════════════════════════════════════════════

def cmd_export(args: argparse.Namespace) -> None:
    """Export trained model to ONNX."""
    from chess_vision.inference.pipeline import export_to_onnx
    from chess_vision.models.classifier import ChessPieceClassifier

    model = ChessPieceClassifier.load_from_checkpoint(args.weights)
    export_to_onnx(model, output_path=args.output)
    log.info("ONNX model saved to %s", args.output)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chess_vision",
        description="Multi-website chessboard recognition system.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ──
    p_train = sub.add_parser("train", help="Train the piece classifier")
    p_train.add_argument("--pieces-root", required=True,
                         help="Path to piece images root dir")
    p_train.add_argument("--fen-file", default=None,
                         help="JSON with FEN positions")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--samples-per-epoch", type=int, default=20_000)
    p_train.add_argument("--output-dir", default="checkpoints")
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                         help="Device: auto (default), cpu, or cuda")
    p_train.add_argument("--resume", default=None,
                         help="Path to last_checkpoint.pt to resume training")

    # ── recognize ──
    p_rec = sub.add_parser("recognize", help="Recognize an image")
    p_rec.add_argument("--image", required=True,
                       help="Path to image image")
    p_rec.add_argument("--weights", required=True,
                       help="Path to classifier .pt checkpoint")
    p_rec.add_argument("--yolo-model", default=None,
                       help="Path to YOLOv8 .pt (optional)")
    p_rec.add_argument("--confidence-threshold", type=float, default=0.6)
    p_rec.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p_rec.add_argument("--tta", action="store_true",
                       help="Enable test-time augmentation")
    p_rec.add_argument("--visualize", action="store_true",
                       help="Show debug visualisation")
    p_rec.add_argument("--save-debug", default=None,
                       help="Save debug image to path")

    # ── export ──
    p_exp = sub.add_parser("export", help="Export model to ONNX")
    p_exp.add_argument("--weights", required=True)
    p_exp.add_argument("--output", default="chess_classifier.onnx")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "train": cmd_train,
        "recognize": cmd_recognize,
        "export": cmd_export,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
