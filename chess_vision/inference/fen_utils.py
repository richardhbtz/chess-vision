"""
FEN Utilities – Reconstruction & Validation
============================================

Responsibilities:
  1. Convert a 64-element prediction list into a FEN position string.
  2. Validate the FEN against basic chess legality rules.
  3. Attempt automatic correction when the position is illegal, using
     per-square confidence scores to decide which predictions to revise.

Legality checks implemented:
  • Exactly 1 white king and 1 black king.
  • At most 8 pawns per side.
  • No pawns on rank 1 or rank 8.

The correction strategy is *conservative*: it only downgrades low-
confidence predictions to ``empty`` (or swaps similar pieces) rather
than inventing pieces that were never predicted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from chess_vision.models.classifier import CLASS_NAMES, CLASS_TO_FEN


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class SquarePrediction:
    """Prediction for a single board square."""
    index: int                 # 0–63, row-major (a8=0, h1=63)
    class_name: str            # e.g. "white_pawn"
    class_index: int           # 0–12
    confidence: float          # softmax probability
    rank: int = 0              # 1–8 (computed)
    file: int = 0              # 0–7 → a–h (computed)

    def __post_init__(self) -> None:
        self.rank = 8 - (self.index // 8)       # index 0 → rank 8
        self.file = self.index % 8              # index 0 → file a (0)


@dataclass
class FENResult:
    """Result of FEN reconstruction."""
    fen: str
    is_valid: bool
    confidence: float                            # average confidence
    was_corrected: bool = False
    violations: List[str] = field(default_factory=list)
    square_predictions: List[SquarePrediction] = field(default_factory=list)


# ── FEN construction ───────────────────────────────────────────────────

def predictions_to_fen(
    predictions: List[SquarePrediction],
    confidence_threshold: float = 0.6,
) -> str:
    """Convert 64 square predictions into a FEN position string.

    Squares with confidence below *confidence_threshold* are treated
    as empty.

    Parameters
    ----------
    predictions : list[SquarePrediction]
        Exactly 64 predictions in FEN row-major order (a8 → h1).
    confidence_threshold : float
        Minimum confidence to accept a non-empty prediction.

    Returns
    -------
    str
        FEN position field (e.g. ``rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR``).
    """
    if len(predictions) != 64:
        raise ValueError(f"Expected 64 predictions, got {len(predictions)}")

    rows: List[str] = []
    for rank_start in range(0, 64, 8):
        row_chars: List[str] = []
        empty_count = 0

        for i in range(8):
            pred = predictions[rank_start + i]
            fen_char = CLASS_TO_FEN.get(pred.class_name, "")

            # Apply confidence threshold (kings are never downgraded)
            if (
                pred.confidence < confidence_threshold
                and pred.class_name not in ("white_king", "black_king")
                and pred.class_name != "empty"
            ):
                fen_char = ""  # treat as empty

            if fen_char == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    row_chars.append(str(empty_count))
                    empty_count = 0
                row_chars.append(fen_char)

        if empty_count > 0:
            row_chars.append(str(empty_count))

        rows.append("".join(row_chars))

    return "/".join(rows)


# ── Validation ─────────────────────────────────────────────────────────

def validate_fen(fen: str) -> Tuple[bool, List[str]]:
    """Check basic legality of a FEN position string.

    Returns ``(is_valid, list_of_violation_strings)``.
    """
    violations: List[str] = []
    rows = fen.split("/")

    if len(rows) != 8:
        violations.append(f"Expected 8 ranks, got {len(rows)}")
        return False, violations

    # Flatten to piece list per rank
    all_pieces: List[str] = []
    for rank_idx, row in enumerate(rows):
        rank_pieces: List[Optional[str]] = []
        for ch in row:
            if ch.isdigit():
                rank_pieces.extend([None] * int(ch))
            else:
                rank_pieces.append(ch)
                all_pieces.append(ch)
        if len(rank_pieces) != 8:
            violations.append(
                f"Rank {8 - rank_idx} has {len(rank_pieces)} squares (expected 8)"
            )

    # King counts
    wk = all_pieces.count("K")
    bk = all_pieces.count("k")
    if wk != 1:
        violations.append(f"White king count = {wk} (expected 1)")
    if bk != 1:
        violations.append(f"Black king count = {bk} (expected 1)")

    # Pawn counts
    wp = all_pieces.count("P")
    bp = all_pieces.count("p")
    if wp > 8:
        violations.append(f"White pawn count = {wp} (max 8)")
    if bp > 8:
        violations.append(f"Black pawn count = {bp} (max 8)")

    # Pawns on rank 1 or 8
    rank8 = rows[0]  # first row in FEN = rank 8
    rank1 = rows[7]  # last row in FEN = rank 1
    for ch in rank8 + rank1:
        if ch in ("P", "p"):
            violations.append("Pawn found on rank 1 or 8 (illegal)")
            break

    is_valid = len(violations) == 0
    return is_valid, violations


# ── Correction ─────────────────────────────────────────────────────────

def correct_fen(
    predictions: List[SquarePrediction],
    confidence_threshold: float = 0.6,
) -> FENResult:
    """Build a FEN string and attempt to fix legality violations.

    Correction strategy (applied iteratively):
      1. **Excess kings** – keep only the highest-confidence king of each
         colour; downgrade others to empty.
      2. **Missing kings** – find the empty square with the highest
         confidence for the missing king class and promote it.
      3. **Excess pawns** – remove the lowest-confidence pawns (set to
         empty) until ≤ 8.
      4. **Pawns on rank 1/8** – downgrade them to empty or promote to
         queen if confidence is high.

    Parameters
    ----------
    predictions : list[SquarePrediction]
        64 square predictions.
    confidence_threshold : float
        Passed through to ``predictions_to_fen``.

    Returns
    -------
    FENResult
    """
    preds = copy.deepcopy(predictions)

    # ── First pass: build raw FEN ──
    raw_fen = predictions_to_fen(preds, confidence_threshold)
    is_valid, violations = validate_fen(raw_fen)

    if is_valid:
        avg_conf = sum(p.confidence for p in preds) / 64
        return FENResult(
            fen=raw_fen,
            is_valid=True,
            confidence=avg_conf,
            was_corrected=False,
            violations=[],
            square_predictions=preds,
        )

    # ── Corrections ──
    corrected = False

    # -- King corrections ------------------------------------------------
    for king_class, fen_char in [("white_king", "K"), ("black_king", "k")]:
        king_preds = [p for p in preds if p.class_name == king_class]
        if len(king_preds) > 1:
            # Keep highest confidence, remove rest
            king_preds.sort(key=lambda p: p.confidence, reverse=True)
            for extra in king_preds[1:]:
                extra.class_name = "empty"
                extra.class_index = 0
                corrected = True
        elif len(king_preds) == 0:
            # Try to find best candidate among empty / low-conf squares
            candidates = [
                p for p in preds
                if p.class_name == "empty" or p.confidence < confidence_threshold
            ]
            if candidates:
                # Pick square whose raw softmax for king class was highest
                # We don't have per-class softmax here, so just pick
                # the lowest-confidence empty square (least certain)
                candidates.sort(key=lambda p: p.confidence)
                promoted = candidates[0]
                promoted.class_name = king_class
                promoted.class_index = CLASS_NAMES.index(king_class)
                corrected = True

    # -- Pawn corrections ------------------------------------------------
    for pawn_class in ("white_pawn", "black_pawn"):
        pawn_preds = [p for p in preds if p.class_name == pawn_class]
        # Remove excess pawns (keep top-8 by confidence)
        if len(pawn_preds) > 8:
            pawn_preds.sort(key=lambda p: p.confidence, reverse=True)
            for excess in pawn_preds[8:]:
                excess.class_name = "empty"
                excess.class_index = 0
                corrected = True

    # -- Pawns on rank 1 / 8 --------------------------------------------
    for pred in preds:
        if pred.class_name in ("white_pawn", "black_pawn"):
            if pred.rank == 1 or pred.rank == 8:
                # Promote to queen if confidence is high, else empty
                if pred.confidence > 0.8:
                    queen_class = (
                        "white_queen" if pred.class_name == "white_pawn" else "black_queen"
                    )
                    pred.class_name = queen_class
                    pred.class_index = CLASS_NAMES.index(queen_class)
                else:
                    pred.class_name = "empty"
                    pred.class_index = 0
                corrected = True

    # ── Rebuild FEN after corrections ──
    final_fen = predictions_to_fen(preds, confidence_threshold)
    is_valid, violations = validate_fen(final_fen)
    avg_conf = sum(p.confidence for p in preds) / 64

    return FENResult(
        fen=final_fen,
        is_valid=is_valid,
        confidence=avg_conf,
        was_corrected=corrected,
        violations=violations,
        square_predictions=preds,
    )


def fen_to_full(position_fen: str) -> str:
    """Append default move / castling / en-passant fields to a position FEN.

    Produces ``"<position> w KQkq - 0 1"`` – the conventional starting
    assumption when only the position is known.
    """
    return f"{position_fen} w KQkq - 0 1"
