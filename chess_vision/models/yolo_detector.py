"""
Board Detection – YOLOv8 primary + Classical CV fallback
========================================================

Strategy:
  • **Primary**: Use a YOLOv8 model (``ultralytics`` package) trained or
    fine-tuned to detect chessboards.  The detector returns the single
    highest-confidence bounding box and crops the image.
  • **Fallback**: If no YOLO model is available (or detection fails) we
    use a multi-strategy classical-CV pipeline that is designed to find
    a chessboard even in a full-screen screenshot containing other UI
    elements (browser chrome, sidebars, chat panels, etc.).

    Strategy A – **Checkerboard pattern detection** *(screenshot-safe)*
        Multi-scale scanning that scores candidate square regions by
        how well they exhibit the 8×8 alternating-colour grid pattern
        unique to chessboards.  The best-scoring candidate is returned.

    Strategy B – **Contour + checkerboard validation**
        Find large roughly-square contours, then validate each candidate
        by computing its checkerboard score, picking the one that looks
        most like a real board.

    Strategy C – **Colour-boundary scan**
        Scan inward from each edge to find where the board's alternating
        colours begin – works when the image is *just* the board with a
        thin dark border.

    Strategy D – **Grid-line detection**
        Find Hough lines, cluster intersections, fit the bounding rect.

    Strategy E – **Centre-crop square fallback**
        Extract the largest centred square from the image.

Design notes:
  • ``detect_board`` is the single public API – it dispatches automatically.
  • The classical pipeline now **validates squareness** (aspect 0.85–1.15)
    before accepting a quadrilateral, preventing trapezoid warps.
  • When the image is already nearly square (aspect within 5 %), we
    short-circuit to a simple resize – avoiding distortion from a
    perspective warp that has nothing useful to correct.
  • Checkerboard-pattern scoring makes the detector robust to full-page
    screenshots where the board is only a portion of the image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class BoardDetection:
    """Result of board detection."""
    cropped: np.ndarray          # Cropped & warped board image
    corners: np.ndarray          # 4×2 array of corner coords in original image
    confidence: float            # Detection confidence (1.0 for classical)
    warped: np.ndarray           # Perspective-normalised 512×512 board
    method: str                  # "yolo" | "classical"


WARP_SIZE: int = 512  # Target size after perspective warp


# ── Public API ─────────────────────────────────────────────────────────

def detect_board(
    image: np.ndarray,
    yolo_model_path: Optional[str] = None,
    warp_size: int = WARP_SIZE,
    yolo_conf: float = 0.25,
) -> BoardDetection:
    """Detect and extract the chessboard from an image.

    Parameters
    ----------
    image : np.ndarray
        BGR image image.
    yolo_model_path : str, optional
        Path to a YOLOv8 ``.pt`` weights file.  If provided and the
        ``ultralytics`` package is installed, YOLO detection is attempted
        first.
    warp_size : int
        Side length of the output square image (default 512).
    yolo_conf : float
        Minimum confidence for YOLO detections.

    Returns
    -------
    BoardDetection
    """
    # ---- Try YOLO first ------------------------------------------------
    if yolo_model_path is not None:
        try:
            result = _detect_yolo(image, yolo_model_path, yolo_conf)
            if result is not None:
                corners = result["corners"]
                warped = _warp_or_resize(image, corners, warp_size)
                cropped = result["crop"]
                return BoardDetection(
                    cropped=cropped,
                    corners=corners,
                    confidence=result["confidence"],
                    warped=warped,
                    method="yolo",
                )
        except Exception:
            pass  # fall through to classical

    # ---- Classical fallback --------------------------------------------
    corners, conf = _detect_classical(image)
    warped = _warp_or_resize(image, corners, warp_size)
    cropped = warped.copy()
    return BoardDetection(
        cropped=cropped,
        corners=corners,
        confidence=conf,
        warped=warped,
        method="classical",
    )


# ── YOLO detection ─────────────────────────────────────────────────────

def _detect_yolo(
    image: np.ndarray,
    model_path: str,
    conf: float,
) -> Optional[dict]:
    """Run YOLOv8 inference and return the best detection, or None."""
    from ultralytics import YOLO  # lazy import – optional dependency

    model = YOLO(model_path)
    results = model.predict(source=image, conf=conf, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return None

    # Pick highest-confidence box
    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    confidence = float(boxes.conf[best_idx].cpu())

    crop = image[y1:y2, x1:x2].copy()

    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)

    return {"crop": crop, "corners": corners, "confidence": confidence}


# ── Classical CV detection (multi-strategy) ────────────────────────────

def _detect_classical(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect board corners using multiple strategies in priority order.

    The pipeline is ordered so that screenshot-friendly strategies run
    first, falling back to simpler heuristics for clean board images.

    Returns (4×2 float32 corners [TL, TR, BR, BL], confidence).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # ── Strategy A: checkerboard pattern detection (screenshot-safe) ───
    corners_a = _strategy_checkerboard_pattern(image, gray, h, w)
    if corners_a is not None:
        log.info("Board detected via checkerboard-pattern scan")
        return _order_corners(corners_a), 0.9

    # ── Strategy B: contour + checkerboard validation ──────────────────
    corners_b = _strategy_contour_validated(image, gray, h, w)
    if corners_b is not None:
        log.info("Board detected via validated-contour strategy")
        return _order_corners(corners_b), 0.85

    # ── Strategy C: colour-boundary scan (best for clean images) ──
    corners_c = _strategy_colour_boundary(gray, h, w)
    if corners_c is not None:
        log.info("Board detected via colour-boundary scan")
        return _order_corners(corners_c), 0.8

    # ── Strategy D: largest square-ish contour (legacy) ────────────────
    corners_d = _strategy_contour(gray, h, w)
    if corners_d is not None:
        log.info("Board detected via contour strategy")
        return _order_corners(corners_d), 0.75

    # ── Strategy E: Hough grid lines ───────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    corners_e = _strategy_hough(edges, h, w)
    if corners_e is not None:
        log.info("Board detected via Hough-line strategy")
        return _order_corners(corners_e), 0.7

    # ── Strategy F: centre-crop largest square ─────────────────────────
    log.info("Board detection fallback: centre square crop")
    corners_f = _strategy_centre_square(h, w)
    return corners_f, 0.5


# ── Checkerboard scoring ───────────────────────────────────────────────

# Pre-compute the checkerboard mask once (True for (r+c) even cells).
_CB_MASK = np.array(
    [[(r + c) % 2 == 0 for c in range(8)] for r in range(8)],
    dtype=bool,
)

def _checkerboard_score(gray_region: np.ndarray) -> float:
    """Score how well a square image region resembles an 8×8 checkerboard.

    A real chessboard has two distinct square colours that alternate in a
    checkerboard pattern: cells where ``(row + col) % 2 == 0`` share one
    colour and cells where ``(row + col) % 2 == 1`` share the other.

    The scoring checks three properties:

    1. **Separation** – the mean brightness of the two groups should
       differ noticeably (the board has light and dark squares).
    2. **Within-group consistency** – cells in each group should have
       similar brightness (low std within each group).  This rejects
       false positives where high adjacent contrast comes from unrelated
       UI elements (dark toolbar next to bright content).
    3. **Alternation regularity** – adjacent cells should alternate; the
       sign of the brightness difference between neighbours should be
       consistent with a checkerboard.

    Fully vectorised with NumPy for speed.

    Parameters
    ----------
    gray_region : np.ndarray
        Greyscale image of the candidate board region (any size).

    Returns
    -------
    float
        Score in [0, 1] – higher means more checkerboard-like.
    """
    if gray_region.size == 0:
        return 0.0

    # Resize to 8×8 directly – each pixel IS the mean of that cell.
    # This is equivalent to resizing to 160×160 and averaging 20×20
    # blocks, but ~400× faster.
    cell_means = cv2.resize(
        gray_region, (8, 8), interpolation=cv2.INTER_AREA,
    ).astype(np.float64)

    # ── 1. Two-group analysis ──────────────────────────────────────────
    group_a = cell_means[_CB_MASK]   # (r+c) even cells
    group_b = cell_means[~_CB_MASK]  # (r+c) odd cells

    mean_a = group_a.mean()
    mean_b = group_b.mean()
    std_a = group_a.std()
    std_b = group_b.std()

    separation = abs(mean_a - mean_b)
    if separation < 5:
        return 0.0

    sep_score = min(separation / 40.0, 1.0)

    # ── 2. Within-group consistency ────────────────────────────────────
    avg_std = (std_a + std_b) / 2.0
    consistency_score = float(np.clip(1.0 - (avg_std - 20) / 50.0, 0.0, 1.0))

    # ── 3. Alternation regularity (vectorised) ─────────────────────────
    # Horizontal diffs: means[:, 1:] - means[:, :-1]
    h_diff = cell_means[:, 1:] - cell_means[:, :-1]
    # Vertical diffs: means[1:, :] - means[:-1, :]
    v_diff = cell_means[1:, :] - cell_means[:-1, :]

    expected_sign = mean_b - mean_a  # sign of a→b transition

    # For cells where (r+c) is even, the diff to the neighbour should
    # have the same sign as expected_sign; for odd cells, opposite sign.
    h_mask = _CB_MASK[:, :-1]  # which cells are "group a" for horiz pairs
    h_correct = np.where(h_mask, h_diff * expected_sign > 0,
                                  h_diff * expected_sign < 0)

    v_mask = _CB_MASK[:-1, :]
    v_correct = np.where(v_mask, v_diff * expected_sign > 0,
                                  v_diff * expected_sign < 0)

    total = h_correct.size + v_correct.size
    correct = int(h_correct.sum() + v_correct.sum())
    alternation_score = correct / max(total, 1)

    # ── Combine ────────────────────────────────────────────────────────
    score = sep_score * consistency_score * alternation_score
    return float(np.clip(score, 0.0, 1.0))


def _strategy_checkerboard_pattern(
    image: np.ndarray,
    gray: np.ndarray,
    h: int,
    w: int,
) -> Optional[np.ndarray]:
    """Multi-scale sliding-window search for the checkerboard pattern.

    Uses a three-pass approach for speed + accuracy:

      1. **Downsampled coarse scan** – shrink the image (e.g. 4×), slide
         square windows at several scales with a stride of one cell-width
         (side/8).  Because ``_checkerboard_score`` internally resizes to
         160×160 anyway, the quality loss from downsampling is negligible.
      2. **Full-resolution coarse confirmation** – re-evaluate the top
         candidates at original resolution to filter false positives.
      3. **Fine refinement** – around the best confirmed candidate, scan
         with a finer stride and nearby sizes to pin-point the exact
         board boundaries at full resolution.

    Designed for screenshots where the board could be anywhere.
    """
    min_board_px = 120  # ignore candidate boards smaller than this

    short_side = min(h, w)
    long_side = max(h, w)
    max_side = min(int(short_side * 0.98), long_side)
    min_side = max(min_board_px, int(short_side * 0.12))

    # ── Downsample for the coarse pass ─────────────────────────────────
    # Choose a factor that keeps the image manageable (~300-500px short
    # side) while preserving enough detail for the score function.
    ds_factor = max(1, short_side // 400)
    if ds_factor > 1:
        small_gray = cv2.resize(
            gray,
            (w // ds_factor, h // ds_factor),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small_gray = gray
    sh, sw = small_gray.shape[:2]

    # Build scale list (in downsampled pixel units)
    scales_ds: list[int] = []
    s = max_side // ds_factor
    min_side_ds = min_side // ds_factor
    while s >= max(min_side_ds, 30):
        scales_ds.append(s)
        s = int(s * 0.85)  # ~15% reduction per step

    if not scales_ds:
        return None

    # ── Pass 1: coarse scan on downsampled image ───────────────────────
    score_threshold = 0.15
    # Keep a small pool of top candidates (position + scale)
    candidates: list[tuple[float, int, int, int]] = []  # (score, x, y, side)

    for side_ds in scales_ds:
        step_ds = max(side_ds // 8, 4)
        for y_ds in range(0, sh - side_ds + 1, step_ds):
            for x_ds in range(0, sw - side_ds + 1, step_ds):
                roi = small_gray[y_ds : y_ds + side_ds, x_ds : x_ds + side_ds]
                sc = _checkerboard_score(roi)
                if sc > score_threshold:
                    candidates.append((sc, x_ds, y_ds, side_ds))

    if not candidates:
        return None

    # Sort by score descending and keep top-N for full-res confirmation
    candidates.sort(key=lambda t: t[0], reverse=True)
    top_n = min(len(candidates), 10)

    # ── Pass 2: confirm at full resolution ─────────────────────────────
    # For each top candidate, test the mapped position *and* small
    # offsets (± ds_factor) to account for rounding.
    best_score = 0.0
    best_x, best_y, best_side = 0, 0, 0

    for _, x_ds, y_ds, side_ds in candidates[:top_n]:
        side_full = side_ds * ds_factor
        # Try a small neighbourhood around the mapped position
        jitter = ds_factor * 2
        for dy in range(-jitter, jitter + 1, max(jitter, 1)):
            for dx in range(-jitter, jitter + 1, max(jitter, 1)):
                x_full = x_ds * ds_factor + dx
                y_full = y_ds * ds_factor + dy
                # Clamp
                x_full = max(0, min(x_full, w - side_full))
                y_full = max(0, min(y_full, h - side_full))
                if side_full <= 0:
                    continue
                roi = gray[
                    y_full : y_full + side_full,
                    x_full : x_full + side_full,
                ]
                sc = _checkerboard_score(roi)
                if sc > best_score:
                    best_score = sc
                    best_x, best_y, best_side = x_full, y_full, side_full

    if best_score < score_threshold:
        return None

    # ── Pass 3: fine refinement around best confirmed hit ──────────────
    # Search nearby positions and sizes at full resolution to pin-point
    # the exact board boundaries.  Uses one-cell-width stride.
    if best_score < 0.92:
        cell_px = max(best_side // 8, 4)
        size_range = [
            int(best_side * f)
            for f in (0.90, 0.95, 1.0, 1.05, 1.10)
        ]

        for side in size_range:
            if side < min_board_px:
                continue
            margin = cell_px * 3
            y_lo = max(0, best_y - margin)
            y_hi = min(h - side, best_y + margin)
            x_lo = max(0, best_x - margin)
            x_hi = min(w - side, best_x + margin)
            for y in range(y_lo, y_hi + 1, cell_px):
                for x in range(x_lo, x_hi + 1, cell_px):
                    roi = gray[y : y + side, x : x + side]
                    sc = _checkerboard_score(roi)
                    if sc > best_score:
                        best_score = sc
                        best_x, best_y, best_side = x, y, side

    if best_score >= score_threshold:
        corners = np.array([
            [best_x, best_y],
            [best_x + best_side, best_y],
            [best_x + best_side, best_y + best_side],
            [best_x, best_y + best_side],
        ], dtype=np.float32)
        log.debug(
            "Checkerboard pattern score=%.3f  region=%s",
            best_score, corners.tolist(),
        )
        return corners

    return None


def _strategy_contour_validated(
    image: np.ndarray,
    gray: np.ndarray,
    h: int,
    w: int,
) -> Optional[np.ndarray]:
    """Find contours and pick the one whose interior is most board-like.

    Unlike the plain contour strategy, this does *not* simply take the
    largest contour – it ranks all plausible candidates by their
    checkerboard score and returns the best one.  This is much more
    robust when the screenshot contains multiple rectangular UI elements
    that could be mistaken for a board.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Try multiple Canny thresholds to increase recall
    candidates: list[np.ndarray] = []
    for lo, hi in [(30, 120), (50, 150), (20, 80)]:
        edges = cv2.Canny(blurred, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            area = cv2.contourArea(approx)
            # Reject tiny contours
            if area < 0.02 * h * w:
                continue
            pts = approx.reshape(4, 2).astype(np.float32)
            if not _is_roughly_square(pts):
                continue
            candidates.append(pts)

    if not candidates:
        return None

    # Score each candidate by checkerboard pattern
    best_pts: Optional[np.ndarray] = None
    best_score: float = 0.0

    for pts in candidates:
        ordered = _order_corners(pts)
        x1 = max(0, int(ordered[0, 0]))
        y1 = max(0, int(ordered[0, 1]))
        x2 = min(w, int(ordered[2, 0]))
        y2 = min(h, int(ordered[2, 1]))
        if x2 - x1 < 40 or y2 - y1 < 40:
            continue
        roi = gray[y1:y2, x1:x2]
        sc = _checkerboard_score(roi)
        if sc > best_score:
            best_score = sc
            best_pts = pts

    # Require a reasonable checkerboard score
    if best_score >= 0.20 and best_pts is not None:
        log.debug(
            "Validated contour checkerboard score=%.3f", best_score,
        )
        return best_pts

    return None


def _strategy_contour(
    gray: np.ndarray, h: int, w: int,
) -> Optional[np.ndarray]:
    """Find the largest roughly-square 4-vertex contour."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )

    best: Optional[np.ndarray] = None
    best_area: float = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        if area < 0.05 * h * w:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        if not _is_roughly_square(pts):
            continue

        if area > best_area:
            best_area = area
            best = pts

    return best


def _strategy_hough(
    edges: np.ndarray, h: int, w: int,
) -> Optional[np.ndarray]:
    """Cluster Hough lines into a bounding rectangle."""
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80, minLineLength=50, maxLineGap=10,
    )
    if lines is None or len(lines) < 4:
        return None

    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if angle < 30 or angle > 150:
            h_lines.append(line[0])
        elif 60 < angle < 120:
            v_lines.append(line[0])

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
    v_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

    top, bottom = h_sorted[0], h_sorted[-1]
    left, right = v_sorted[0], v_sorted[-1]

    corners = []
    for hl in [top, bottom]:
        for vl in [left, right]:
            pt = _line_intersection(hl, vl)
            if pt is not None:
                corners.append(pt)

    if len(corners) != 4:
        return None

    pts = np.array(corners, dtype=np.float32)
    if not _is_roughly_square(pts):
        return None

    return pts


def _strategy_colour_boundary(
    gray: np.ndarray, h: int, w: int,
) -> Optional[np.ndarray]:
    """Scan inward from each edge to find where board colours begin.

    Works by finding the first/last column and row whose mean brightness
    is clearly above the dark border level.  This is effective when the
    image is the board with minimal chrome.
    """
    # Compute per-row and per-column mean brightness
    col_means = gray.mean(axis=0)  # shape (w,)
    row_means = gray.mean(axis=1)  # shape (h,)

    # Threshold: board squares are typically >100, borders < 80
    threshold = 100.0

    # Find left/right board boundary from column means
    bright_cols = np.where(col_means > threshold)[0]
    bright_rows = np.where(row_means > threshold)[0]

    if len(bright_cols) < 10 or len(bright_rows) < 10:
        return None

    x1, x2 = int(bright_cols[0]), int(bright_cols[-1])
    y1, y2 = int(bright_rows[0]), int(bright_rows[-1])

    bw = x2 - x1
    bh = y2 - y1

    if bw < 50 or bh < 50:
        return None

    # Force square: take the smaller dimension centred on the midpoint
    aspect = bw / bh
    if aspect < 0.8 or aspect > 1.25:
        # Too far from square – trim to square
        side = min(bw, bh)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

    corners = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=np.float32)
    return corners


def _strategy_centre_square(h: int, w: int) -> np.ndarray:
    """Extract the largest centred square from the image dimensions."""
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side
    return np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=np.float32)


# ── Geometry helpers ───────────────────────────────────────────────────

def _is_roughly_square(pts: np.ndarray, tol: float = 0.15) -> bool:
    """Check if 4 points form a roughly square/rectangular shape.

    Validates:
      1. Aspect ratio close to 1:1 (within *tol*).
      2. Opposite sides are parallel (direction cosine > 0.95).
      3. All four interior angles are close to 90° (within 10°).
    """
    ordered = _order_corners(pts)
    sides = [
        ordered[1] - ordered[0],  # top
        ordered[2] - ordered[1],  # right
        ordered[3] - ordered[2],  # bottom
        ordered[0] - ordered[3],  # left
    ]
    lengths = [float(np.linalg.norm(s)) for s in sides]
    if any(l < 1 for l in lengths):
        return False

    # Aspect ratio: average of top/bottom vs left/right
    width = (lengths[0] + lengths[2]) / 2
    height = (lengths[1] + lengths[3]) / 2
    aspect = width / height
    if not ((1 - tol) <= aspect <= (1 + tol)):
        return False

    # Opposite sides must be roughly parallel
    def _parallel(a: np.ndarray, b: np.ndarray) -> bool:
        """Return True if two vectors are roughly parallel."""
        cos = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        return cos > 0.95  # within ~18°

    if not _parallel(sides[0], -sides[2]):  # top vs bottom (reversed)
        return False
    if not _parallel(sides[1], -sides[3]):  # right vs left (reversed)
        return False

    # All 4 angles near 90°
    for i in range(4):
        v1 = sides[i]
        v2 = sides[(i + 1) % 4]
        cos_angle = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        # cos(90°) = 0, so cos_angle should be near 0; angle from perpendicular
        angle_from_perp = np.degrees(np.arcsin(np.clip(cos_angle, 0, 1)))
        if angle_from_perp > 5:
            return False

    return True


def _line_intersection(
    line1: np.ndarray, line2: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Compute intersection of two line segments (extended to infinity)."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses the sum / difference heuristic:
      • TL has the smallest sum  (x+y)
      • BR has the largest sum
      • TR has the smallest diff (y-x)
      • BL has the largest diff
    """
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # TL
    ordered[1] = pts[np.argmin(d)]   # TR
    ordered[2] = pts[np.argmax(s)]   # BR
    ordered[3] = pts[np.argmax(d)]   # BL
    return ordered


def _warp_or_resize(
    image: np.ndarray,
    corners: np.ndarray,
    size: int = WARP_SIZE,
) -> np.ndarray:
    """Warp/resize the detected region to a square image.

    If the detected region is already an axis-aligned rectangle (i.e.
    corners form a rectangle with sides parallel to image axes), we use
    a simple crop + resize which avoids interpolation artefacts from
    an unnecessary perspective transform.

    For non-axis-aligned quadrilaterals (actual perspective distortion)
    we apply a full perspective warp.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image.
    corners : np.ndarray
        4×2 ordered corners (TL, TR, BR, BL).
    size : int
        Side-length of the output square (default 512).

    Returns
    -------
    np.ndarray
        Square image of shape ``(size, size, 3)``.
    """
    src = _order_corners(corners.astype(np.float32))

    # Check if corners are axis-aligned (simple rectangle)
    tl, tr, br, bl = src
    is_axis_aligned = (
        abs(tl[1] - tr[1]) < 3 and   # top edge horizontal
        abs(bl[1] - br[1]) < 3 and   # bottom edge horizontal
        abs(tl[0] - bl[0]) < 3 and   # left edge vertical
        abs(tr[0] - br[0]) < 3       # right edge vertical
    )

    if is_axis_aligned:
        # Simple crop + resize (no warp artefacts)
        x1 = max(0, int(round(min(tl[0], bl[0]))))
        y1 = max(0, int(round(min(tl[1], tr[1]))))
        x2 = min(image.shape[1], int(round(max(tr[0], br[0]))))
        y2 = min(image.shape[0], int(round(max(bl[1], br[1]))))
        crop = image[y1:y2, x1:x2]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    # Full perspective warp for non-rectangular regions
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped


def extract_squares(
    board_img: np.ndarray,
    square_size: int = 64,
) -> list[np.ndarray]:
    """Split a warped board image into 64 square images.

    The output order is **FEN row-major**: rank 8 (top row of image)
    through rank 1 (bottom row), files a–h left-to-right within each rank.

    Parameters
    ----------
    board_img : np.ndarray
        Warped square board image (e.g. 512×512).
    square_size : int
        Each extracted square is resized to ``(square_size, square_size)``.

    Returns
    -------
    list[np.ndarray]
        64 images in FEN order (index 0 = a8, index 63 = h1).
    """
    h, w = board_img.shape[:2]
    cell_h = h / 8
    cell_w = w / 8
    squares: list[np.ndarray] = []

    for row in range(8):       # rank 8 → rank 1
        for col in range(8):   # file a → file h
            y1 = int(row * cell_h)
            y2 = int((row + 1) * cell_h)
            x1 = int(col * cell_w)
            x2 = int((col + 1) * cell_w)
            cell = board_img[y1:y2, x1:x2]
            cell = cv2.resize(cell, (square_size, square_size))
            squares.append(cell)

    return squares
