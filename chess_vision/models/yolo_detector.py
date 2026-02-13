"""
Board Detection – YOLOv8 primary + Classical CV fallback
========================================================

Strategy:
  • **Primary**: Use a YOLOv8 model (``ultralytics`` package) trained or
    fine-tuned to detect chessboards.  The detector returns the single
    highest-confidence bounding box and crops the image.
  • **Fallback**: If no YOLO model is available (or detection fails) we
    use a multi-strategy classical-CV pipeline:

    Strategy A – **Largest square-ish contour**
        Find the biggest 4-sided contour whose aspect ratio is close to
        1:1 (a board is square).

    Strategy B – **Grid-line detection**
        Find Hough lines, cluster intersections, fit the bounding rect.

    Strategy C – **Colour-boundary scan**
        Scan inward from each edge to find where the board's alternating
        colours begin – works even when there is no clear contour (e.g.
        the image is *just* the board with a thin dark border).

    Strategy D – **Centre-crop square fallback**
        Extract the largest centred square from the image.

Design notes:
  • ``detect_board`` is the single public API – it dispatches automatically.
  • The classical pipeline now **validates squareness** (aspect 0.85–1.15)
    before accepting a quadrilateral, preventing trapezoid warps.
  • When the image is already nearly square (aspect within 5 %), we
    short-circuit to a simple resize – avoiding distortion from a
    perspective warp that has nothing useful to correct.
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

    Returns (4×2 float32 corners [TL, TR, BR, BL], confidence).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # ── Strategy A: colour-boundary scan (best for clean images) ──
    corners_a = _strategy_colour_boundary(gray, h, w)
    if corners_a is not None:
        log.info("Board detected via colour-boundary scan")
        return _order_corners(corners_a), 0.8

    # ── Strategy B: largest square-ish contour ─────────────────────────
    corners_b = _strategy_contour(gray, h, w)
    if corners_b is not None:
        log.info("Board detected via contour strategy")
        return _order_corners(corners_b), 0.75

    # ── Strategy C: Hough grid lines ───────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    corners_c = _strategy_hough(edges, h, w)
    if corners_c is not None:
        log.info("Board detected via Hough-line strategy")
        return _order_corners(corners_c), 0.7

    # ── Strategy D: centre-crop largest square ─────────────────────────
    log.info("Board detection fallback: centre square crop")
    corners_d = _strategy_centre_square(h, w)
    return corners_d, 0.5


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
