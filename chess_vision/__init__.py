"""
Chess Recognition System
========================

A production-grade chessboard recognition pipeline that processes images
from multiple chess websites and outputs FEN strings.

Architecture:
    1. Board Detection   – YOLO or classical CV fallback
    2. Perspective Norm.  – corner detection + warp to 512×512
    3. Square Extraction  – 8×8 grid → 64 images at 64×64
    4. Classification     – MobileNetV3-Small transfer-learned on 13 classes
    5. FEN Reconstruction – validated, corrected FEN output
"""

__version__ = "1.0.0"
