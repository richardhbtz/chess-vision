"""
Root entry point – delegates to the chess_vision package.

Usage:
    python chess_vision.py train     --pieces-root dataset/pieces --epochs 10
    python chess_vision.py recognize  --image game.png --weights checkpoints/best_classifier.pt
    python chess_vision.py export    --weights checkpoints/best_classifier.pt
"""

from chess_vision.main import main

if __name__ == "__main__":
    main()

