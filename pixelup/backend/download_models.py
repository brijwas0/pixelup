"""Pre-download the Real-ESRGAN weights:  python -m backend.download_models [--fast-only]"""
import sys

from .weights import MODELS, ensure_weights

if __name__ == "__main__":
    names = ["fast"] if "--fast-only" in sys.argv else list(MODELS)
    for n in names:
        ensure_weights(n)
    print("[PixelUp] Models ready:", ", ".join(names))
