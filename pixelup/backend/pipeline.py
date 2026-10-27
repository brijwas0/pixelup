"""
The PixelUp enhancement pipeline. Everything happens in memory: the uploaded
bytes go in, the enhanced image bytes come out, nothing is written to disk.

decode/validate -> colour-manage to sRGB -> noise reduction (only if noisy)
-> Real-ESRGAN tiled super-resolution (2x / 4x) -> colour grade -> encode
"""
import io
import os
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .grading import apply_color_grade
from .weights import load_model, weights_path

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_DECODE_MP = float(os.environ.get("PIXELUP_MAX_DECODE_MP", 100))   # reject anything bigger
MAX_OUTPUT_MP = float(os.environ.get("PIXELUP_MAX_OUTPUT_MP", 48))    # input is shrunk to fit this
MIN_SIDE = 16
Image.MAX_IMAGE_PIXELS = int(MAX_DECODE_MP * 1_000_000)


class ImageError(Exception):
    """A problem with the user's image that we can explain to them."""


class Cancelled(Exception):
    pass


@dataclass
class Result:
    data: bytes
    mime: str
    ext: str
    width: int
    height: int
    notice: Optional[str] = None


# --------------------------------------------------------------------------- #
# Decoding & validation
# --------------------------------------------------------------------------- #
def _sniff(data: bytes) -> Optional[str]:
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


_SRGB = None


def _srgb_profile():
    global _SRGB
    if _SRGB is None:
        _SRGB = ImageCms.createProfile("sRGB")
    return _SRGB


def decode_image(data: bytes):
    """Return (rgb ndarray uint8, alpha ndarray|None, notes list)."""
    notes = []
    if _sniff(data) is None:
        raise ImageError("That file isn't a JPG, PNG or WEBP image.")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()                       # cheap integrity check
        im = Image.open(io.BytesIO(data))    # verify() invalidates the handle
        if im.format not in ALLOWED_FORMATS:
            raise ImageError("Only JPG, PNG and WEBP images are supported.")
        w, h = im.size
        if min(w, h) < MIN_SIDE:
            raise ImageError(f"The image is too small (minimum {MIN_SIDE}px per side).")
        if w * h > MAX_DECODE_MP * 1_000_000:
            raise ImageError(f"The image is too large ({w * h / 1e6:.0f} MP). Maximum is {MAX_DECODE_MP:.0f} MP.")
        if getattr(im, "n_frames", 1) > 1:
            notes.append("Animated image: only the first frame was enhanced.")
        im.load()
    except ImageError:
        raise
    except Image.DecompressionBombError:
        raise ImageError(f"The image is too large. Maximum is {MAX_DECODE_MP:.0f} megapixels.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageError("The image appears to be corrupted or unreadable.")

    icc = im.info.get("icc_profile")
    im = ImageOps.exif_transpose(im)         # honour phone-camera rotation

    if im.mode.startswith("I"):              # 16-bit greyscale PNG
        arr = np.asarray(im).astype(np.float32)
        arr = arr / (65535.0 if arr.max() > 255 else 255.0) * 255.0
        im = Image.fromarray(arr.astype(np.uint8), "L")

    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    alpha = None
    if has_alpha:
        im = im.convert("RGBA")
        alpha = np.asarray(im.getchannel("A"))
    rgb = im.convert("RGB")

    if icc:                                   # e.g. iPhone "Display P3" -> sRGB
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            rgb = ImageCms.profileToProfile(rgb, src, _srgb_profile(), outputMode="RGB")
        except Exception:
            pass                              # unknown profile: assume sRGB
    return np.ascontiguousarray(np.asarray(rgb)), alpha, notes


# --------------------------------------------------------------------------- #
# Noise reduction
# --------------------------------------------------------------------------- #
def estimate_noise(rgb: np.ndarray) -> float:
    """Immerkaer fast noise-sigma estimate (0-255 scale) on a central crop."""
    h, w = rgb.shape[:2]
    s = 1024
    y0, x0 = max(0, (h - s) // 2), max(0, (w - s) // 2)
    gray = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y0 + s, x0:x0 + s]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(gray, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv).mean() / 6.0)


def denoise_if_noisy(rgb: np.ndarray) -> np.ndarray:
    sigma = estimate_noise(rgb)
    if sigma < 2.5:
        return rgb
    strength = float(np.clip(sigma * 0.8, 3.0, 9.0))
    return cv2.fastNlMeansDenoisingColored(rgb, None, strength, strength, 5, 15)


# --------------------------------------------------------------------------- #
# Real-ESRGAN, tiled
# --------------------------------------------------------------------------- #
def super_resolve(rgb: np.ndarray, scale: int, quality: str, progress: Callable[[float], None]) -> np.ndarray:
    engine = load_model(quality)
    H, W = rgb.shape[:2]
    tile, pad = engine.tile, engine.pad
    out = np.empty((H * scale, W * scale, 3), np.uint8)

    ys = list(range(0, H, tile))
    xs = list(range(0, W, tile))
    total, n = len(ys) * len(xs), 0

    for y in ys:
        for x in xs:
            y1, x1 = min(H, y + tile), min(W, x + tile)
            py0, py1 = max(0, y - pad), min(H, y1 + pad)      # padded patch avoids visible tile seams
            px0, px1 = max(0, x - pad), min(W, x1 + pad)
            patch = np.ascontiguousarray(rgb[py0:py1, px0:px1])
            res = engine.run(patch)                           # 4x result
            if scale != 4:                                    # models are x4; 2x = area-downsample the tile
                res = cv2.resize(res, (patch.shape[1] * scale, patch.shape[0] * scale), interpolation=cv2.INTER_AREA)
            oy0, ox0 = (y - py0) * scale, (x - px0) * scale
            out[y * scale:y1 * scale, x * scale:x1 * scale] = res[oy0:oy0 + (y1 - y) * scale, ox0:ox0 + (x1 - x) * scale]
            n += 1
            progress(n / total)
    return out


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def encode(rgb: np.ndarray, alpha: Optional[np.ndarray]):
    buf = io.BytesIO()
    if alpha is not None:
        im = Image.fromarray(rgb, "RGB")
        im.putalpha(Image.fromarray(alpha, "L"))
        im.save(buf, "PNG", compress_level=4)
        return buf.getvalue(), "image/png", "png"
    Image.fromarray(rgb, "RGB").save(buf, "JPEG", quality=96, subsampling=0, optimize=True)
    return buf.getvalue(), "image/jpeg", "jpg"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def enhance(data: bytes, scale: int, quality: str, grade: bool, denoise: bool,
            on_progress: Callable[[float, str], None]) -> Result:
    if scale not in (2, 4):
        raise ImageError("Scale must be 2x or 4x.")
    if quality not in ("fast", "best"):
        raise ImageError("Unknown quality mode.")

    on_progress(0.01, "Reading your photo…")
    rgb, alpha, notes = decode_image(data)
    del data

    # Keep the output within the memory budget by shrinking oversized inputs first.
    h, w = rgb.shape[:2]
    if (h * scale) * (w * scale) > MAX_OUTPUT_MP * 1e6:
        f = (MAX_OUTPUT_MP * 1e6 / ((h * scale) * (w * scale))) ** 0.5
        nw, nh = max(MIN_SIDE, int(w * f)), max(MIN_SIDE, int(h * f))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        notes.append(f"Large photo: it was reduced to {nw}×{nh} before enhancing so the {scale}× result stays within "
                     f"{MAX_OUTPUT_MP:g} megapixels.")
        h, w = nh, nw

    if denoise:
        on_progress(0.06, "Reducing noise…")
        rgb = denoise_if_noisy(rgb)

    if not weights_path(quality).exists():
        on_progress(0.08, "Downloading the AI model (one-time)…")
    on_progress(0.10, "Enhancing detail with AI…")
    rgb = super_resolve(rgb, scale, quality,
                        lambda f: on_progress(0.10 + 0.72 * f, f"Enhancing detail with AI… {int(f * 100)}%"))

    if alpha is not None:
        alpha = cv2.resize(alpha, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    if grade:
        on_progress(0.83, "Applying natural colour…")
        rgb = apply_color_grade(rgb, 1.0, lambda f: on_progress(0.83 + 0.13 * f, "Applying natural colour…"))

    on_progress(0.97, "Preparing your download…")
    payload, mime, ext = encode(rgb, alpha)
    return Result(payload, mime, ext, w * scale, h * scale, " ".join(notes) or None)
