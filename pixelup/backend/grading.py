"""
Natural, "iPhone-inspired" colour grade.

Not a filter preset: every parameter is derived from the photo itself, then
applied conservatively so the result stays realistic.

  1. Gentle white balance  - partial gray-world correction on mid-tones (clamped).
  2. Smart-HDR-style tone  - the image is split into an edge-aware base layer and
                             detail; the base gets a soft S-curve + shadow lift,
                             the detail gets a small boost (local contrast).
  3. Auto exposure         - gentle gamma toward a balanced mean luminance.
  4. Vibrance              - boosts muted colours more than saturated ones.
  5. Skin protection       - a soft skin-hue mask reduces the vibrance boost and
                             warmth so faces never turn orange or plastic.
  6. Fine sharpening       - small unsharp mask on luminance only.

Work is done in horizontal strips so huge images do not need many full-size
float32 copies in RAM.
"""
import math

import cv2
import numpy as np

STRIP_ROWS = 768
OVERLAP = 16  # rows of context around each strip (>= 3 * sharpen sigma)


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _analyse(rgb_u8: np.ndarray) -> dict:
    """Derive global parameters from a small thumbnail of the image."""
    h, w = rgb_u8.shape[:2]
    f = 640.0 / max(h, w)
    thumb = rgb_u8 if f >= 1 else cv2.resize(rgb_u8, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    tf = thumb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(tf, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]

    # --- white balance: gray-world on low-chroma mid-tones, heavily damped ---
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    sel = (L > 25) & (L < 80) & (chroma < 30)
    gains = np.ones(3, np.float32)
    if sel.sum() > 500:
        mean = tf[sel].mean(axis=0)
        gray = float(mean.mean())
        raw = gray / np.maximum(mean, 1e-4)
        gains = 1.0 + 0.35 * (raw - 1.0)          # only correct 35% of the cast
        gains = np.clip(gains, 0.93, 1.07).astype(np.float32)

    # --- exposure: nudge mean L* toward ~50, never more than a gentle amount ---
    mean_l = float(np.clip(L.mean(), 8, 92)) / 100.0
    gamma = math.log(0.50) / math.log(mean_l)      # <1 brightens, >1 darkens
    gamma = 1.0 + 0.45 * (gamma - 1.0)
    gamma = float(np.clip(gamma, 0.86, 1.14))

    # --- edge-preserving base layer (low-res) for local tone mapping ---
    base = cv2.bilateralFilter(L.astype(np.float32), d=9, sigmaColor=14, sigmaSpace=6)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    return {"gains": gains, "gamma": gamma, "base": base, "thumb_hw": L.shape}


def _tone_curve(b: np.ndarray, gamma: float) -> np.ndarray:
    """b in 0..1 (L*/100). Soft S-curve + shadow lift + highlight roll-off."""
    b = np.power(np.clip(b, 0, 1), gamma)
    s = b * b * (3.0 - 2.0 * b)
    out = b + 0.30 * (s - b)                       # balanced contrast
    out = out + 0.035 * (1.0 - out) ** 3           # open up shadows slightly
    out = out - 0.020 * out ** 5                   # protect highlights
    return np.clip(out, 0.0, 1.0)


def _base_strip(base_small: np.ndarray, full_hw, y0: int, y1: int) -> np.ndarray:
    """Bilinearly sample the low-res base layer for output rows [y0, y1)."""
    H, W = full_hw
    sh, sw = base_small.shape
    sx, sy = sw / W, sh / H
    M = np.float32([[sx, 0, 0.5 * sx - 0.5], [0, sy, (y0 + 0.5) * sy - 0.5]])
    return cv2.warpAffine(base_small, M, (W, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _grade_strip(rgb_u8, base_up, p, strength):
    f = rgb_u8.astype(np.float32) / 255.0
    f *= p["gains"][None, None, :]
    np.clip(f, 0, 1, out=f)

    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # ---- local tone mapping: curve the base, keep (and lightly boost) detail ----
    detail = L - base_up
    new_base = _tone_curve(base_up / 100.0, p["gamma"]) * 100.0
    L2 = new_base + detail * (1.0 + 0.18 * strength)
    L2 = L + (L2 - L) * strength

    # ---- fine sharpening on luminance only ----
    blur = cv2.GaussianBlur(L2, (0, 0), 1.1)
    L2 = L2 + (0.30 * strength) * (L2 - blur)
    lab[..., 0] = np.clip(L2, 0, 100)

    # ---- skin mask (soft) in Lab hue/chroma space ----
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    skin = (_smoothstep(hue, 10, 24) * (1 - _smoothstep(hue, 58, 74))
            * _smoothstep(chroma, 8, 16) * (1 - _smoothstep(chroma, 48, 62))
            * _smoothstep(L2, 22, 38) * (1 - _smoothstep(L2, 93, 99)))

    # ---- vibrance: lift muted colours, leave saturated ones alone ----
    muted = 1.0 - np.clip(chroma / 85.0, 0, 1)
    boost = (0.04 + 0.22 * muted ** 1.5) * strength
    boost *= 1.0 - 0.75 * skin
    a *= 1.0 + boost
    b *= 1.0 + boost
    b += 1.4 * strength * (1.0 - 0.6 * skin)       # a whisper of warmth (iPhone look)

    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    np.clip(out, 0, 1, out=out)
    out *= 255.0
    return (out + 0.5).astype(np.uint8)


def apply_color_grade(rgb_u8: np.ndarray, strength: float = 1.0, progress=None) -> np.ndarray:
    """rgb_u8: HxWx3 uint8 -> graded HxWx3 uint8 (new array)."""
    H, W = rgb_u8.shape[:2]
    params = _analyse(rgb_u8)
    out = np.empty_like(rgb_u8)
    y = 0
    while y < H:
        y1 = min(H, y + STRIP_ROWS)
        c0, c1 = max(0, y - OVERLAP), min(H, y1 + OVERLAP)
        base_up = _base_strip(params["base"], (H, W), c0, c1)
        graded = _grade_strip(rgb_u8[c0:c1], base_up, params, strength)
        out[y:y1] = graded[y - c0:y - c0 + (y1 - y)]
        y = y1
        if progress:
            progress(y / H)
    return out
