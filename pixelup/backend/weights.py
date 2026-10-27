"""
Model weight management: downloads the official, MIT/BSD-licensed Real-ESRGAN
checkpoints from GitHub on first use (no account, no API key) and loads them.
"""
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .archs import RRDBNet, SRVGGNetCompact

log = logging.getLogger("pixelup")

MODELS_DIR = Path(os.environ.get("PIXELUP_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))

_BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"
MODELS = {
    # Fast: ~5 MB, good general-purpose model with built-in denoising.
    "fast": {
        "file": "realesr-general-x4v3.pth",
        "url": f"{_BASE}/v0.2.5.0/realesr-general-x4v3.pth",
        "min_bytes": 3_000_000,
        "key": "params",
        "build": lambda: SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4),
    },
    # Best: ~64 MB, the flagship RealESRGAN_x4plus. Slower, finest detail.
    "best": {
        "file": "RealESRGAN_x4plus.pth",
        "url": f"{_BASE}/v0.1.0/RealESRGAN_x4plus.pth",
        "min_bytes": 50_000_000,
        "key": "params_ema",
        "build": lambda: RRDBNet(num_feat=64, num_block=23, num_grow_ch=32),
    },
}

_download_lock = threading.Lock()
_cache: dict = {}
_cache_lock = threading.Lock()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def weights_path(name: str) -> Path:
    return MODELS_DIR / MODELS[name]["file"]


def ensure_weights(name: str, log=print) -> Path:
    """Download the checkpoint if missing. Safe to call from several threads."""
    spec = MODELS[name]
    path = weights_path(name)
    with _download_lock:
        if path.exists() and path.stat().st_size >= spec["min_bytes"]:
            return path
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log(f"[PixelUp] Downloading {spec['file']} (one-time) ...")
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "PixelUp/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last = 0, -1
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    if pct // 10 != last // 10:
                        log(f"[PixelUp]   {pct}%")
                        last = pct
        if tmp.stat().st_size < spec["min_bytes"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Download of {spec['file']} was incomplete. Please try again.")
        tmp.replace(path)
        log(f"[PixelUp] Saved {path}")
    return path


TILE, PAD = 232, 12          # a padded patch is exactly 256x256 -> fixed shape = fastest for ONNX Runtime / DirectML
FIXED = TILE + 2 * PAD


class TorchEngine:
    """Runs one x4 Real-ESRGAN model on uint8 RGB patches with PyTorch."""
    name = "PyTorch"

    def __init__(self, model, device, quality):
        self.model, self.device = model, device
        cuda = device.type == "cuda"
        self.tile, self.pad = (512, 16) if cuda else (TILE, PAD)
        self.sec_per_mp = None
        if not cuda:
            self.model = self.model.to(memory_format=torch.channels_last)
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    def run(self, patch: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 -> (4H)x(4W)x3 uint8."""
        with torch.inference_mode():
            t = torch.from_numpy(patch).to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            if self.device.type == "cpu":
                t = t.contiguous(memory_format=torch.channels_last)
            res = self.model(t).clamp_(0, 1).mul_(255.0).round_().byte()
            return res.squeeze(0).permute(1, 2, 0).cpu().numpy()


class OrtEngine:
    """Same network through ONNX Runtime (DirectML GPU on Windows, or optimised CPU)."""

    def __init__(self, session, label):
        self.session, self.name = session, label
        self.tile, self.pad = TILE, PAD
        self.sec_per_mp = None

    def run(self, patch: np.ndarray) -> np.ndarray:
        h, w = patch.shape[:2]
        if (h, w) != (FIXED, FIXED):                       # fixed-shape model: pad edge patches, crop afterwards
            patch = np.pad(patch, ((0, FIXED - h), (0, FIXED - w), (0, 0)), mode="edge")
        x = np.ascontiguousarray(patch.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        y = self.session.run(None, {"input": x})[0][0]
        y = (np.clip(y, 0, 1) * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)
        return np.ascontiguousarray(y[:h * 4, :w * 4])


def _export_onnx(cpu_model, path: Path):
    tmp = path.with_suffix(".tmp")
    dummy = torch.zeros(1, 3, FIXED, FIXED)
    kw = dict(input_names=["input"], output_names=["output"], opset_version=17)
    with torch.no_grad():
        try:
            torch.onnx.export(cpu_model, dummy, str(tmp), dynamo=False, **kw)
        except TypeError:                                   # older torch has no `dynamo` argument
            torch.onnx.export(cpu_model, dummy, str(tmp), **kw)
    tmp.replace(path)


def _ort_engines(name: str, cpu_model) -> dict:
    """Build every ONNX Runtime variant available on this machine (may be empty)."""
    import onnxruntime as ort
    path = MODELS_DIR / (MODELS[name]["file"].rsplit(".", 1)[0] + f"-{FIXED}.onnx")
    if not path.exists():
        log.info("[PixelUp] Converting %s to ONNX (one-time)...", name)
        _export_onnx(cpu_model, path)
    avail = ort.get_available_providers()
    variants = []
    if "DmlExecutionProvider" in avail:
        variants.append(("GPU (DirectML)", ["DmlExecutionProvider", "CPUExecutionProvider"]))
    if "CUDAExecutionProvider" in avail:
        variants.append(("GPU (CUDA)", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    variants.append(("CPU (ONNX Runtime)", ["CPUExecutionProvider"]))
    out = {}
    for label, providers in variants:
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = max(1, os.cpu_count() or 1)
            if providers[0] == "DmlExecutionProvider":      # settings recommended by Microsoft for DirectML
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            out[label] = OrtEngine(ort.InferenceSession(str(path), so, providers=providers), label)
        except Exception as e:
            log.warning("[PixelUp] %s unavailable: %s", label, e)
    return out


def _test_patch() -> np.ndarray:
    yy, xx = np.mgrid[0:FIXED, 0:FIXED]
    base = np.stack([xx * 255 // FIXED, yy * 255 // FIXED, (xx + yy) * 127 // FIXED], -1)
    noise = np.random.default_rng(0).integers(-12, 12, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _time(engine, patch, repeats=2) -> float:
    engine.run(patch)                                       # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        engine.run(patch)
    return (time.perf_counter() - t0) / repeats


def _choose_engine(name: str, torch_engine: TorchEngine, cpu_model):
    """Benchmark all engines on this machine and keep the fastest one that matches PyTorch's output."""
    patch = _test_patch()
    ref = torch_engine.run(patch).astype(np.int16)
    best, best_t = torch_engine, _time(torch_engine, patch)
    log.info("[PixelUp] %-20s %.2f s per tile", "PyTorch", best_t)
    if torch_engine.device.type == "cpu":                   # a CUDA PyTorch engine is already the fastest option
        try:
            candidates = _ort_engines(name, cpu_model)
        except Exception as e:
            log.info("[PixelUp] ONNX Runtime not used (%s). Run: pip install -r requirements-fast.txt", e)
            candidates = {}
        for label, eng in candidates.items():
            try:
                diff = float(np.abs(eng.run(patch).astype(np.int16) - ref).mean())
                if diff > 4.0:
                    log.warning("[PixelUp] %s output differs from PyTorch (%.1f) - skipped", label, diff)
                    continue
                t = _time(eng, patch)
                log.info("[PixelUp] %-20s %.2f s per tile", label, t)
                if t < best_t * 0.9:                        # must be clearly faster to switch
                    best, best_t = eng, t
            except Exception as e:
                log.warning("[PixelUp] %s failed (%s) - skipped", label, e)
    best.sec_per_mp = best_t / (best.tile * best.tile / 1e6)
    log.info("[PixelUp] Using %s for the '%s' model (~%.0f s per megapixel)", best.name, name, best.sec_per_mp)
    return best


def speeds() -> dict:
    """Seconds per input megapixel for each model that is already loaded."""
    return {k: round(e.sec_per_mp, 2) for k, e in _cache.items() if getattr(e, "sec_per_mp", None)}


def engine_names() -> dict:
    return {k: e.name for k, e in _cache.items()}


def load_model(name: str):
    """Return a ready engine, downloading + loading weights on first use."""
    with _cache_lock:
        if name in _cache:
            return _cache[name]
        spec = MODELS[name]
        path = ensure_weights(name)
        device = pick_device()
        model = spec["build"]()
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get(spec["key"], state)
        model.load_state_dict(state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        cpu_model = model
        if device.type != "cpu":
            import copy
            model = copy.deepcopy(model).to(device)
        _cache[name] = _choose_engine(name, TorchEngine(model, device, name), cpu_model)
        return _cache[name]
