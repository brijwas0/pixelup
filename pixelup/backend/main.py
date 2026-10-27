"""
PixelUp - FastAPI server.

Flow (all in memory; photos are never written to disk):
  POST   /api/enhance            upload + options  -> {"job_id"}
  GET    /api/jobs/{id}          progress / status
  GET    /api/jobs/{id}/result   enhanced image (deleted from the server right after it is sent)
  DELETE /api/jobs/{id}          cancel / discard
  GET    /api/health
Static frontend is served from ../frontend.
"""
import asyncio
import logging
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from . import pipeline
from .pipeline import Cancelled, ImageError

log = logging.getLogger("pixelup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_UPLOAD_MB = float(os.environ.get("PIXELUP_MAX_UPLOAD_MB", 25))
JOB_TTL_SECONDS = int(os.environ.get("PIXELUP_JOB_TTL", 600))     # finished results are discarded after this
MAX_ACTIVE_JOBS = int(os.environ.get("PIXELUP_MAX_ACTIVE_JOBS", 3))  # running + waiting
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

def _warmup():
    try:
        from . import weights
        weights.load_model("fast")
    except Exception:
        log.exception("Model warm-up failed (it will be retried on the first photo)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_janitor())
    threading.Thread(target=_warmup, daemon=True).start()   # load + benchmark the model before the first photo
    yield
    task.cancel()


app = FastAPI(title="PixelUp", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="enhance")  # one heavy job at a time


class Job:
    def __init__(self):
        self.status = "queued"          # queued | processing | done | error | cancelled
        self.progress = 0.0
        self.message = "Waiting in line…"
        self.error: Optional[str] = None
        self.notice: Optional[str] = None
        self.result: Optional[pipeline.Result] = None
        self.cancelled = False
        self.created = time.time()
        self.updated = time.time()


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def _active_count() -> int:
    return sum(1 for j in jobs.values() if j.status in ("queued", "processing"))


def _run_job(job: Job, data: bytes, scale: int, quality: str, grade: bool, denoise: bool):
    def on_progress(frac: float, msg: str):
        if job.cancelled:
            raise Cancelled()
        job.progress, job.message, job.updated = max(job.progress, min(frac, 0.99)), msg, time.time()

    try:
        if job.cancelled:
            raise Cancelled()
        job.status = "processing"
        job.result = pipeline.enhance(data, scale, quality, grade, denoise, on_progress)
        job.notice = job.result.notice
        job.progress, job.message, job.status = 1.0, "Done", "done"
    except Cancelled:
        job.status, job.message = "cancelled", "Cancelled"
    except ImageError as e:
        job.status, job.error = "error", str(e)
    except MemoryError:
        job.status, job.error = "error", "Not enough memory for this photo. Try a smaller image or the 2× option."
    except Exception as e:  # never leak internals to the browser
        log.exception("Enhancement failed")
        msg = "Something went wrong while enhancing the photo."
        if "download" in str(e).lower() or "urlopen" in repr(e).lower():
            msg = "The AI model could not be downloaded. Check your internet connection and try again."
        job.status, job.error = "error", msg
    finally:
        job.updated = time.time()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


async def _janitor():
    """Drop abandoned jobs (and the photos inside them) after the TTL."""
    while True:
        await asyncio.sleep(30)
        now = time.time()
        with jobs_lock:
            for jid in [k for k, j in jobs.items()
                        if j.status in ("done", "error", "cancelled") and now - j.updated > JOB_TTL_SECONDS]:
                jobs.pop(jid, None)


@app.get("/api/health")
def health():
    from . import weights  # imported lazily so the server can start even if torch is slow to load
    return {"status": "ok", "device": str(weights.pick_device()),
            "models": {k: weights.weights_path(k).exists() for k in weights.MODELS},
            "engines": getattr(weights, "engine_names", lambda: {})(),
            "sec_per_mp": getattr(weights, "speeds", lambda: {})(),
            "max_upload_mb": MAX_UPLOAD_MB}


@app.post("/api/enhance", status_code=202)
async def enhance(request: Request,
                  file: UploadFile = File(...),
                  scale: int = Form(2),
                  quality: str = Form("fast"),
                  grade: bool = Form(True),
                  denoise: bool = Form(True)):
    if scale not in (2, 4):
        raise HTTPException(400, "Scale must be 2× or 4×.")
    if quality not in ("fast", "best"):
        raise HTTPException(400, "Unknown quality mode.")

    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + 64 * 1024:
        raise HTTPException(413, f"That file is too large. Maximum size is {MAX_UPLOAD_MB:.0f} MB.")

    chunks, size = [], 0
    while True:                                   # stream in, enforcing the limit as we go
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"That file is too large. Maximum size is {MAX_UPLOAD_MB:.0f} MB.")
        chunks.append(chunk)
    data = b"".join(chunks)
    del chunks
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")

    with jobs_lock:
        if _active_count() >= MAX_ACTIVE_JOBS:
            raise HTTPException(429, "PixelUp is busy right now. Please try again in a minute.")
        job_id = secrets.token_urlsafe(16)
        job = jobs[job_id] = Job()

    # Cheap pre-validation so obviously bad files fail instantly, not after queueing.
    try:
        if pipeline._sniff(data) is None:
            raise ImageError("That file isn't a JPG, PNG or WEBP image.")
    except ImageError as e:
        with jobs_lock:
            jobs.pop(job_id, None)
        raise HTTPException(415, str(e))

    executor.submit(_run_job, job, data, scale, quality, grade, denoise)
    return {"job_id": job_id}


def _get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "This job no longer exists. Please upload your photo again.")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _get_job(job_id)
    body = {"status": job.status, "progress": round(job.progress, 3), "message": job.message,
            "error": job.error, "notice": job.notice}
    if job.result:
        body.update(width=job.result.width, height=job.result.height, bytes=len(job.result.data))
    return body


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _get_job(job_id)
    if job.status != "done" or not job.result:
        raise HTTPException(409, "The photo isn't ready yet.")
    res = job.result

    def discard():                                 # one-shot: the server forgets the photo once delivered
        with jobs_lock:
            jobs.pop(job_id, None)

    return Response(res.data, media_type=res.mime,
                    headers={"Content-Disposition": f'attachment; filename="pixelup.{res.ext}"',
                             "X-Width": str(res.width), "X-Height": str(res.height)},
                    background=BackgroundTask(discard))


@app.delete("/api/jobs/{job_id}", status_code=204)
def job_cancel(job_id: str):
    job = jobs.get(job_id)
    if job:
        job.cancelled = True
        if job.status != "processing":
            with jobs_lock:
                jobs.pop(job_id, None)
    return Response(status_code=204)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
