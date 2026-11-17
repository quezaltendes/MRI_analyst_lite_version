from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import asyncio
from contextlib import suppress
import importlib.metadata
import json
import logging
from pathlib import Path
import tempfile
import threading
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import MAX_UPLOAD, NOTICE, ROOT, device_name
from .errors import InputError, ModelUnavailable
from .inputs import prepare_knee, unpack_zip
from .models import availability, checkpoint_path, predict_brain, predict_knee
from .reporting import EvidencePanels, description_status, generate_description
from .chat import ChatError, ChatRequest, chat_status, generate_reply
from .spine import predict_spine, spine_available
from .anatomy import detect_region

log = logging.getLogger("mri_doctor")
jobs = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mri-inference")
JOB_TTL = 3600
chat_gate = threading.BoundedSemaphore(1)
detect_gate = threading.BoundedSemaphore(1)


@asynccontextmanager
async def lifespan(app):
    import torch
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    async def cleanup_loop():
        while True:
            await asyncio.sleep(30)
            with lock:
                expire_jobs()
    cleanup_task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        executor.shutdown(wait=True, cancel_futures=False)


app = FastAPI(title="MRI Doctor — research inference", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "mri_doctor" / "static"), name="static")


@app.middleware("http")
async def local_boundary(request: Request, call_next):
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Cross-origin uploads are disabled."}, status_code=403)
        try:
            if int(request.headers.get("content-length", "0")) > MAX_UPLOAD + 1024**2:
                return JSONResponse({"detail": "The request exceeds the upload size limit."}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "mri_doctor" / "static" / "index.html")


@app.get("/api/status")
def status():
    versions = {}
    for name in ("torch", "monai", "timm", "pydicom", "nibabel"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"models": availability(), "device": device_name(), "notice": NOTICE,
            "max_upload_mb": MAX_UPLOAD // 1024**2, "versions": versions,
            "description": description_status(), "chat": chat_status(),
            "spine": {"installed": spine_available(), "kind":"segmentation"}}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if not detect_gate.acquire(blocking=False):
        await file.close()
        raise HTTPException(429,"Another region detection is in progress. Please wait.")
    workspace = None
    submitted = False
    try:
        with lock:
            if any(j["status"] in {"uploading","queued","running"} for j in jobs.values()):
                raise HTTPException(429,"Wait for study analysis to finish before detecting another region.")
        workspace = tempfile.TemporaryDirectory(prefix="mri-region-")
        path, _ = await save_upload(file,Path(workspace.name),"scan",0)
        future = executor.submit(detect_region,path,workspace.name)
        def finished(_):
            workspace.cleanup()
            detect_gate.release()
        future.add_done_callback(finished)
        submitted = True
        return await asyncio.shield(asyncio.wrap_future(future))
    except InputError as exc:
        raise HTTPException(422,str(exc)) from exc
    finally:
        await file.close()
        if not submitted:
            if workspace is not None: workspace.cleanup()
            detect_gate.release()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not chat_gate.acquire(blocking=False):
        raise HTTPException(429, "The assistant is answering another question. Please wait.")
    with lock:
        inference_busy = any(j["status"] in {"uploading", "queued", "running"} for j in jobs.values())
    if inference_busy:
        chat_gate.release()
        raise HTTPException(429, "Please wait for study analysis to finish before using chat.")
    try:
        future = executor.submit(generate_reply, [m.model_dump() for m in request.messages])
    except BaseException:
        chat_gate.release()
        raise
    # Even if a tab disconnects, release capacity only when its worker has actually stopped.
    future.add_done_callback(lambda _: chat_gate.release())
    try:
        return await asyncio.shield(asyncio.wrap_future(future))
    except ChatError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


def expire_jobs():
    now = time.time()
    for key in list(jobs):
        if jobs[key]["status"] in {"complete", "error"} and now - jobs[key]["updated_at"] > JOB_TTL:
            del jobs[key]


def update_job(job_id, **data):
    with lock:
        jobs[job_id].update(data, updated_at=time.time())


def process(job_id, workspace, mode, task, paths, include_description=True):
    start = time.perf_counter()
    evidence = EvidencePanels()
    try:
        update_job(job_id, status="running", phase="Validating study")
        folder = Path(workspace.name)
        if mode == "knee":
            unpacked = folder / "unpacked"
            unpacked.mkdir()
            unpack_zip(paths[0], unpacked)
            series, warnings = prepare_knee(unpacked, folder / "series")
            update_job(job_id, phase="Preparing slices and running inference")
            result = predict_knee(series, warnings, evidence_sink=evidence if include_description and description_status()["available"] else None)
        elif mode == "spine":
            update_job(job_id, phase="Segmenting the spinal cord")
            result = predict_spine(paths[0])
        else:
            update_job(job_id, phase="Preparing brain images and running inference")
            result = predict_brain(task, paths, evidence_sink=evidence if include_description and description_status()["available"] else None)
        if include_description and mode != "spine":
            update_job(job_id, phase="Writing description")
            result["description"] = generate_description(mode, task, evidence, folder)
        else:
            result["description"] = {"status": "not_requested"}
        result.update(mode=mode, task=task if mode == "brain" else "cord_segmentation" if mode == "spine" else "12_findings", notice=NOTICE,
                      elapsed_seconds=round(time.perf_counter() - start, 2),
                      created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        update_job(job_id, status="complete", phase="Complete", result=result)
    except (InputError, ModelUnavailable) as e:
        update_job(job_id, status="error", error=str(e), phase="Processing stopped")
    except Exception as e:
        # Do not log uploaded filenames, headers, or patient identifiers.
        log.error("Inference failed: %s", type(e).__name__)
        update_job(job_id, status="error", error="Processing failed. Check the input format, dependencies, and model weight compatibility.",
                   phase="Processing stopped")
    finally:
        for panel in evidence:
            panel.image.close()
        workspace.cleanup()


async def save_upload(upload, folder, stem, total):
    if upload is None:
        raise HTTPException(422, "A required file is missing.")
    suffix = ".nii.gz" if (upload.filename or "").lower().endswith(".nii.gz") else Path(upload.filename or "").suffix.lower()
    if suffix not in {".zip", ".nii", ".nii.gz"}:
        raise HTTPException(422, "Use ZIP for knee studies and .nii/.nii.gz for brain scans.")
    path = folder / (stem + suffix)
    n = total
    with path.open("wb") as f:
        while chunk := await upload.read(1024**2):
            n += len(chunk)
            if n > MAX_UPLOAD:
                raise HTTPException(413, "The combined file size exceeds the upload limit.")
            f.write(chunk)
    if path.stat().st_size == 0:
        raise HTTPException(422, "The uploaded file is empty.")
    return path, n


@app.post("/api/jobs", status_code=202)
async def create_job(mode: Literal["knee", "brain", "spine"] = Form(...),
                     task: Literal["mci", "idh"] = Form("mci"),
                     preprocessed: bool = Form(False),
                     include_description: bool = Form(True),
                     spine_confirmed: bool = Form(False),
                     file: UploadFile | None = File(None),
                     flair: UploadFile | None = File(None),
                     t1ce: UploadFile | None = File(None)):
    key = "knee" if mode == "knee" else task
    uploads = [x for x in (file, flair, t1ce) if x is not None]
    workspace = None
    reserved_id = None
    try:
        if mode == "brain" and not preprocessed:
            raise HTTPException(422, "For BrainIAC, confirm N4 correction, template registration, and skull stripping.")
        if mode == "spine" and not spine_confirmed:
            raise HTTPException(422,"Confirm that this is a T2 MRI covering the spinal cord.")
        expected = [flair, t1ce] if mode == "brain" and task == "idh" else [file]
        if any(x is None for x in expected) or len(uploads) != len(expected):
            raise HTTPException(422, "Knee/MCI: one file. IDH: flair and t1ce only, both are required.")
        for upload in expected:
            name = (upload.filename or "").lower()
            if mode == "knee" and not name.endswith(".zip"):
                raise HTTPException(422, "Knee mode requires a ZIP containing DICOM series.")
            if mode in {"brain","spine"} and not name.endswith((".nii", ".nii.gz")):
                raise HTTPException(422, "This mode requires NIfTI (.nii/.nii.gz). Extract ZIP files before selecting a volume.")
        if mode == "spine" and not spine_available():
            raise HTTPException(503,"Spinal cord weights are missing. Run scripts/setup_spine.py.")
        if mode != "spine" and not availability()[key]["installed"]:
            raise HTTPException(503, "The selected model weights are not installed yet. Run scripts/download_models.py.")
        with lock:
            expire_jobs()
            if sum(j["status"] in {"uploading", "queued", "running"} for j in jobs.values()) >= 2:
                raise HTTPException(429, "The model is busy; wait for the current job to finish.")
            reserved_id = uuid4().hex
            jobs[reserved_id] = {"id": reserved_id, "status": "uploading", "phase": "Uploading",
                                 "updated_at": time.time()}
        workspace = tempfile.TemporaryDirectory(prefix="mri-doctor-")
        paths, total = [], 0
        for i, upload in enumerate(expected):
            path, total = await save_upload(upload, Path(workspace.name), str(i), total)
            paths.append(path)
        update_job(reserved_id, status="queued", phase="Queued")
        executor.submit(process, reserved_id, workspace, mode, task, paths, include_description)
        return {"id": reserved_id, "status": "queued"}
    except BaseException:
        if workspace is not None:
            workspace.cleanup()
        if reserved_id:
            with lock:
                jobs.pop(reserved_id, None)
        raise
    finally:
        for upload in uploads:
            await upload.close()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with lock:
        expire_jobs()
        if job_id not in jobs:
            raise HTTPException(404, "The job was not found or its result has expired (after 1 hour).")
        snapshot = dict(jobs[job_id])
        if "result" in snapshot:
            snapshot["result"] = {k:v for k,v in snapshot["result"].items() if k != "_mask_nifti_gz"}
        return snapshot


@app.get("/api/jobs/{job_id}/mask")
def download_mask(job_id: str):
    with lock:
        expire_jobs()
        result = jobs.get(job_id,{}).get("result",{})
        payload = result.get("_mask_nifti_gz")
        if payload is None: raise HTTPException(404,"The segmentation mask is unavailable or expired.")
    return Response(content=payload,media_type="application/gzip",
                    headers={"Content-Disposition":'attachment; filename="spinal-cord-mask.nii.gz"'})


@app.delete("/api/jobs/{job_id}")
def forget_job(job_id: str):
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job not found.")
        if jobs[job_id]["status"] not in {"complete", "error"}:
            raise HTTPException(409, "Wait for processing to finish.")
        jobs.pop(job_id)
    return {"deleted": True}
