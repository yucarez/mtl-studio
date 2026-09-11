"""
HTTP API + static frontend.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000     (from ./backend)
  or: python run.py                                          (from the project root)
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config
from .config import PipelineOptions
from .engines import inpaint_engines as ie
from .engines import ocr_engines as oe
from .jobs import JobManager
from .pipeline import fonts
from .pipeline.pipeline import STAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
app = FastAPI(title="MTL Studio", version="1.0")
jobs = JobManager(config.DATA_DIR)


def _item(iid: str):
    it = jobs.items.get(iid)
    if not it:
        raise HTTPException(404, "Image not found")
    return it


def _item_json(it) -> dict:
    d = {k: getattr(it, k) for k in ("id", "batch_id", "filename", "status", "stage", "stage_label", "progress",
                                     "error", "warnings", "errors", "regions", "kept_original", "width",
                                     "height", "meta", "seconds")}
    d["has_result"] = bool(it.result_ext)
    d["download_name"] = jobs.download_name(it) if it.result_ext else None
    return d


# ------------------------------------------------------------------ config
@app.get("/api/config")
def get_config():
    st = oe.engine_status()
    warnings = []
    if not (st["paddle"] or st["easyocr"]):
        warnings.append("No OCR engine installed. Install PaddleOCR (recommended) or EasyOCR - see README.")
    if not st["lama"]:
        warnings.append("LaMa inpainting is not installed; textured backgrounds use the OpenCV fallback.")
    if config.configured_translators() == ["passthrough"]:
        warnings.append("No translation API key configured in .env - only the layout test mode is available.")
    if not fonts.available_families():
        warnings.append("No usable fonts found. Run: python scripts/download_fonts.py")
    return {
        "languages": [{"code": l.code, "name": l.name} for l in config.LANGS.values()],
        "translators": [{"id": t, "configured": t in config.configured_translators()}
                        for t in ("anthropic", "openai", "deepl", "google", "passthrough")],
        "engines": st,
        "fonts": fonts.available_families(),
        "defaults": PipelineOptions().to_dict(),
        "stages": [{"id": s, "label": l} for s, l in STAGES],
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "models": {"anthropic": config.ANTHROPIC_MODEL, "openai": config.OPENAI_MODEL},
        "warnings": warnings,
    }


# ------------------------------------------------------------------ batches
@app.post("/api/batches")
async def create_batch(files: list[UploadFile] = File(...), options: str = Form("{}")):
    try:
        opts = json.loads(options or "{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "options must be JSON")
    payload = []
    for f in files:
        data = await f.read()
        payload.append((f.filename or "image.png", data))
    try:
        b = jobs.create_batch(payload, opts)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": b.id, "items": [_item_json(jobs.items[i]) for i in b.item_ids]}


@app.get("/api/batches/{bid}")
def get_batch(bid: str):
    b = jobs.batches.get(bid)
    if not b:
        raise HTTPException(404, "Batch not found")
    return {"id": b.id, "options": b.options,
            "items": [_item_json(jobs.items[i]) for i in b.item_ids if i in jobs.items]}


@app.get("/api/batches")
def list_batches():
    return [{"id": b.id, "created": b.created, "count": len(b.item_ids)}
            for b in sorted(jobs.batches.values(), key=lambda b: -b.created)]


@app.post("/api/batches/{bid}/retry-failed")
def retry_failed(bid: str, options: Optional[dict] = Body(None)):
    b = jobs.batches.get(bid)
    if not b:
        raise HTTPException(404, "Batch not found")
    n = 0
    for iid in b.item_ids:
        it = jobs.items.get(iid)
        if it and it.status == "failed":
            jobs.retry(iid, options)
            n += 1
    return {"retried": n}


@app.delete("/api/batches/{bid}")
def delete_batch(bid: str):
    jobs.delete_batch(bid)
    return {"ok": True}


@app.get("/api/batches/{bid}/zip")
def batch_zip(bid: str):
    b = jobs.batches.get(bid)
    if not b:
        raise HTTPException(404, "Batch not found")
    data = jobs.zip_bytes(b.item_ids)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="translated-{bid}.zip"'})


@app.post("/api/zip")
def items_zip(item_ids: list[str] = Body(..., embed=True)):
    data = jobs.zip_bytes(item_ids)
    if len(data) <= 22:
        raise HTTPException(400, "None of those images are finished yet")
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="translated.zip"'})


# ------------------------------------------------------------------ items
@app.get("/api/items/{iid}")
def get_item(iid: str):
    it = _item(iid)
    d = _item_json(it)
    d["state"] = jobs.state(it)
    return d


@app.get("/api/items/{iid}/original")
def item_original(iid: str):
    return FileResponse(jobs.original_path(_item(iid)))


@app.get("/api/items/{iid}/work")
def item_work(iid: str):
    """The upright (EXIF/page-rotation corrected) source used for processing."""
    it = _item(iid)
    p = jobs.item_dir(it) / "work" / "work.png"
    return FileResponse(p if p.exists() else jobs.original_path(it))


@app.get("/api/items/{iid}/result")
def item_result(iid: str, download: bool = False):
    it = _item(iid)
    p = jobs.result_path(it)
    if not p:
        raise HTTPException(404, "No result yet")
    return FileResponse(p, filename=jobs.download_name(it) if download else None,
                        headers={"Cache-Control": "no-store"})


@app.get("/api/items/{iid}/thumb")
def item_thumb(iid: str, which: str = "original"):
    it = _item(iid)
    p = jobs.item_dir(it) / ("result_thumb.jpg" if which == "result" else "thumb.jpg")
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.get("/api/items/{iid}/overlay")
def item_overlay(iid: str):
    p = jobs.item_dir(_item(iid)) / "work" / "overlay.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.post("/api/items/{iid}/retry")
def retry_item(iid: str, options: Optional[dict] = Body(None)):
    _item(iid)
    try:
        return _item_json(jobs.retry(iid, options))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/items/{iid}/rerender")
def rerender_item(iid: str, edits: dict = Body(..., embed=True)):
    it = _item(iid)
    try:
        jobs.rerender(it.id, {int(k): str(v) for k, v in edits.items()})
    except ValueError as e:
        raise HTTPException(400, str(e))
    d = _item_json(it)
    d["state"] = jobs.state(it)
    return d


@app.get("/api/health")
def health():
    return {"ok": True, "queue": jobs.q.qsize(), "lama": ie.lama_available()}


@app.exception_handler(Exception)
async def unhandled(request, exc):  # pragma: no cover
    logging.getLogger("mtl").exception("unhandled error")
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
