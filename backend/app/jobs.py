"""
Batch/job manager.

* One background worker processes items in upload order (natural filename sort, so a manga
  chapter's page 2 comes after page 1 and translation context flows page to page).
* Models are loaded once and shared; engines are lock-guarded.
* Every item is persisted to DATA_DIR so results survive a server restart.
"""
from __future__ import annotations

import io
import json
import logging
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from . import config
from .config import PipelineOptions
from .pipeline import pipeline as P
from .engines.translators import TranslationError
from .pipeline.datatypes import PipelineError

log = logging.getLogger("mtl.jobs")
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


@dataclass
class Item:
    id: str
    batch_id: str
    filename: str
    status: str = "queued"          # queued | processing | done | failed
    stage: str = "queued"
    stage_label: str = "Waiting"
    progress: float = 0.0
    error: str = ""
    warnings: int = 0
    errors: int = 0
    regions: int = 0
    kept_original: int = 0
    result_ext: str = ""
    width: int = 0
    height: int = 0
    meta: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    seconds: float = 0.0


@dataclass
class Batch:
    id: str
    options: dict
    item_ids: list[str] = field(default_factory=list)
    history: list = field(default_factory=list)
    created: float = field(default_factory=time.time)


class JobManager:
    def __init__(self, root: Path):
        self.root = root / "batches"
        self.root.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, Item] = {}
        self.batches: dict[str, Batch] = {}
        self.lock = threading.RLock()
        self.q: "queue.Queue[str]" = queue.Queue()
        self._load()
        self.worker = threading.Thread(target=self._run, name="mtl-worker", daemon=True)
        self.worker.start()

    # ---------------------------------------------------------------- paths
    def item_dir(self, it: Item) -> Path:
        return self.root / it.batch_id / it.id

    def original_path(self, it: Item) -> Path:
        d = self.item_dir(it)
        return next(iter(sorted(d.glob("original.*"))), d / "original.png")

    def result_path(self, it: Item) -> Optional[Path]:
        p = self.item_dir(it) / f"result.{it.result_ext}"
        return p if it.result_ext and p.exists() else None

    # ---------------------------------------------------------------- persistence
    def _save_item(self, it: Item):
        d = self.item_dir(it)
        d.mkdir(parents=True, exist_ok=True)
        (d / "item.json").write_text(json.dumps(asdict(it), ensure_ascii=False, default=str), encoding="utf-8")

    def _save_batch(self, b: Batch):
        d = self.root / b.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "batch.json").write_text(json.dumps(asdict(b), ensure_ascii=False, default=str), encoding="utf-8")

    def _load(self):
        for bj in sorted(self.root.glob("*/batch.json")):
            try:
                b = Batch(**json.loads(bj.read_text(encoding="utf-8")))
            except Exception:
                continue
            self.batches[b.id] = b
            for iid in b.item_ids:
                ij = bj.parent / iid / "item.json"
                if not ij.exists():
                    continue
                try:
                    it = Item(**json.loads(ij.read_text(encoding="utf-8")))
                except Exception:
                    continue
                if it.status in ("queued", "processing"):
                    it.status, it.error, it.stage_label = "failed", "Interrupted by a server restart. Retry it.", "Interrupted"
                self.items[it.id] = it

    # ---------------------------------------------------------------- API
    def create_batch(self, files: list[tuple[str, bytes]], options: dict) -> Batch:
        opts = PipelineOptions.from_dict(options)          # validates languages
        if opts.translator not in ("auto", *config.configured_translators()):
            raise ValueError(f"Translator '{opts.translator}' is not configured. "
                             f"Add its API key to .env (see README).")
        b = Batch(id=uuid.uuid4().hex[:12], options=opts.to_dict())
        files = sorted(files, key=lambda f: natural_key(f[0]))
        for name, data in files:
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_EXT:
                raise ValueError(f"{name}: unsupported type (use PNG, JPG, JPEG or WEBP)")
            if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"{name}: larger than {config.MAX_UPLOAD_MB} MB")
            try:
                im = Image.open(io.BytesIO(data))
                im.verify()
            except Exception:
                raise ValueError(f"{name}: not a readable image")
            it = Item(id=uuid.uuid4().hex[:12], batch_id=b.id, filename=Path(name).name)
            d = self.item_dir(it)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"original{ext}").write_bytes(data)
            try:
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
                it.width, it.height = im.size
                self._thumb(im, d / "thumb.jpg")
            except Exception:
                pass
            b.item_ids.append(it.id)
            with self.lock:
                self.items[it.id] = it
            self._save_item(it)
        with self.lock:
            self.batches[b.id] = b
        self._save_batch(b)
        for iid in b.item_ids:
            self.q.put(iid)
        return b

    def retry(self, iid: str, options: Optional[dict] = None):
        it = self.items[iid]
        if it.status in ("queued", "processing"):
            return it
        if options:
            b = self.batches[it.batch_id]
            b.options = PipelineOptions.from_dict({**b.options, **options}).to_dict()
            self._save_batch(b)
        it.status, it.stage, it.stage_label, it.progress, it.error = "queued", "queued", "Waiting", 0.0, ""
        self._save_item(it)
        self.q.put(iid)
        return it

    def rerender(self, iid: str, edits: dict[int, str]) -> Item:
        it = self.items[iid]
        if it.status != "done":
            raise ValueError("Only finished images can be re-rendered")
        b = self.batches[it.batch_id]
        opts = PipelineOptions.from_dict(b.options)
        res = P.rerender(self.item_dir(it) / "work", edits, opts, self.original_path(it).read_bytes())
        self._store_result(it, res)
        return it

    def delete_batch(self, bid: str):
        with self.lock:
            b = self.batches.pop(bid, None)
            if b:
                for iid in b.item_ids:
                    self.items.pop(iid, None)
        shutil.rmtree(self.root / bid, ignore_errors=True)

    def state(self, it: Item) -> Optional[dict]:
        p = self.item_dir(it) / "work" / "state.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def zip_bytes(self, item_ids: list[str]) -> bytes:
        buf = io.BytesIO()
        used = set()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for iid in item_ids:
                it = self.items.get(iid)
                if not it or it.status != "done":
                    continue
                rp = self.result_path(it)
                if not rp:
                    continue
                name = self.download_name(it)
                k, base = 2, name
                while name in used:
                    stem, ext = base.rsplit(".", 1)
                    name, k = f"{stem}-{k}.{ext}", k + 1
                used.add(name)
                z.write(rp, name)
        return buf.getvalue()

    def download_name(self, it: Item) -> str:
        tgt = self.batches[it.batch_id].options.get("target_lang", "en")
        return f"{Path(it.filename).stem}.{tgt}.{it.result_ext}"

    # ---------------------------------------------------------------- worker
    def _thumb(self, im: Image.Image, path: Path, size: int = 420):
        t = im.convert("RGB")
        t.thumbnail((size, size * 3))
        t.save(path, "JPEG", quality=82)

    def _store_result(self, it: Item, res):
        d = self.item_dir(it)
        for old in d.glob("result.*"):
            old.unlink()
        (d / f"result.{res.ext}").write_bytes(res.image_bytes)
        it.result_ext = res.ext
        self._thumb(Image.open(io.BytesIO(res.image_bytes)), d / "result_thumb.jpg")
        all_issues = [i for r in res.regions for i in r.issues] + list(res.issues)
        it.warnings = sum(1 for i in all_issues if i.severity == "warning" and not i.fixed)
        it.errors = sum(1 for i in all_issues if i.severity == "error" and not i.fixed)
        it.regions = len(res.regions)
        it.kept_original = sum(1 for r in res.regions if r.status == "kept_original")
        it.meta = res.meta
        self._save_item(it)

    def _run(self):
        stage_labels = dict(P.STAGES)
        while True:
            iid = self.q.get()
            it = self.items.get(iid)
            if it is None or it.status != "queued":
                continue
            b = self.batches[it.batch_id]
            it.status, it.progress, t0 = "processing", 0.0, time.time()

            def progress(stage, frac, it=it):
                it.stage, it.stage_label, it.progress = stage, stage_labels.get(stage, stage), round(frac, 3)

            try:
                opts = PipelineOptions.from_dict(b.options)
                res = P.process(self.original_path(it).read_bytes(), opts, progress, b.history,
                                self.item_dir(it) / "work")
                self._store_result(it, res)
                it.status, it.stage, it.stage_label, it.progress = "done", "done", "Done", 1.0
            except TranslationError as e:
                it.status, it.error = "failed", f"Translation failed: {e}. Check the API key/model in .env and retry."
            except (PipelineError, ValueError, RuntimeError) as e:
                it.status, it.error = "failed", str(e)
            except Exception as e:  # pragma: no cover
                log.error("item %s failed:\n%s", iid, traceback.format_exc())
                it.status, it.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                if it.status == "failed":
                    it.stage_label = "Failed"
                if len(b.history) > 200:
                    del b.history[:-200]
                it.finished = time.time()
                it.seconds = round(it.finished - t0, 1)
                try:
                    self._save_item(it)
                    self._save_batch(b)
                except Exception:   # a failed save must never stop the worker for later pages
                    log.error("could not save status for %s:\n%s", iid, traceback.format_exc())
