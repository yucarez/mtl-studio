"""
Text detection + recognition adapters.

Engines (install what you want, the app picks the best available):
  * PaddleOCR 3.x  (pip install paddleocr paddlepaddle)   detection + multilingual recognition
  * manga-ocr      (pip install manga-ocr)                best-in-class Japanese manga OCR
  * EasyOCR        (pip install easyocr)                  fallback detector/recognizer

Every recognizer returns (text, confidence) where confidence is in [0, 1] or None when
the engine cannot provide one (manga-ocr).
"""
from __future__ import annotations

import importlib.util
import logging
import threading
from typing import Optional

import cv2
import numpy as np

from .. import config

log = logging.getLogger("mtl.ocr")
_lock = threading.RLock()
_cache: dict = {}


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def engine_status() -> dict:
    return {
        "paddle": available("paddleocr"),
        "easyocr": available("easyocr"),
        "manga-ocr": available("manga_ocr"),
        "lama": available("simple_lama_inpainting") and config.ENABLE_LAMA,
    }


def _use_gpu() -> bool:
    if config.DEVICE == "cpu":
        return False
    try:
        import torch  # noqa
        return bool(torch.cuda.is_available())
    except Exception:
        return config.DEVICE == "cuda"


def _cached(key, factory):
    with _lock:
        if key not in _cache:
            log.info("loading model %s", key)
            _cache[key] = factory()
        return _cache[key]


# ============================================================ detectors
class Detector:
    name = "base"

    def detect(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        raise NotImplementedError


class PaddleDetector(Detector):
    name = "paddle"

    def __init__(self):
        from paddleocr import TextDetection

        def make():
            try:
                return TextDetection(model_name=config.PADDLE_DET_MODEL, box_thresh=0.5, unclip_ratio=1.6)
            except TypeError:
                return TextDetection(model_name=config.PADDLE_DET_MODEL)

        self.model = _cached(("paddle-det", config.PADDLE_DET_MODEL), make)

    def detect(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with _lock:
            out = list(self.model.predict(bgr, batch_size=1))
        res = out[0]
        polys = _field(res, "dt_polys")
        polys = [] if polys is None else list(polys)
        scores = _field(res, "dt_scores")
        scores = [1.0] * len(polys) if scores is None else list(scores)
        result = []
        for p, s in zip(polys, scores):
            q = np.asarray(p, np.float32).reshape(-1, 2)
            if len(q) != 4:  # polygon -> min area rect
                q = cv2.boxPoints(cv2.minAreaRect(q)).astype(np.float32)
            result.append((q, float(s)))
        return result


class EasyOCRDetector(Detector):
    name = "easyocr"

    def __init__(self, langs=("en",)):
        import easyocr
        self.reader = _cached(("easyocr", tuple(langs)), lambda: easyocr.Reader(list(langs), gpu=_use_gpu()))

    def detect(self, rgb):
        with _lock:
            hlist, flist = self.reader.detect(
                rgb, text_threshold=0.6, low_text=0.35, link_threshold=0.4, canvas_size=2560,
                mag_ratio=1.5, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5,
                add_margin=0.05)
        result = []
        for x0, x1, y0, y1 in (hlist[0] if hlist else []):
            result.append((np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), 0.9))
        for pts in (flist[0] if flist else []):
            result.append((np.asarray(pts, np.float32).reshape(4, 2), 0.9))
        return result


# ============================================================ recognizers
class Recognizer:
    name = "base"
    gives_confidence = True

    def recognize(self, crops: list[np.ndarray], vertical: list[bool]) -> list[tuple[str, Optional[float]]]:
        raise NotImplementedError


def _rot_for_vertical(crop: np.ndarray) -> np.ndarray:
    # Same convention as the PaddleOCR pipeline: lay a vertical column on its side (CCW)
    return np.ascontiguousarray(np.rot90(crop, k=1))


class PaddleRecognizer(Recognizer):
    name = "paddle"

    def __init__(self, model_name: str):
        from paddleocr import TextRecognition
        self.model_name = model_name
        self.model = _cached(("paddle-rec", model_name), lambda: TextRecognition(model_name=model_name))

    def recognize(self, crops, vertical):
        if not crops:
            return []
        batch = []
        for c, v in zip(crops, vertical):
            c = _rot_for_vertical(c) if v else c
            batch.append(cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
        with _lock:
            out = list(self.model.predict(batch, batch_size=8))
        return [(str(_field(r, "rec_text") or ""), float(_field(r, "rec_score") or 0.0)) for r in out]


class EasyOCRRecognizer(Recognizer):
    name = "easyocr"

    def __init__(self, langs):
        import easyocr
        self.reader = _cached(("easyocr", tuple(langs)), lambda: easyocr.Reader(list(langs), gpu=_use_gpu()))

    def recognize(self, crops, vertical):
        out = []
        for c, v in zip(crops, vertical):
            c = _rot_for_vertical(c) if v else c
            g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
            h, w = g.shape
            with _lock:
                res = self.reader.recognize(g, horizontal_list=[[0, w, 0, h]], free_list=[], detail=1)
            if not res:
                out.append(("", 0.0))
                continue
            text = " ".join(r[1] for r in res)
            conf = float(np.mean([r[2] for r in res]))
            out.append((text, conf))
        return out


class MangaOcrRecognizer(Recognizer):
    name = "manga-ocr"
    gives_confidence = False

    def __init__(self):
        from manga_ocr import MangaOcr
        self.model = _cached(("manga-ocr",), lambda: MangaOcr(force_cpu=not _use_gpu()))

    def recognize(self, crops, vertical):
        from PIL import Image
        out = []
        for c in crops:
            with _lock:
                out.append((self.model(Image.fromarray(c)), None))
        return out


# ============================================================ factories
_EASY_LANGS = {
    "ja": ["ja", "en"], "zh": ["ch_sim", "en"], "zh-TW": ["ch_tra", "en"], "ko": ["ko", "en"],
    "th": ["th", "en"], "ru": ["ru", "en"], "uk": ["uk", "en"], "fil": ["tl", "en"],
}


def easy_langs(code: str) -> list[str]:
    return _EASY_LANGS.get(code, [code, "en"] if code != "en" else ["en"])


def paddle_model_for(code: str) -> str:
    L = config.lang(code)
    if L.script in config.CJK_SCRIPTS or code == "en":
        return config.PADDLE_REC_MODEL
    return {
        "hangul": config.PADDLE_REC_MODEL_KO,
        "cyrillic": config.PADDLE_REC_MODEL_CYRILLIC,
        "thai": config.PADDLE_REC_MODEL_TH,
    }.get(L.script, config.PADDLE_REC_MODEL_LATIN)


def make_detector(choice: str, lang_hint: str = "en") -> Detector:
    st = engine_status()
    order = [choice] if choice != "auto" else ["paddle", "easyocr"]
    errors = []
    for name in order:
        try:
            if name == "paddle" and st["paddle"]:
                return PaddleDetector()
            if name == "easyocr" and st["easyocr"]:
                return EasyOCRDetector(easy_langs(lang_hint))
        except Exception as e:  # pragma: no cover
            errors.append(f"{name}: {e}")
    raise RuntimeError("No text detector available. Install PaddleOCR or EasyOCR "
                       f"(see README). {'; '.join(errors)}")


def make_recognizer(choice: str, code: str, content_type: str = "auto") -> Recognizer:
    """Recognizer for a *known* language code."""
    st = engine_status()
    if choice == "auto":
        order = []
        if code == "ja" and st["manga-ocr"] and content_type in ("auto", "manga", "comic", "webtoon"):
            order.append("manga-ocr")
        order += ["paddle", "easyocr"]
    else:
        order = [choice]
    errors = []
    for name in order:
        try:
            if name == "manga-ocr" and st["manga-ocr"] and code == "ja":
                return MangaOcrRecognizer()
            if name == "paddle" and st["paddle"]:
                model = paddle_model_for(code)
                try:
                    return PaddleRecognizer(model)
                except Exception as e:
                    log.warning("paddle model %s failed (%s); falling back to %s",
                                model, e, config.PADDLE_REC_MODEL)
                    return PaddleRecognizer(config.PADDLE_REC_MODEL)
            if name == "easyocr" and st["easyocr"]:
                return EasyOCRRecognizer(easy_langs(code))
        except Exception as e:  # pragma: no cover
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"No OCR engine available for '{code}'. {'; '.join(errors)}")


def make_confidence_recognizer(code: str) -> Optional[Recognizer]:
    """A recognizer that reports confidence (orientation voting, language probing, re-read checks)."""
    for engine in ("paddle", "easyocr"):
        try:
            r = make_recognizer(engine, code)
            if r.gives_confidence:
                return r
        except Exception:
            continue
    return None


def _field(res, key):
    """PaddleOCR 3.x result objects are dict-like; fall back to their .json payload."""
    try:
        return res[key]
    except (KeyError, TypeError, IndexError):
        j = getattr(res, "json", None) or {}
        return (j.get("res", j) if isinstance(j, dict) else {}).get(key)
