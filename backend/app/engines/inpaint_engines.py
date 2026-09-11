"""
Content-aware inpainting.

  * LaMa (big-lama) via `simple-lama-inpainting`  -> textures, screentone, gradients, artwork
  * OpenCV Telea / Navier-Stokes                  -> automatic fallback (thin text on smooth areas)
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from .. import config
from .ocr_engines import available, _use_gpu

log = logging.getLogger("mtl.inpaint")
_lock = threading.Lock()
_lama = None
_lama_failed = False


def lama_available() -> bool:
    return config.ENABLE_LAMA and available("simple_lama_inpainting") and not _lama_failed


def _get_lama():
    global _lama, _lama_failed
    with _lock:
        if _lama is None and not _lama_failed:
            try:
                import torch
                from simple_lama_inpainting import SimpleLama
                dev = torch.device("cuda" if _use_gpu() else "cpu")
                _lama = SimpleLama(device=dev)
            except Exception as e:  # pragma: no cover
                log.warning("LaMa unavailable (%s); using OpenCV inpainting", e)
                _lama_failed = True
        return _lama


def inpaint_lama(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """rgb uint8 HxWx3, mask bool HxW. Returns inpainted crop or None if LaMa unavailable."""
    model = _get_lama()
    if model is None:
        return None
    from PIL import Image
    h, w = mask.shape
    # LaMa works best around 512-1024px; scale large crops down, then back up.
    scale = min(1.0, 1024.0 / max(h, w))
    if scale < 1.0:
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        img_s = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(mask.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
        m_s = cv2.dilate(m_s, np.ones((3, 3), np.uint8))
    else:
        img_s, m_s = rgb, mask.astype(np.uint8) * 255
    with _lock:
        res = model(Image.fromarray(img_s), Image.fromarray(m_s))
    res = np.asarray(res.convert("RGB"))[: img_s.shape[0], : img_s.shape[1]]
    if scale < 1.0:
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
    return res


def inpaint_cv(rgb: np.ndarray, mask: np.ndarray, radius: int = 3, method: str = "telea") -> np.ndarray:
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask.astype(np.uint8) * 255, radius, flag)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
