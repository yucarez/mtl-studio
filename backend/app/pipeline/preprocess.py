"""
Step 1: preprocessing.

The single most common cause of "sideways" output in image translators is ignoring the
EXIF Orientation tag: phone photos are stored rotated and only *displayed* upright. We
bake the orientation into the pixels first so every later stage sees what the user sees.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from .datatypes import PipelineError

Image.MAX_IMAGE_PIXELS = 250_000_000  # tall webtoon strips are legitimately huge


@dataclass
class Page:
    rgb: np.ndarray                 # HxWx3 uint8, upright
    alpha: Optional[np.ndarray]     # HxW uint8 or None
    icc: Optional[bytes]
    fmt: str                        # PNG | JPEG | WEBP ...
    exif_rotated: bool
    rotation: int = 0               # extra page rotation applied by orientation stage (deg CW)


def load_image(data: bytes) -> Page:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise PipelineError(f"Could not decode image: {e}")
    fmt = (im.format or "PNG").upper()
    icc = im.info.get("icc_profile")
    before = im.size
    exif_orient = im.getexif().get(0x0112, 1) if hasattr(im, "getexif") else 1
    im = ImageOps.exif_transpose(im)
    rotated = exif_orient not in (0, 1) or im.size != before

    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        a = np.asarray(im)[..., 3]
        if a.min() < 255:
            alpha = a.copy()
            # flatten on white for analysis; alpha is restored on save
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    rgb = np.ascontiguousarray(np.asarray(im))
    if rgb.shape[0] < 16 or rgb.shape[1] < 16:
        raise PipelineError("Image is too small to process")
    return Page(rgb=rgb, alpha=alpha, icc=icc, fmt=fmt, exif_rotated=rotated)


def encode_image(rgb: np.ndarray, page: Page, output_format: str) -> tuple[bytes, str]:
    """Encode at full original resolution. Returns (bytes, extension)."""
    arr = rgb
    if page.alpha is not None and page.rotation == 0 and page.alpha.shape == rgb.shape[:2]:
        arr = np.dstack([rgb, page.alpha])
    im = Image.fromarray(arr)
    fmt = page.fmt if output_format == "same" else "PNG"
    kw = {}
    if page.icc:
        kw["icc_profile"] = page.icc
    buf = io.BytesIO()
    if fmt in ("JPEG", "JPG", "MPO"):
        im.convert("RGB").save(buf, "JPEG", quality=95, subsampling=0, **kw)
        ext = "jpg"
    elif fmt == "WEBP":
        im.save(buf, "WEBP", lossless=True, **kw)
        ext = "webp"
    else:
        im.save(buf, "PNG", optimize=False, **kw)
        ext = "png"
    return buf.getvalue(), ext
