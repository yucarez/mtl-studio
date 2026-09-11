"""
Step 8: text removal.

Only the actual glyph pixels (plus anti-aliasing / outline margin) are removed, never the
whole box, and the fill method is picked per region:
  * uniform speech bubble / flat background -> exact background colour (pixel perfect)
  * smooth gradient                          -> OpenCV Telea
  * texture, screentone, artwork             -> LaMa (fallback: OpenCV Navier-Stokes)
Bubble outlines are protected by intersecting the mask with the bubble interior.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..engines import inpaint_engines as ie
from .analysis import _kernel, ring_stats, text_pixel_mask
from .geometry import polygon_mask
from .datatypes import Issue, Region


def _window(shape, bbox, pad):
    H, W = shape[:2]
    x0, y0, x1, y1 = bbox
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def region_mask(rgb: np.ndarray, r: Region, closed) -> tuple[np.ndarray, str]:
    """Full-size bool mask of pixels to remove for region r, and the fill method."""
    g = max(4.0, r.glyph_px)
    pad = int(1.5 * g) + 6
    x0, y0, x1, y1 = _window(rgb.shape, r.bbox, pad)
    win = rgb[y0:y1, x0:x1]
    off = np.array([x0, y0], np.float32)
    core = np.zeros(win.shape[:2], bool)
    for l in r.lines:
        core |= polygon_mask(win.shape, l.quad - off)
    area = cv2.dilate(core.astype(np.uint8), _kernel(0.22 * g)).astype(bool)

    interior = None
    if closed is not None:
        full = closed.full_mask(rgb.shape)[y0:y1, x0:x1]
        interior = cv2.erode(full.astype(np.uint8), _kernel(1)).astype(bool)

    local = False
    limit = cv2.dilate(area.astype(np.uint8), _kernel(0.15 * g)).astype(bool)
    if closed is not None and closed.uniform:
        bg, method = np.array(closed.bg, np.float32), "flat"
    else:
        bg, mad, mean_dev = ring_stats(win, area, 0.5 * g)
        core_px = win[core].astype(np.float32)
        core_med = np.median(core_px, axis=0) if len(core_px) else bg
        core_dev = np.abs(core_px - core_med).max(axis=1) if len(core_px) else np.array([99.0])
        plate = len(core_px) > 30 and float(np.percentile(core_dev, 55)) < 6
        if plate and not (mad < 4 and mean_dev < 14 and float(np.abs(core_med - bg).max()) < 12):
            # the text's own box is a flat colour (label / plate / flat panel) while the
            # surroundings are not: fill flat, strictly inside the box
            bg, method = core_med, "flat"
            area = core
            limit = cv2.dilate(core.astype(np.uint8), _kernel(2)).astype(bool)
        elif mad < 4 and mean_dev < 14:
            method = "flat"
        elif mad < 10:
            method, local = "telea", True
        else:
            method, local = "lama", True

    tmask, _ = text_pixel_mask(win, area, bg, g, local=local)
    tmask = cv2.dilate(tmask.astype(np.uint8), _kernel(max(2.0, 0.12 * g))).astype(bool)
    if method == "flat" and closed is not None and closed.uniform:
        m = (tmask | area) & interior          # wipe the whole text area inside the bubble
    else:
        m = tmask & limit
        if interior is not None:
            m &= interior
    out = np.zeros(rgb.shape[:2], bool)
    out[y0:y1, x0:x1] = m
    r.render["fill_bg"] = [int(v) for v in bg]
    return out, method


def _apply(clean: np.ndarray, mask: np.ndarray, method: str, bg) -> str:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return method
    if method == "flat":
        clean[mask] = np.array(bg, np.uint8)
        return method
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    ctx = max(48, int(0.8 * max(bx1 - bx0, by1 - by0)))
    x0, y0, x1, y1 = _window(clean.shape, (bx0, by0, bx1, by1), ctx)
    crop, m = clean[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    res = None
    if method == "lama" and ie.lama_available():
        res = ie.inpaint_lama(crop, m)
        used = "lama"
    if res is None:
        g_radius = 3 if method == "telea" else 5
        res = ie.inpaint_cv(crop, m, g_radius, "telea" if method == "telea" else "ns")
        used = "telea" if method == "telea" else "opencv"
    crop[m] = res[m]
    return used


def remove_text(rgb: np.ndarray, regions: list[Region], closed_by_id: dict) -> tuple[np.ndarray, np.ndarray]:
    clean = rgb.copy()
    labels = np.zeros(rgb.shape[:2], np.uint16)
    for r in regions:
        if r.status == "skipped":
            continue
        mask, method = region_mask(rgb, r, closed_by_id.get(r.id))
        mask &= labels == 0
        labels[mask] = r.id
        r.render["fill_method"] = _apply(clean, mask, method, r.render["fill_bg"])
    return clean, labels


def residual_pass(clean: np.ndarray, rgb: np.ndarray, labels: np.ndarray, regions: list[Region],
                  detector, detect_fn) -> list[Issue]:
    """Re-run detection on the cleaned image; re-inpaint any leftover glyphs."""
    issues = []
    try:
        leftovers = detect_fn(detector, clean)
    except Exception:
        return issues
    by_id = {r.id: r for r in regions}
    for l in leftovers:
        m = polygon_mask(clean.shape, l.quad)
        ids = labels[m]
        ids = ids[ids > 0]
        if len(ids) < 0.3 * m.sum() or len(ids) == 0:
            continue                      # not in a processed area (real artwork text is left alone)
        rid = int(np.bincount(ids).argmax())
        r = by_id.get(rid)
        if r is None:
            continue
        g = max(4.0, r.glyph_px)
        grow = cv2.dilate(m.astype(np.uint8), _kernel(0.25 * g)).astype(bool)
        region_area = cv2.dilate((labels == rid).astype(np.uint8), _kernel(0.4 * g)).astype(bool)
        extra = grow & region_area
        method = "flat" if r.render.get("fill_method") == "flat" else "lama"
        used = _apply(clean, extra, method, r.render.get("fill_bg", (255, 255, 255)))
        labels[extra & (labels == 0)] = rid
        issues.append(Issue("residual_text", "info",
                            f"Leftover glyph traces were found after removal and cleaned again ({used}).",
                            rid, fixed=True))
    return issues
