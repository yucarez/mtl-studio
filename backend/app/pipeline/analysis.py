from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .geometry import polygon_mask


@dataclass
class Closed:
    mask: np.ndarray            
    raw: np.ndarray             
    offset: tuple[int, int]     
    bg: tuple[int, int, int]
    uniform: bool
    area: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        x0, y0 = self.offset
        return int(xs.min() + x0), int(ys.min() + y0), int(xs.max() + x0 + 1), int(ys.max() + y0 + 1)

    def full_mask(self, shape) -> np.ndarray:
        m = np.zeros(shape[:2], bool)
        x0, y0 = self.offset
        h, w = self.mask.shape
        m[y0:y0 + h, x0:x0 + w] = self.mask
        return m

    def contains(self, x: float, y: float) -> bool:
        xi, yi = int(x) - self.offset[0], int(y) - self.offset[1]
        return 0 <= yi < self.mask.shape[0] and 0 <= xi < self.mask.shape[1] and bool(self.mask[yi, xi])


def _kernel(r: float) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_holes(m: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def ring_stats(rgb: np.ndarray, area: np.ndarray, width: float) -> tuple[np.ndarray, float, float]:
    ring = cv2.dilate(area.astype(np.uint8), _kernel(width)).astype(bool) & ~area
    px = rgb[ring]
    if len(px) < 10:
        px = rgb.reshape(-1, 3)
    med = np.median(px, axis=0)
    dev = np.abs(px.astype(np.float32) - med).max(axis=1)
    return med, float(np.median(dev)), float(np.mean(dev))


def dominant_color(px: np.ndarray) -> tuple[np.ndarray, float]:
    if len(px) < 10:
        return np.array([255, 255, 255], np.float32), 0.0
    q = (px // 16).astype(np.int32)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    top = np.bincount(key).argmax()
    sel = px[key == top].astype(np.float32)
    mode = np.median(sel, axis=0)
    near = np.abs(px.astype(np.float32) - mode).max(axis=1) < 20
    return np.median(px[near].astype(np.float32), axis=0), float(near.mean())


def closed_region(rgb: np.ndarray, quads: list[np.ndarray], glyph: float, expand: float) -> Optional[Closed]:
    H, W = rgb.shape[:2]
    pts = np.concatenate(quads)
    bx0, by0 = pts.min(axis=0)
    bx1, by1 = pts.max(axis=0)
    x0, y0 = max(0, int(bx0 - expand)), max(0, int(by0 - expand))
    x1, y1 = min(W, int(bx1 + expand) + 1), min(H, int(by1 + expand) + 1)
    win = rgb[y0:y1, x0:x1]
    area = np.zeros(win.shape[:2], bool)
    for q in quads:
        area |= polygon_mask(win.shape, q - np.array([x0, y0], np.float32))
    area = cv2.dilate(area.astype(np.uint8), _kernel(0.15 * glyph)).astype(bool)
    
    ring = cv2.dilate(area.astype(np.uint8), _kernel(max(2.0, 0.2 * glyph))).astype(bool) & ~area
    bg, frac = dominant_color(win[ring])
    if frac < 0.5:                       
        return None
    tol = 30.0
    diff = np.abs(win.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    sim = (diff < tol) | area
    n, labels = cv2.connectedComponents(sim.astype(np.uint8), connectivity=4)
    seed_labels = labels[area]
    seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0:
        return None
    lab = np.bincount(seed_labels).argmax()
    comp = labels == lab
    edge = np.concatenate([comp[0, :], comp[-1, :], comp[:, 0], comp[:, -1]])
    
    touches = (comp[0, :].any() and y0 > 0) or (comp[-1, :].any() and y1 < H) or \
              (comp[:, 0].any() and x0 > 0) or (comp[:, -1].any() and x1 < W)
    if touches or edge.mean() > 0.5:
        return None
    filled = fill_holes(comp)
    raw = comp & ~area
    vals = win[raw]
    dev = np.abs(vals.astype(np.float32) - bg).max(axis=1) if len(vals) else np.array([99.0])
    uniform = len(vals) > 20 and float(np.percentile(dev, 90)) < 14.0
    return Closed(mask=filled, raw=raw, offset=(x0, y0), bg=tuple(int(v) for v in bg),
                  uniform=uniform, area=int(filled.sum()))


def is_bubble(closed: Closed, block_area: float) -> bool:
    if closed is None or block_area <= 0:
        return False
    ratio = closed.area / block_area
    if not (1.08 <= ratio <= 40):
        return False
    cnts, _ = cv2.findContours(closed.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    solidity = cv2.contourArea(c) / max(1.0, cv2.contourArea(hull))
    return solidity >= 0.7


def text_pixel_mask(rgb_win: np.ndarray, area: np.ndarray, bg, glyph: float = 20.0,
                    local: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if local:
        k = int(max(5, min(41, round(0.7 * glyph)))) | 1
        ref = cv2.medianBlur(rgb_win, k).astype(np.int16)
    else:
        ref = np.asarray(bg, np.int16)
    diff = np.abs(rgb_win.astype(np.int16) - ref).max(axis=2).astype(np.uint8)
    vals = diff[area]
    if len(vals) < 5:
        return np.zeros_like(area), diff
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(28.0, float(thr) * 0.8)
    raw = diff > thr
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
    if n <= 1:
        return raw & area, diff
    inside = np.bincount(lab[area], minlength=n)
    total = stats[:, cv2.CC_STAT_AREA]
    keep = (inside >= 0.6 * np.maximum(total, 1))
    keep[0] = False
    return keep[lab] & area, diff


def estimate_style(rgb: np.ndarray, quad: np.ndarray, glyph: float) -> dict:
    H, W = rgb.shape[:2]
    pad = int(glyph * 1.2) + 4
    x0, y0 = max(0, int(quad[:, 0].min()) - pad), max(0, int(quad[:, 1].min()) - pad)
    x1, y1 = min(W, int(quad[:, 0].max()) + pad), min(H, int(quad[:, 1].max()) + pad)
    win = rgb[y0:y1, x0:x1]
    area = polygon_mask(win.shape, quad - np.array([x0, y0], np.float32))
    area = cv2.dilate(area.astype(np.uint8), _kernel(0.12 * glyph)).astype(bool)
    bg, spread, mean_dev = ring_stats(win, area, 0.5 * glyph)
    core = polygon_mask(win.shape, quad - np.array([x0, y0], np.float32))
    core_px = win[core].astype(np.float32)
    plate = False
    if len(core_px) > 30:
        cmed = np.median(core_px, axis=0)
        if float(np.percentile(np.abs(core_px - cmed).max(axis=1), 55)) < 6:
            plate = not (spread < 4 and mean_dev < 14)
            if plate:            
                bg, spread, mean_dev, area = cmed, 0.0, 0.0, core
    tmask, diff = text_pixel_mask(win, area, bg, glyph, local=spread >= 6)
    if tmask.sum() > 10:
        d = diff[tmask]
        strong = tmask & (diff >= np.percentile(d, 60))
        fg = np.median(win[strong], axis=0)
        cnts, _ = cv2.findContours(tmask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        stroke = 2.0 * tmask.sum() / max(perim, 1.0)
        bold = stroke / max(glyph, 1.0) > 0.12
    else:
        stroke = 0.0
        fg = np.array([0, 0, 0]) if bg.mean() > 128 else np.array([255, 255, 255])
        bold = False
    return {"bg": tuple(int(v) for v in bg), "fg": tuple(int(v) for v in fg),
            "uniform": bool(spread < 4 and mean_dev < 14), "spread": spread, "bold": bool(bold),
            "plate": plate, "stroke": float(stroke)}


def luminance(c) -> float:
    def ch(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = c
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(a, b) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def ink_extent(crop: np.ndarray, vertical: bool) -> float:
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    med = float(np.median(g))
    diff = np.abs(g.astype(np.int16) - med).astype(np.uint8)
    if diff.max() < 30:
        return 0.0
    thr, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = diff > max(30, thr)
    prof = ink.sum(axis=0 if vertical else 1)
    idx = np.nonzero(prof > max(1, 0.02 * prof.max()))[0]
    return float(idx[-1] - idx[0] + 1) if len(idx) else 0.0
