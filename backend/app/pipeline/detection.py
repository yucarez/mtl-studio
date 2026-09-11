"""
Step 3: text detection (line level).

* Small images are upscaled so thin text survives the detector's internal resize.
* Very tall (webtoon) or very wide images are split into overlapping tiles; otherwise the
  detector would shrink a 800x20000 strip until the text disappears.
* Boxes cut by a tile seam are dropped in favour of the complete box from the neighbour tile.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import bbox_overlap_small, quad_bbox
from .datatypes import TextLine


def _tiles(H: int, W: int) -> list[tuple[int, int, int, int]]:
    if H > 2.5 * W:
        th = int(W * 1.6)
        ov = int(th * 0.25)
        ys = list(range(0, max(1, H - ov), th - ov))
        return [(0, y, W, min(H, y + th)) for y in ys]
    if W > 2.5 * H:
        tw = int(H * 1.6)
        ov = int(tw * 0.25)
        xs = list(range(0, max(1, W - ov), tw - ov))
        return [(x, 0, min(W, x + tw), H) for x in xs]
    return [(0, 0, W, H)]


def detect_lines(detector, rgb: np.ndarray) -> list[TextLine]:
    H, W = rgb.shape[:2]
    raw: list[tuple[np.ndarray, float, bool]] = []
    for (x0, y0, x1, y1) in _tiles(H, W):
        tile = rgb[y0:y1, x0:x1]
        th, tw = tile.shape[:2]
        scale = 1.0
        if max(th, tw) < 1000:
            scale = min(2.0, 1600.0 / max(th, tw))
            tile = cv2.resize(tile, (int(tw * scale), int(th * scale)), interpolation=cv2.INTER_CUBIC)
        for quad, score in detector.detect(tile):
            q = quad.astype(np.float32) / scale + np.array([x0, y0], np.float32)
            bx = quad_bbox(q)
            m = 3
            cut = ((y0 > 0 and bx[1] <= y0 + m) or (y1 < H and bx[3] >= y1 - m) or
                   (x0 > 0 and bx[0] <= x0 + m) or (x1 < W and bx[2] >= x1 - m))
            raw.append((q, score, cut))

    # de-duplicate (tile overlaps, detector double hits)
    raw.sort(key=lambda t: (t[2], -t[1] * _area(t[0])))   # complete boxes first, then big/confident
    kept: list[tuple[np.ndarray, float]] = []
    kept_boxes: list[tuple] = []
    for q, s, cut in raw:
        b = quad_bbox(q)
        thr = 0.3 if cut else 0.75
        if any(bbox_overlap_small(b, kb) > thr for kb in kept_boxes):
            continue
        kept.append((q, s))
        kept_boxes.append(b)

    lines = []
    for q, s in kept:
        rect = cv2.minAreaRect(q)
        if min(rect[1]) < 5 or _area(q) < 60:      # specks, hairlines
            continue
        lines.append(TextLine(quad=q, det_score=s))
    return lines


def _area(q: np.ndarray) -> float:
    return float(abs(cv2.contourArea(q.astype(np.float32))))
