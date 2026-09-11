"""
Step 4: orientation detection.

For every line we decide, *before* OCR:
  * its tilt (theta) and whether it is a vertical CJK column or a horizontal line,
  * whether it is upside down (theta vs theta+180), by letting a confidence-reporting
    recognizer vote on both readings,
and for the whole page whether it was scanned/photographed sideways or upside down.

Output text is ALWAYS rendered upright: see typeset.render_angle(). The analysis here only
exists so the OCR reads the source correctly.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .geometry import norm90, norm180, rect_crop
from .datatypes import TextLine


def analyze_line(line: TextLine, cjk_source: Optional[bool]) -> None:
    rect = cv2.minAreaRect(line.quad.astype(np.float32))
    pts = cv2.boxPoints(rect)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    n1, n2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    long_v, L, T = (e1, n1, n2) if n1 >= n2 else (e2, n2, n1)
    axis = norm90(math.degrees(math.atan2(long_v[1], long_v[0])))
    line.center = (float(rect[0][0]), float(rect[0][1]))
    aspect = L / max(T, 1e-3)

    if aspect < 1.3:                       # 1-2 glyphs: direction unknown, decided by neighbours
        t = axis
        while t > 45:
            t -= 90
        while t <= -45:
            t += 90
        line.theta, line.length, line.thickness = t, L, T
        line.vertical = None
    elif abs(axis) <= 45:                  # horizontal-ish line
        line.theta, line.length, line.thickness, line.vertical = axis, L, T, False
    elif cjk_source is not False:          # long axis vertical: CJK column (tategaki)
        line.theta, line.length, line.thickness, line.vertical = norm90(axis - 90), L, T, True
    else:                                  # rotated latin line reading along the long axis
        line.theta, line.length, line.thickness, line.vertical = axis, L, T, False


def line_crop(rgb: np.ndarray, line: TextLine, theta: Optional[float] = None,
              vertical: Optional[bool] = None) -> np.ndarray:
    t = line.theta if theta is None else theta
    v = line.vertical if vertical is None else vertical
    pad = max(2.0, 0.1 * line.thickness)
    if v:  # tall crop, upright glyphs
        return rect_crop(rgb, line.center, line.thickness + 2 * pad, line.length + 2 * pad, t)
    return rect_crop(rgb, line.center, line.length + 2 * pad, line.thickness + 2 * pad, t)


def _conf(rec, crops, vflags) -> list[float]:
    return [c or 0.0 for _, c in rec.recognize(crops, vflags)]


def page_rotation_vote(rgb: np.ndarray, lines: list[TextLine], rec, cjk_possible: bool) -> int:
    """Returns the clockwise rotation (0/90/180/270) to APPLY to the page so text is upright."""
    if rec is None or not getattr(rec, "gives_confidence", False):
        return 0
    strong = [l for l in lines if l.length / max(l.thickness, 1) >= 2.2]
    if len(strong) < 3:
        return 0
    strong.sort(key=lambda l: -l.length * l.thickness)
    vert = [l for l in strong if abs(l.theta) > 45 or l.vertical]
    horiz = [l for l in strong if not (abs(l.theta) > 45 or l.vertical)]
    w_vert = sum(l.length for l in vert) / max(1e-6, sum(l.length for l in strong))

    if w_vert >= 0.6:
        sample = vert[:8]
        crops_down, crops_up, crops_col = [], [], []
        for l in sample:
            axis = l.theta if not l.vertical else norm90(l.theta + 90)
            base = 90.0 if axis > 0 else -90.0
            tilt = axis - base
            crops_down.append(line_crop(rgb, l, theta=90.0 + tilt, vertical=False))
            crops_up.append(line_crop(rgb, l, theta=-90.0 + tilt, vertical=False))
            if cjk_possible:
                crops_col.append(line_crop(rgb, l, theta=tilt, vertical=True))
        c_down = np.mean(_conf(rec, crops_down, [False] * len(sample)))
        c_up = np.mean(_conf(rec, crops_up, [False] * len(sample)))
        c_col = np.mean(_conf(rec, crops_col, [True] * len(sample))) if cjk_possible else 0.0
        best = max(c_down, c_up, c_col)
        if best == c_col or best < 0.5 or best - c_col < 0.12:
            return 0
        # reading direction pointing down  -> page was rotated CW  -> rotate CCW (270 CW)
        return 270 if c_down >= c_up else 90

    sample = horiz[:8]
    if not sample:
        return 0
    c0 = _conf(rec, [line_crop(rgb, l) for l in sample], [False] * len(sample))
    c180 = _conf(rec, [line_crop(rgb, l, theta=l.theta + 180) for l in sample], [False] * len(sample))
    wins = sum(1 for a, b in zip(c0, c180) if b > a + 0.1)
    if wins >= max(2, int(0.7 * len(sample))) and np.mean(c180) - np.mean(c0) > 0.15:
        return 180
    return 0


def rotate_page(rgb: np.ndarray, cw: int) -> np.ndarray:
    if cw == 90:
        return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    if cw == 180:
        return cv2.rotate(rgb, cv2.ROTATE_180)
    if cw == 270:
        return cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rgb


def resolve_ambiguous(lines: list[TextLine], default_vertical: bool) -> None:
    """Give 1-2 glyph lines the direction of their nearest confident neighbour."""
    known = [l for l in lines if l.vertical is not None]
    for l in lines:
        if l.vertical is not None:
            continue
        best, bd = None, 1e18
        for k in known:
            d = (k.center[0] - l.center[0]) ** 2 + (k.center[1] - l.center[1]) ** 2
            if d < bd and d < (6 * max(k.thickness, l.thickness)) ** 2:
                best, bd = k, d
        l.vertical = best.vertical if best is not None else default_vertical
        if best is not None:
            l.theta = best.theta
        if l.vertical:
            l.length, l.thickness = max(l.length, l.thickness), min(l.length, l.thickness)


def resolve_upside_down(rgb: np.ndarray, lines: list[TextLine], rec, thorough: bool) -> int:
    """Per-line 0/180 disambiguation for horizontal lines. Returns #lines flipped."""
    if rec is None or not getattr(rec, "gives_confidence", False):
        # no confidence available: keep the geometric reading (-90..90) which is upright
        return 0
    # CJK: a tall box may be a vertical column OR a horizontal line rotated 90 deg; let OCR decide
    cols = [l for l in lines if l.vertical and l.length / max(l.thickness, 1) >= 2.5]
    if cols:
        c_col = _conf(rec, [line_crop(rgb, l) for l in cols], [True] * len(cols))
        c_dn = _conf(rec, [line_crop(rgb, l, theta=l.theta + 90, vertical=False) for l in cols], [False] * len(cols))
        c_up = _conf(rec, [line_crop(rgb, l, theta=l.theta - 90, vertical=False) for l in cols], [False] * len(cols))
        for l, a, b, c in zip(cols, c_col, c_dn, c_up):
            if max(b, c) > a + 0.2:
                l.vertical = False
                l.theta = norm180(l.theta + (90 if b >= c else -90))
    cand = [l for l in lines if not l.vertical and (thorough or abs(l.theta) > 20)]
    if not cand:
        return 0
    c0 = _conf(rec, [line_crop(rgb, l) for l in cand], [False] * len(cand))
    c1 = _conf(rec, [line_crop(rgb, l, theta=l.theta + 180) for l in cand], [False] * len(cand))
    flipped = 0
    for l, a, b in zip(cand, c0, c1):
        if b > a + 0.15:
            l.theta = norm180(l.theta + 180)
            flipped += 1
    return flipped
