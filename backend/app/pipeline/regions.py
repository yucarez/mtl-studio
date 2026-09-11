"""
Step 6: region building and reading-order analysis.

Lines are merged into logical regions (bubble / caption / paragraph) only when they are
geometrically consistent (same direction, similar tilt and glyph size, adjacent) AND
do not belong to different closed regions (two neighbouring speech bubbles never merge).
Regions are then ordered with a recursive XY-cut that reads right-to-left for manga.
"""
from __future__ import annotations

import numpy as np

from .analysis import closed_region, estimate_style
from .geometry import axes, norm180, oriented_extent, rect_quad
from .datatypes import Region, TextLine


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _mergeable(a: TextLine, b: TextLine) -> bool:
    if a.vertical != b.vertical:
        return False
    wa, wb = getattr(a, "weight", None), getattr(b, "weight", None)
    if not a.vertical and wa and wb and max(wa, wb) / min(wa, wb) > 1.35:
        return False     # a bold heading and regular body text are separate regions
    if abs(norm180(a.theta - b.theta)) > 12:
        return False
    Ta, Tb = a.thickness, b.thickness
    if max(Ta, Tb) / max(1e-3, min(Ta, Tb)) > 1.7:
        return False
    Tm = (Ta + Tb) / 2
    u, n = axes(a.theta)
    d = np.array(b.center) - np.array(a.center)
    du, dn = float(d @ u), float(d @ n)
    if not a.vertical:
        # same row (detector split words) or stacked lines of a paragraph
        if abs(dn) < 0.5 * Tm and abs(du) - (a.length + b.length) / 2 <= 1.2 * Tm:
            return True
        gap = abs(dn) - Tm
        a0, a1 = -a.length / 2, a.length / 2
        b0, b1 = du - b.length / 2, du + b.length / 2
        ov = min(a1, b1) - max(a0, b0)
        return -0.6 * Tm <= gap <= 0.85 * Tm and ov > 0.1 * min(a.length, b.length)
    # vertical columns: same column split, or neighbouring columns
    if abs(du) < 0.5 * Tm and abs(dn) - (a.length + b.length) / 2 <= 1.2 * Tm:
        return True
    gap = abs(du) - Tm
    a0, a1 = -a.length / 2, a.length / 2
    b0, b1 = dn - b.length / 2, dn + b.length / 2
    ov = min(a1, b1) - max(a0, b0)
    return -0.6 * Tm <= gap <= 1.0 * Tm and ov > 0.1 * min(a.length, b.length)


def group_lines(rgb: np.ndarray, lines: list[TextLine], join_cjk: bool) -> list[Region]:
    n = len(lines)
    for l in lines:        # relative stroke weight per line (hierarchy cue)
        st = estimate_style(rgb, l.quad, l.thickness * 0.72)
        l.weight = max(1e-3, st["stroke"] / max(1.0, l.thickness))
    closed = [closed_region(rgb, [l.quad], l.thickness, max(6 * l.thickness, 0.6 * l.length)) for l in lines]
    dsu = _DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = lines[i], lines[j]
            reach = 3 * max(a.thickness, b.thickness) + (a.length + b.length) / 2
            if abs(a.center[0] - b.center[0]) > reach or abs(a.center[1] - b.center[1]) > reach:
                continue
            if not _mergeable(a, b):
                continue
            ci, cj = closed[i], closed[j]
            if ci is not None and cj is not None:
                if not (ci.contains(*b.center) and cj.contains(*a.center)):
                    continue       # different bubbles
            elif (ci is None) != (cj is None):
                other = ci or cj
                probe = b if ci is not None else a
                if not other.contains(*probe.center):
                    continue       # one inside a bubble, one outside
            dsu.union(i, j)

    groups: dict[int, list[TextLine]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(lines[i])
    regions = [build_region(k, g, join_cjk) for k, g in enumerate(groups.values())]
    return regions


def build_region(rid: int, lines: list[TextLine], join_cjk: bool) -> Region:
    w = np.array([max(1.0, l.length) for l in lines])
    ref = lines[int(np.argmax(w))].theta
    th = ref + float(np.sum(w * np.array([norm180(l.theta - ref) for l in lines])) / w.sum())
    vertical = sum(w[i] for i, l in enumerate(lines) if l.vertical) > w.sum() / 2
    pts = np.concatenate([l.quad for l in lines])
    center, W, H = oriented_extent(pts, th)
    u, nvec = axes(th)
    # line order inside the region
    if vertical:   # columns right -> left, then top -> bottom
        lines.sort(key=lambda l: (-round((np.array(l.center) @ u) / max(1, l.thickness * 0.6)),
                                  np.array(l.center) @ nvec))
    else:          # rows top -> bottom, then left -> right
        lines.sort(key=lambda l: (round((np.array(l.center) @ nvec) / max(1, l.thickness * 0.6)),
                                  np.array(l.center) @ u))
    T = float(np.median([l.thickness for l in lines]))
    ems = [em_from_ink(l.ink, l.text, bool(l.vertical)) for l in lines if l.ink > 0 and l.text]
    if ems:    # em size from measured ink extent, corrected for the letters present
        glyph = float(np.clip(max(ems), 0.5 * T, 1.0 * T))
    else:      # box-based fallback: CJK column width ~1.15 em, latin line height ~1.4 em
        glyph = T * (0.85 if (join_cjk or vertical) else 0.72)
    r = Region(id=rid, lines=lines, quad=rect_quad(center, W, H, th), center=center,
               width=W, height=H, theta=th, vertical=vertical, glyph_px=glyph)
    r.source_text = join_lines([l.text for l in lines], join_cjk)
    confs = [l.conf for l in lines if l.conf >= 0]
    r.ocr_conf = float(np.mean(confs)) if confs else -1.0
    return r


_DESC = set("gjpqyQ,;()[]{}|/@$")
_ASC = set("bdfhklt0123456789!?ABCDEFGHIJKLMNOPRSTUVWXYZ\"'&%#")


def em_from_ink(ink: float, text: str, vertical: bool) -> float:
    """Convert measured ink extent to an approximate em (font px) size.
    Latin ink height depends on which letters occur: 'exit' is x-height only (~0.53 em),
    'Save' reaches cap height (~0.73 em), 'play' spans ascender..descender (~0.95 em)."""
    if vertical or any(ord(c) > 0x2E80 for c in text):
        return ink / 0.9
    has_d = any(c in _DESC for c in text)
    has_a = any(c in _ASC or (c.isalpha() and c.isupper()) for c in text)
    return ink / (0.95 if (has_d and has_a) else 0.76 if has_d else 0.73 if has_a else 0.53)


def join_lines(parts: list[str], join_cjk: bool) -> str:
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not out:
            out = p
        elif join_cjk:
            out += p
        elif out.endswith("-") and p[:1].islower():
            out = out[:-1] + p
        else:
            out += " " + p
    return out


# ------------------------------------------------------------------ reading order
def _split(items, lo, hi, tol_frac=0.15):
    """Group items whose [lo,hi) intervals overlap (beyond a small tolerance)."""
    order = sorted(items, key=lambda it: lo(it))
    groups, cur, cur_hi = [], [], None
    for it in order:
        a, b = lo(it), hi(it)
        tol = tol_frac * (b - a)
        if cur and a < cur_hi - tol:
            cur.append(it)
            cur_hi = max(cur_hi, b)
        else:
            if cur:
                groups.append(cur)
            cur, cur_hi = [it], b
    if cur:
        groups.append(cur)
    return groups


def xy_cut(regs: list[Region], rtl: bool, depth: int = 0) -> list[Region]:
    if len(regs) <= 1 or depth > 40:
        return list(regs)
    rows = _split(regs, lambda r: r.bbox[1], lambda r: r.bbox[3])
    if len(rows) > 1:
        return [x for g in rows for x in xy_cut(g, rtl, depth + 1)]
    cols = _split(regs, lambda r: r.bbox[0], lambda r: r.bbox[2])
    if len(cols) > 1:
        if rtl:
            cols = cols[::-1]
        return [x for g in cols for x in xy_cut(g, rtl, depth + 1)]
    # no clean cut: fall back to top edge, then horizontal position
    return sorted(regs, key=lambda r: (r.bbox[1], -r.bbox[2] if rtl else r.bbox[0]))


def _segments(nonblank: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    idx = np.nonzero(nonblank)[0]
    if len(idx) == 0:
        return []
    segs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            segs.append((start, prev + 1))
            start = i
        prev = i
    segs.append((start, prev + 1))
    return segs


def detect_panels(rgb: np.ndarray, rtl: bool) -> list[tuple[int, int, int, int]]:
    """Recursive gutter cutting (rows of panels, then panels in a row). Works for manga/comic
    pages, and degrades gracefully to paragraph/column blocks for screenshots and documents."""
    H, W = rgb.shape[:2]
    frame = np.concatenate([rgb[:2].reshape(-1, 3), rgb[-2:].reshape(-1, 3),
                            rgb[:, :2].reshape(-1, 3), rgb[:, -2:].reshape(-1, 3)])
    bg = np.median(frame, axis=0).astype(np.int16)
    blank = np.abs(rgb.astype(np.int16) - bg).max(axis=2) < 30
    min_gap = max(4, int(0.006 * max(H, W)))
    out: list[tuple[int, int, int, int]] = []

    def cut(x0, y0, x1, y1, depth):
        sub = ~blank[y0:y1, x0:x1]
        if depth > 8 or sub.size == 0:
            out.append((x0, y0, x1, y1))
            return
        rows = sub.sum(axis=1) > max(1, 0.002 * sub.shape[1])   # gutter = (almost) no ink at all
        segs = _segments(rows, min_gap)
        if len(segs) > 1:
            for a, b in segs:
                if b - a >= 12:
                    cut(x0, y0 + a, x1, y0 + b, depth + 1)
            return
        cols = sub.sum(axis=0) > max(1, 0.002 * sub.shape[0])
        segs = _segments(cols, min_gap)
        if len(segs) > 1:
            for a, b in (segs[::-1] if rtl else segs):
                if b - a >= 12:
                    cut(x0 + a, y0, x0 + b, y1, depth + 1)
            return
        out.append((x0, y0, x1, y1))

    cut(0, 0, W, H, 0)
    return out or [(0, 0, W, H)]


def order_regions(regs: list[Region], rtl: bool, rgb: np.ndarray | None = None) -> list[Region]:
    if rgb is not None and len(regs) > 1:
        panels = detect_panels(rgb, rtl)
        buckets: dict[int, list[Region]] = {}
        for r in regs:
            b = r.bbox
            best, best_ov = len(panels), 0
            for i, (x0, y0, x1, y1) in enumerate(panels):
                ov = max(0, min(b[2], x1) - max(b[0], x0)) * max(0, min(b[3], y1) - max(b[1], y0))
                if ov > best_ov:
                    best, best_ov = i, ov
            buckets.setdefault(best, []).append(r)
        ordered = [x for k in sorted(buckets) for x in xy_cut(buckets[k], rtl)]
    else:
        ordered = xy_cut(regs, rtl)
    for i, r in enumerate(ordered):
        r.order = i
        r.id = i + 1
    return ordered
