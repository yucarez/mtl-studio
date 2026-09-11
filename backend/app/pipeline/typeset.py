"""
Step 9: typography / layout reconstruction.

Rules that make sideways / upside-down / mirrored output impossible by construction:
  * Translated text is always laid out as upright horizontal lines in a local frame.
  * The frame may follow the original tilt only within +-follow_tilt_max (<= 40 deg);
    anything steeper (vertical CJK columns, 90-deg signs, upside-down text) is rendered level.
  * The frame transform is a pure rotation (determinant +1): no mirroring is possible.

Fitting is shape-aware: each line gets the width actually available at its height inside
the bubble mask, so text follows oval bubbles instead of overflowing their edges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import config
from . import fonts
from .analysis import _kernel, contrast_ratio
from .geometry import axes, frame_matrix, norm180, rect_quad, warp_mask_to_frame, polygon_mask
from .datatypes import Region

NO_START = set("、。，．・：；？！ー）」』】〕〉》…‥々ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ,.!?;:)]}”’%")
NO_END = set("（「『【〔〈《([{“‘")


def render_angle(r: Region, follow_max: float) -> float:
    t = norm180(r.theta)
    if abs(t) > 90:                 # upside-down source -> render upright
        t = norm180(t + 180)
    if abs(t) > min(follow_max, 40.0):
        t = 0.0                     # steep / vertical / sideways source -> level text
    if abs(t) < 2.0:
        t = 0.0                     # snap near-level text for crisp, unresampled glyphs
    return t


# ------------------------------------------------------------------ allowed areas
def base_area(shape, r: Region, closed, pad_scale: float = 1.0) -> tuple[np.ndarray, str]:
    """Image-space bool mask where the translation of r may be drawn."""
    g = max(4.0, r.glyph_px)
    if closed is not None and r.kind == "bubble":
        m = closed.full_mask(shape)
        bw = closed.bbox[2] - closed.bbox[0]
        bh = closed.bbox[3] - closed.bbox[1]
        pad = max(3.0, 0.07 * min(bw, bh)) * pad_scale
        m = cv2.erode(m.astype(np.uint8), _kernel(pad)).astype(bool)
        if m.sum() > 0.5 * r.width * r.height:
            return m, "bubble"
    if r.kind == "plate":          # flat label: letter strictly inside it
        inset = max(1.0, 0.06 * min(r.width, r.height))
        q = rect_quad(r.center, r.width - 2 * inset, r.height - 2 * inset, r.theta)
        return polygon_mask(shape, q), "plate"
    grow = 0.3 * g
    q = rect_quad(r.center, r.width + 2 * grow, r.height + 2 * grow, r.theta)
    return polygon_mask(shape, q), "rect"


def expanded_area(clean: np.ndarray, r: Region, base: np.ndarray, blockers: np.ndarray) -> np.ndarray:
    """Grow a non-bubble area into surrounding *flat* background (never over artwork)."""
    g = max(4.0, r.glyph_px)
    if r.align == "left":   # grow mostly to the right and downward, keep the left margin
        u, n = axes(r.theta)
        if len(r.lines) == 1:      # heading / label: extend along the line
            grow_r, grow_d = 0.9 * r.width + 2 * g, 0.5 * r.height + g
        else:                      # paragraph: keep its measure, grow downward
            grow_r, grow_d = 0.15 * r.width + g, 0.8 * r.height + 2 * g
        c = np.array(r.center) + u * (grow_r - 0.3 * g) / 2 + n * grow_d / 2
        q = rect_quad(tuple(c), r.width + 0.3 * g + grow_r, r.height + grow_d + 0.6 * g, r.theta)
    else:
        q = rect_quad(r.center, r.width * 1.35 + 2 * g, r.height * 1.35 + 2 * g, r.theta)
    cand = polygon_mask(clean.shape, q) & ~blockers
    bg = np.array(r.render.get("fill_bg", r.bg_color), np.int16)
    diff = np.abs(clean.astype(np.int16) - bg).max(axis=2)
    flat = diff < 24
    grown = (cand & flat) | base
    grown = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, _kernel(0.3 * g)).astype(bool)
    return grown | base


def partition(areas: dict[int, np.ndarray], centers: dict[int, tuple]) -> dict[int, np.ndarray]:
    """Make allowed areas disjoint: contested pixels go to the nearest region centre."""
    ids = list(areas)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            inter = areas[a] & areas[b]
            if not inter.any():
                continue
            ys, xs = np.nonzero(inter)
            da = (xs - centers[a][0]) ** 2 + (ys - centers[a][1]) ** 2
            db = (xs - centers[b][0]) ** 2 + (ys - centers[b][1]) ** 2
            to_a = da <= db
            areas[b][ys[to_a], xs[to_a]] = False
            areas[a][ys[~to_a], xs[~to_a]] = False
    return areas


# ------------------------------------------------------------------ frame
@dataclass
class Frame:
    center: tuple[float, float]
    w: int
    h: int
    theta: float
    mask: np.ndarray          # frame-space bool

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        M = frame_matrix(self.center, self.w, self.h, self.theta)
        p = M @ np.array([x, y, 1.0])
        return float(p[0]), float(p[1])


def make_frame(area: np.ndarray, theta: float) -> Optional[Frame]:
    ys, xs = np.nonzero(area)
    if len(xs) < 20:
        return None
    if theta == 0.0:
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        return Frame(((x0 + x1) / 2, (y0 + y1) / 2), int(x1 - x0), int(y1 - y0), 0.0,
                     area[y0:y1, x0:x1].copy())
    u, n = axes(theta)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    a, b = pts @ u, pts @ n
    a0, a1, b0, b1 = a.min(), a.max() + 1, b.min(), b.max() + 1
    c = ((a0 + a1) / 2) * u + ((b0 + b1) / 2) * n
    w, h = int(math.ceil(a1 - a0)) + 2, int(math.ceil(b1 - b0)) + 2
    m = warp_mask_to_frame(area, (c[0], c[1]), w, h, theta)
    return Frame((float(c[0]), float(c[1])), w, h, theta, m)


def _anchor(frame: Frame, kind: str, r: Region) -> tuple[int, int]:
    m = frame.mask
    if kind == "bubble":
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        core = dt >= 0.6 * dt.max()
        ys, xs = np.nonzero(core)
        return int(np.median(xs)), int(np.median(ys))
    # rect / expanded: keep the text where the original was
    M = frame_matrix(frame.center, frame.w, frame.h, frame.theta)
    Minv = cv2.invertAffineTransform(M)
    p = Minv @ np.array([r.center[0], r.center[1], 1.0])
    x = int(np.clip(p[0], 0, frame.w - 1))
    y = int(np.clip(p[1], 0, frame.h - 1))
    if not m[y, x]:
        ys, xs = np.nonzero(m)
        k = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        x, y = int(xs[k]), int(ys[k])
    return x, y


def _spans(mask: np.ndarray, cx: int) -> tuple[np.ndarray, np.ndarray]:
    H, W = mask.shape
    left = np.zeros(H, np.int32)
    right = np.zeros(H, np.int32)
    for y in range(H):
        row = mask[y]
        if not row[cx]:
            continue
        lf = np.nonzero(~row[:cx])[0]
        rt = np.nonzero(~row[cx:])[0]
        left[y] = lf[-1] + 1 if len(lf) else 0
        right[y] = cx + rt[0] if len(rt) else W
    return left, right


# ------------------------------------------------------------------ tokens & wrapping
def tokenize(text: str, spaced: bool) -> list[str]:
    toks: list[str] = []
    for pi, para in enumerate(text.split("\n")):
        if pi:
            toks.append("\n")
        if spaced:
            toks += [w for w in para.split(" ") if w]
        else:
            for ch in para:
                if ch == " ":
                    continue
                if toks and toks[-1] != "\n" and (ch in NO_START or toks[-1][-1] in NO_END):
                    toks[-1] += ch
                else:
                    toks.append(ch)
    return toks


def wrap(tokens: list[str], widths: list[float], measure, space_w: float, spaced: bool,
         hyphenate: bool = True) -> Optional[list[tuple[str, float]]]:
    lines: list[tuple[str, float]] = []
    i, li = 0, 0
    toks = list(tokens)
    while i < len(toks):
        if li >= len(widths):
            return None
        avail = widths[li]
        cur, cur_w = "", 0.0
        while i < len(toks):
            t = toks[i]
            if t == "\n":
                i += 1
                break
            tw = measure(t)
            add = tw + (space_w if cur and spaced else 0.0)
            if cur_w + add <= avail:
                cur = (cur + " " + t) if (cur and spaced) else cur + t
                cur_w += add
                i += 1
                continue
            if not cur and tw > avail:
                if not (hyphenate and spaced and len(t) >= 6):
                    if not spaced and len(t) > 1:   # split a kinsoku cluster as last resort
                        toks[i:i + 1] = list(t)
                        continue
                    return None
                k = len(t) - 1
                while k >= 3 and (measure(t[:k] + "-") > avail or
                                  not (t[k - 1].isalpha() and t[k].isalpha())):
                    k -= 1   # only break between two letters, never at punctuation
                if (k < 3 or sum(ch.isalpha() for ch in t[:k]) < 3
                        or sum(ch.isalpha() for ch in t[k:]) < 3):
                    return None
                toks[i:i + 1] = [t[:k] + "-", t[k:]]
                continue
            break
        lines.append((cur, cur_w))
        li += 1
    return lines


@dataclass
class Layout:
    size: int
    line_h: int
    ascent: int
    lines: list[dict]              # {text, x, baseline, w}
    font_path: str
    bold: bool
    stroke: int
    ok: bool = True
    widths_scale: float = 1.0
    meta: dict = field(default_factory=dict)


def _orig_left(frame: Frame, r: Region) -> float:
    """Frame x of the original text's left edge (keeps left-aligned text in place)."""
    Minv = cv2.invertAffineTransform(frame_matrix(frame.center, frame.w, frame.h, frame.theta))
    return float(min((Minv @ np.array([p[0], p[1], 1.0]))[0] for p in r.quad))


def fit(frame: Frame, kind: str, r: Region, text: str, font_path: str, bold: bool, spaced: bool,
        s_min: int, s_max: int, align: str, stroke_ratio: float, hyphenate: bool = False) -> Optional[Layout]:
    mask = frame.mask
    cx, cy = _anchor(frame, kind, r)
    left, right = _spans(mask, cx)
    if align == "left":
        x_left0 = int(max(0, _orig_left(frame, r)))
        left = np.where(right > left, np.maximum(left, min(x_left0, cx)), left)
    top_anchor = None
    if align == "left" and kind not in ("bubble", "plate"):
        M = frame_matrix(frame.center, frame.w, frame.h, frame.theta)
        Minv = cv2.invertAffineTransform(M)
        fq = np.c_[r.quad, np.ones(4)] @ Minv.T
        x_left0 = int(max(0, np.floor(fq[:, 0].min())))
        left = np.maximum(left, np.minimum(x_left0, right))
        top_anchor = float(fq[:, 1].min())   # left-aligned text grows downward from its top
    rows = np.nonzero(right > left)[0]
    if len(rows) == 0:
        return None
    y_first, y_last = int(rows.min()), int(rows.max())
    tokens = tokenize(text, spaced)
    if not tokens:
        return None

    def attempt(s: int, scale: float = 1.0, n_fixed: Optional[int] = None):
        font = fonts.load(font_path, s, bold)
        asc, desc = font.getmetrics()
        stroke = int(round(stroke_ratio * s)) if stroke_ratio else 0
        lh = int(math.ceil((asc + desc) * (1.0 if spaced else 1.08))) + 2 * stroke
        cache: dict[str, float] = {}

        def meas(t):
            if t not in cache:
                cache[t] = font.getlength(t) + 2 * stroke
            return cache[t]

        space_w = font.getlength(" ") if spaced else 0.0
        n_max = max(1, (y_last - y_first + 1) // lh)
        ns = [n_fixed] if n_fixed else range(1, n_max + 1)
        for n in ns:
            block_h = n * lh
            base_top = cy - block_h / 2 if top_anchor is None else top_anchor + max(0, (r.glyph_px * 1.1 - lh) / 2)
            for off in (0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0):
                top = int(round(base_top + off * lh))
                top = max(y_first, min(top, y_last + 1 - block_h))
                if top < y_first:
                    continue
                spans = []
                for i in range(n):
                    a, b = top + i * lh, top + (i + 1) * lh
                    L, R = left[a:b].max(), right[a:b].min()
                    spans.append((int(L), int(R)))
                widths = [max(0.0, (R - L) * scale - 1) for L, R in spans]
                if min(widths) <= 0:
                    continue
                out = wrap(tokens, widths, meas, space_w, spaced, hyphenate)
                if out is None:
                    continue
                lines = []
                for i, (t, w) in enumerate(out):
                    L, R = spans[i]
                    x = L + stroke if align == "left" else (L + R) / 2 - w / 2 + stroke
                    lines.append({"text": t, "x": float(x), "baseline": float(top + i * lh + stroke + asc),
                                  "w": float(w)})
                # drop trailing empty lines produced by forced breaks
                while lines and not lines[-1]["text"]:
                    lines.pop()
                return Layout(s, lh, asc, lines, font_path, bold, stroke, widths_scale=scale), n
        return None

    lo, hi, best = s_min, max(s_min, s_max), None
    while lo <= hi:
        mid = (lo + hi) // 2
        res = attempt(mid)
        if res:
            best, lo = res, mid + 1
        else:
            hi = mid - 1
    if not best:
        return None
    layout, n = best
    if n > 1 and align == "center":      # balance line lengths (nicer rag, same size & line count)
        lo_s, hi_s, good = 0.45, 1.0, None
        for _ in range(7):
            mid = (lo_s + hi_s) / 2
            res = attempt(layout.size, mid, n)
            if res:
                good, hi_s = res[0], mid
            else:
                lo_s = mid
        if good:
            layout = good
    layout.meta["anchor"] = (cx, cy)
    return layout


def draw_layout(frame: Frame, layout: Layout, fg, stroke_color) -> np.ndarray:
    patch = Image.new("RGBA", (frame.w, frame.h), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)
    font = fonts.load(layout.font_path, layout.size, layout.bold)
    for ln in layout.lines:
        if not ln["text"]:
            continue
        d.text((ln["x"], ln["baseline"]), ln["text"], font=font, fill=tuple(fg) + (255,), anchor="ls",
               stroke_width=layout.stroke, stroke_fill=tuple(stroke_color) + (255,))
    return np.asarray(patch).copy()


def overflow_ratio(frame: Frame, patch: np.ndarray) -> float:
    ink = patch[..., 3] > 40
    if not ink.any():
        return 1.0
    allowed = cv2.dilate(frame.mask.astype(np.uint8), _kernel(1)).astype(bool)
    return float((ink & ~allowed).sum()) / float(ink.sum())


def pick_colors(r: Region, bubble: bool) -> tuple[tuple, tuple, float]:
    bg = tuple(r.render.get("fill_bg", r.bg_color))
    fg = tuple(r.fg_color)
    if contrast_ratio(fg, bg) < 3.0:
        fg = (0, 0, 0) if contrast_ratio((0, 0, 0), bg) >= contrast_ratio((255, 255, 255), bg) else (255, 255, 255)
    stroke_ratio = 0.0
    stroke_color = bg
    if not bubble and not r.bg_uniform:
        stroke_ratio = 0.12                 # text over artwork: add a halo for legibility
        stroke_color = (255, 255, 255) if sum(fg) < 384 else (0, 0, 0)
    return fg, stroke_color, stroke_ratio


def size_bounds(r: Region, opts, spaced_src: bool) -> tuple[int, int]:
    g = max(4.0, r.glyph_px)
    s_max = int(round(g * (1.0 if spaced_src else 1.08)))
    s_min = int(max(opts.min_font_px, math.floor(g * opts.min_font_ratio)))
    return min(s_min, s_max), max(s_min, s_max)


def prepare_text(r: Region, opts) -> str:
    t = r.translation.strip()
    if opts.text_case == "upper" and config.lang(opts.target_lang).script in ("latin", "cyrillic"):
        t = t.upper()
    return t
