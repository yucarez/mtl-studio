"""
Steps 9-10: typeset every region, validate, auto-correct, and composite the final image.

Correction loop per page (max 4 rounds):
  orientation out of bounds -> force level text          (sideways / upside-down / mirrored)
  ink outside allowed area  -> shrink & re-fit           (outside_region / clipped)
  ink of two regions touch  -> shrink both & re-fit      (overlap)
  does not fit at min size  -> grow into flat background -> policy (keep original / best effort)
  re-OCR of rendered text   -> re-render level + high contrast if it fails (strict mode)
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

import cv2
import numpy as np

from .. import config
from . import fonts
from .analysis import _kernel
from .geometry import frame_matrix, paste_frame, rect_crop, polygon_mask
from .typeset import (Frame, base_area, draw_layout, expanded_area, fit, make_frame, overflow_ratio,
                      partition, pick_colors, prepare_text, render_angle, size_bounds)
from .datatypes import Issue, Region
from .validate import ink_overlaps, orientation_ok, similarity

log = logging.getLogger("mtl.compose")


def _ink_in_image(frame: Frame, patch: np.ndarray, shape) -> tuple[tuple[int, int, int, int], np.ndarray]:
    M = frame_matrix(frame.center, frame.w, frame.h, frame.theta)
    corners = np.array([[0, 0, 1], [frame.w, 0, 1], [frame.w, frame.h, 1], [0, frame.h, 1]], float) @ M.T
    x0 = max(0, int(math.floor(corners[:, 0].min())))
    y0 = max(0, int(math.floor(corners[:, 1].min())))
    x1 = min(shape[1], int(math.ceil(corners[:, 0].max())) + 1)
    y1 = min(shape[0], int(math.ceil(corners[:, 1].max())) + 1)
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    a = cv2.warpAffine(patch[..., 3], M2, (max(1, x1 - x0), max(1, y1 - y0)), flags=cv2.INTER_NEAREST)
    return (x0, y0, x1, y1), a > 40


class RegionRender:
    def __init__(self, r: Region, frame: Frame, layout, patch, fg, stroke_color, area_kind):
        self.r, self.frame, self.layout, self.patch = r, frame, layout, patch
        self.fg, self.stroke_color, self.area_kind = fg, stroke_color, area_kind


def typeset_page(clean: np.ndarray, original: np.ndarray, labels: np.ndarray, regions: list[Region],
                 closed_by_id: dict, opts, verifier=None, source_lang: str = "en") -> tuple[np.ndarray, list[Issue]]:
    shape = clean.shape
    tgt = config.lang(opts.target_lang)
    spaced_src = config.lang(source_lang).spaced if source_lang in config.LANGS else True
    prefer_comic = opts.content_type in ("manga", "comic", "webtoon")
    page_issues: list[Issue] = []

    todo = [r for r in regions if r.status not in ("skipped",) and r.translation.strip()]
    for r in todo:   # drop stale typesetting data (re-render after edits)
        r.render = {k: v for k, v in r.render.items() if k in ("fill_bg", "fill_method")}
        r.status = "pending"
    for r in regions:
        if r.status != "skipped" and not r.translation.strip() and r.source_text.strip():
            r.status = "kept_original"

    areas, kinds = {}, {}
    for r in todo:
        areas[r.id], kinds[r.id] = base_area(shape, r, closed_by_id.get(r.id))
    areas = partition(areas, {r.id: r.center for r in todo})

    caps: dict[int, int] = {}          # per-region max font size imposed by corrections
    forced_level: set[int] = set()
    force_contrast: set[int] = set()
    renders: dict[int, RegionRender] = {}
    failed: dict[int, Issue] = {}
    pending = list(todo)

    for round_ in range(4):
        for r in pending:
            renders.pop(r.id, None)
            res = _render_region(r, clean, areas, kinds, closed_by_id, opts, tgt, spaced_src, prefer_comic,
                                 caps.get(r.id), r.id in forced_level, r.id in force_contrast, regions)
            if isinstance(res, Issue):
                failed[r.id] = res
            else:
                failed.pop(r.id, None)
                renders[r.id] = res
        pending = []

        # --- page-level: ink overlap between regions
        inks = {rid: _ink_in_image(rr.frame, rr.patch, shape) for rid, rr in renders.items()}
        for a, b, k in ink_overlaps(inks):
            for rid in (a, b):
                rr = renders[rid]
                caps[rid] = max(4, int(rr.layout.size * 0.9))
                if rr.r not in pending:
                    pending.append(rr.r)
                    rr.r.issues.append(Issue("overlap", "warning",
                                             f"Rendered text touched region {b if rid == a else a}; resized.",
                                             rid, fixed=True))
        # --- hierarchy: keep bubble text sizes consistent on the page
        if round_ == 0:
            bub = [rr for rr in renders.values() if rr.area_kind == "bubble"]
            if len(bub) >= 3:
                med_size = float(np.median([rr.layout.size for rr in bub]))
                med_glyph = float(np.median([rr.r.glyph_px for rr in bub]))
                for rr in bub:
                    if rr.layout.size > 1.25 * med_size and rr.r.glyph_px < 1.2 * med_glyph:
                        caps[rr.r.id] = int(1.25 * med_size)
                        if rr.r not in pending:
                            pending.append(rr.r)
        # --- strict verification by re-reading the rendered text
        if verifier is not None and opts.strict_verify and not pending and round_ < 3:
            for rid, rr in list(renders.items()):
                if rr.r.render.get("verified"):
                    continue
                verdict = _verify(rr, clean, verifier)
                rr.r.render["verify"] = verdict
                if verdict["upside_down"] and rid in forced_level:
                    # already level (upright by construction): the re-reader is unreliable here
                    rr.r.issues.append(Issue("verify_failed", "warning",
                                             "Re-read check disagrees with the (level, upright) render. "
                                             "Please check this region.", rid))
                    rr.r.render["verified"] = True
                elif verdict["upside_down"]:
                    rr.r.issues.append(Issue("upside_down", "error",
                                             "Re-reading suggested the text is upside down; re-rendered level.",
                                             rid, fixed=True))
                    forced_level.add(rid)
                    pending.append(rr.r)
                elif verdict["similarity"] < 0.45 and rid in force_contrast:
                    rr.r.issues.append(Issue("verify_failed", "warning",
                                             f"Rendered text still re-reads poorly (similarity "
                                             f"{verdict['similarity']:.2f}). Please check this region.", rid))
                    rr.r.render["verified"] = True
                elif verdict["similarity"] < 0.45:
                    rr.r.issues.append(Issue("verify_failed", "warning",
                                             f"Rendered text was hard to re-read (similarity "
                                             f"{verdict['similarity']:.2f}); re-rendered with higher contrast.",
                                             rid, fixed=True))
                    force_contrast.add(rid)
                    pending.append(rr.r)
                else:
                    rr.r.render["verified"] = True
        if not pending:
            break

    # --- composite
    final = clean.copy()
    for r in regions:
        if r.id in failed or r.status == "kept_original":
            if r.id in failed:
                r.issues.append(failed[r.id])
            m = labels == r.id            # restore the untouched original pixels
            final[m] = original[m]
            r.status = "kept_original"
    for rid, rr in renders.items():
        ok, code = orientation_ok(rr.frame.theta, opts.follow_tilt_max)
        det = np.linalg.det(frame_matrix(rr.frame.center, rr.frame.w, rr.frame.h, rr.frame.theta)[:, :2])
        if not ok or det <= 0:  # cannot happen by construction; belt and braces
            rr.r.issues.append(Issue(code or "mirrored", "error", "Invalid orientation blocked.", rid))
            m = labels == rid
            final[m] = original[m]
            rr.r.status = "kept_original"
            continue
        paste_frame(final, rr.patch, rr.frame.center, rr.frame.theta)
        rr.r.status = "rendered"
        L = rr.layout
        rr.r.render.update({
            "font": os.path.basename(rr.layout.font_path),
            "size": L.size, "line_height": L.line_h, "bold": L.bold, "stroke": L.stroke,
            "fg": list(rr.fg), "stroke_color": list(rr.stroke_color), "theta": rr.frame.theta,
            "area": rr.area_kind, "lines": [ln["text"] for ln in L.lines],
            "frame": {"cx": rr.frame.center[0], "cy": rr.frame.center[1], "w": rr.frame.w,
                      "h": rr.frame.h},
        })
    return final, page_issues


def _render_region(r: Region, clean, areas, kinds, closed_by_id, opts, tgt, spaced_src, prefer_comic,
                   cap: Optional[int], force_level: bool, force_contrast: bool, all_regions):
    text = prepare_text(r, opts)
    choice = fonts.choose_font(tgt.script, text, prefer_comic, opts.font)
    if choice is None:
        return Issue("glyph_missing", "error", "No installed font can display this language. "
                     "Run scripts/download_fonts.py.", r.id)
    missing = fonts.missing_glyphs(choice.regular, text)
    if missing:
        r.issues.append(Issue("glyph_missing", "warning",
                              f"Font '{choice.family}' lacks: {missing[:12]}", r.id))
    bold_flag = r.bold or prefer_comic
    path = choice.path(bold_flag)

    theta = 0.0 if force_level else render_angle(r, opts.follow_tilt_max)
    ok, code = orientation_ok(theta, opts.follow_tilt_max)
    if not ok:
        r.issues.append(Issue(code, "error", "Render angle out of bounds; forced level.", r.id, fixed=True))
        theta = 0.0

    bubble = kinds[r.id] in ("bubble", "plate")
    fg, stroke_color, stroke_ratio = pick_colors(r, bubble)
    if force_contrast:
        bg = tuple(r.render.get("fill_bg", r.bg_color))
        fg = (0, 0, 0) if sum(bg) > 384 else (255, 255, 255)
        stroke_color = (255, 255, 255) if fg == (0, 0, 0) else (0, 0, 0)
        stroke_ratio = max(stroke_ratio, 0.1)
    s_min, s_max = size_bounds(r, opts, spaced_src)
    if cap:
        s_max = max(s_min, min(s_max, cap))
    align = "center" if bubble or r.vertical else r.align

    def try_area(area, kind, smin, smax, hyph):
        fr = make_frame(area, theta)
        if fr is None:
            return None
        lay = fit(fr, kind, r, text, path, bold_flag, tgt.spaced, smin, smax, align, stroke_ratio, hyph)
        if lay is None:
            return None
        patch = draw_layout(fr, lay, fg, stroke_color)
        shrink = 0
        while overflow_ratio(fr, patch) > 0.003 and lay.size > smin and shrink < 12:
            lay2 = fit(fr, kind, r, text, path, bold_flag, tgt.spaced, smin, lay.size - 1, align,
                       stroke_ratio, hyph)
            if lay2 is None:
                break
            lay, patch, shrink = lay2, draw_layout(fr, lay2, fg, stroke_color), shrink + 1
        if overflow_ratio(fr, patch) > 0.003:
            return None
        if shrink:
            r.issues.append(Issue("clipped", "info", "Glyph extents exceeded the area; font reduced.", r.id, True))
        return fr, lay, patch

    area, kind = areas[r.id], kinds[r.id]
    candidates = [(area, kind, "base")]
    if kind == "bubble":
        a2, k2 = base_area(clean.shape, r, closed_by_id.get(r.id), pad_scale=0.45)
        candidates.append((_minus_others(areas, r.id, a2), k2, "bubble_tight"))
    elif kind == "rect":
        blockers = np.zeros(clean.shape[:2], bool)
        for rid, a in areas.items():
            if rid != r.id:
                blockers |= cv2.dilate(a.astype(np.uint8), _kernel(3)).astype(bool)
        for o in all_regions:
            if o.id != r.id:
                blockers |= polygon_mask(clean.shape, o.quad)
        candidates.append((expanded_area(clean, r, area, blockers), "rect", "expanded"))

    got, used = None, None
    for hyph in (False, True):            # hyphenation is a last resort
        for a, k, label in candidates:
            got = try_area(a, k, s_min, s_max, hyph)
            if got is not None:
                used = label
                break
        if (got is not None and used == "base" and kind == "rect" and r.bg_uniform
                and got[1].size < 0.85 * s_max):
            # text on flat background (screens, documents): growing into the empty space
            # around it beats shrinking/wrapping, as long as nothing else is there
            alt = try_area(candidates[-1][0], "rect", s_min, s_max, hyph)
            if alt is not None and (alt[1].size >= 1.08 * got[1].size or
                                    len(alt[1].lines) < len(got[1].lines)):
                got, used = alt, "expanded"
        if got is not None and used == "base" and kind == "rect" and r.bg_uniform:
            # flat surroundings: extend into free background rather than shrink/wrap a lot
            alt = try_area(candidates[-1][0], "rect", s_min, s_max, hyph)
            if alt is not None and (alt[1].size >= 1.08 * got[1].size or
                                    (alt[1].size >= got[1].size and len(alt[1].lines) < len(got[1].lines))):
                got, used = alt, "expanded"
        if got is not None:
            if hyph:
                r.issues.append(Issue("hyphenated", "info", "A long word was hyphenated to fit.", r.id, True))
            break
    if used == "expanded":
        # claim only the space actually used, so neighbours keep theirs
        (bx0, by0, bx1, by1), ink = _ink_in_image(got[0], got[2], clean.shape)
        claim = np.zeros(clean.shape[:2], bool)
        claim[by0:by1, bx0:bx1] = ink
        claim = cv2.dilate(claim.astype(np.uint8), _kernel(0.35 * max(4.0, r.glyph_px))).astype(bool)
        areas[r.id] = areas[r.id] | (claim & candidates[-1][0])
        r.issues.append(Issue("outside_region", "info",
                              "Text was extended into the surrounding flat background to stay legible.",
                              r.id, fixed=True))
    if got is None and opts.on_uncertain == "best_effort":
        floor = max(6, int(0.3 * r.glyph_px))
        got = try_area(candidates[-1][0], candidates[-1][1], floor, s_min, True)
        if got is not None:
            r.issues.append(Issue("too_small", "warning",
                                  f"Font shrunk to {got[1].size}px (original ~{r.glyph_px:.0f}px).", r.id))
    if got is None:
        return Issue("too_small", "error",
                     "Translation does not fit legibly in the available space; original text kept. "
                     "Edit it shorter and re-render.", r.id)
    fr, lay, patch = got
    return RegionRender(r, fr, lay, patch, fg, stroke_color, kind)


def _minus_others(areas, rid, a):
    other = np.zeros_like(a)
    for k, m in areas.items():
        if k != rid:
            other |= m
    return a & ~other


def _verify(rr: RegionRender, clean: np.ndarray, verifier) -> dict:
    """Re-read rendered lines from a scratch composite and compare with the intended text."""
    scratch = clean.copy()
    paste_frame(scratch, rr.patch, rr.frame.center, rr.frame.theta)
    L = rr.layout
    crops, crops180, texts = [], [], []
    for ln in L.lines:
        if not ln["text"].strip():
            continue
        cx = ln["x"] + ln["w"] / 2 - L.stroke
        cy = ln["baseline"] - L.ascent + L.line_h / 2 - L.stroke
        c = rr.frame.to_image(cx, cy)
        w, h = ln["w"] + 0.5 * L.line_h, L.line_h * 1.15
        crop = rect_crop(scratch, c, w, h, rr.frame.theta)
        crops.append(crop)
        crops180.append(np.ascontiguousarray(crop[::-1, ::-1]))
        texts.append(ln["text"])
    if not crops:
        return {"similarity": 1.0, "upside_down": False}
    try:
        up = verifier.recognize(crops, [False] * len(crops))
        dn = verifier.recognize(crops180, [False] * len(crops))
    except Exception as e:  # pragma: no cover
        log.warning("verification failed: %s", e)
        return {"similarity": 1.0, "upside_down": False, "error": str(e)}
    sims = [similarity(t, u[0]) for t, u in zip(texts, up)]
    flips = sum(1 for u, d in zip(up, dn) if (d[1] or 0) > (u[1] or 0) + 0.25)
    return {"similarity": float(np.mean(sims)), "upside_down": flips > len(crops) / 2,
            "read": [u[0] for u in up]}
