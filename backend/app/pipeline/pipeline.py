"""
Image -> preprocessing -> text detection -> orientation detection -> OCR -> reading-order
analysis -> translation -> text removal/inpainting -> typography/layout reconstruction ->
quality validation -> final image
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import config
from ..engines import ocr_engines as oe
from ..engines.translators import PageContext, make_translator
from . import detection, fonts, ocr, orientation, regions as regmod
from .analysis import closed_region, estimate_style, is_bubble
from .compose import typeset_page
from .inpaint import remove_text, residual_pass
from .preprocess import Page, encode_image, load_image
from .datatypes import Issue, PipelineError, Region
from .validate import check_order, check_translations

log = logging.getLogger("mtl.pipeline")

STAGES = [
    ("preprocess", "Preparing image"), ("detect", "Finding text"), ("orientation", "Checking orientation"),
    ("ocr", "Reading text"), ("order", "Ordering regions"), ("translate", "Translating"),
    ("inpaint", "Removing original text"), ("typeset", "Lettering"), ("validate", "Validating"),
    ("save", "Saving"),
]

Progress = Callable[[str, float], None]


@dataclass
class PageResult:
    image_bytes: bytes
    ext: str
    regions: list[Region]
    issues: list[Issue]
    meta: dict = field(default_factory=dict)


def _stage(progress: Optional[Progress], name: str, frac: float = 0.0):
    if progress:
        idx = [s for s, _ in STAGES].index(name)
        progress(name, (idx + frac) / len(STAGES))


# ------------------------------------------------------------------ helpers
def _context_images(rgb: np.ndarray, regions: list[Region]) -> list[bytes]:
    """Downscaled page (or tiles for tall strips) with numbered region outlines, as JPEG."""
    H, W = rgb.shape[:2]
    tiles = []
    if H > 2.2 * W:
        th = int(W * 1.8)
        ys = list(range(0, H, int(th * 0.85)))[:8]
        tiles = [(0, y, W, min(H, y + th)) for y in ys]
    elif W > 2.2 * H:
        tw = int(H * 1.8)
        xs = list(range(0, W, int(tw * 0.85)))[:8]
        tiles = [(x, 0, min(W, x + tw), H) for x in xs]
    else:
        tiles = [(0, 0, W, H)]
    out = []
    for x0, y0, x1, y1 in tiles:
        crop = Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGB")
        scale = min(1.0, 1568.0 / max(crop.size))
        if scale < 1:
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        d = ImageDraw.Draw(crop)
        fs = max(14, int(min(crop.size) * 0.028))
        try:
            f = ImageFont.truetype(fonts.choose_font("latin", "0", False).regular, fs)
        except Exception:
            f = ImageFont.load_default()
        vis = False
        for r in regions:
            q = (r.quad - np.array([x0, y0])) * scale
            if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() > crop.width or q[:, 1].min() > crop.height:
                continue
            vis = True
            d.polygon([tuple(p) for p in q], outline=(230, 30, 60), width=2)
            tx, ty = float(q[:, 0].min()), float(q[:, 1].min()) - fs
            d.rectangle([tx, ty, tx + fs * (0.7 * len(str(r.id)) + 0.4), ty + fs * 1.1], fill=(230, 30, 60))
            d.text((tx + 2, ty), str(r.id), fill=(255, 255, 255), font=f)
        if vis:
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=88)
            out.append(buf.getvalue())
    return out


def _classify(rgb: np.ndarray, r: Region) -> Optional[object]:
    """Closed-region + style analysis. Returns the Closed object for bubbles."""
    g = max(4.0, r.glyph_px)
    closed = closed_region(rgb, [l.quad for l in r.lines], g, max(1.2 * max(r.width, r.height), 6 * g))
    style = estimate_style(rgb, r.quad, g)
    r.bg_color, r.fg_color, r.bg_uniform, r.bold = style["bg"], style["fg"], style["uniform"], style["bold"]
    if closed is not None and is_bubble(closed, r.width * r.height):
        r.kind = "bubble"
        r.bubble_bbox = closed.bbox
        r.bg_color = closed.bg
        r.bg_uniform = closed.uniform
    else:
        closed = None
        r.kind = "plate" if style.get("plate") else ("caption" if r.bg_uniform else "overlay")
        if r.kind == "plate":
            r.bg_uniform = True
    # alignment for multi-line horizontal text outside bubbles
    if not r.vertical and len(r.lines) >= 2 and r.kind != "bubble":
        from .geometry import axes
        u, _ = axes(r.theta)
        lefts = [float(np.min(l.quad @ u)) for l in r.lines]
        mids = [float(np.array(l.center) @ u) for l in r.lines]
        r.align = "left" if np.std(lefts) < 0.5 * np.std(mids) + 1 else "center"
    return closed


def _resolve_content_type(opts, regions: list[Region]) -> str:
    if opts.content_type != "auto":
        return opts.content_type
    if not regions:
        return "document"
    vert = sum(r.vertical for r in regions) / len(regions)
    bub = sum(r.kind == "bubble" for r in regions) / len(regions)
    return "manga" if vert > 0.3 or bub > 0.35 else "document"


# ------------------------------------------------------------------ main entry
def process(data: bytes, opts, progress: Optional[Progress] = None, history: Optional[list] = None,
            workdir: Optional[Path] = None) -> PageResult:
    issues: list[Issue] = []
    meta: dict = {}

    # 1. preprocessing ---------------------------------------------------------
    _stage(progress, "preprocess")
    page = load_image(data)
    meta["exif_rotated"] = page.exif_rotated
    rgb = page.rgb

    # 2. detection -------------------------------------------------------------
    _stage(progress, "detect")
    hint = opts.source_lang if opts.source_lang != "auto" else "ja"
    detector = oe.make_detector(opts.detector, hint)
    lines = detection.detect_lines(detector, rgb)
    meta["detector"] = detector.name

    # 3. orientation -----------------------------------------------------------
    _stage(progress, "orientation")
    src = opts.source_lang
    cjk = None if src == "auto" else config.lang(src).script in config.CJK_SCRIPTS
    for l in lines:
        orientation.analyze_line(l, cjk)
    conf_rec = oe.make_confidence_recognizer(src if src != "auto" else "ja")
    if opts.auto_rotate_pages and lines:
        rot = orientation.page_rotation_vote(rgb, lines, conf_rec, cjk_possible=cjk is not False)
        if rot:
            rgb = orientation.rotate_page(rgb, rot)
            page.rotation = rot
            issues.append(Issue("page_rotated", "info",
                                f"Page was rotated {rot} deg so its text reads upright.", None, True))
            lines = detection.detect_lines(detector, rgb)
            for l in lines:
                orientation.analyze_line(l, cjk)
    meta["page_rotation"] = page.rotation

    # source language (auto) -------------------------------------------------
    if src == "auto":
        src = ocr.probe_language(rgb, lines, oe.make_confidence_recognizer) if lines else "en"
        if src not in config.LANGS:
            src = "en"
        cjk = config.lang(src).script in config.CJK_SCRIPTS
        for l in lines:           # re-run geometry with the now-known script
            orientation.analyze_line(l, cjk)
        conf_rec = oe.make_confidence_recognizer(src)
    meta["source_lang"] = src
    vertical_default = cjk and sum(1 for l in lines if l.vertical) >= sum(1 for l in lines if l.vertical is False)
    orientation.resolve_ambiguous(lines, bool(vertical_default))
    flipped = orientation.resolve_upside_down(rgb, lines, conf_rec, thorough=False)
    if flipped:
        issues.append(Issue("upside_down", "info", f"{flipped} upside-down source line(s) were read correctly.",
                            None, True))

    # 4. OCR -------------------------------------------------------------------
    _stage(progress, "ocr")
    rec = oe.make_recognizer(opts.ocr_engine, src, opts.content_type)
    meta["ocr"] = rec.name
    ocr.recognize_lines(rgb, lines, rec)
    kept = []
    for l in lines:
        if not l.text or not ocr.has_letters(l.text):
            continue                                       # artwork / noise guard
        if 0 <= l.conf < 0.35:
            continue
        kept.append(l)
    dropped = len(lines) - len(kept)
    if dropped:
        issues.append(Issue("artwork", "info",
                            f"{dropped} detection(s) did not look like text and were left untouched.", None, True))
    lines = kept

    # 5. regions + reading order ----------------------------------------------
    _stage(progress, "order")
    join_cjk = config.lang(src).script in config.CJK_SCRIPTS or src == "th"
    regs = regmod.group_lines(rgb, lines, join_cjk)
    closed_tmp = {}
    for r in regs:
        closed_tmp[id(r)] = _classify(rgb, r)
    content = _resolve_content_type(opts, regs)
    meta["content_type"] = content
    if content in ("document", "screenshot"):
        for r in regs:
            if r.kind in ("caption", "overlay") and not r.vertical and len(r.lines) == 1:
                r.align = "left"
    if content in ("document", "screenshot"):
        for r in regs:
            if r.kind not in ("bubble", "plate") and not r.vertical and len(r.lines) == 1:
                r.align = "left"
    if opts.reading_direction == "auto":
        vert = sum(r.vertical for r in regs)
        rtl = (src in ("ja", "zh-TW") and (content == "manga" and vert >= max(1, len(regs) / 3)))
    else:
        rtl = opts.reading_direction == "rtl"
    meta["reading_direction"] = "rtl" if rtl else "ltr"
    regs = regmod.order_regions(regs, rtl, rgb)
    closed_by_id = {r.id: closed_tmp[id(r)] for r in regs}
    issues += check_order(regs)
    opts_eff = type(opts).from_dict({**opts.to_dict(), "content_type": content})

    # 6. translation -----------------------------------------------------------
    _stage(progress, "translate")
    translator = make_translator(opts.translator)
    meta["translator"] = translator.name
    if regs:
        ctx = PageContext(
            regions=[{"id": r.id, "text": r.source_text, "vertical": r.vertical, "kind_hint": r.kind,
                      "lines": [l.text for l in r.lines] if len(r.lines) > 1 else None}
                     for r in regs],
            source_lang=src, target_lang=opts.target_lang,
            images=_context_images(rgb, regs) if (opts.use_page_context and translator.vision) else [],
            glossary=opts.glossary, instructions=opts.instructions, history=list(history or []),
            sfx=opts.sfx, content_type=content)
        results, detected = translator.translate_page(ctx)
        _apply_translations(regs, results, opts, translator.smart)
        t_issues, retry = check_translations(regs, src, opts.target_lang)
        if retry:
            ctx.regions = [c for c in ctx.regions if c["id"] in retry]
            ctx.instructions = (ctx.instructions + "\nSome regions came back untranslated or empty. "
                                "Translate every listed region fully into the target language.").strip()
            again, _ = translator.translate_page(ctx)
            _apply_translations([r for r in regs if r.id in retry], again, opts, translator.smart)
            t_issues2, _ = check_translations([r for r in regs if r.id in retry], src, opts.target_lang)
            fixed_ids = retry - {i.region_id for i in t_issues2}
            for i in t_issues:
                if i.region_id in fixed_ids:
                    i.fixed, i.severity = True, "info"
            t_issues = [i for i in t_issues if i.fixed or i.region_id not in retry] + t_issues2
            for i in t_issues2:     # still bad: do not render it
                if i.severity == "error":
                    r = next(x for x in regs if x.id == i.region_id)
                    r.translation = ""
        for i in t_issues:
            r = next((x for x in regs if x.id == i.region_id), None)
            (r.issues if r else issues).append(i)
        if history is not None:
            history.extend((r.source_text, r.translation) for r in regs if r.translation and r.is_text)

    # 7. text removal ----------------------------------------------------------
    _stage(progress, "inpaint")
    clean, labels = remove_text(rgb, regs, closed_by_id)
    if opts.residual_check and regs:
        _stage(progress, "inpaint", 0.6)
        for i in residual_pass(clean, rgb, labels, regs, detector, detection.detect_lines):
            r = next((x for x in regs if x.id == i.region_id), None)
            (r.issues if r else issues).append(i)
    meta["lama"] = oe.engine_status()["lama"]

    # 8-9. typesetting + validation -----------------------------------------------
    _stage(progress, "typeset")
    verifier = None
    if opts.strict_verify:
        verifier = oe.make_confidence_recognizer(opts.target_lang)
        if verifier is None:
            issues.append(Issue("verify_unavailable", "info",
                                "No OCR engine with confidence scores for the target language; "
                                "re-read verification skipped.", None))
    final, page_issues = typeset_page(clean, rgb, labels, regs, closed_by_id, opts_eff, verifier, src)
    issues += page_issues
    _stage(progress, "validate")

    # 10. save --------------------------------------------------------------------
    _stage(progress, "save")
    img_bytes, ext = encode_image(final, page, opts.output_format)
    if workdir is not None:
        _persist(workdir, rgb, clean, labels, regs, issues, meta, page, opts_eff)
        _overlay(rgb, regs).save(workdir / "overlay.jpg", quality=88)
    return PageResult(img_bytes, ext, regs, issues, meta)


def _apply_translations(regs: list[Region], results: dict, opts, smart: bool) -> None:
    for r in regs:
        res = results.get(r.id)
        if res is None:
            continue
        if smart and res.source and res.source.strip():
            r.source_text = res.source.strip()
        r.translation = (res.translation or "").strip()
        if smart:
            r.is_text = bool(res.is_text)
            r.role = (res.kind or "").lower()
        if not r.is_text:
            r.status, r.skip_reason = "skipped", "Not text (left untouched)"
        elif r.role == "sfx" and opts.sfx == "skip":
            r.status, r.skip_reason = "skipped", "Sound effect (left untouched by setting)"
        else:
            r.status, r.skip_reason = "pending", ""


# ------------------------------------------------------------------ persistence / re-render
def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not serializable: {type(o)}")


def _persist(workdir: Path, rgb, clean, labels, regs, issues, meta, page: Page, opts):
    workdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(workdir / "work.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(workdir / "clean.png"), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(workdir / "labels.png"), labels)
    state = {"regions": [r.to_dict() for r in regs], "issues": [i.to_dict() for i in issues], "meta": meta,
             "page": {"fmt": page.fmt, "rotation": page.rotation, "has_alpha": page.alpha is not None},
             "options": opts.to_dict()}
    (workdir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1, default=_np_default), encoding="utf-8")


def load_state(workdir: Path) -> dict:
    return json.loads((workdir / "state.json").read_text(encoding="utf-8"))


def rerender(workdir: Path, edits: dict[int, str], opts, original_bytes: bytes) -> PageResult:
    """Re-typeset after the user edited translations; reuses the stored clean plate."""
    state = load_state(workdir)
    rgb = cv2.cvtColor(cv2.imread(str(workdir / "work.png")), cv2.COLOR_BGR2RGB)
    clean = cv2.cvtColor(cv2.imread(str(workdir / "clean.png")), cv2.COLOR_BGR2RGB)
    labels = cv2.imread(str(workdir / "labels.png"), cv2.IMREAD_UNCHANGED)
    regs = [Region.from_dict(d) for d in state["regions"]]
    for r in regs:
        for l in r.lines:
            orientation.analyze_line(l, None if r.vertical is None else r.vertical)
            l.vertical = r.vertical
        r.issues = []
        if r.id in edits:
            r.translation = edits[r.id].strip()
            if r.status == "skipped" and r.translation:
                r.status, r.skip_reason, r.is_text = "pending", "", True
    # regions that were skipped before inpainting have no clean plate: inpaint them now
    newly = [r for r in regs if r.id in edits and r.translation and not (labels == r.id).any()]
    closed_by_id = {}
    for r in regs:
        g = max(4.0, r.glyph_px)
        c = closed_region(rgb, [l.quad for l in r.lines], g, max(1.2 * max(r.width, r.height), 6 * g))
        closed_by_id[r.id] = c if (r.kind == "bubble" and c is not None) else None
    if newly:
        clean2, labels2 = remove_text(clean, newly, closed_by_id)
        m = labels2 > 0
        clean[m], labels[m] = clean2[m], labels2[m]
    src = state["meta"].get("source_lang", "en")
    opts_eff = type(opts).from_dict({**opts.to_dict(), "content_type": state["meta"].get("content_type", "auto")})
    verifier = oe.make_confidence_recognizer(opts.target_lang) if opts.strict_verify else None
    final, page_issues = typeset_page(clean, rgb, labels, regs, closed_by_id, opts_eff, verifier, src)
    page = load_image(original_bytes)
    page.rotation = state["page"]["rotation"]
    img_bytes, ext = encode_image(final, page, opts.output_format)
    issues = [Issue(**{k: i[k] for k in ("code", "severity", "message", "region_id", "fixed")})
              for i in state["issues"] if i.get("region_id") is None] + page_issues
    cv2.imwrite(str(workdir / "clean.png"), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(workdir / "labels.png"), labels)
    state["regions"] = [r.to_dict() for r in regs]
    (workdir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1, default=_np_default), encoding="utf-8")
    return PageResult(img_bytes, ext, regs, issues, state["meta"])


def _overlay(rgb: np.ndarray, regs: list[Region]) -> Image.Image:
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    colors = {"rendered": (43, 123, 185), "kept_original": (214, 51, 58), "skipped": (150, 150, 150),
              "pending": (224, 161, 0)}
    fs = max(12, int(min(im.size) * 0.02))
    try:
        f = ImageFont.truetype(fonts.choose_font("latin", "0", False).regular, fs)
    except Exception:
        f = ImageFont.load_default()
    for r in regs:
        c = colors.get(r.status, (224, 161, 0))
        if r.bubble_bbox:
            d.rectangle(r.bubble_bbox, outline=c + (110,), width=1)
        d.polygon([tuple(p) for p in r.quad], outline=c + (255,), fill=c + (40,), width=2)
        x, y = float(r.quad[:, 0].min()), float(r.quad[:, 1].min())
        d.rectangle([x, y - fs - 4, x + fs * 0.7 * len(str(r.order + 1)) + 8, y], fill=c + (230,))
        d.text((x + 4, y - fs - 3), str(r.order + 1), fill=(255, 255, 255), font=f)
    return im
