"""
Step 10: quality validation.

Translation-level checks run before layout (so bad regions can be re-translated);
layout/render-level checks run inside compose.typeset_page, which auto-corrects and
re-renders offending regions.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np

from .. import config
from .ocr import script_counts
from .datatypes import Issue, Region


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in ("L", "N"))


def similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def foreign_script_ratio(text: str, target_script: str) -> float:
    c = script_counts(text)
    total = sum(c.values())
    if not total:
        return 0.0
    allowed = {
        "latin": {"latin"}, "cyrillic": {"cyrillic", "latin"}, "thai": {"thai", "latin"},
        "hangul": {"hangul", "han", "latin"}, "cjk-ja": {"kana", "han", "latin"},
        "cjk-zh": {"han", "latin"}, "cjk-zht": {"han", "latin"},
    }.get(target_script, {"latin"})
    return sum(v for k, v in c.items() if k not in allowed) / total


def check_translations(regions: list[Region], source_lang: str, target_lang: str) -> tuple[list[Issue], set[int]]:
    """Returns issues and the ids that deserve a retry."""
    issues, retry = [], set()
    tgt = config.lang(target_lang)
    src_script = config.lang(source_lang).script if source_lang in config.LANGS else None
    seen: dict[str, Region] = {}
    for r in regions:
        if r.status == "skipped" or not r.is_text:
            continue
        tr = r.translation.strip()
        if r.source_text.strip() and not tr:
            issues.append(Issue("missing", "error", "No translation was returned for this region.", r.id))
            retry.add(r.id)
            continue
        ratio = foreign_script_ratio(tr, tgt.script)
        if ratio > 0.25:
            issues.append(Issue("untranslated", "error",
                                f"Translation still contains {int(ratio * 100)}% source-script characters.", r.id))
            retry.add(r.id)
        elif src_script and src_script != tgt.script and _norm(tr) == _norm(r.source_text) and _norm(tr):
            issues.append(Issue("untranslated", "warning", "Translation is identical to the source text.", r.id))
            retry.add(r.id)
        key = _norm(tr)
        if len(key) >= 4:
            if key in seen and _norm(seen[key].source_text) != _norm(r.source_text):
                issues.append(Issue("duplicate", "warning",
                                    f"Same translation as region {seen[key].id} although the source differs.", r.id))
            seen.setdefault(key, r)
    return issues, retry


def check_order(regions: list[Region]) -> list[Issue]:
    orders = [r.order for r in regions]
    if sorted(orders) != list(range(len(regions))) or len({r.id for r in regions}) != len(regions):
        return [Issue("reading_order", "error", "Reading order was inconsistent and has been rebuilt.", None, True)]
    return []


def ink_overlaps(inks: dict[int, tuple[tuple[int, int, int, int], np.ndarray]]) -> list[tuple[int, int, int]]:
    """inks: id -> (bbox, bool mask in bbox). Returns (id_a, id_b, overlapping pixels)."""
    out = []
    ids = list(inks)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (ax0, ay0, ax1, ay1), ma = inks[ids[i]]
            (bx0, by0, bx1, by1), mb = inks[ids[j]]
            x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
            if x1 <= x0 or y1 <= y0:
                continue
            sa = ma[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
            sb = mb[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
            k = int((sa & sb).sum())
            if k > 0:
                out.append((ids[i], ids[j], k))
    return out


def orientation_ok(theta: float, follow_max: float) -> tuple[bool, str]:
    if abs(theta) > 90:
        return False, "upside_down"
    if abs(theta) > min(40.0, follow_max) + 1e-6:
        return False, "sideways"
    return True, ""
