"""
Step 5: OCR (per line / per column, on rectified upright crops).
Also: automatic source-language detection.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from .analysis import ink_extent
from .orientation import line_crop
from .datatypes import TextLine

_RANGES = {
    "kana": [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D)],
    "han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)],
    "hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "cyrillic": [(0x0400, 0x04FF)],
    "thai": [(0x0E00, 0x0E7F)],
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
}


def script_counts(text: str) -> dict[str, int]:
    c = {k: 0 for k in _RANGES}
    for ch in text:
        o = ord(ch)
        for k, rs in _RANGES.items():
            if any(a <= o <= b for a, b in rs):
                c[k] += 1
                break
    return c


def guess_lang(texts: list[str]) -> str | None:
    c = script_counts("".join(texts))
    total = sum(c.values())
    if total < 2:
        return None
    if c["kana"] >= max(1, 0.05 * total):
        return "ja"
    if c["hangul"] >= 0.3 * total:
        return "ko"
    if c["han"] >= 0.3 * total:
        return "zh"
    if c["thai"] >= 0.3 * total:
        return "th"
    if c["cyrillic"] >= 0.3 * total:
        return "ru"
    if c["latin"] >= 0.3 * total:
        return "en"
    return None


def has_letters(text: str) -> bool:
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in text)


def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t) if not re.search(r"[\u3000-\u30ff]", t) else t
    t = re.sub(r"\s+", " ", t).strip()
    return t


def recognize_lines(rgb: np.ndarray, lines: list[TextLine], rec) -> None:
    if not lines:
        return
    crops = [line_crop(rgb, l) for l in lines]
    res = rec.recognize(crops, [bool(l.vertical) for l in lines])
    for l, crop in zip(lines, crops):
        l.ink = ink_extent(crop, bool(l.vertical))
    for l, (t, c) in zip(lines, res):
        l.text = clean_text(t or "")
        l.conf = float(c) if c is not None else -1.0   # -1 = engine gives no confidence


def probe_language(rgb: np.ndarray, lines: list[TextLine], make_rec) -> str:
    """Try a few recognizer families on the biggest lines and keep the most confident."""
    sample = sorted(lines, key=lambda l: -l.length * l.thickness)[:8]
    if not sample:
        return "en"
    crops = [line_crop(rgb, l) for l in sample]
    vflags = [bool(l.vertical) for l in sample]
    best_lang, best_conf, best_texts = "en", -1.0, []
    for probe in ("ja", "ko", "en"):
        try:
            rec = make_rec(probe)
        except Exception:
            continue
        if rec is None or not getattr(rec, "gives_confidence", True):
            continue
        res = rec.recognize(crops, vflags)
        conf = float(np.mean([c or 0.0 for _, c in res]))
        if conf > best_conf:
            best_lang, best_conf, best_texts = probe, conf, [t for t, _ in res]
    return guess_lang(best_texts) or best_lang
