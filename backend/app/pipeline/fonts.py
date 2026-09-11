"""
Font discovery with glyph-coverage checks (no tofu boxes).

Put extra .ttf/.otf/.ttc files in ./fonts (run `python scripts/download_fonts.py` for a good
free set). System font folders are searched too.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import ImageFont

from .. import config

_DIRS = [config.FONT_DIR, Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
         Path.home() / ".fonts", Path.home() / ".local/share/fonts", Path("/Library/Fonts"),
         Path("/System/Library/Fonts"), Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"]


@functools.lru_cache(maxsize=1)
def _index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for d in _DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
                idx.setdefault(p.name.lower(), str(p))
    return idx


@dataclass(frozen=True)
class Family:
    name: str
    regular: tuple[str, ...]
    bold: tuple[str, ...] = ()
    comic: bool = False


FAMILIES = [
    Family("Comic Neue", ("comicneue-bold.ttf", "comicneue-regular.ttf"), ("comicneue-bold.ttf",), True),
    Family("Anime Ace", ("animeace2_reg.ttf", "animeace2_bld.ttf"), ("animeace2_bld.ttf",), True),
    Family("Noto Sans", ("notosans-regular.ttf", "notosans[wdth,wght].ttf"),
           ("notosans-bold.ttf", "notosans[wdth,wght].ttf")),
    Family("Noto Sans JP", ("notosanscjk-regular.ttc", "notosanscjkjp-regular.otf", "notosansjp[wght].ttf",
                            "notosansjp-regular.ttf"),
           ("notosanscjk-bold.ttc", "notosanscjkjp-bold.otf", "notosansjp[wght].ttf")),
    Family("Noto Sans KR", ("notosanskr[wght].ttf", "notosanscjk-regular.ttc", "notosanskr-regular.ttf"),
           ("notosanskr[wght].ttf", "notosanscjk-bold.ttc")),
    Family("Noto Sans SC", ("notosanssc[wght].ttf", "notosanscjk-regular.ttc", "notosanssc-regular.ttf"),
           ("notosanssc[wght].ttf", "notosanscjk-bold.ttc")),
    Family("Noto Sans TC", ("notosanstc[wght].ttf", "notosanscjk-regular.ttc", "notosanstc-regular.ttf"),
           ("notosanstc[wght].ttf", "notosanscjk-bold.ttc")),
    Family("Noto Sans Thai", ("notosansthai[wdth,wght].ttf", "notosansthai-regular.ttf"),
           ("notosansthai-bold.ttf", "notosansthai[wdth,wght].ttf")),
    Family("DejaVu Sans", ("dejavusans.ttf",), ("dejavusans-bold.ttf",)),
    Family("Liberation Sans", ("liberationsans-regular.ttf",), ("liberationsans-bold.ttf",)),
    Family("Arial", ("arial.ttf",), ("arialbd.ttf",)),
    Family("Yu Gothic", ("yugothm.ttc", "msgothic.ttc"), ("yugothb.ttc",)),
    Family("Malgun Gothic", ("malgun.ttf",), ("malgunbd.ttf",)),
    Family("Microsoft YaHei", ("msyh.ttc",), ("msyhbd.ttc",)),
]

_PREFS = {
    "latin": ["Comic Neue", "Anime Ace", "Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial"],
    "cyrillic": ["Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial", "Comic Neue"],
    "cjk-ja": ["Noto Sans JP", "Yu Gothic", "Noto Sans SC"],
    "cjk-zh": ["Noto Sans SC", "Microsoft YaHei", "Noto Sans JP"],
    "cjk-zht": ["Noto Sans TC", "Noto Sans JP", "Noto Sans SC"],
    "hangul": ["Noto Sans KR", "Malgun Gothic", "Noto Sans JP"],
    "thai": ["Noto Sans Thai", "Noto Sans"],
}


def _resolve(names: tuple[str, ...]) -> Optional[str]:
    idx = _index()
    for n in names:
        if n in idx:
            return idx[n]
    return None


@functools.lru_cache(maxsize=64)
def _cmap(path: str) -> frozenset:
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(path, fontNumber=0, lazy=True)
        return frozenset(f.getBestCmap().keys())
    except Exception:
        return frozenset()


def covers(path: str, text: str) -> bool:
    cm = _cmap(path)
    if not cm:
        return True  # cannot check: assume ok
    return all(ord(ch) in cm or ch.isspace() for ch in text)


def missing_glyphs(path: str, text: str) -> str:
    cm = _cmap(path)
    return "".join(sorted({ch for ch in text if cm and not ch.isspace() and ord(ch) not in cm}))


@dataclass
class FontChoice:
    family: str
    regular: str
    bold: Optional[str]

    def path(self, bold: bool) -> str:
        return (self.bold if bold and self.bold else self.regular)


def available_families() -> list[str]:
    return [f.name for f in FAMILIES if _resolve(f.regular)]


def choose_font(script: str, text: str, prefer_comic: bool, requested: str = "auto") -> Optional[FontChoice]:
    names = list(_PREFS.get(script, _PREFS["latin"]))
    if not prefer_comic:
        names = [n for n in names if not next(f for f in FAMILIES if f.name == n).comic] + \
                [n for n in names if next(f for f in FAMILIES if f.name == n).comic]
    if requested and requested != "auto":
        names = [requested] + [n for n in names if n != requested]
    fams = {f.name: f for f in FAMILIES}
    fallback = None
    for n in names + [f.name for f in FAMILIES]:
        f = fams.get(n)
        if not f:
            continue
        reg = _resolve(f.regular)
        if not reg:
            continue
        choice = FontChoice(f.name, reg, _resolve(f.bold))
        if covers(reg, text):
            return choice
        fallback = fallback or choice
    return fallback


@functools.lru_cache(maxsize=512)
def load(path: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size=size, index=0, layout_engine=ImageFont.Layout.BASIC
                              if not _has_raqm() else ImageFont.Layout.RAQM)
    if bold and "[" in os.path.basename(path):     # variable font: pick the bold instance
        try:
            font.set_variation_by_name("Bold")
        except Exception:
            pass
    return font


@functools.lru_cache(maxsize=1)
def _has_raqm() -> bool:
    try:
        from PIL import features
        return bool(features.check("raqm"))
    except Exception:
        return False
