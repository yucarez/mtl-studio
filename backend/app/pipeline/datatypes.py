from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TextLine:
    """One detected line (horizontal) or column (vertical CJK)."""
    quad: np.ndarray                 # (4,2) float32 image coords
    det_score: float = 1.0
    # geometry (filled by orientation.analyze_line)
    center: tuple[float, float] = (0.0, 0.0)
    length: float = 0.0              # extent along reading axis
    thickness: float = 0.0           # extent across reading axis (~ glyph size)
    theta: float = 0.0               # rotation (deg) of the upright-glyph frame vs image x-axis
    vertical: Optional[bool] = None  # True = top-to-bottom CJK column
    # recognition
    text: str = ""
    conf: float = 0.0
    ink: float = 0.0                 # measured ink extent across the reading axis (px)
    weight: float = 0.0              # stroke width / line thickness (bold vs regular)


@dataclass
class Issue:
    code: str            # sideways | upside_down | mirrored | reading_order | overlap | outside_region |
                         # clipped | too_small | untranslated | artwork | duplicate | missing |
                         # residual_text | glyph_missing | low_contrast | verify_failed | ...
    severity: str        # info | warning | error
    message: str
    region_id: Optional[int] = None
    fixed: bool = False  # True if the validator auto-corrected it

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "message": self.message,
                "region_id": self.region_id, "fixed": self.fixed}


@dataclass
class Region:
    """A logical text block (speech bubble, caption, sign, paragraph...)."""
    id: int
    lines: list[TextLine]
    quad: np.ndarray                  # (4,2) oriented block rectangle
    center: tuple[float, float] = (0.0, 0.0)
    width: float = 0.0                # in block frame
    height: float = 0.0
    theta: float = 0.0                # glyph-frame rotation of the original text
    vertical: bool = False
    glyph_px: float = 0.0             # estimated original glyph size
    order: int = 0                    # reading order index
    kind: str = "text"                # layout: bubble | plate | caption | overlay
    role: str = ""                    # semantic (from translator): speech | thought | narration | sfx | sign | ui ...
    bubble_bbox: Optional[tuple[int, int, int, int]] = None   # x0,y0,x1,y1 (image coords)
    bg_color: tuple[int, int, int] = (255, 255, 255)
    fg_color: tuple[int, int, int] = (0, 0, 0)
    bg_uniform: bool = True
    bold: bool = False
    align: str = "center"             # center | left
    source_text: str = ""
    ocr_conf: float = 0.0
    translation: str = ""
    is_text: bool = True
    skip_reason: str = ""             # why a region is intentionally left untouched
    status: str = "pending"           # pending | rendered | kept_original | skipped
    render: dict = field(default_factory=dict)   # final typesetting params
    issues: list[Issue] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x0, y0 = self.quad.min(axis=0)
        x1, y1 = self.quad.max(axis=0)
        return int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "order": self.order,
            "kind": self.kind,
            "role": self.role,
            "quad": np.round(self.quad, 1).tolist(),
            "bbox": list(self.bbox),
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "theta": round(self.theta, 2),
            "vertical": self.vertical,
            "glyph_px": round(self.glyph_px, 1),
            "bubble_bbox": list(self.bubble_bbox) if self.bubble_bbox else None,
            "bg_color": list(self.bg_color),
            "fg_color": list(self.fg_color),
            "bg_uniform": self.bg_uniform,
            "bold": self.bold,
            "align": self.align,
            "source_text": self.source_text,
            "ocr_conf": round(self.ocr_conf, 3),
            "translation": self.translation,
            "is_text": self.is_text,
            "skip_reason": self.skip_reason,
            "status": self.status,
            "render": self.render,
            "issues": [i.to_dict() for i in self.issues],
            "lines": [
                {"quad": np.round(l.quad, 1).tolist(), "text": l.text, "conf": round(l.conf, 3),
                 "vertical": l.vertical, "theta": round(l.theta, 2)}
                for l in self.lines
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        lines = [TextLine(quad=np.array(l["quad"], np.float32), text=l.get("text", ""),
                          conf=l.get("conf", 0.0), vertical=l.get("vertical"),
                          theta=l.get("theta", 0.0)) for l in d.get("lines", [])]
        r = cls(id=d["id"], lines=lines, quad=np.array(d["quad"], np.float32))
        for k in ("order", "kind", "role", "width", "height", "theta", "vertical", "glyph_px", "bg_uniform",
                  "bold", "align", "source_text", "ocr_conf", "translation", "is_text",
                  "skip_reason", "status", "render"):
            if k in d:
                setattr(r, k, d[k])
        r.center = tuple(d.get("center", (0, 0)))
        r.bubble_bbox = tuple(d["bubble_bbox"]) if d.get("bubble_bbox") else None
        r.bg_color = tuple(d.get("bg_color", (255, 255, 255)))
        r.fg_color = tuple(d.get("fg_color", (0, 0, 0)))
        return r


class PipelineError(Exception):
    """Raised for user-facing failures (missing engine, bad image...)."""
