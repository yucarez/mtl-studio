"""
Central configuration.

=====================================================================
  WHERE TO PUT API KEYS / MODEL SETTINGS
  Copy `.env.example` to `.env` in the project root and fill it in.
  Every value below can also be set as a normal environment variable.
=====================================================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    return default if not v else v.lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------- API keys
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-sonnet-5")
ANTHROPIC_BASE_URL = _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-5")
# Point this at any OpenAI-compatible server (Ollama, vLLM, LM Studio, OpenRouter...)
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_SUPPORTS_VISION = _env_bool("OPENAI_SUPPORTS_VISION", True)

DEEPL_API_KEY = _env("DEEPL_API_KEY")          # keys ending in ":fx" use the free endpoint
GOOGLE_TRANSLATE_API_KEY = _env("GOOGLE_TRANSLATE_API_KEY")

# ---------------------------------------------------------------- runtime
DEVICE = _env("DEVICE", "auto")                 # auto | cpu | cuda
DATA_DIR = Path(_env("DATA_DIR", str(ROOT_DIR / "data")))
FONT_DIR = Path(_env("FONT_DIR", str(ROOT_DIR / "fonts")))
FRONTEND_DIR = ROOT_DIR / "frontend"
MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "60"))
ENABLE_LAMA = _env_bool("ENABLE_LAMA", True)
LLM_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "180"))

# PaddleOCR 3.x model names (override if you want mobile/faster models)
PADDLE_DET_MODEL = _env("PADDLE_DET_MODEL", "PP-OCRv5_server_det")
PADDLE_REC_MODEL = _env("PADDLE_REC_MODEL", "PP-OCRv5_server_rec")        # zh / ja / en
PADDLE_REC_MODEL_KO = _env("PADDLE_REC_MODEL_KO", "korean_PP-OCRv5_mobile_rec")
PADDLE_REC_MODEL_LATIN = _env("PADDLE_REC_MODEL_LATIN", "latin_PP-OCRv5_mobile_rec")
PADDLE_REC_MODEL_CYRILLIC = _env("PADDLE_REC_MODEL_CYRILLIC", "eslav_PP-OCRv5_mobile_rec")
PADDLE_REC_MODEL_TH = _env("PADDLE_REC_MODEL_TH", "th_PP-OCRv5_mobile_rec")

DATA_DIR.mkdir(parents=True, exist_ok=True)
FONT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- languages
@dataclass(frozen=True)
class Lang:
    code: str
    name: str
    script: str        # latin | cjk-ja | cjk-zh | cjk-zht | hangul | cyrillic | thai | arabic
    spaced: bool       # words separated by spaces?
    deepl: str = ""
    google: str = ""


LANGS: dict[str, Lang] = {
    l.code: l
    for l in [
        Lang("ja", "Japanese", "cjk-ja", False, "JA", "ja"),
        Lang("zh", "Chinese (Simplified)", "cjk-zh", False, "ZH-HANS", "zh-CN"),
        Lang("zh-TW", "Chinese (Traditional)", "cjk-zht", False, "ZH-HANT", "zh-TW"),
        Lang("ko", "Korean", "hangul", True, "KO", "ko"),
        Lang("en", "English", "latin", True, "EN-US", "en"),
        Lang("es", "Spanish", "latin", True, "ES", "es"),
        Lang("fr", "French", "latin", True, "FR", "fr"),
        Lang("de", "German", "latin", True, "DE", "de"),
        Lang("it", "Italian", "latin", True, "IT", "it"),
        Lang("pt", "Portuguese", "latin", True, "PT-BR", "pt"),
        Lang("nl", "Dutch", "latin", True, "NL", "nl"),
        Lang("pl", "Polish", "latin", True, "PL", "pl"),
        Lang("tr", "Turkish", "latin", True, "TR", "tr"),
        Lang("id", "Indonesian", "latin", True, "ID", "id"),
        Lang("ms", "Malay", "latin", True, "", "ms"),
        Lang("fil", "Filipino", "latin", True, "", "tl"),
        Lang("vi", "Vietnamese", "latin", True, "VI", "vi"),
        Lang("ru", "Russian", "cyrillic", True, "RU", "ru"),
        Lang("uk", "Ukrainian", "cyrillic", True, "UK", "uk"),
        Lang("th", "Thai", "thai", False, "TH", "th"),
    ]
}

CJK_SCRIPTS = {"cjk-ja", "cjk-zh", "cjk-zht"}


def lang(code: str) -> Lang:
    if code not in LANGS:
        raise ValueError(f"Unsupported language code: {code}")
    return LANGS[code]


# ---------------------------------------------------------------- per-job options
@dataclass
class PipelineOptions:
    source_lang: str = "auto"
    target_lang: str = "en"
    translator: str = "auto"            # auto | anthropic | openai | deepl | google | passthrough
    detector: str = "auto"              # auto | paddle | easyocr
    ocr_engine: str = "auto"            # auto | paddle | easyocr | manga-ocr
    content_type: str = "auto"          # auto | manga | comic | webtoon | screenshot | document
    reading_direction: str = "auto"     # auto | rtl | ltr
    sfx: str = "skip"                   # skip | translate
    text_case: str = "as_is"            # as_is | upper
    font: str = "auto"
    follow_tilt_max: float = 30.0       # rendered text may follow original tilt up to this angle
    min_font_px: int = 10
    min_font_ratio: float = 0.45        # never shrink below 45% of the original glyph size
    on_uncertain: str = "keep_original" # keep_original | best_effort
    strict_verify: bool = True          # re-OCR rendered text to confirm it is legible & upright
    residual_check: bool = True         # re-detect after inpainting to catch leftover glyphs
    auto_rotate_pages: bool = True
    use_page_context: bool = True       # send the page image to vision LLMs
    output_format: str = "png"          # png | same
    glossary: str = ""
    instructions: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> "PipelineOptions":
        d = d or {}
        kw = {}
        for f in fields(cls):
            if f.name in d and d[f.name] is not None:
                default = getattr(cls, f.name)
                v = d[f.name]
                try:
                    kw[f.name] = type(default)(v) if not isinstance(default, bool) else (
                        v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"})
                except (TypeError, ValueError):
                    pass
        o = cls(**kw)
        if o.source_lang != "auto":
            lang(o.source_lang)
        lang(o.target_lang)
        o.follow_tilt_max = max(0.0, min(40.0, o.follow_tilt_max))  # hard ceiling: never sideways
        o.min_font_px = max(6, min(40, o.min_font_px))
        return o

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def configured_translators() -> list[str]:
    out = []
    if ANTHROPIC_API_KEY:
        out.append("anthropic")
    if OPENAI_API_KEY or "api.openai.com" not in OPENAI_BASE_URL:
        out.append("openai")
    if DEEPL_API_KEY:
        out.append("deepl")
    if GOOGLE_TRANSLATE_API_KEY:
        out.append("google")
    out.append("passthrough")
    return out
