"""
Translation providers.

LLM providers (Anthropic / OpenAI-compatible) translate a whole page at once, see the
page image with numbered region outlines, fix OCR slips, classify regions (speech, sfx,
non-text...) and keep names/pronouns consistent across the batch. This is by far the
most accurate option. DeepL / Google are supported as plain text translators.

API keys are read from config.py (.env).
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .. import config

log = logging.getLogger("mtl.translate")


@dataclass
class PageContext:
    regions: list[dict]                 # {id, text, kind_hint, vertical}
    source_lang: str                    # code or "auto"
    target_lang: str
    images: list[bytes] = field(default_factory=list)   # JPEG bytes with numbered outlines
    glossary: str = ""
    instructions: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)
    sfx: str = "skip"
    content_type: str = "auto"


@dataclass
class RegionResult:
    translation: str
    source: Optional[str] = None
    is_text: bool = True
    kind: Optional[str] = None


class TranslationError(Exception):
    pass


class Translator:
    name = "base"
    vision = False
    smart = False     # can correct OCR / classify regions

    def translate_page(self, ctx: PageContext) -> tuple[dict[int, RegionResult], Optional[str]]:
        """Returns ({region_id: result}, detected_source_lang_or_None)."""
        raise NotImplementedError


# ----------------------------------------------------------------- helpers
def _post(url: str, headers: dict, payload: dict, params: dict | None = None) -> dict:
    last = None
    for attempt in range(4):
        try:
            r = httpx.post(url, headers=headers, json=payload, params=params, timeout=config.LLM_TIMEOUT_S)
            if r.status_code in (429, 500, 502, 503, 504, 529):
                last = TranslationError(f"HTTP {r.status_code}: {r.text[:300]}")
                time.sleep(2 ** attempt * 2)
                continue
            if r.status_code >= 400:
                raise TranslationError(f"HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
        except httpx.HTTPError as e:
            last = TranslationError(f"Network error: {e}")
            time.sleep(2 ** attempt)
    raise last or TranslationError("request failed")


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        return json.loads(text[a:b + 1])
    raise TranslationError("Model did not return JSON")


def _lang_name(code: str) -> str:
    return "the source language (detect it)" if code == "auto" else config.lang(code).name


SYSTEM_PROMPT = """You are an expert localizer for manga, manhwa, webtoons, comics, screenshots and scanned documents.
You receive OCR output for numbered text regions of ONE image, listed in reading order, and (when provided) the image itself with every region outlined and labeled with its number.

For every region:
1. Correct obvious OCR mistakes using the image and context (look-alike characters, missing small kana, stray symbols, broken words). Never invent text that is not visible.
2. Set "is_text": false for detections that are not real text (textures, hatching, hair strands, stray marks, decorative squiggles, isolated punctuation with no meaning).
3. Set "kind" to one of: speech, thought, narration, sfx, sign, ui, title, credit, other.
4. Translate into the target language naturally and idiomatically, like a professional localizer: keep meaning, tone, register and each character's voice. Use neighbouring regions and the image to resolve speaker, pronouns, gender, plurality and politeness. Keep names and terms consistent with the glossary and the previous context.
5. Keep regions separate: never merge regions, move text between them, reorder them or drop any.
6. Keep each translation about as compact as the original so it fits the same space. "visual_lines" shows how the original was broken into lines: use "\\n" only where a break is meaningful (title vs. body, list items, separate paragraphs); never add line breaks just for wrapping.
7. Sound effects: {sfx_rule}

Return ONLY a JSON object, no prose, no code fences:
{{"source_language": "<ISO code of the detected source language>",
  "regions": [{{"id": 1, "is_text": true, "kind": "speech", "source": "<corrected original text>", "translation": "<translation>"}}]}}
The "regions" array must contain exactly one object for every input id."""


def _user_prompt(ctx: PageContext, only_ids: Optional[set] = None) -> str:
    regs = []
    for r in ctx.regions:
        if only_ids is not None and r["id"] not in only_ids:
            continue
        item = {"id": r["id"], "text": r["text"], "layout": "vertical" if r.get("vertical") else "horizontal",
                "hint": r.get("kind_hint", "text")}
        if r.get("lines"):
            item["visual_lines"] = r["lines"]   # lets the model spot title/body or list structure
        regs.append(item)
    parts = [f"Source language: {_lang_name(ctx.source_lang)}",
             f"Target language: {_lang_name(ctx.target_lang)}",
             f"Content type: {ctx.content_type}"]
    if ctx.glossary.strip():
        parts.append("Glossary (source = target), always follow it:\n" + ctx.glossary.strip())
    if ctx.instructions.strip():
        parts.append("Extra instructions from the user:\n" + ctx.instructions.strip())
    if ctx.history:
        hist = "\n".join(f"- {s}  =>  {t}" for s, t in ctx.history[-30:])
        parts.append("Previous pages in this batch (for consistency only, do not translate again):\n" + hist)
    parts.append("Regions (reading order):\n" + json.dumps(regs, ensure_ascii=False, indent=1))
    return "\n\n".join(parts)


def _system(ctx: PageContext) -> str:
    rule = ("translate them into short, punchy equivalents." if ctx.sfx == "translate" else
            'still translate them, but always set "kind": "sfx" so they can be left untouched.')
    return SYSTEM_PROMPT.format(sfx_rule=rule)


def _parse_regions(data: dict) -> tuple[dict[int, RegionResult], Optional[str]]:
    out = {}
    for item in data.get("regions", []):
        try:
            rid = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        out[rid] = RegionResult(
            translation=str(item.get("translation", "") or ""),
            source=item.get("source"),
            is_text=bool(item.get("is_text", True)),
            kind=item.get("kind"),
        )
    src = data.get("source_language")
    return out, (str(src) if src else None)


class _LLMTranslator(Translator):
    smart = True

    def _complete(self, system: str, user: str, images: list[bytes]) -> str:
        raise NotImplementedError

    def translate_page(self, ctx):
        imgs = ctx.images if self.vision else []
        text = self._complete(_system(ctx), _user_prompt(ctx), imgs)
        results, detected = _parse_regions(_extract_json(text))
        missing = {r["id"] for r in ctx.regions} - set(results)
        if missing:  # one targeted retry for anything the model skipped
            log.info("retrying %d missing regions", len(missing))
            text = self._complete(_system(ctx), _user_prompt(ctx, missing), imgs)
            more, _ = _parse_regions(_extract_json(text))
            results.update({k: v for k, v in more.items() if k in missing})
        return results, detected


class AnthropicTranslator(_LLMTranslator):
    name = "anthropic"
    vision = True

    def _complete(self, system, user, images):
        if not config.ANTHROPIC_API_KEY:
            raise TranslationError("ANTHROPIC_API_KEY is not set (.env)")
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                "data": base64.b64encode(b).decode()}} for b in images]
        content.append({"type": "text", "text": user})
        data = _post(
            f"{config.ANTHROPIC_BASE_URL.rstrip('/')}/v1/messages",
            {"x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
            {"model": config.ANTHROPIC_MODEL, "max_tokens": 8000, "system": system,
             "messages": [{"role": "user", "content": content}]},
        )
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class OpenAITranslator(_LLMTranslator):
    name = "openai"

    @property
    def vision(self):
        return config.OPENAI_SUPPORTS_VISION

    def _complete(self, system, user, images):
        content = [{"type": "text", "text": user}]
        for b in images:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
        headers = {"Content-Type": "application/json"}
        if config.OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {config.OPENAI_API_KEY}"
        payload = {"model": config.OPENAI_MODEL,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": content if images else user}],
                   "response_format": {"type": "json_object"}}
        url = f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
        try:
            data = _post(url, headers, payload)
        except TranslationError as e:
            if "response_format" not in str(e):
                raise
            payload.pop("response_format")       # some compatible servers don't support it
            data = _post(url, headers, payload)
        return data["choices"][0]["message"]["content"] or ""


class DeepLTranslator(Translator):
    name = "deepl"

    def translate_page(self, ctx):
        if not config.DEEPL_API_KEY:
            raise TranslationError("DEEPL_API_KEY is not set (.env)")
        tgt = config.lang(ctx.target_lang).deepl
        if not tgt:
            raise TranslationError(f"DeepL does not support target {ctx.target_lang}")
        host = "api-free.deepl.com" if config.DEEPL_API_KEY.endswith(":fx") else "api.deepl.com"
        payload = {"text": [r["text"] for r in ctx.regions], "target_lang": tgt, "preserve_formatting": True}
        if ctx.source_lang != "auto" and config.lang(ctx.source_lang).deepl:
            payload["source_lang"] = config.lang(ctx.source_lang).deepl.split("-")[0]
        if ctx.history:
            payload["context"] = " ".join(s for s, _ in ctx.history[-10:])
        data = _post(f"https://{host}/v2/translate",
                     {"Authorization": f"DeepL-Auth-Key {config.DEEPL_API_KEY}"}, payload)
        tr = data.get("translations", [])
        out = {r["id"]: RegionResult(t["text"]) for r, t in zip(ctx.regions, tr)}
        det = tr[0].get("detected_source_language", "").lower() if tr else None
        return out, det


class GoogleTranslator(Translator):
    name = "google"

    def translate_page(self, ctx):
        if not config.GOOGLE_TRANSLATE_API_KEY:
            raise TranslationError("GOOGLE_TRANSLATE_API_KEY is not set (.env)")
        payload = {"q": [r["text"] for r in ctx.regions], "target": config.lang(ctx.target_lang).google,
                   "format": "text"}
        if ctx.source_lang != "auto":
            payload["source"] = config.lang(ctx.source_lang).google
        data = _post("https://translation.googleapis.com/language/translate/v2", {}, payload,
                     params={"key": config.GOOGLE_TRANSLATE_API_KEY})
        tr = data.get("data", {}).get("translations", [])
        out = {r["id"]: RegionResult(t.get("translatedText", "")) for r, t in zip(ctx.regions, tr)}
        det = tr[0].get("detectedSourceLanguage") if tr else None
        return out, det


class PassthroughTranslator(Translator):
    """Copies the OCR text. Useful to test detection/inpainting/layout without an API key."""
    name = "passthrough"

    def translate_page(self, ctx):
        return {r["id"]: RegionResult(r["text"]) for r in ctx.regions}, None


def make_translator(name: str) -> Translator:
    if name == "auto":
        name = config.configured_translators()[0]
    cls = {"anthropic": AnthropicTranslator, "openai": OpenAITranslator, "deepl": DeepLTranslator,
           "google": GoogleTranslator, "passthrough": PassthroughTranslator}.get(name)
    if cls is None:
        raise TranslationError(f"Unknown translator '{name}'")
    return cls()
