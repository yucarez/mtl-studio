# MTL Studio

Bulk image translation for manga, manhwa, webtoons, comics, screenshots and scans. Drop in a
chapter, and each page goes through a full detection → OCR → translation → clean-up →
lettering → validation pipeline. The output keeps the original resolution and layout, and
anything the app is unsure about is left in the original language and flagged, never
rendered wrong.

```
Image → preprocessing → text detection → orientation detection → OCR → reading-order analysis
      → translation → text removal / inpainting → typography & layout → quality validation → final image
```

---

## 1. Quick start

Requires **Python 3.10–3.12** (3.11 recommended).

```bash
cd mtl-studio
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt       # web app + image pipeline
pip install -r requirements-ocr.txt   # OCR + inpainting models (large download, see §3)

python scripts/download_fonts.py      # free fonts for every target language
cp .env.example .env                  # then put your API key(s) in .env  (see §2)

python scripts/selftest.py            # optional: checks the pipeline, no models/keys needed
python run.py                         # open http://127.0.0.1:8000
```

Models are downloaded automatically the first time they are used (PaddleOCR, manga-ocr and
LaMa each fetch their weights once), so the very first page takes longer.

---

## 2. Where API keys and models are configured

**Everything is configured in one file: `.env` in the project root** (copy it from
`.env.example`). The values are read by `backend/app/config.py`, which is the only place in
the code that touches keys. You can also set them as ordinary environment variables.

| What | `.env` variable | Notes |
|---|---|---|
| **Claude (recommended)** | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Default model `claude-sonnet-5`. Sees the page image, corrects OCR slips, keeps names/pronouns consistent across the batch. Key: https://console.anthropic.com |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_MODEL` | Default `gpt-5`; set to any model your account has. |
| Any OpenAI-compatible server | `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_SUPPORTS_VISION` | Ollama, vLLM, LM Studio, OpenRouter. No key needed for local servers. |
| DeepL | `DEEPL_API_KEY` | Keys ending in `:fx` automatically use the free endpoint. Text only (no page context). |
| Google Translate | `GOOGLE_TRANSLATE_API_KEY` | Cloud Translation v2 key. Text only. |
| GPU / CPU | `DEVICE` | `auto`, `cpu` or `cuda`. |
| PaddleOCR models | `PADDLE_DET_MODEL`, `PADDLE_REC_MODEL`, `PADDLE_REC_MODEL_KO` … | Defaults are the accurate PP-OCRv5 *server* models. |
| LaMa inpainting | `ENABLE_LAMA` | `true` uses LaMa when installed. |
| Storage / fonts | `DATA_DIR`, `FONT_DIR`, `MAX_UPLOAD_MB` | Defaults `./data`, `./fonts`, 60 MB. |

Restart the server after editing `.env`. The Settings panel in the app shows which
translators are configured, and a notice appears at the top if a key, OCR engine or font is
missing. With no key at all, the "No translation (layout test)" mode still runs the whole
pipeline so you can check detection and clean-up.

Local example, fully offline translation with a vision model through Ollama:

```ini
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5vl:32b
OPENAI_SUPPORTS_VISION=true
```

---

## 3. OCR and inpainting engines

Install at least one OCR engine. The app picks the best available per page (Settings can force one).

| Engine | Install | Used for |
|---|---|---|
| **PaddleOCR 3.x** (recommended) | `pip install paddlepaddle paddleocr` | Text detection for every language; recognition for zh/ja/en plus Korean, Latin, Cyrillic and Thai models. Supplies the confidence scores used for orientation voting and re-read checks. |
| **manga-ocr** | `pip install manga-ocr` | Japanese recognition in manga (vertical columns, stylised bubbles). Chosen automatically for Japanese comics. |
| **LaMa** | `pip install simple-lama-inpainting` | Content-aware fill for screentone, textures and artwork. Without it, OpenCV inpainting is used for those areas. |
| EasyOCR (fallback) | `pip install easyocr` | Alternative detector/recognizer if PaddleOCR is unavailable. |

**GPU:** manga-ocr and LaMa use PyTorch; install the CUDA build of `torch` from
https://pytorch.org first. For PaddleOCR on GPU install `paddlepaddle-gpu` following
https://www.paddlepaddle.org.cn/install instead of `paddlepaddle`. Everything also runs on CPU, just slower.

---

## 4. How accuracy is achieved

**Detection.** Line-level detection (never whole-image). Small images are upscaled first;
very tall webtoon strips and very wide images are tiled with overlap so text is not shrunk away,
and boxes cut by tile seams are replaced by the complete box from the neighbouring tile.

**Orientation, before OCR.** For every line the app measures its tilt and decides whether it is
a vertical CJK column, a horizontal line, or a rotated line. Ambiguous cases (tall CJK boxes
that could be a rotated horizontal line, tilted lines that could be upside down) are resolved
by letting the OCR engine read both candidates and keeping the more confident one. Sideways or
upside-down pages are detected by a line-level vote and straightened. EXIF orientation from
phone photos is applied first, which is the most common cause of sideways results.

**OCR.** Each line is cut out upright along its own angle and recognised individually.
Detections that do not contain letters, or read with very low confidence, are treated as
artwork and left alone.

**Regions and reading order.** Lines are merged into regions only when direction, tilt, glyph
size and stroke weight agree, and never across a speech-bubble boundary, so two neighbouring
bubbles or a bold heading and body text are never mixed. Panels are found from page gutters,
and regions are ordered panel by panel (right-to-left for manga, detected automatically).

**Translation.** AI translators receive the whole page at once: every region in reading order
plus the page image with numbered outlines, the glossary, your instructions and the last pages
of the same batch. They correct OCR mistakes, mark non-text detections and sound effects, and
must return exactly one translation per region; missing or untranslated regions are retried.

**Text removal.** Only glyph pixels are removed, not boxes. Uniform bubbles and flat labels get
an exact background fill; gradients use OpenCV; texture and art use LaMa. Bubble outlines and
panel borders are protected, and the cleaned image is re-scanned for leftover glyphs.

**Lettering.** Text is fitted to the real shape of the bubble (each line gets the width
available at its height, so text follows oval balloons), with balanced line breaks, kinsoku rules
for CJK, and hyphenation only as a last resort between letters. Font size is capped relative to
the original glyph size so hierarchy is preserved, bubble sizes are kept consistent across the
page, and left-aligned text keeps its original margin. Fonts are checked for glyph coverage.

---

## 5. Orientation and layout safeguards

Translated text is **structurally** unable to come out sideways, upside down or mirrored:
it is always laid out as upright horizontal lines, the lettering may follow a tilt of at most
±40° (default ±30°, adjustable), steeper originals (vertical columns, 90° signs, upside-down
text) are lettered level, and the only transform used is a pure rotation. This is still
re-checked before compositing.

The validation pass covers every check in the brief, and corrects automatically where it can:

| Check | What happens |
|---|---|
| Sideways / upside-down / mirrored | Angle bounds and rotation matrix asserted; optional re-read of the rendered text (upright vs rotated) forces a level re-render. |
| Incorrect reading order | Order rebuilt from panels + XY-cut; translator must return every id once. |
| Overlapping translated text | Ink of every pair of regions compared; both shrunk and re-lettered. |
| Text outside its region / clipped | Rendered ink compared with the allowed area; font reduced until it fits. |
| Excessive shrinking | Minimum size (px and % of original); tries a tighter bubble margin or nearby flat background first, then keeps the original text and flags it. |
| Untranslated / missing text | Source-script characters or empty output trigger a retry; if still wrong, the original is kept and flagged. |
| Duplicated text | Identical translations of different sources are flagged. |
| Accidentally translated artwork | Non-letter detections, low-confidence reads and translator `is_text: false` are left untouched. |
| Leftover original glyphs | Cleaned areas re-scanned and re-filled. |
| Legibility | Contrast checked; re-read similarity below threshold triggers a high-contrast re-render. |

When a region cannot be lettered safely, its original pixels are restored exactly and the page
shows a red "kept original" badge. Open the page, shorten the translation, and choose
**Re-letter edited regions**; only the lettering is redone, the cleaned background is reused.

---

## 6. Using the app

* Drop images (or paste) onto the panel. Pages are processed in natural filename order
  (`p2` before `p10`), which keeps translation context flowing through a chapter.
* Each card shows the current stage, progress, and badges for errors, warnings and regions kept
  in the original language. Failed pages show the reason and a Retry button.
* **Review** opens the page: drag the slider to compare, or switch to Original / Translated /
  Regions (numbered reading order, blue = lettered, red = kept original, grey = skipped).
  Every region's translation is editable.
* **Download ZIP** downloads every finished page; each card also downloads individually.
* Settings are remembered in the browser and apply to new pages and retries.

---

## 7. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/config` | Languages, configured translators, installed engines, fonts, defaults, warnings |
| POST | `/api/batches` | multipart `files[]` + `options` (JSON) → batch |
| GET | `/api/batches/{id}` | Status of every image in a batch |
| GET | `/api/batches/{id}/zip` | ZIP of the batch results |
| POST | `/api/batches/{id}/retry-failed` | Retry failed images (optional options JSON) |
| DELETE | `/api/batches/{id}` | Delete a batch and its files |
| GET | `/api/items/{id}` | Item status + full region data (geometry, text, issues) |
| GET | `/api/items/{id}/original`, `/work`, `/result`, `/overlay`, `/thumb` | Images |
| POST | `/api/items/{id}/retry` | Re-run one image (optional options JSON) |
| POST | `/api/items/{id}/rerender` | `{"edits": {"3": "new text"}}` → re-letter edited regions |
| POST | `/api/zip` | `{"item_ids": [...]}` → ZIP |

Options accepted by `/api/batches` (all optional): `source_lang` (`auto` or a code),
`target_lang`, `translator`, `detector`, `ocr_engine`, `content_type`, `reading_direction`,
`sfx`, `text_case`, `font`, `follow_tilt_max`, `min_font_px`, `min_font_ratio`, `on_uncertain`,
`strict_verify`, `residual_check`, `auto_rotate_pages`, `use_page_context`, `output_format`,
`glossary`, `instructions`. See `PipelineOptions` in `backend/app/config.py`.

---

## 8. Project layout

```
run.py                         start the server
.env.example                   all configuration (copy to .env)
requirements.txt               core dependencies
requirements-ocr.txt           OCR / inpainting models
scripts/download_fonts.py      free font set
scripts/selftest.py            offline pipeline self-test
frontend/                      index.html, styles.css, app.js (no build step)
backend/app/
  config.py                    settings, API keys, languages, per-job options
  main.py                      FastAPI routes + static frontend
  jobs.py                      batch queue, worker, persistence, ZIP
  engines/ocr_engines.py       PaddleOCR / EasyOCR / manga-ocr adapters
  engines/translators.py       Claude / OpenAI-compatible / DeepL / Google
  engines/inpaint_engines.py   LaMa + OpenCV
  pipeline/preprocess.py       EXIF, colour modes, alpha, ICC
  pipeline/detection.py        tiled line detection
  pipeline/orientation.py      per-line and per-page orientation
  pipeline/ocr.py              recognition, language detection
  pipeline/regions.py          grouping, panel detection, reading order
  pipeline/analysis.py         bubble detection, colours, weight
  pipeline/inpaint.py          glyph masks and fill selection
  pipeline/typeset.py          shape-aware fitting and rendering
  pipeline/compose.py          correction loop, verification, compositing
  pipeline/validate.py         validation checks
  pipeline/pipeline.py         orchestration, persistence, re-render
```

---

## 9. Troubleshooting

* **"No OCR engine installed"** — install `requirements-ocr.txt` (or at least `paddleocr` + `paddlepaddle`).
* **A PaddleOCR language model fails to load** — the app falls back to `PP-OCRv5_server_rec`
  and logs a warning. Set the model names in `.env` to ones your PaddleOCR version provides.
* **Boxes instead of letters** — run `scripts/download_fonts.py`; the region will also show a missing-glyph warning.
* **Arabic/Hebrew targets** need Pillow built with libraqm for correct shaping; they are not in the default language list.
* **Many regions kept original** — the translation is too long for the space. Lower
  "Smallest font", switch "When a translation will not fit" to "Letter it smaller", ask the AI
  translator for concise wording in the instructions, or edit and re-letter.
* **Slow on CPU** — the accurate server models are the default; set
  `PADDLE_DET_MODEL=PP-OCRv5_mobile_det` and `PADDLE_REC_MODEL=PP-OCRv5_mobile_rec` for speed.

Run one server process (`workers=1`, as `run.py` does): models are loaded once and shared by the
job worker.
