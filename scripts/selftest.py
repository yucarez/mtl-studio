#!/usr/bin/env python3
"""
Offline self-test: builds a synthetic manga page (speech bubbles with vertical Japanese,
a caption box, a 15-degree tilted sign on artwork, a sideways label on screentone) and runs
the REAL pipeline stages (grouping, reading order, text removal, typesetting, validation)
with stand-in detector/OCR/translator, so it needs no models and no API keys.

    python scripts/selftest.py        -> writes selftest_input.png / selftest_output.png
"""
import io
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config  # noqa: E402
from app.engines import ocr_engines as oe, translators as tr  # noqa: E402
from app.pipeline import fonts, pipeline as P  # noqa: E402
from app.pipeline.typeset import render_angle  # noqa: E402
from app.pipeline.validate import orientation_ok  # noqa: E402


def build_page():
    cjk = fonts.choose_font("cjk-ja", "おまえ", False)
    if cjk is None or not fonts.covers(cjk.regular, "おまえはもうしんでいる立入禁止"):
        sys.exit("A Japanese font is needed for the self-test page. Run: python scripts/download_fonts.py")
    W, H = 900, 1300
    arr = np.full((H, W, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    tone = (((xx % 8) - 4) ** 2 + ((yy % 8) - 4) ** 2) < 5
    arr[tone & (yy > 380) & (yy < 630) & (xx > 30) & (xx < 870)] = (90, 90, 90)
    z = (yy > 670) & (yy < 1270) & (xx > 470) & (xx < 870)
    g = (120 + 100 * np.sin(xx / 40.0) * np.cos(yy / 55.0)).astype(np.uint8)
    arr[z, 0], arr[z, 1], arr[z, 2] = g[z], (g[z] * 0.8).astype(np.uint8), 180
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for box in [(20, 20, 880, 640), (20, 660, 440, 1280), (460, 660, 880, 1280)]:
        d.rectangle(box, outline="black", width=4)
    truth = []
    f = ImageFont.truetype(cjk.regular, 30)

    def bubble(cx, cy, rx, ry, cols):
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill="white", outline="black", width=3)
        xr = cx + (len(cols) - 1) * 38 / 2
        for i, col in enumerate(cols):
            x, top = xr - i * 38, cy - len(col) * 33 / 2
            for j, ch in enumerate(col):
                d.text((x, top + j * 33), ch, font=f, fill="black", anchor="mt")
            truth.append((np.array([[x - 18, top - 3], [x + 18, top - 3], [x + 18, top + len(col) * 33 + 2],
                                    [x - 18, top + len(col) * 33 + 2]], np.float32), col))

    bubble(700, 200, 120, 150, ["おまえは", "もう", "しんでいる"])
    bubble(330, 210, 95, 130, ["なに", "っ！？"])
    bubble(230, 900, 110, 160, ["ここは", "どこだ", "ろう…"])
    d.rectangle((520, 700, 850, 790), fill=(255, 255, 235), outline="black", width=2)
    f2 = ImageFont.truetype(cjk.regular, 26)
    for i, t in enumerate(["その日、", "世界は変わった。"]):
        y = 715 + i * 34
        d.text((540, y), t, font=f2, fill="black")
        w = f2.getlength(t)
        truth.append((np.array([[536, y - 2], [540 + w + 4, y - 2], [540 + w + 4, y + 34], [536, y + 34]],
                               np.float32), t))
    sign = Image.new("RGBA", (260, 60), (0, 0, 0, 0))
    ImageDraw.Draw(sign).text((10, 8), "立入禁止", font=ImageFont.truetype(cjk.regular, 38), fill=(250, 250, 40, 255))
    rot = sign.rotate(-15, expand=True, resample=Image.BICUBIC)
    im.paste(rot, (560, 1000), rot)
    t = math.radians(15)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    pts = np.array([[6, 4], [170, 4], [170, 58], [6, 58]], float) - [130, 30]
    truth.append(((pts @ R.T + [560 + rot.width / 2, 1000 + rot.height / 2]).astype(np.float32), "立入禁止"))
    lab = Image.new("RGBA", (220, 44), (255, 255, 255, 255))
    ImageDraw.Draw(lab).text((8, 4), "ドカーン", font=ImageFont.truetype(cjk.regular, 32), fill=(0, 0, 0, 255))
    im.paste(lab.rotate(90, expand=True), (80, 400))
    truth.append((np.array([[80, 400], [124, 400], [124, 620], [80, 620]], np.float32), "ドカーン"))
    return np.array(im), truth


def main():
    rgb, truth = build_page()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    Image.fromarray(rgb).save(ROOT / "selftest_input.png")

    class Det(oe.Detector):
        name = "stand-in"

        def detect(self, img):
            s = img.shape[1] / rgb.shape[1]
            out = []
            for q, _ in truth:
                x0, y0 = np.maximum(q.min(0).astype(int), 0)
                x1, y1 = q.max(0).astype(int)
                c = img[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)].astype(int)
                if c.size and (np.abs(c - np.median(c.reshape(-1, 3), 0)).max(2) > 80).mean() > 0.02:
                    out.append((q * s, 0.95))
            return out

    class Rec(oe.Recognizer):
        name = "stand-in"
        gives_confidence = False

        def recognize(self, crops, vertical):
            return [("", None) for _ in crops]

    def fake_ocr(img, lines, rec):
        for l in lines:
            best = min(truth, key=lambda t: np.linalg.norm(t[0].mean(0) - np.array(l.center)))
            l.text, l.conf = best[1], 0.95

    EN = {"おまえはもうしんでいる": "You are already dead.", "なにっ！？": "What?!",
          "ここはどこだろう…": "Where... is this place?", "その日、世界は変わった。": "That day, the world changed forever.",
          "立入禁止": "KEEP OUT", "ドカーン": "KABOOM"}

    class Tr(tr.Translator):
        name = "stand-in"
        smart = True

        def translate_page(self, ctx):
            return {r["id"]: tr.RegionResult(EN.get(r["text"], r["text"]), r["text"], True,
                                             "sfx" if r["text"] == "ドカーン" else "speech")
                    for r in ctx.regions}, "ja"

    P.ocr.recognize_lines = fake_ocr
    oe.make_detector = lambda *a, **k: Det()
    oe.make_recognizer = lambda *a, **k: Rec()
    oe.make_confidence_recognizer = lambda *a, **k: None
    P.make_translator = lambda n: Tr()

    opts = config.PipelineOptions.from_dict({"source_lang": "ja", "target_lang": "en", "sfx": "translate"})
    res = P.process(buf.getvalue(), opts, None, [])
    Image.open(io.BytesIO(res.image_bytes)).save(ROOT / "selftest_output.png")

    ok = True
    print(f"\nreading direction: {res.meta['reading_direction']}  content: {res.meta['content_type']}")
    order = [r.source_text for r in res.regions]
    expect_first = ["おまえはもうしんでいる", "なにっ！？"]
    if order[:2] != expect_first:
        ok = False
        print("FAIL reading order:", order)
    for r in res.regions:
        a = r.render.get("theta", 0.0)
        good, code = orientation_ok(a, opts.follow_tilt_max)
        flag = "ok " if good else "BAD"
        ok &= good
        print(f"  [{flag}] {r.order + 1}. {r.source_text:<14} -> {r.translation!r:<40} {r.status:<14} "
              f"angle {a:5.1f}  {' / '.join(r.render.get('lines', []))}")
    bad = sum(1 for t in np.arange(-360, 360, 0.5)
              if not orientation_ok(render_angle(type(res.regions[0])(0, [], np.zeros((4, 2)), theta=float(t)),
                                                 opts.follow_tilt_max), opts.follow_tilt_max)[0])
    ok &= bad == 0
    print(f"\norientation guard over 1440 source angles: {bad} violations")
    print("wrote selftest_input.png and selftest_output.png")
    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
