#!/usr/bin/env python3
"""Download a free (SIL OFL) font set into ./fonts so every target language renders correctly."""
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "fonts"
BASE = "https://github.com/google/fonts/raw/main/ofl/"
FONTS = {
    "ComicNeue-Bold.ttf": BASE + "comicneue/ComicNeue-Bold.ttf",          # comic lettering (latin)
    "ComicNeue-Regular.ttf": BASE + "comicneue/ComicNeue-Regular.ttf",
    "NotoSans[wdth,wght].ttf": BASE + "notosans/NotoSans%5Bwdth,wght%5D.ttf",   # latin/cyrillic/greek/vi
    "NotoSansJP[wght].ttf": BASE + "notosansjp/NotoSansJP%5Bwght%5D.ttf",
    "NotoSansKR[wght].ttf": BASE + "notosanskr/NotoSansKR%5Bwght%5D.ttf",
    "NotoSansSC[wght].ttf": BASE + "notosanssc/NotoSansSC%5Bwght%5D.ttf",
    "NotoSansTC[wght].ttf": BASE + "notosanstc/NotoSansTC%5Bwght%5D.ttf",
    "NotoSansThai[wdth,wght].ttf": BASE + "notosansthai/NotoSansThai%5Bwdth,wght%5D.ttf",
}

def main():
    DEST.mkdir(exist_ok=True)
    failed = []
    for name, url in FONTS.items():
        out = DEST / name
        if out.exists() and out.stat().st_size > 10_000:
            print(f"  ok      {name}")
            continue
        try:
            print(f"  fetch   {name}")
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            failed.append(name)
            print(f"  FAILED  {name}: {e}")
    if failed:
        print("\nSome fonts failed. Download them manually from https://fonts.google.com and put the"
              f" .ttf/.otf files in {DEST}. Any Noto Sans / Noto Sans CJK file works.")
        sys.exit(1)
    print(f"\nFonts ready in {DEST}")

if __name__ == "__main__":
    main()
