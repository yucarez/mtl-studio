#!/usr/bin/env python3
"""Start MTL Studio:  python run.py  [--host 0.0.0.0] [--port 8000]"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    import uvicorn
    print(f"\n  MTL Studio  ->  http://{a.host}:{a.port}\n")
    uvicorn.run("app.main:app", host=a.host, port=a.port, workers=1)  # one worker: models are shared
