#!/usr/bin/env python3
"""Capture dashboard screenshots for the README (docs/images/*.png) with a headless Chromium browser.

Start the dashboard first (`python run.py --no-browser` or `failforge serve --no-browser`), then:

    python scripts/screenshots.py --url http://127.0.0.1:8765 --run reference

Needs Microsoft Edge or Google Chrome installed (uses its --headless --screenshot mode).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]
TABS = ["pipeline", "failures", "synthesis", "results"]


def browser() -> str:
    for c in CANDIDATES:
        p = shutil.which(c) or (c if Path(c).is_file() else None)
        if p:
            return p
    sys.exit("no Edge / Chrome found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--run", default="reference")
    ap.add_argument("--out", default=str(REPO / "docs" / "images"))
    ap.add_argument("--size", default="1440,1000")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exe = browser()
    for tab in TABS:
        target = out / f"dashboard_{tab}.png"
        url = f"{a.url}/?run={a.run}&tab={tab}"
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={a.size}",
                        "--virtual-time-budget=8000", f"--screenshot={target}", url], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(target, "ok" if target.is_file() else "FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
