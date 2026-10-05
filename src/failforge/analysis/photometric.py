"""Cheap, interpretable descriptors of a crop and its image.

They sit next to the CLIP embedding in the clustering features (so "dark" or "blurry" are directly
expressible) and they make cluster cards readable ("brightness -1.9 sigma vs. the population").
"""

from __future__ import annotations

import cv2
import numpy as np

ATTRS = ("brightness", "contrast", "sharpness", "saturation", "img_brightness", "log_size", "aspect", "cy")


def context_box(box: list[float], w: int, h: int, context: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    s = max(bw, bh) * (1 + context)  # square crop around the box: CLIP sees the object and its surroundings
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(s, 24.0)
    a, b = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    c, d = int(min(cx + s / 2, w)), int(min(cy + s / 2, h))
    return a, b, max(c, a + 1), max(d, b + 1)


def describe_crop(img_bgr: np.ndarray, box: list[float], img_gray_mean: float) -> dict[str, float]:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(max(x2, x1 + 1), w), min(max(y2, y1 + 1), h)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_bgr[:1, :1]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lap = cv2.Laplacian(g, cv2.CV_32F).var() if min(g.shape) >= 3 else 0.0
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    return {
        "brightness": float(g.mean() / 255.0),
        "contrast": float(g.std() / 255.0),
        "sharpness": float(np.log1p(lap)),
        "saturation": float(hsv[..., 1].mean() / 255.0),
        "img_brightness": float(img_gray_mean / 255.0),
        "log_size": float(np.log(np.sqrt(bw * bh) / np.hypot(w, h))),
        "aspect": float(np.log(bw / bh)),
        "cy": float((y1 + y2) / 2 / h),
    }


def standardize(m: np.ndarray, ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = m if ref is None else ref
    mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
    return (m - mu) / sd, mu, sd
