"""Classical, physics-flavoured relighting: the "standard augmentation" ablation arm.

This is what a team would do *without* a generative model: darken with a gamma curve, shift the
white balance, add sensor noise and a little blur, and sprinkle a few light blooms. It is the
honest baseline the synthetic arm has to beat ("is diffusion worth the GPU hours?"), and it is the
cheap backend CI uses to exercise the loop end to end without downloading Stable Diffusion.
"""

from __future__ import annotations

import cv2
import numpy as np


def relight(rgb: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    img = rgb.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if condition in ("night", "glare", "dusk"):
        gamma = rng.uniform(1.8, 2.8) if condition != "dusk" else rng.uniform(1.2, 1.6)
        gain = rng.uniform(0.25, 0.45) if condition != "dusk" else rng.uniform(0.55, 0.8)
        img = gain * np.power(img, gamma)
        tint = np.array([0.85, 0.95, 1.15]) if condition != "dusk" else np.array([1.15, 0.95, 0.8])  # RGB
        img *= tint * rng.uniform(0.9, 1.1, 3)
        # Light blooms (street lamps / headlights) in the upper half / along the horizon.
        n = int(rng.integers(3, 9)) if condition != "dusk" else 0
        if condition == "glare":
            n += 4
        bloom = np.zeros((h, w), np.float32)
        for _ in range(n):
            cx, cy = rng.uniform(0, w), rng.uniform(0.3 * h, 0.65 * h)
            cv2.circle(bloom, (int(cx), int(cy)), int(rng.uniform(3, 10)), 1.0, -1)
        if n:
            bloom = cv2.GaussianBlur(bloom, (0, 0), sigmaX=rng.uniform(8, 20)) * rng.uniform(2, 5)
            img += bloom[..., None] * np.array([1.0, 0.9, 0.7])
    elif condition == "rain":
        img = 0.75 * img + 0.1
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
    elif condition == "fog":
        a = rng.uniform(0.35, 0.6)
        img = (1 - a) * img + a * 0.8
    elif condition == "snow":
        img = 0.8 * img + 0.15
    # Sensor noise (stronger in the dark) and mild defocus.
    img += rng.normal(0, rng.uniform(0.01, 0.04), img.shape).astype(np.float32)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.2))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
