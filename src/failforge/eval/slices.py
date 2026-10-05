"""Named image slices of an evaluation set (image-attribute predicates)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SLICES: dict[str, Callable[[dict], bool]] = {
    "overall": lambda a: True,
    "day": lambda a: a.get("timeofday") == "day",
    "night": lambda a: a.get("timeofday") == "night",
    "dawn": lambda a: a.get("timeofday") == "dawn",
    "clear": lambda a: a.get("weather") == "clear",
    "overcast": lambda a: a.get("weather") in ("overcast", "partly_cloudy"),
    "rainy": lambda a: a.get("weather") == "rainy",
    "snowy": lambda a: a.get("weather") == "snowy",
    "night_rainy": lambda a: a.get("timeofday") == "night" and a.get("weather") == "rainy",
    "highway": lambda a: a.get("scene") == "highway",
    "city_street": lambda a: a.get("scene") == "city_street",
}

MIN_SLICE_IMAGES = 25  # smaller slices are reported but never gated on (too noisy)


def slice_mask(attrs: list[dict], name: str) -> np.ndarray:
    if name not in SLICES:
        raise KeyError(f"unknown slice {name!r}; known: {sorted(SLICES)}")
    f = SLICES[name]
    return np.array([bool(f(a)) for a in attrs], dtype=bool)
