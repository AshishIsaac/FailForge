"""Prompt recipes per target condition (what a failure mode's synthesis target turns into)."""

from __future__ import annotations

import random

BASE = "a realistic dashcam photo of a {scene}"
CONDITIONS = {
    "night": [
        "at night, dark sky, street lights, glowing car headlights and red tail lights",
        "at night, dimly lit, sodium street lamps, light reflections on the road",
        "late at night, very dark, only headlights and shop lights illuminate the street",
    ],
    "dusk": [
        "at dusk, twilight sky, low orange sun, long shadows, headlights on",
        "at dawn, soft blue light, low contrast, street lights still on",
    ],
    "rain": [
        "in heavy rain, wet reflective asphalt, raindrops on the windshield, grey sky",
        "on a rainy evening, puddles, blurry reflections, spray from cars",
    ],
    "snow": [
        "in winter, snow on the road and sidewalks, falling snow, overcast",
        "during a snowstorm, snow-covered cars, white sky",
    ],
    "fog": [
        "in dense fog, hazy, low visibility, washed-out colors",
    ],
    "glare": [
        "at night with strong oncoming headlight glare, lens flare, bright light streaks",
    ],
}
TAIL = "photorealistic, 4k, sharp focus, natural colors"
NEGATIVE = {
    "default": "cartoon, painting, illustration, cgi, render, anime, deformed, lowres, text, watermark, logo, frame",
    "night": "daylight, sunny, blue sky, bright day",
    "dusk": "midday, harsh noon sun",
    "rain": "dry road, sunny",
    "snow": "summer, green trees",
    "fog": "clear sky, crisp",
    "glare": "daylight, sunny",
}
SCENE = {"city_street": "city street", "highway": "highway", "residential": "residential street",
         "parking_lot": "parking lot", "tunnel": "tunnel", "gas_stations": "gas station"}

# How a synthesized condition is tagged (so golden-style slices can be computed on synthetic sets too).
CONDITION_ATTRS = {
    "night": {"timeofday": "night"}, "dusk": {"timeofday": "dawn"}, "glare": {"timeofday": "night"},
    "rain": {"weather": "rainy"}, "snow": {"weather": "snowy"}, "fog": {"weather": "foggy"},
}


def make_prompt(condition: str, scene_attr: str, rng: random.Random) -> tuple[str, str]:
    if condition not in CONDITIONS:
        raise KeyError(f"no prompt recipe for condition {condition!r}")
    scene = SCENE.get(scene_attr, "city street")
    pos = f"{BASE.format(scene=scene)} {rng.choice(CONDITIONS[condition])}, {TAIL}"
    neg = f"{NEGATIVE['default']}, {NEGATIVE.get(condition, '')}".strip(", ")
    return pos, neg
