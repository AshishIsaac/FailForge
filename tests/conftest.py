"""Shared fixtures: an isolated workspace with a tiny synthetic "BDD-like" dataset.

Images are procedurally drawn (coloured rectangles on a road-ish background); "night" images are
dark. Nothing touches the network or the real workspace.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from failforge.data.schema import Box, Sample, write_manifest


def draw_dataset(root: Path, n_per_tod: dict[str, int], seed: int = 0, size=(320, 180)) -> list[Sample]:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    (root / "data").mkdir(parents=True, exist_ok=True)
    W, H = size
    out = []
    k = 0
    for tod, n in n_per_tod.items():
        for _ in range(n):
            img = np.full((H, W, 3), (120, 130, 140), np.uint8)
            img[int(H * 0.55):] = (90, 90, 90)
            boxes = []
            for _ in range(rng.randint(1, 5)):
                w, h = rng.randint(18, 70), rng.randint(18, 60)
                x, y = rng.randint(0, W - w - 1), rng.randint(int(H * 0.3), H - h - 1)
                c = rng.choice([0, 2, 3])
                color = {0: (40, 200, 40), 2: (40, 40, 220), 3: (220, 120, 40)}[c]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
                boxes.append(Box(c, float(x), float(y), float(x + w), float(y + h)))
            if tod == "night":
                img = (img.astype(np.float32) * 0.25).astype(np.uint8)
            elif tod == "dawn":
                img = (img.astype(np.float32) * 0.6).astype(np.uint8)
            img = np.clip(img + nrng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
            vid = f"{k:08x}"
            p = root / "data" / f"{vid}-{k:08x}.jpg"
            cv2.imwrite(str(p), img)
            weather = rng.choice(["clear", "clear", "overcast", "rainy", "snowy"])
            out.append(Sample(id=p.stem, image=str(p.resolve()), width=W, height=H, boxes=boxes,
                              attrs={"timeofday": tod, "weather": weather, "scene": "city_street", "source": "real"}))
            k += 1
    return out


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    home = tmp_path / "ws"
    monkeypatch.setenv("FAILFORGE_HOME", str(home))
    return home


@pytest.fixture
def tiny_data(workspace):
    """A prepared workspace: manifests + frozen golden set over ~150 synthetic images."""
    from failforge.config import DataConfig
    from failforge.data import splits

    samples = draw_dataset(workspace / "data" / "bdd100k", {"day": 90, "night": 50, "dawn": 15})
    write_manifest(workspace / "data" / "manifests" / "all.jsonl", samples)
    cfg = DataConfig(golden_day=15, golden_night=20, golden_dawn=5, probe_day=5, probe_night=15, probe_dawn=4,
                     val_day=10, train_day=0)
    sp = splits.make_splits(samples, cfg)
    splits.write_splits(sp)
    return {"samples": samples, "splits": sp, "cfg": cfg}
