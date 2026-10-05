import json
import random

import cv2
import numpy as np

from failforge.config import SynthesisConfig
from failforge.data.schema import read_manifest
from failforge.synthesis import engine
from failforge.synthesis.augment import relight
from failforge.synthesis.prompts import make_prompt
from failforge.synthesis.quality import box_structure, kid


def test_allocate_sums_and_is_proportional():
    a = engine.allocate([{"excess_errors": 30, "condition": "night", "classes": []},
                         {"excess_errors": 10, "condition": "rain", "classes": []}], 101)
    assert sum(x["quota"] for x in a) == 101 and a[0]["quota"] in (75, 76)
    assert engine.allocate([], 10) == []


def test_prompts_are_condition_specific():
    p, n = make_prompt("night", "highway", random.Random(0))
    assert "night" in p and "highway" in p and "daylight" in n


def test_relight_darkens_and_keeps_shape():
    img = np.full((90, 160, 3), 180, np.uint8)
    out = relight(img, "night", np.random.default_rng(0))
    assert out.shape == img.shape and out.mean() < img.mean() * 0.6


def test_box_structure_same_vs_destroyed():
    rng = np.random.default_rng(0)
    img = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(img, (40, 30), (110, 100), (255, 255, 255), 3)
    cv2.circle(img, (75, 65), 15, (200, 200, 200), 2)
    box = [[35, 25, 115, 105]]
    assert box_structure(img, img, box)[0] > 0.99
    assert box_structure(img, (img * 0.3).astype(np.uint8), box)[0] > 0.9  # darker, same structure
    noise = rng.integers(0, 255, img.shape, dtype=np.uint8)
    assert box_structure(img, noise, box)[0] < 0.3
    assert np.isnan(box_structure(img, img, [[0, 0, 10, 10]])[0])  # too small to judge


def test_kid_orders_distributions():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(200, 16)), rng.normal(size=(200, 16)), rng.normal(size=(200, 16)) + 0.7
    assert kid(a, b, subsets=10, subset_size=100)[0] < kid(a, c, subsets=10, subset_size=100)[0]


def test_engine_augment_run_is_resumable(tiny_data, workspace):
    train = tiny_data["splits"]["train_day"]
    targets = [{"condition": "night", "classes": ["car"], "excess_errors": 5.0, "name": "t"}]
    out = workspace / "syn"
    cfg = SynthesisConfig(backend="augment")
    s1 = engine.run("augment", targets, train, out, cfg, count=6, id_prefix="aug")
    assert s1["accepted"] == 6
    m = read_manifest(out / "manifest.jsonl")
    assert len(m) == 6 and all(x.attrs["timeofday"] == "night" and x.attrs["source"] == "aug" for x in m)
    parents = {s.id: s for s in train}
    for x in m:  # labels are inherited from the source image
        assert x.boxes == parents[x.attrs["parent"]].boxes
    # Resume: nothing is regenerated, nothing duplicated.
    n_lines = len((out / "qc.jsonl").read_text().splitlines())
    s2 = engine.run("augment", targets, train, out, cfg, count=6, id_prefix="aug")
    assert s2["accepted"] == 6 and len((out / "qc.jsonl").read_text().splitlines()) == n_lines
    assert json.loads((out / "summary.json").read_text())["accepted"] == 6
