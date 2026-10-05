import json

import pytest

from failforge.config import DataConfig
from failforge.data import splits
from failforge.data.bdd import parse_fiftyone
from failforge.data.schema import Box, Sample, manifest_digest, read_manifest, validate_samples, write_manifest
from failforge.data.yolo import build_dataset, label_lines


def test_manifest_roundtrip_and_validation(tmp_path):
    s = [Sample("a-1", "a.jpg", 100, 50, [Box(2, 1, 2, 30, 40)], {"timeofday": "day"})]
    p = tmp_path / "m.jsonl"
    write_manifest(p, s)
    back = read_manifest(p)
    assert back == s
    assert validate_samples(back) == []
    bad = [Sample("a", "a.jpg", 100, 50, [Box(99, 10, 10, 5, 5)]), Sample("a", "a.jpg", 100, 50, [])]
    probs = validate_samples(bad)
    assert any("out of range" in x for x in probs) and any("degenerate" in x for x in probs) and any("duplicate" in x for x in probs)


def test_digest_is_order_independent():
    a = Sample("x", "x", 10, 10, [Box(0, 1, 1, 5, 5)])
    b = Sample("y", "y", 10, 10, [])
    assert manifest_digest([a, b]) == manifest_digest([b, a])
    assert manifest_digest([a]) != manifest_digest([Sample("x", "x", 10, 10, [Box(0, 1, 1, 6, 5)])])


def test_yolo_labels():
    s = Sample("a", "a.jpg", 200, 100, [Box(3, 50, 25, 150, 75)])
    assert label_lines(s) == ["3 0.500000 0.500000 0.500000 0.500000"]


def test_parse_fiftyone(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "v1-f1.jpg").write_bytes(b"x")
    rec = {"filepath": "data/v1-f1.jpg", "metadata": {"width": 1000, "height": 500},
           "timeofday": {"label": "dawn/dusk"}, "weather": {"label": "partly cloudy"}, "scene": {"label": "city street"},
           "detections": {"detections": [
               {"label": "car", "bounding_box": [0.1, 0.2, 0.3, 0.4]},
               {"label": "bicycle", "bounding_box": [0.5, 0.5, 0.1, 0.1]},
               {"label": "train", "bounding_box": [0.0, 0.0, 0.5, 0.5]},  # unmapped -> dropped
               {"label": "car", "bounding_box": [0.0, 0.0, 0.001, 0.001]},  # tiny -> dropped
           ]}}
    sj = tmp_path / "samples.json"
    sj.write_text(json.dumps({"samples": [rec]}))
    (s,) = parse_fiftyone(sj, tmp_path / "data")
    assert s.attrs == {"timeofday": "dawn", "weather": "partly_cloudy", "scene": "city_street", "source": "real"}
    assert [b.cls for b in s.boxes] == [2, 5]
    assert s.boxes[0].xyxy() == [100.0, 100.0, 400.0, 300.0]


def test_splits_disjoint_deterministic_and_frozen(tiny_data, workspace):
    sp = tiny_data["splits"]
    splits.check_disjoint(sp)
    assert all(s.attrs["timeofday"] == "day" for s in sp["train_day"] + sp["val_day"])
    assert {s.attrs["timeofday"] for s in sp["golden"]} == {"day", "night", "dawn"}
    # Same inputs -> same splits; the golden set comes from the lock.
    again = splits.make_splits(tiny_data["samples"], tiny_data["cfg"])
    assert [s.id for s in again["golden"]] == [s.id for s in sp["golden"]]
    # Changing the requested sizes later must NOT change the frozen golden set.
    bigger = DataConfig(golden_day=40, golden_night=40, golden_dawn=10)
    assert [s.id for s in splits.make_splits(tiny_data["samples"], bigger)["golden"]] == [s.id for s in sp["golden"]]


def test_golden_lock_detects_label_tampering(tiny_data):
    samples = tiny_data["samples"]
    gid = tiny_data["splits"]["golden"][0].id
    tampered = [Sample(s.id, s.image, s.width, s.height, s.boxes[:-1] if s.id == gid else s.boxes, s.attrs) for s in samples]
    with pytest.raises(splits.GoldenLockError):
        splits.make_splits(tampered, tiny_data["cfg"])


def test_stratified_take_is_proportional():
    ss = [Sample(f"{i:04d}-a", "x", 1, 1, [], {"timeofday": "night", "weather": "rainy" if i < 200 else "clear"})
          for i in range(1000)]
    got = splits.stratified_take(ss, 100, seed=1)
    rainy = sum(s.attrs["weather"] == "rainy" for s in got)
    assert len(got) == 100 and 18 <= rainy <= 22


def test_build_dataset(tiny_data, workspace):
    sp = tiny_data["splits"]
    y = build_dataset(workspace / "ds", sp["train_day"][:5], sp["val_day"][:3])
    lines = (workspace / "ds" / "train.txt").read_text().split()
    assert len(lines) == 5 and all("/images/" in x for x in lines)
    lbl = workspace / "datasets" / "store" / "labels" / f"{sp['train_day'][0].id}.txt"
    assert lbl.is_file() and len(lbl.read_text().split("\n")) >= 1
    assert "names" in y.read_text()
