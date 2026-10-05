import contextlib
import io

import numpy as np
import pytest

from failforge.data.schema import Box, Sample
from failforge.detections import Detections
from failforge.eval.matching import iou_matrix, match_coco
from failforge.eval.metrics import EvalIndex, bootstrap_delta


def _random_set(seed=0, n=120, ncls=4):
    rng = np.random.default_rng(seed)
    samples, preds = [], {}
    for i in range(n):
        W, H = 640, 480
        boxes = []
        for _ in range(rng.integers(0, 10)):
            w, h = rng.uniform(5, 200), rng.uniform(5, 200)
            x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
            boxes.append(Box(int(rng.integers(0, ncls)), x, y, x + w, y + h))
        pb, ps, pc = [], [], []
        for b in boxes:
            if rng.random() < 0.8:
                pb.append(np.array(b.xyxy()) + rng.normal(0, 0.12, 4) * [b.w, b.h, b.w, b.h])
                ps.append(rng.uniform(0.2, 1))
                pc.append(b.cls if rng.random() < 0.9 else int(rng.integers(0, ncls)))
        for _ in range(rng.integers(0, 5)):
            w, h = rng.uniform(5, 150), rng.uniform(5, 150)
            x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
            pb.append([x, y, x + w, y + h])
            ps.append(rng.uniform(0, 0.7))
            pc.append(int(rng.integers(0, ncls)))
        samples.append(Sample(str(i), "x", W, H, boxes, {"timeofday": "night" if i % 3 == 0 else "day"}))
        preds[str(i)] = Detections(np.array(pb, np.float32).reshape(-1, 4), np.array(ps, np.float32),
                                   np.array(pc, np.int64))
    return samples, preds


def test_iou_basic():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0])


def test_match_prefers_regular_gt_over_ignored():
    ious = np.array([[0.9, 0.95]])  # one prediction overlapping an ignored GT more
    tp, m_ign = match_coco(ious, np.array([False, True]), np.array([0.5]))
    assert tp[0, 0] and not m_ign[0, 0]


def test_perfect_predictions_score_one():
    s = [Sample("a", "x", 100, 100, [Box(0, 10, 10, 50, 50), Box(1, 60, 60, 90, 95)])]
    p = {"a": Detections(np.array([[10, 10, 50, 50], [60, 60, 90, 95]], np.float32), np.array([0.9, 0.8], np.float32),
                         np.array([0, 1]))}
    assert EvalIndex(s, p, num_classes=2).map() == pytest.approx(1.0)


def test_no_predictions_score_zero_and_absent_class_is_ignored():
    s = [Sample("a", "x", 100, 100, [Box(0, 10, 10, 50, 50)])]
    idx = EvalIndex(s, {}, num_classes=3)
    assert idx.map() == 0.0
    assert idx.summary()["per_class"]["car"] is None  # class 2 has no ground truth -> excluded


def test_weight_equals_duplication():
    samples, preds = _random_set(1, n=40)
    idx = EvalIndex(samples, preds, num_classes=4)
    w = np.ones(len(samples))
    w[:10] = 2
    dpreds = dict(preds)
    dup = samples + [Sample(s.id + "_d", s.image, s.width, s.height, s.boxes) for s in samples[:10]]
    for s in samples[:10]:
        dpreds[s.id + "_d"] = preds[s.id]
    assert idx.map(w) == pytest.approx(EvalIndex(dup, dpreds, num_classes=4).map(), abs=1e-9)


def test_matches_pycocotools():
    pc = pytest.importorskip("pycocotools.cocoeval")
    from pycocotools.coco import COCO

    samples, preds = _random_set(0)
    images = [{"id": i, "width": s.width, "height": s.height} for i, s in enumerate(samples)]
    anns, dets, aid = [], [], 1
    for i, s in enumerate(samples):
        for b in s.boxes:
            anns.append({"id": aid, "image_id": i, "category_id": b.cls + 1, "bbox": [b.x1, b.y1, b.w, b.h],
                         "area": b.w * b.h, "iscrowd": 0})
            aid += 1
        d = preds[s.id]
        for bb, sc, c in zip(d.boxes, d.scores, d.classes):
            dets.append({"image_id": i, "category_id": int(c) + 1, "score": float(sc),
                         "bbox": [float(bb[0]), float(bb[1]), float(bb[2] - bb[0]), float(bb[3] - bb[1])]})
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = {"images": images, "annotations": anns, "categories": [{"id": k + 1, "name": str(k)} for k in range(4)]}
        g.createIndex()
        e = pc.COCOeval(g, g.loadRes(dets), "bbox")
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = EvalIndex(samples, preds, num_classes=4).summary()
    ours = [s["map"], s["map50"], s["map75"], s["map_small"], s["map_medium"], s["map_large"]]
    np.testing.assert_allclose(ours, e.stats[:6], atol=1e-6)


def test_bootstrap_identical_models_zero_delta():
    samples, preds = _random_set(2, n=60)
    a = EvalIndex(samples, preds, num_classes=4)
    b = EvalIndex(samples, preds, num_classes=4)
    r = bootstrap_delta(a, b, n=50)
    assert r["delta"] == pytest.approx(0) and r["ci_low"] == pytest.approx(0) and r["ci_high"] == pytest.approx(0)


def test_bootstrap_detects_real_improvement():
    samples, preds = _random_set(3, n=80)
    worse = {k: Detections(v.boxes, v.scores * np.random.default_rng(0).uniform(0, 1, len(v.scores)).astype(np.float32),
                           v.classes) for k, v in preds.items()}
    # perfect model = GT boxes
    perfect = {s.id: Detections(np.array([b.xyxy() for b in s.boxes], np.float32).reshape(-1, 4),
                                np.ones(len(s.boxes), np.float32), np.array([b.cls for b in s.boxes], np.int64))
               for s in samples}
    r = bootstrap_delta(EvalIndex(samples, worse, 4), EvalIndex(samples, perfect, 4), n=100)
    assert r["delta"] > 0.3 and r["ci_low"] > 0


def test_batched_bootstrap_matches_scalar_map():
    from failforge.eval.metrics import _boot_maps

    samples, preds = _random_set(4, n=50)
    idx_obj = EvalIndex(samples, preds, num_classes=4)
    sel = np.flatnonzero([s.attrs["timeofday"] == "night" for s in samples])
    rng = np.random.default_rng(1)
    W = np.stack([np.bincount(rng.integers(0, len(sel), len(sel)), minlength=len(sel)) for _ in range(7)]).astype(float)
    batched = _boot_maps(idx_obj, sel, W, batch=3)
    for k in range(len(W)):
        full = np.zeros(len(samples))
        full[sel] = W[k]
        assert batched[k] == pytest.approx(idx_obj.map(full), abs=1e-9)
