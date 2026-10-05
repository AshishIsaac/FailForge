import numpy as np

from failforge.data.schema import Box, Sample
from failforge.detections import Detections
from failforge.eval.tide import summarize, type_image


def _s():
    return Sample("img", "x", 200, 200, [Box(0, 10, 10, 60, 60), Box(2, 100, 100, 180, 180)])


def _d(rows):
    a = np.array(rows, np.float32).reshape(-1, 6)
    return Detections(a[:, :4], a[:, 4], a[:, 5].astype(np.int64))


def kinds(rows):
    return sorted(r.kind for r in type_image(_s(), _d(rows), conf=0.25))


def test_tp_and_miss():
    assert kinds([[10, 10, 60, 60, 0.9, 0]]) == ["miss", "tp"]


def test_duplicate():
    assert kinds([[10, 10, 60, 60, 0.9, 0], [11, 11, 60, 60, 0.8, 0]]) == ["dupe", "miss", "tp"]


def test_class_confusion_and_covered_miss():
    recs = type_image(_s(), _d([[100, 100, 180, 180, 0.9, 0]]), conf=0.25)
    k = {r.kind: r for r in recs}
    assert "cls" in k and k["cls"].gt_cls == 2
    misses = [r for r in recs if r.kind == "miss"]
    assert {m.cls for m in misses} == {0, 2}
    assert next(m for m in misses if m.cls == 2).covered_by == "cls"


def test_localization_and_background():
    got = kinds([[10, 10, 30, 60, 0.9, 0], [150, 10, 190, 40, 0.7, 0]])  # IoU 0.4 -> loc
    assert got.count("loc") == 1 and got.count("bkg") == 1


def test_below_operating_point_ignored():
    assert kinds([[10, 10, 60, 60, 0.1, 0]]) == ["miss", "miss"]


def test_summary_precision_recall():
    s = summarize(type_image(_s(), _d([[10, 10, 60, 60, 0.9, 0], [150, 10, 190, 40, 0.7, 0]]), conf=0.25))
    assert s["tp"] == 1 and s["miss"] == 1 and s["bkg"] == 1
    assert s["precision"] == 0.5 and s["recall"] == 0.5
