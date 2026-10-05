"""TIDE-style error typing at a deployment operating point.

"False positives and false negatives" hides what to fix. Following TIDE (Bolya et al., 2020) every
prediction above the operating confidence is typed, in priority order:

    tp    IoU >= t_fg with an unmatched GT of the same class
    dupe  IoU >= t_fg with a same-class GT that a higher-scoring prediction already took
    cls   IoU >= t_fg with a GT of another class (right place, wrong label)
    loc   t_bg <= IoU < t_fg with a same-class GT (right label, sloppy box)
    both  t_bg <= IoU < t_fg with a GT of another class
    bkg   IoU < t_bg with every GT (hallucinated object)

and every GT not matched by a TP is a ``miss``, with ``covered_by`` recording whether some wrong
prediction overlapped it (so a Loc/Cls error and its missed GT are not counted as two problems).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from failforge.data.schema import Sample
from failforge.detections import Detections
from failforge.eval.matching import iou_matrix

ERROR_TYPES = ("cls", "loc", "both", "dupe", "bkg", "miss")


@dataclass
class Record:
    image_id: str
    kind: str  # tp | cls | loc | both | dupe | bkg | miss
    cls: int  # predicted class (GT class for miss)
    box: list[float]
    score: float  # 0 for miss
    iou: float  # best IoU with any GT (same class for tp/dupe/loc)
    gt_cls: int = -1
    covered_by: str = ""  # miss only

    def to_dict(self) -> dict:
        return asdict(self)


def type_image(s: Sample, d: Detections, conf: float, t_fg: float = 0.5, t_bg: float = 0.1) -> list[Record]:
    d = d.above(conf)
    order = np.argsort(-d.scores, kind="mergesort")
    p_box, p_sc, p_cls = d.boxes[order], d.scores[order], d.classes[order]
    g_box = np.array([b.xyxy() for b in s.boxes], np.float64).reshape(-1, 4)
    g_cls = np.array([b.cls for b in s.boxes], np.int64)
    ious = iou_matrix(p_box, g_box)
    taken = np.zeros(len(g_box), bool)
    covered: dict[int, str] = {}
    out: list[Record] = []
    for i in range(len(p_sc)):
        same = g_cls == p_cls[i]
        row = ious[i] if len(g_box) else np.zeros(0)
        best_any = float(row.max()) if len(row) else 0.0
        kind, iou_v, gt_c = "bkg", best_any, -1
        if same.any():
            free = same & ~taken
            if free.any() and row[free].max() >= t_fg:
                g = int(np.flatnonzero(free)[np.argmax(row[free])])
                taken[g] = True
                out.append(Record(s.id, "tp", int(p_cls[i]), p_box[i].tolist(), float(p_sc[i]), float(row[g]), int(g_cls[g])))
                continue
            if row[same].max() >= t_fg:
                kind, iou_v = "dupe", float(row[same].max())
        if kind == "bkg":
            other = ~same
            if other.any() and row[other].max() >= t_fg:
                g = int(np.flatnonzero(other)[np.argmax(row[other])])
                kind, iou_v, gt_c = "cls", float(row[g]), int(g_cls[g])
            elif same.any() and row[same].max() >= t_bg:
                g = int(np.flatnonzero(same)[np.argmax(row[same])])
                kind, iou_v, gt_c = "loc", float(row[g]), int(g_cls[g])
            elif other.any() and row[other].max() >= t_bg:
                g = int(np.flatnonzero(other)[np.argmax(row[other])])
                kind, iou_v, gt_c = "both", float(row[g]), int(g_cls[g])
        if kind in ("cls", "loc", "both"):
            covered.setdefault(gt_index(row, g_cls, gt_c, kind, t_bg), kind)
        out.append(Record(s.id, kind, int(p_cls[i]), p_box[i].tolist(), float(p_sc[i]), iou_v, gt_c))
    for g in np.flatnonzero(~taken):
        best = float(ious[:, g].max()) if len(p_sc) else 0.0
        out.append(Record(s.id, "miss", int(g_cls[g]), g_box[g].tolist(), 0.0, best, int(g_cls[g]),
                          covered_by=covered.get(int(g), "")))
    return out


def gt_index(row: np.ndarray, g_cls: np.ndarray, c: int, kind: str, t_bg: float) -> int:
    """Index of the GT a cls/loc/both error overlapped (best IoU among GTs of class ``c``)."""
    k = np.flatnonzero(g_cls == c)
    return int(k[np.argmax(row[k])]) if len(k) else -1


def summarize(records: list[Record]) -> dict:
    counts = {k: 0 for k in ("tp", *ERROR_TYPES)}
    for r in records:
        counts[r.kind] += 1
    n_gt = counts["tp"] + counts["miss"]
    n_pred = sum(v for k, v in counts.items() if k != "miss")
    counts["miss_uncovered"] = sum(1 for r in records if r.kind == "miss" and not r.covered_by)
    counts["precision"] = counts["tp"] / n_pred if n_pred else 0.0
    counts["recall"] = counts["tp"] / n_gt if n_gt else 0.0
    return counts
