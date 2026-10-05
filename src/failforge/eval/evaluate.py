"""Score a prediction set on a labeled image set: overall, per class, per slice, per size, TIDE."""

from __future__ import annotations

import math

import numpy as np

from failforge.config import CLASSES
from failforge.data.schema import Sample
from failforge.detections import Detections
from failforge.eval import tide
from failforge.eval.metrics import EvalIndex, bootstrap_delta
from failforge.eval.slices import MIN_SLICE_IMAGES, SLICES, slice_mask


def _clean(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def evaluate(samples: list[Sample], preds: dict[str, Detections], operating_conf: float = 0.25,
             slices: list[str] | None = None) -> tuple[dict, EvalIndex]:
    idx = EvalIndex(samples, preds)
    attrs = [s.attrs for s in samples]
    res: dict = {"overall": idx.summary(), "slices": {}}
    for name in slices or list(SLICES):
        m = slice_mask(attrs, name)
        if name == "overall" or not m.any():
            continue
        r = idx.summary(m.astype(float))
        r["gated"] = bool(m.sum() >= MIN_SLICE_IMAGES)
        res["slices"][name] = r
    recs = [r for s in samples for r in tide.type_image(s, preds.get(s.id, Detections.empty()), operating_conf)]
    res["tide"] = tide.summarize(recs)
    res["tide"]["operating_conf"] = operating_conf
    tod = {}
    for name in ("day", "night", "dawn"):
        m = slice_mask(attrs, name)
        if m.any():
            ids = {samples[i].id for i in np.flatnonzero(m)}
            tod[name] = tide.summarize([r for r in recs if r.image_id in ids])
    res["tide_by_slice"] = tod
    res["classes"] = CLASSES
    return _clean(res), idx


def compare(base: EvalIndex, cand: EvalIndex, slices: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired-bootstrap Delta mAP@50:95 (candidate - base) for each slice."""
    out = {}
    for name in slices:
        m = slice_mask(base.attrs, name)
        if not m.any():
            continue
        out[name] = _clean(bootstrap_delta(base, cand, m, n=n_boot, seed=seed))
    return out
