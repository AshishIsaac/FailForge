"""Failure mining: type every error the deployed model makes on the production probe set.

Output is one record per error (and a random sample of true positives, so that each failure
cluster's *error rate* can be estimated, not just its error count). Records carry the image
attributes, so later stages can relate a cluster to time of day / weather without peeking at
those tags during clustering itself (the clustering never sees them: it must rediscover them).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from failforge.config import MiningConfig
from failforge.data.schema import Sample
from failforge.detections import Detections
from failforge.eval import tide

MINED_KINDS = ("miss", "bkg", "cls", "loc", "both")  # dupes are an NMS issue, not a data issue


@dataclass
class MinedSet:
    errors: list[dict] = field(default_factory=list)
    tps: list[dict] = field(default_factory=list)
    tp_rate: float = 1.0  # fraction of all TPs kept in ``tps``
    n_tp_total: int = 0
    summary: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"errors": self.errors, "tps": self.tps, "tp_rate": self.tp_rate,
                                    "n_tp_total": self.n_tp_total, "summary": self.summary}), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> MinedSet:
        d = json.loads(path.read_text(encoding="utf-8"))
        return MinedSet(d["errors"], d["tps"], d["tp_rate"], d["n_tp_total"], d.get("summary", {}))


def mine(samples: list[Sample], preds: dict[str, Detections], conf: float, cfg: MiningConfig, seed: int = 0) -> MinedSet:
    by_id = {s.id: s for s in samples}
    errors, tps = [], []
    all_recs = []
    for s in samples:
        recs = tide.type_image(s, preds.get(s.id, Detections.empty()), conf, cfg.iou_match, cfg.iou_background)
        all_recs += recs
        for r in recs:
            x1, y1, x2, y2 = r.box
            if min(x2 - x1, y2 - y1) < 2:
                continue
            d = {**r.to_dict(), "attrs": s.attrs, "width": s.width, "height": s.height, "image": s.image}
            if r.kind == "tp":
                tps.append(d)
            elif r.kind in MINED_KINDS:
                errors.append(d)
    rng = random.Random(seed)
    if len(errors) > cfg.max_errors:
        errors = rng.sample(errors, cfg.max_errors)
    n_tp = len(tps)
    if len(tps) > cfg.tp_sample:
        tps = rng.sample(tps, cfg.tp_sample)
    summ = tide.summarize(all_recs)
    summ["images"] = len(by_id)
    return MinedSet(errors, tps, (len(tps) / n_tp) if n_tp else 1.0, n_tp, summ)
