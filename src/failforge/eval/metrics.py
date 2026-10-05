"""COCO-style detection metrics with slices and paired bootstrap confidence intervals.

The numbers follow the COCO protocol (pycocotools semantics): 10 IoU thresholds 0.50:0.05:0.95,
101-point interpolated precision, at most ``max_det`` (100) detections per image, small / medium /
large area ranges with ignore semantics, mAP = mean over classes that have ground truth.

The implementation is our own (numpy) for one reason: an image-level *weight vector*. Matching is
done once; any subset of images (a slice like ``night``) is a 0/1 weight vector, and a bootstrap
resample is a multinomial weight vector. So per-slice mAP and the bootstrap distribution of
Delta mAP between two models are cheap and exact (a weight of k is identical to duplicating the
image k times), without re-matching or re-running pycocotools thousands of times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from failforge.config import CLASSES
from failforge.data.schema import Sample
from failforge.detections import Detections
from failforge.eval.matching import IOU_THRESHOLDS, iou_matrix, match_coco

AREA_RANGES = {"all": (0.0, 1e10), "small": (0.0, 32.0**2), "medium": (32.0**2, 96.0**2), "large": (96.0**2, 1e10)}
REC_THRS = np.linspace(0.0, 1.0, 101)


@dataclass
class _ClassTable:
    scores: np.ndarray  # (N,) descending
    img: np.ndarray  # (N,) image index
    tp: np.ndarray  # (T, N) bool
    ign: np.ndarray  # (T, N) bool - prediction does not count (matched ignored GT / outside area range)
    npos: np.ndarray  # (n_images,) non-ignored GT per image


class EvalIndex:
    """All matching results for one model on one image set."""

    def __init__(self, samples: list[Sample], preds: dict[str, Detections], num_classes: int = len(CLASSES),
                 max_det: int = 100, areas: tuple[str, ...] = ("all", "small", "medium", "large")) -> None:
        self.ids = [s.id for s in samples]
        self.attrs = [s.attrs for s in samples]
        self.num_classes = num_classes
        self.areas = areas
        n_img, n_t = len(samples), len(IOU_THRESHOLDS)
        acc: dict[tuple[int, str], dict[str, list]] = {
            (c, a): {"scores": [], "img": [], "tp": [], "ign": []} for c in range(num_classes) for a in areas}
        npos = {(c, a): np.zeros(n_img, np.int64) for c in range(num_classes) for a in areas}
        for i, s in enumerate(samples):
            d = preds.get(s.id, Detections.empty()).top(max_det)
            g_cls = np.array([b.cls for b in s.boxes], np.int64)
            g_box = np.array([b.xyxy() for b in s.boxes], np.float64).reshape(-1, 4)
            g_area = (g_box[:, 2] - g_box[:, 0]) * (g_box[:, 3] - g_box[:, 1])
            p_area = (d.boxes[:, 2] - d.boxes[:, 0]) * (d.boxes[:, 3] - d.boxes[:, 1]) if len(d) else np.zeros(0)
            for c in range(num_classes):
                gk = g_cls == c
                pk = np.flatnonzero(d.classes == c)
                if not gk.any() and len(pk) == 0:
                    continue
                pk = pk[np.argsort(-d.scores[pk], kind="mergesort")]
                ious = iou_matrix(d.boxes[pk], g_box[gk])
                for a in areas:
                    lo, hi = AREA_RANGES[a]
                    g_ign = (g_area[gk] < lo) | (g_area[gk] > hi)
                    npos[(c, a)][i] = int((~g_ign).sum())
                    if len(pk) == 0:
                        continue
                    tp, m_ign = match_coco(ious, g_ign)
                    out_rng = (p_area[pk] < lo) | (p_area[pk] > hi)
                    ign = m_ign | (~tp & out_rng[None, :])
                    t = acc[(c, a)]
                    t["scores"].append(d.scores[pk])
                    t["img"].append(np.full(len(pk), i, np.int64))
                    t["tp"].append(tp)
                    t["ign"].append(ign)
        self.tables: dict[tuple[int, str], _ClassTable] = {}
        for key, t in acc.items():
            if t["scores"]:
                sc = np.concatenate(t["scores"])
                order = np.argsort(-sc, kind="mergesort")
                tab = _ClassTable(sc[order], np.concatenate(t["img"])[order],
                                  np.concatenate(t["tp"], axis=1)[:, order], np.concatenate(t["ign"], axis=1)[:, order],
                                  npos[key])
            else:
                tab = _ClassTable(np.zeros(0), np.zeros(0, np.int64), np.zeros((n_t, 0), bool),
                                  np.zeros((n_t, 0), bool), npos[key])
            self.tables[key] = tab

    # ------------------------------------------------------------------ AP

    @staticmethod
    def _ap(tab: _ClassTable, w: np.ndarray) -> np.ndarray:
        """AP at every IoU threshold, (T,), NaN if the class has no (weighted) ground truth."""
        n_pos = float((tab.npos * w).sum())
        n_t = tab.tp.shape[0]
        if n_pos <= 0:
            return np.full(n_t, np.nan)
        if tab.tp.shape[1] == 0:
            return np.zeros(n_t)
        wp = w[tab.img][None, :] * ~tab.ign
        tps = np.cumsum(tab.tp * wp, axis=1)
        fps = np.cumsum(~tab.tp * wp, axis=1)
        rec = tps / n_pos
        prec = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
        prec = np.maximum.accumulate(prec[:, ::-1], axis=1)[:, ::-1]
        out = np.zeros(n_t)
        n = rec.shape[1]
        for t in range(n_t):
            idx = np.searchsorted(rec[t], REC_THRS, side="left")
            q = np.where(idx < n, prec[t][np.minimum(idx, n - 1)], 0.0)
            out[t] = q.mean()
        return out

    def ap_table(self, weights: np.ndarray | None = None, area: str = "all") -> np.ndarray:
        """(num_classes, T) AP matrix (NaN rows for classes without ground truth)."""
        w = np.ones(len(self.ids)) if weights is None else np.asarray(weights, dtype=np.float64)
        return np.stack([self._ap(self.tables[(c, area)], w) for c in range(self.num_classes)])

    def map(self, weights: np.ndarray | None = None, area: str = "all") -> float:
        t = self.ap_table(weights, area)
        rows = ~np.isnan(t[:, 0])
        return float(t[rows].mean()) if rows.any() else float("nan")

    def summary(self, weights: np.ndarray | None = None) -> dict:
        t = self.ap_table(weights)
        rows = ~np.isnan(t[:, 0])
        w = np.ones(len(self.ids)) if weights is None else np.asarray(weights, dtype=np.float64)

        def m(x: np.ndarray) -> float:
            return float(np.nanmean(x)) if rows.any() else float("nan")

        out = {
            "map": m(t[rows].mean(axis=1)) if rows.any() else float("nan"),
            "map50": m(t[rows, 0]), "map75": m(t[rows, 5]),
            "per_class": {CLASSES[c]: (None if np.isnan(t[c, 0]) else float(t[c].mean())) for c in range(self.num_classes)},
            "n_images": int((w > 0).sum()),
            "n_gt": int(sum((self.tables[(c, "all")].npos * (w > 0)).sum() for c in range(self.num_classes))),
        }
        for a in ("small", "medium", "large"):
            if a in self.areas:
                v = self.map(w, a)
                out[f"map_{a}"] = None if np.isnan(v) else v
        return out


# ---------------------------------------------------------------------- bootstrap


def _restrict(tab: _ClassTable, idx: np.ndarray, n_img: int) -> _ClassTable:
    """The table restricted to the images ``idx`` (re-indexed 0..len(idx)-1)."""
    pos = np.full(n_img, -1, np.int64)
    pos[idx] = np.arange(len(idx))
    keep = pos[tab.img] >= 0
    return _ClassTable(tab.scores[keep], pos[tab.img[keep]], tab.tp[:, keep], tab.ign[:, keep], tab.npos[idx])


def _ap_batch(tab: _ClassTable, W: np.ndarray) -> np.ndarray:
    """AP averaged over IoU thresholds for a batch of image-weight vectors ``W`` (B, m) -> (B,), NaN = no GT."""
    B = W.shape[0]
    n_pos = W @ tab.npos.astype(np.float64)
    out = np.full(B, np.nan)
    has = n_pos > 0
    if not has.any():
        return out
    if tab.tp.shape[1] == 0:
        out[has] = 0.0
        return out
    wimg = W[:, tab.img]
    n = wimg.shape[1]
    acc = np.zeros(B)
    for t in range(tab.tp.shape[0]):
        w = wimg * ~tab.ign[t]
        tps = np.cumsum(w * tab.tp[t], axis=1)
        fps = np.cumsum(w * ~tab.tp[t], axis=1)
        rec = tps / np.maximum(n_pos, 1e-12)[:, None]
        prec = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
        prec = np.maximum.accumulate(prec[:, ::-1], axis=1)[:, ::-1]
        for b in range(B):
            k = np.searchsorted(rec[b], REC_THRS, side="left")
            acc[b] += np.where(k < n, prec[b][np.minimum(k, n - 1)], 0.0).mean()
    out[has] = acc[has] / tab.tp.shape[0]
    return out


def _boot_maps(ix: EvalIndex, idx: np.ndarray, W: np.ndarray, batch: int = 32) -> np.ndarray:
    """mAP for every row of the resample-count matrix ``W`` over the images ``idx``."""
    tabs = [_restrict(ix.tables[(c, "all")], idx, len(ix.ids)) for c in range(ix.num_classes)]
    out = np.empty(len(W))
    for s in range(0, len(W), batch):
        aps = np.stack([_ap_batch(tb, W[s:s + batch]) for tb in tabs], axis=1)  # (B, C)
        with np.errstate(all="ignore"):
            out[s:s + batch] = np.nanmean(np.where(np.isnan(aps).all(axis=1, keepdims=True), 0.0, aps), axis=1)
    return out


def bootstrap_delta(a: EvalIndex, b: EvalIndex, mask: np.ndarray | None = None, n: int = 1000, seed: int = 0,
                    ci: float = 0.95) -> dict:
    """Paired image-level bootstrap of ``mAP(b) - mAP(a)`` over the images in ``mask``.

    Both indexes must cover the same images in the same order. The same resample is applied to both
    models, so per-image difficulty cancels and the interval reflects the *difference* only.
    Resamples are evaluated in vectorized batches on tables restricted to the slice, and the
    baseline's resampled mAPs are cached on it, so comparing several arms to one baseline costs
    one baseline pass.
    """
    if a.ids != b.ids:
        raise ValueError("bootstrap_delta: the two evaluations cover different images")
    idx = np.flatnonzero(np.ones(len(a.ids), bool) if mask is None else mask)
    w0 = np.zeros(len(a.ids))
    w0[idx] = 1.0
    point = b.map(w0) - a.map(w0)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)) for _ in range(n)]).astype(np.float64)
    key = (idx.tobytes(), n, seed)
    cache = a.__dict__.setdefault("_boot_cache", {})
    if key not in cache:
        cache[key] = _boot_maps(a, idx, W)
    deltas = _boot_maps(b, idx, W) - cache[key]
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.quantile(deltas, [(1 - ci) / 2, 1 - (1 - ci) / 2]) if len(deltas) else (np.nan, np.nan)
    return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_zero": float((deltas <= 0).mean()) if len(deltas) else float("nan"), "n_boot": int(len(deltas)),
            "n_images": int(len(idx))}
