"""Box overlap and COCO-style greedy matching."""

from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``a`` (N,4) and ``b`` (M,4) xyxy boxes -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_coco(ious: np.ndarray, gt_ignore: np.ndarray, thresholds: np.ndarray = IOU_THRESHOLDS
               ) -> tuple[np.ndarray, np.ndarray]:
    """Greedy COCO matching of score-sorted predictions to ground truth, at every threshold.

    ``ious`` is (P, G) with predictions already sorted by descending score. Ground truth flagged in
    ``gt_ignore`` (outside the area range) can absorb a prediction without counting it as a TP or FP.
    Returns ``tp`` (T, P) bool and ``matched_ignored`` (T, P) bool.
    Mirrors pycocotools: non-ignored GT is preferred; once a prediction has a regular match it never
    switches to an ignored one.
    """
    thr = np.minimum(np.asarray(thresholds, dtype=np.float64), 1 - 1e-10)
    n_t, (n_p, n_g) = len(thr), ious.shape
    tp = np.zeros((n_t, n_p), dtype=bool)
    m_ign = np.zeros((n_t, n_p), dtype=bool)
    if n_p == 0 or n_g == 0:
        return tp, m_ign
    ign = np.asarray(gt_ignore, dtype=bool)
    taken = np.zeros((n_t, n_g), dtype=bool)
    rows = np.arange(n_t)
    for p in range(n_p):
        row = ious[p]
        if row.max() < thr[0]:
            continue  # overlaps nothing at the loosest threshold
        cand = ~taken & (row[None, :] >= thr[:, None])  # (T, G)
        reg = cand & ~ign[None, :]
        has_reg = reg.any(axis=1)
        has_any = cand.any(axis=1)
        # Best regular GT if there is one, else best ignored GT (pycocotools order).
        pick_reg = np.argmax(np.where(reg, row[None, :], -1.0), axis=1)
        pick_ign = np.argmax(np.where(cand & ign[None, :], row[None, :], -1.0), axis=1)
        pick = np.where(has_reg, pick_reg, pick_ign)
        ok = has_any
        taken[rows[ok], pick[ok]] = True
        tp[ok & has_reg, p] = True
        m_ign[ok & ~has_reg, p] = True
    return tp, m_ign
