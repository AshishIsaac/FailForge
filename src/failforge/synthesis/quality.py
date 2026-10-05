"""Quality gates for generated training images.

Bad synthetic data is worse than none: a "night" image that is still daylight teaches nothing, and
an image whose car moved teaches the detector to put boxes on empty road. Every candidate passes:

1. **Prompt adherence**: CLIP cosine(image, prompt) >= ``min_clip_score``.
2. **Condition check**: CLIP zero-shot P(target condition) among {target, daylight, overcast}
   >= ``min_target_prob`` (did it actually become night?).
3. **Label validity**: an independent *teacher* detector (COCO-pretrained YOLO11s, never trained on
   BDD) is run on the source and on the generated image. Every sizeable labeled object the teacher
   finds in the source (class-agnostic IoU >= 0.5) must still be found at the same place in the
   generated image (IoU >= ``teacher_iou``); at least ``min_box_pass`` of them must survive. This
   is what licenses reusing the source's labels. It is semantic and lighting-invariant, unlike
   edge or pixel similarity, which drop sharply for a correct night re-rendering. Without a
   teacher (CI), a blurred-edge correlation per box is used instead (``min_box_structure``).

Plus a *batch* diagnostic: KID (kernel inception distance, here on CLIP features) of the accepted
set against real images of the target condition, next to the KID of the untouched day sources and
of the classical augmentation, so the report can show which is closer to real night.
"""

from __future__ import annotations

import cv2
import numpy as np


def _edges(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.equalizeHist(g)  # compare structure, not brightness: night and day edges have different contrast
    e = cv2.Canny(g, 60, 160).astype(np.float32)
    return cv2.GaussianBlur(e, (0, 0), 2.0)


def box_structure(src_rgb: np.ndarray, gen_rgb: np.ndarray, boxes_xyxy: list[list[float]], min_side: int = 24
                  ) -> list[float]:
    """Edge correlation per box (NaN for boxes too small to judge)."""
    es, eg = _edges(src_rgb), _edges(gen_rgb)
    out = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1, x2, y2 = int(max(x1, 0)), int(max(y1, 0)), int(x2), int(y2)
        if min(x2 - x1, y2 - y1) < min_side:
            out.append(float("nan"))
            continue
        a, b = es[y1:y2, x1:x2].ravel(), eg[y1:y2, x1:x2].ravel()
        if a.std() < 1e-6 or b.std() < 1e-6:
            out.append(0.0 if a.std() >= 1e-6 or b.std() >= 1e-6 else 1.0)
            continue
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def kid(x: np.ndarray, y: np.ndarray, subsets: int = 50, subset_size: int = 200, seed: int = 0) -> tuple[float, float]:
    """Unbiased KID (cubic polynomial kernel MMD^2) between feature sets; mean and std over subsets."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    d = x.shape[1]
    m = min(subset_size, len(x), len(y))
    if m < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(subsets):
        a = x[rng.choice(len(x), m, replace=False)]
        b = y[rng.choice(len(y), m, replace=False)]
        kaa, kbb, kab = (a @ a.T / d + 1) ** 3, (b @ b.T / d + 1) ** 3, (a @ b.T / d + 1) ** 3
        vals.append((kaa.sum() - np.trace(kaa)) / (m * (m - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
                    - 2 * kab.mean())
    return float(np.mean(vals)), float(np.std(vals))


class Teacher:
    """COCO-pretrained detector used only to verify that labeled objects survived generation."""

    def __init__(self, weights: str, conf: float = 0.1, imgsz: int = 640) -> None:
        from ultralytics import YOLO

        from failforge.detector import pretrained_weights, resolve_device

        self.model = YOLO(pretrained_weights(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = resolve_device("auto")

    def boxes(self, rgb: np.ndarray) -> np.ndarray:
        r = self.model.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=self.imgsz, conf=self.conf,
                               device=self.device, half=self.device != "cpu", verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))


def box_survival(teacher: Teacher, src_rgb: np.ndarray, gen_rgb: np.ndarray, boxes: list[list[float]],
                 min_side: int = 24, iou_src: float = 0.5, iou_gen: float = 0.4) -> tuple[float, int]:
    """Fraction of teacher-verifiable labeled boxes still found in the generated image, and their count."""
    from failforge.eval.matching import iou_matrix

    big = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) >= min_side]
    if not big:
        return 1.0, 0
    gt = np.array(big)
    in_src = iou_matrix(gt, teacher.boxes(src_rgb)).max(axis=1, initial=0) >= iou_src
    if not in_src.any():
        return 1.0, 0
    in_gen = iou_matrix(gt[in_src], teacher.boxes(gen_rgb)).max(axis=1, initial=0) >= iou_gen
    return float(in_gen.mean()), int(in_src.sum())


class QualityFilter:
    COND_PROMPTS = {
        "night": "a dashcam photo of a road at night, dark sky, artificial lights",
        "dusk": "a dashcam photo at dusk or dawn, low sun, twilight sky",
        "rain": "a dashcam photo in the rain, wet road",
        "snow": "a dashcam photo of a snowy road",
        "fog": "a dashcam photo in dense fog",
        "glare": "a dashcam photo at night with strong headlight glare",
    }
    REFS = ["a dashcam photo in bright daylight, blue sky", "a dashcam photo on an overcast grey day"]

    def __init__(self, embedder, cfg, teacher: Teacher | None = None) -> None:
        self.emb = embedder
        self.cfg = cfg
        self.teacher = teacher
        self._cond = {k: embedder.texts([v, *self.REFS]) for k, v in self.COND_PROMPTS.items()}

    def check(self, src_rgb: np.ndarray, gen_rgb: np.ndarray, boxes: list[list[float]], prompt: str,
              condition: str) -> dict:
        small = cv2.resize(gen_rgb, (398, 224), interpolation=cv2.INTER_AREA)
        e = self.emb.images([small])
        clip_score = float((e @ self.emb.texts([prompt]).T)[0, 0])
        p_target = float(self.emb.zero_shot(e, self._cond[condition])[0, 0])
        if self.teacher is not None:
            box_pass, n_judged = box_survival(self.teacher, src_rgb, gen_rgb, boxes, iou_gen=self.cfg.teacher_iou)
        else:
            st = [v for v in box_structure(src_rgb, gen_rgb, boxes) if not np.isnan(v)]
            box_pass = float(np.mean([v >= self.cfg.min_box_structure for v in st])) if st else 1.0
            n_judged = len(st)
        reasons = []
        if clip_score < self.cfg.min_clip_score:
            reasons.append("prompt")
        if p_target < self.cfg.min_target_prob:
            reasons.append("condition")
        if box_pass < self.cfg.min_box_pass:
            reasons.append("structure")
        return {"clip_score": clip_score, "p_target": p_target, "box_pass": box_pass, "boxes_judged": n_judged,
                "accepted": not reasons, "reasons": reasons, "embedding": e[0]}
