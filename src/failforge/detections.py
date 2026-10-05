"""Model output for one image, plus (de)serialization of a whole prediction set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) float32 xyxy, pixels of the original image
    scores: np.ndarray  # (N,) float32
    classes: np.ndarray  # (N,) int64

    @staticmethod
    def empty() -> Detections:
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.scores)

    def above(self, conf: float) -> Detections:
        k = self.scores >= conf
        return Detections(self.boxes[k], self.scores[k], self.classes[k])

    def top(self, n: int) -> Detections:
        if len(self) <= n:
            return self
        k = np.argsort(-self.scores, kind="stable")[:n]
        return Detections(self.boxes[k], self.scores[k], self.classes[k])


def save_predictions(path: Path, preds: dict[str, Detections]) -> None:
    """One compressed .npz: concatenated arrays + per-image offsets."""
    ids = sorted(preds)
    counts = np.array([len(preds[i]) for i in ids], dtype=np.int64)

    def cat(f, shape, dt):
        return np.concatenate([f(preds[i]) for i in ids]).astype(dt) if ids else np.zeros(shape, dt)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids), counts=counts,
                        boxes=cat(lambda d: d.boxes.reshape(-1, 4), (0, 4), np.float32),
                        scores=cat(lambda d: d.scores, (0,), np.float32),
                        classes=cat(lambda d: d.classes, (0,), np.int64))


def load_predictions(path: Path) -> dict[str, Detections]:
    z = np.load(path, allow_pickle=False)
    out, o = {}, 0
    for i, n in zip(z["ids"].tolist(), z["counts"].tolist()):
        out[str(i)] = Detections(z["boxes"][o:o + n], z["scores"][o:o + n], z["classes"][o:o + n])
        o += n
    return out
