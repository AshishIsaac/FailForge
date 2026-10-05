"""The detector under test: an Ultralytics YOLO11 wrapper with a small, explicit surface.

Everything else in FailForge talks to :class:`Detector` (``train`` / ``predict``), so swapping in
RT-DETR or another family is one class. Training budgets are fixed (no early stopping, the *last*
checkpoint is used, not the best on day-only validation) so that every ablation arm gets exactly
the same optimization budget and no arm benefits from checkpoint selection.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from failforge import paths
from failforge.config import DetectorConfig, TrainConfig
from failforge.data.schema import Sample
from failforge.detections import Detections

log = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def safe_workers(requested: int) -> int:
    """Dataloader workers capped by RAM: on Windows each worker is a full process with its own copy of
    torch (about 1.5-2.5 GB each with mosaic buffers), and 16 GB machines hit the commit limit at 4."""
    try:
        import psutil

        gb = psutil.virtual_memory().total / 2**30
    except ImportError:
        return requested
    return max(1, min(requested, int(gb // 6)))


def pretrained_weights(name: str) -> str:
    """COCO weights live in the workspace model cache (downloaded on first use)."""
    p = Path(name)
    if p.is_file():
        return str(p)
    dest = paths.models_dir() / "ultralytics" / p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(dest))
    return str(dest)


class Detector:
    def __init__(self, weights: str | Path, cfg: DetectorConfig) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.weights = str(weights)
        self.device = resolve_device(cfg.device)
        self.model = YOLO(self.weights)

    # ------------------------------------------------------------------ training

    @staticmethod
    def train(data_yaml: Path, out_dir: Path, init_weights: str, dcfg: DetectorConfig, tcfg: TrainConfig,
              epochs: int, lr0: float, name: str = "train", warmup_epochs: float = 3.0, progress=None) -> Path:
        """Train / fine-tune; returns ``out_dir/weights.pt`` (the final-epoch checkpoint).

        ``progress(epoch, epochs, metrics)`` is called after every epoch.
        """
        from ultralytics import YOLO

        out_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(pretrained_weights(init_weights))
        if progress is not None:
            def _cb(trainer) -> None:
                m = {k: float(v) for k, v in (trainer.metrics or {}).items() if isinstance(v, (int, float))}
                # Ultralytics fires this once more after the final validation: clamp to the budget.
                progress(min(trainer.epoch + 1, trainer.epochs), trainer.epochs, m)

            model.add_callback("on_fit_epoch_end", _cb)
        model.train(
            data=str(data_yaml), epochs=epochs, imgsz=dcfg.imgsz, batch=dcfg.batch, workers=safe_workers(dcfg.workers),
            device=resolve_device(dcfg.device), project=str(out_dir), name=name, exist_ok=True,
            optimizer=tcfg.optimizer, lr0=lr0, cos_lr=tcfg.cos_lr, patience=tcfg.patience or epochs + 1,
            close_mosaic=min(tcfg.close_mosaic, max(epochs - 1, 0)), seed=tcfg.seed, deterministic=True,
            amp=dcfg.amp, plots=False, verbose=False, warmup_epochs=warmup_epochs, cache=dcfg.cache or False,
            # Photometric jitter is part of every arm's recipe (the Ultralytics default HSV jitter),
            # so the "augment" arm measures *targeted* extra data, not the presence of jitter itself.
        )
        last = out_dir / name / "weights" / "last.pt"
        if not last.is_file():
            raise RuntimeError(f"training produced no checkpoint at {last}")
        dst = out_dir / "weights.pt"
        shutil.copy2(last, dst)
        return dst

    # ------------------------------------------------------------------ inference

    def predict(self, samples: list[Sample], conf: float | None = None, batch: int = 16,
                progress=None) -> dict[str, Detections]:
        conf = self.cfg.conf if conf is None else conf
        out: dict[str, Detections] = {}
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            results = self.model.predict([s.image for s in chunk], imgsz=self.cfg.imgsz, conf=conf, iou=self.cfg.iou,
                                         max_det=self.cfg.max_det, device=self.device, half=self.device != "cpu",
                                         verbose=False, batch=len(chunk))
            for s, r in zip(chunk, results):
                b = r.boxes
                if b is None or len(b) == 0:
                    out[s.id] = Detections.empty()
                    continue
                out[s.id] = Detections(b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                                       b.cls.cpu().numpy().astype(np.int64))
            if progress:
                progress(min(i + batch, len(samples)), len(samples))
        return out
