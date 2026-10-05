"""YOLO views of manifests.

Ultralytics finds a label by swapping ``/images/`` for ``/labels/`` in the image path, so every
image used for training is placed into a content store ``datasets/store_<imgsz>/images/<id>.jpg``
with its label next door in ``datasets/store_<imgsz>/labels/<id>.txt``. A training set is then just
a text file listing store paths, so the five ablation arms share one store instead of copying
5,000 images five times.

Store images are **pre-resized** so their long side equals the training resolution: BDD frames are
1280x720 and training runs at 640, so every epoch of every arm would otherwise decode 4x more
pixels than it uses. Labels are normalized, so they are unaffected. (Evaluation always predicts on
the original images, at their original resolution, so scores are in native pixel coordinates.)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import yaml

from failforge import paths
from failforge.config import CLASSES
from failforge.data.schema import Sample, resolve_image


def store_dir(imgsz: int = 0) -> Path:
    return paths.datasets_dir() / (f"store_{imgsz}" if imgsz else "store")


def label_lines(s: Sample) -> list[str]:
    out = []
    for b in s.boxes:
        cx = (b.x1 + b.x2) / 2 / s.width
        cy = (b.y1 + b.y2) / 2 / s.height
        w = (b.x2 - b.x1) / s.width
        h = (b.y2 - b.y1) / s.height
        out.append(f"{b.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:  # FAT32 / cross-device: fall back to a copy
        shutil.copy2(src, dst)


def _place(src: Path, dst: Path, s: Sample, imgsz: int) -> None:
    if imgsz <= 0 or max(s.width, s.height) <= imgsz:
        _link_or_copy(src, dst)
        return
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(src)
    h, w = im.shape[:2]
    r = imgsz / max(h, w)
    small = cv2.resize(im, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    tmp = dst.with_name(dst.stem + ".tmp.jpg")
    cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 93])
    tmp.replace(dst)


def materialize(samples: list[Sample], manifest_path: Path | None = None, root: Path | None = None,
                imgsz: int = 0) -> list[Path]:
    """Place images + labels in the store; returns the store image paths in input order."""
    root = root or store_dir(imgsz)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    out = []
    for s in samples:
        src = resolve_image(s, manifest_path)
        dst = root / "images" / f"{s.id}.jpg"
        _place(src, dst, s, imgsz)
        lbl = root / "labels" / f"{s.id}.txt"
        text = "\n".join(label_lines(s)) + ("\n" if s.boxes else "")
        if not lbl.is_file() or lbl.read_text(encoding="utf-8") != text:
            lbl.write_text(text, encoding="utf-8")
        out.append(dst)
    return out


def write_list(paths_: list[Path], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(p.resolve().as_posix() for p in paths_) + "\n", encoding="utf-8")
    return path


def write_data_yaml(path: Path, train_list: Path, val_list: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "path": path.parent.resolve().as_posix(),
        "train": train_list.resolve().as_posix(),
        "val": val_list.resolve().as_posix(),
        "names": dict(enumerate(CLASSES)),
    }, sort_keys=False), encoding="utf-8")
    return path


def build_dataset(name_dir: Path, train: list[Sample], val: list[Sample], imgsz: int = 0) -> Path:
    """Materialize train/val sets (pre-resized to ``imgsz``) and write ``data.yaml`` for Ultralytics."""
    tr = write_list(materialize(train, imgsz=imgsz), name_dir / "train.txt")
    va = write_list(materialize(val, imgsz=imgsz), name_dir / "val.txt")
    return write_data_yaml(name_dir / "data.yaml", tr, va)
