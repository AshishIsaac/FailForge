"""Where things live.

Everything FailForge downloads or produces goes under one *workspace* folder, so uninstalling is
deleting one folder and nothing is scattered across the machine. The default is ``workspace/``
next to the code; set ``FAILFORGE_HOME`` (or ``python run.py --home DIR``) to put it elsewhere,
e.g. on a bigger or faster internal drive.

    workspace/
      data/bdd100k/          raw download (images + FiftyOne labels)
      data/manifests/        normalized labels, attributes and the frozen splits (golden.lock.json)
      datasets/              YOLO views: labels/*.txt next to images, per-arm image lists
      models/                hub caches (Hugging Face, Ultralytics weights)
      runs/<run_id>/         one active-learning loop: every stage's outputs, metrics, report
      registry/              promoted models (champion.json + weights)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def home() -> Path:
    env = os.environ.get("FAILFORGE_HOME", "").strip()
    return Path(env).expanduser().resolve() if env else REPO / "workspace"


def data_dir() -> Path:
    return home() / "data"


def raw_dir() -> Path:
    return data_dir() / "bdd100k"


def manifests_dir() -> Path:
    return data_dir() / "manifests"


def datasets_dir() -> Path:
    return home() / "datasets"


def models_dir() -> Path:
    return home() / "models"


def runs_dir() -> Path:
    return home() / "runs"


def registry_dir() -> Path:
    return home() / "registry"


def reference_dir() -> Path:
    """The committed reference run (results of the full loop on the author's machine)."""
    return REPO / "reports" / "reference"


def configure_caches() -> None:
    """Point the Hugging Face / Ultralytics caches into the workspace (unless the user set them)."""
    m = models_dir()
    # Ultralytics silently falls back to ./Ultralytics when its settings folder does not exist yet.
    (m / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(m / "hf"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(m / "ultralytics"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
