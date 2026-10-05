"""BDD100K: download and normalization.

The data comes from the Hugging Face mirror ``dgural/bdd100k``: 10,000 real 1280x720 dashcam
frames (the BDD100K validation split) with detection boxes *and* per-image ``timeofday`` /
``weather`` / ``scene`` tags, about 700 MB, no account needed. The tags are what make the
experiment controlled: we know which images are night, rain or snow, so the baseline can be
trained on daytime only and the drift is real, known and measurable.

BDD100K is distributed for non-commercial research and education under the Berkeley license
(see docs/data.md). FailForge downloads it at first run; it is never committed to git.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from failforge import paths
from failforge.config import CLASSES, DataConfig
from failforge.data.schema import Box, Sample, write_manifest

log = logging.getLogger(__name__)

# BDD100K label -> FailForge class. Labels not listed (train, trailer, other vehicle) are dropped.
LABEL_MAP = {
    "pedestrian": "pedestrian",
    "other person": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
TIMEOFDAY = {"daytime": "day", "night": "night", "dawn/dusk": "dawn", "undefined": "undefined"}


def download(cfg: DataConfig, dest: Path | None = None) -> Path:
    """Fetch images + labels (resumable: already downloaded files are skipped)."""
    from huggingface_hub import snapshot_download

    dest = dest or paths.raw_dir()
    dest.mkdir(parents=True, exist_ok=True)
    patterns = ["data/*", "samples.json", "metadata.json", "README.md"]
    if cfg.subset > 0:
        # CI / smoke: the labels first, then a deterministic, time-of-day-stratified subset of the images.
        from huggingface_hub import hf_hub_download

        from failforge.data.splits import stratified_take

        sj = Path(hf_hub_download(cfg.hf_repo, "samples.json", repo_type="dataset", revision=cfg.hf_revision,
                                  local_dir=str(dest)))
        alls = [s for s in parse_fiftyone(sj, dest / "data", cfg.min_box_px) if s.attrs["timeofday"] != "undefined"]
        pick: list[Sample] = []
        for tod, frac in (("day", 0.5), ("night", 0.35), ("dawn", 0.15)):
            pick += stratified_take([s for s in alls if s.attrs["timeofday"] == tod], int(cfg.subset * frac), cfg.seed)
        patterns = ["samples.json", *(f"data/{Path(s.image).name}" for s in pick)]
        log.info("downloading a %d-image subset of %s -> %s", len(pick), cfg.hf_repo, dest)
    else:
        log.info("downloading %s (about 700 MB, once) -> %s", cfg.hf_repo, dest)
    snapshot_download(repo_id=cfg.hf_repo, repo_type="dataset", revision=cfg.hf_revision, local_dir=str(dest),
                      allow_patterns=patterns,
                      max_workers=64)  # 10k small files: latency-bound, so many parallel requests
    n = sum(1 for _ in (dest / "data").glob("*.jpg"))
    log.info("%d images on disk", n)
    return dest


def _label(v) -> str:
    if isinstance(v, dict):
        return str(v.get("label") or "undefined")
    return str(v) if v else "undefined"


def parse_fiftyone(samples_json: Path, image_root: Path, min_box_px: float = 4.0) -> list[Sample]:
    """FiftyOne export -> Samples. Boxes there are normalized ``[x, y, w, h]`` (top-left origin)."""
    raw = json.loads(samples_json.read_text(encoding="utf-8"))["samples"]
    out: list[Sample] = []
    dropped = 0
    for r in raw:
        md = r.get("metadata") or {}
        w, h = int(md.get("width") or 1280), int(md.get("height") or 720)
        fp = Path(r["filepath"])
        boxes = []
        for d in (r.get("detections") or {}).get("detections", []) or []:
            name = LABEL_MAP.get(d.get("label", ""))
            if name is None:
                continue
            bx, by, bw, bh = d["bounding_box"]
            x1, y1 = max(bx * w, 0.0), max(by * h, 0.0)
            x2, y2 = min((bx + bw) * w, w), min((by + bh) * h, h)
            if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
                dropped += 1
                continue
            boxes.append(Box(CLASS_INDEX[name], round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                             occluded=bool(d.get("occluded", False)), truncated=bool(d.get("truncated", False))))
        attrs = {"timeofday": TIMEOFDAY.get(_label(r.get("timeofday")), "undefined"),
                 "weather": _label(r.get("weather")).replace(" ", "_"),
                 "scene": _label(r.get("scene")).replace(" ", "_"),
                 "source": "real"}
        out.append(Sample(id=fp.stem, image=str((image_root / fp.name).resolve()), width=w, height=h,
                          boxes=boxes, attrs=attrs))
    log.info("parsed %d images, %d boxes (%d tiny boxes dropped)", len(out), sum(len(s.boxes) for s in out), dropped)
    return out


def build_manifest(cfg: DataConfig, raw: Path | None = None, out: Path | None = None) -> Path:
    raw = raw or paths.raw_dir()
    out = out or paths.manifests_dir() / "all.jsonl"
    sj = raw / "samples.json"
    if not sj.is_file():
        raise FileNotFoundError(f"{sj} not found - run `failforge data download` first")
    samples = [s for s in parse_fiftyone(sj, raw / "data", cfg.min_box_px) if Path(s.image).is_file()]
    write_manifest(out, samples)
    return out
