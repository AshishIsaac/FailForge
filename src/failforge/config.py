"""Typed configuration.

Every option is a field of a nested dataclass, with a default and a type. YAML files are
loaded strictly: an unknown key raises :class:`ConfigError`, so a typo in a config fails at
start-up instead of silently falling back to a default. ``--set a.b=value`` overrides any
field from the command line (the value is parsed as YAML: ``--set synthesis.count=300``).

A profile file may start with ``extends: other.yaml`` (resolved relative to itself); the
child's keys are deep-merged over the parent's.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml


class ConfigError(ValueError):
    """Raised for malformed or unknown configuration values."""


# Detection classes used throughout (BDD100K labels are mapped onto these in data/bdd.py).
CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "two_wheeler", "traffic_light", "traffic_sign"]


@dataclass
class DataConfig:
    hf_repo: str = "dgural/bdd100k"  # 10k BDD100K images with timeofday / weather / scene tags
    hf_revision: str = "main"
    # Split sizes. The golden set is frozen on first creation (golden.lock.json) and never changes.
    golden_day: int = 300
    golden_night: int = 600
    golden_dawn: int = 200
    probe_day: int = 100  # "production triage" sample: labeled, used for mining / drift, never trained on
    probe_night: int = 300
    probe_dawn: int = 100
    val_day: int = 400  # model selection during training (day only: the baseline team never saw night)
    train_day: int = 0  # 0 = every remaining daytime image
    seed: int = 2026
    min_box_px: float = 4.0  # boxes thinner than this (at native resolution) are dropped as label noise
    subset: int = 0  # download only this many images (stratified; CI). 0 = all 10,000


@dataclass
class DetectorConfig:
    weights: str = "yolo11n.pt"  # COCO-pretrained starting point for the baseline
    imgsz: int = 640
    batch: int = 16
    workers: int = 4
    # Ultralytics image cache during training: "" (off), "disk" (.npy files) or "ram" (~4 GB at 640).
    # Off by default: the content store already holds images pre-resized to imgsz (see data/yolo.py).
    cache: str = ""
    device: str = "auto"  # auto | cpu | 0 | 1 ...
    conf: float = 0.001  # evaluation threshold (mAP needs the full PR curve)
    iou: float = 0.6  # NMS IoU
    max_det: int = 300
    mine_conf: float = 0.25  # operating point for failure mining (what a deployed model would emit)
    amp: bool = True


@dataclass
class TrainConfig:
    baseline_epochs: int = 30
    finetune_epochs: int = 12  # every ablation arm fine-tunes the baseline for exactly this long
    patience: int = 0  # 0 = no early stopping (equal budgets across arms)
    lr0: float = 0.002
    finetune_lr0: float = 0.001
    optimizer: str = "AdamW"
    cos_lr: bool = True
    close_mosaic: int = 3
    seed: int = 0
    # Fraction of the day training set mixed into each fine-tuning arm (1.0 = all of it). Keeping the
    # original data in every arm guards against forgetting and keeps the arms comparable.
    replay_fraction: float = 1.0


@dataclass
class MiningConfig:
    iou_match: float = 0.5  # TP threshold at the operating point
    iou_background: float = 0.1  # TIDE: below this overlap with every GT a FP is "background"
    context: float = 0.6  # crops are expanded by this fraction of the box size (context matters for "night")
    min_crop_px: int = 12
    max_errors: int = 6000
    tp_sample: int = 3000  # true positives embedded alongside errors, to compute per-cluster failure lift


@dataclass
class AnalysisConfig:
    embedder: str = "openai/clip-vit-base-patch32"
    batch: int = 64
    reducer: str = "umap"  # umap | tsne | pca
    n_components: int = 10  # clustering space (2-D is computed separately for plots)
    n_neighbors: int = 30
    min_dist: float = 0.0
    min_cluster_size: int = 40
    min_samples: int = 10
    photometric_weight: float = 0.5  # weight of the standardized photometric attributes next to CLIP
    top_modes: int = 3  # failure modes handed to synthesis
    min_lift: float = 1.2  # a cluster is a failure mode only if its error rate beats the population's
    seed: int = 0


@dataclass
class DriftConfig:
    reference_samples: int = 500
    window: int = 300
    permutations: int = 500
    alpha: float = 0.01


@dataclass
class SynthesisConfig:
    backend: str = "controlnet"  # controlnet | augment (classical relighting; CI / CPU stand-in)
    count: int = 500  # accepted images to produce (the same N is used by the augment and real arms)
    oversample: float = 2.0  # candidates generated per accepted image (filters reject the rest)
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnets: list[str] = field(default_factory=lambda: ["lllyasviel/control_v11p_sd15_canny",
                                                            "lllyasviel/control_v11f1p_sd15_depth"])
    control_scales: list[float] = field(default_factory=lambda: [0.9, 0.6])
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    width: int = 640
    height: int = 360
    steps: int = 15
    strength: float = 0.75  # img2img: how far from the source photo (structure is pinned by ControlNet)
    guidance: float = 6.5
    # img2img starts from a classically relit copy of the photo (augment.py) instead of the daylight
    # original: the init image's luminance dominates img2img, so without this "night" stays dusk.
    # ControlNet still sees the original's edges and depth.
    relit_init: bool = True
    batch: int = 1
    cpu_offload: bool = True  # fits SD1.5 + 2 ControlNets in 6 GB
    seed: int = 1234
    # Quality filters
    min_clip_score: float = 0.22  # CLIP image-prompt cosine
    min_target_prob: float = 0.6  # CLIP zero-shot probability that the image shows the target condition
    teacher: str = "yolo11s.pt"  # independent COCO detector verifying that labeled objects survived ("" = edge check)
    teacher_iou: float = 0.4
    min_box_pass: float = 0.5  # fraction of teacher-verifiable boxes that must survive
    min_box_structure: float = 0.25  # fallback without a teacher: per-box blurred-edge correlation
    kid_reference: int = 300  # real target-condition images for the batch KID diagnostic


@dataclass
class GateConfig:
    min_target_gain: float = 0.02  # target slice mAP@50:95 must improve by >= 2 points
    max_overall_drop: float = 0.005  # golden overall mAP may drop by at most 0.5 points
    max_slice_drop: float = 0.02  # no golden slice may drop by more than 2 points
    require_ci: bool = True  # the target gain's bootstrap 95% CI lower bound must be > 0
    bootstrap: int = 1000
    target_slices: list[str] = field(default_factory=lambda: ["night"])


@dataclass
class TriggerConfig:
    # The loop runs when drift is detected or a golden slice is below its SLO (mAP@50:95).
    slo: dict[str, float] = field(default_factory=lambda: {"night": 0.20, "overall": 0.20})
    force: bool = False


@dataclass
class BenchConfig:
    runs: int = 200
    warmup: int = 20
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8])
    onnx: bool = True


@dataclass
class Config:
    profile: str = "full"
    data: DataConfig = field(default_factory=DataConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    arms: list[str] = field(default_factory=lambda: ["baseline_ft", "augment", "synthetic", "real"])
    # Subsample every split to at most this many images (0 = off). For smoke tests / CI.
    limit_images: int = 0
    log_level: str = "INFO"

    def fingerprint(self, *sections: str) -> str:
        """Stable hash of the given sections or dotted fields (all if none): used to cache pipeline stages."""
        d = to_dict(self)
        if sections:
            picked = {}
            for s in sections:
                node = d
                for part in s.split("."):
                    node = node[part]
                picked[s] = node
            d = picked
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- loading


def _check_type(value: Any, tp: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if tp is Any:
        return value
    if origin in (Union, types.UnionType):
        for arg in typing.get_args(tp):
            try:
                return _check_type(value, arg, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: {value!r} does not match {tp}")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        (arg,) = typing.get_args(tp)
        return [_check_type(v, arg, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {value!r}")
        kt, vt = typing.get_args(tp)
        return {_check_type(k, kt, path): _check_type(v, vt, f"{path}.{k}") for k, v in value.items()}
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {value!r}")
        return _from_dict(tp, value, path)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp is str and value is None:
        return ""
    if tp in (int, float) and isinstance(value, bool):
        raise ConfigError(f"{path}: expected {tp.__name__}, got a boolean")
    if not isinstance(value, tp):
        raise ConfigError(f"{path}: expected {getattr(tp, '__name__', tp)}, got {value!r}")
    return value


def _from_dict(cls: type, data: dict, path: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in {path or 'config'}; valid: {sorted(names)}")
    obj = cls()
    for k, v in data.items():
        setattr(obj, k, _check_type(v, hints[k], f"{path}.{k}" if path else k))
    return obj


def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def from_dict(data: dict) -> Config:
    cfg = _from_dict(Config, data)
    validate(cfg)
    return cfg


def apply_override(data: dict, expr: str) -> None:
    if "=" not in expr:
        raise ConfigError(f"override {expr!r} must look like section.key=value")
    key, raw = expr.split("=", 1)
    parts = key.strip().split(".")
    node = data
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {expr!r}: {p} is not a section")
    node[parts[-1]] = yaml.safe_load(raw) if raw.strip() else ""


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(p: Path, seen: tuple = ()) -> dict:
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    if p.resolve() in seen:
        raise ConfigError(f"circular 'extends' at {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    parent = data.pop("extends", None)
    if parent:
        data = _deep_merge(_read_yaml((p.parent / parent), (*seen, p.resolve())), data)
    return data


def resolve_profile(name_or_path: str | Path | None) -> Path | None:
    """``quick`` -> configs/quick.yaml in the repository; a path is returned as is."""
    if not name_or_path:
        return None
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p
    from failforge.paths import REPO

    return REPO / "configs" / f"{name_or_path}.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    data: dict = {}
    p = resolve_profile(path)
    if p is not None:
        data = _read_yaml(p)
    for o in overrides or []:
        apply_override(data, o)
    return from_dict(data)


def validate(cfg: Config) -> None:
    def one_of(path: str, value: str, options: tuple) -> None:
        if value not in options:
            raise ConfigError(f"{path}: {value!r} is not one of {options}")

    one_of("analysis.reducer", cfg.analysis.reducer, ("umap", "tsne", "pca"))
    one_of("synthesis.backend", cfg.synthesis.backend, ("controlnet", "augment"))
    for a in cfg.arms:
        one_of("arms[]", a, ("baseline_ft", "augment", "synthetic", "real"))
    if len(cfg.synthesis.controlnets) != len(cfg.synthesis.control_scales):
        raise ConfigError("synthesis.controlnets and synthesis.control_scales must have the same length")
    if cfg.synthesis.width % 8 or cfg.synthesis.height % 8:
        raise ConfigError("synthesis.width/height must be multiples of 8")
    if not 0 < cfg.synthesis.strength <= 1:
        raise ConfigError("synthesis.strength must be in (0, 1]")
    if cfg.synthesis.oversample < 1:
        raise ConfigError("synthesis.oversample must be >= 1")
    if not 0 <= cfg.train.replay_fraction <= 1:
        raise ConfigError("train.replay_fraction must be in [0, 1]")
    if cfg.synthesis.count < 0:
        raise ConfigError("synthesis.count must be >= 0")
