"""Deterministic, leakage-free splits and the frozen golden set.

    golden      frozen evaluation set, stratified over time of day x weather. Created once and pinned
                by ``golden.lock.json`` (ids + label digest). Every model, now and later, is scored on
                exactly these images; if the lock and the data disagree, FailForge refuses to run.
    probe       a labeled "production triage" sample (mostly night / dawn, some day): what a QA team
                would label from the deployment stream. Failure mining and drift run on it. Never trained on.
    val_day     day-only validation for training (the baseline team only had daytime data).
    train_day   daytime training images.
    target_pool remaining night / dawn images. Only the *real* upper-bound arm trains on it, and the
                KID diagnostic uses it as the reference distribution for synthetic images.

Assignment is by hashing a *group key* (the BDD video id prefix) with the seed, so images that could
come from the same drive never straddle two splits. Stratification is proportional: within each
(time of day, weather) cell, groups are ranked by hash and the globally smallest normalized ranks
are taken, which reproduces the cell proportions without rounding drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from failforge import paths
from failforge.config import DataConfig
from failforge.data.schema import Sample, manifest_digest, read_manifest, write_manifest

log = logging.getLogger(__name__)

SPLITS = ("golden", "probe", "val_day", "train_day", "target_pool")
LOCK_NAME = "golden.lock.json"


class GoldenLockError(RuntimeError):
    pass


def group_key(s: Sample) -> str:
    return s.id.split("-")[0]


def _h(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def stratified_take(samples: list[Sample], n: int, seed: int, strata=("timeofday", "weather")) -> list[Sample]:
    """Proportionally stratified, group-aware, deterministic selection of about ``n`` samples."""
    if n <= 0 or not samples:
        return []
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[group_key(s)].append(s)
    cells: dict[tuple, list[str]] = defaultdict(list)
    for g, members in groups.items():
        cells[tuple(members[0].attrs.get(k, "") for k in strata)].append(g)
    keyed = []
    for gs in cells.values():
        gs.sort(key=lambda g: _h(seed, g))
        keyed += [((i + 0.5) / len(gs), _h(seed + 1, g), g) for i, g in enumerate(gs)]
    keyed.sort()
    out: list[Sample] = []
    for _, _, g in keyed:
        if len(out) >= n:
            break
        out += groups[g]
    return out


def _by_tod(samples: list[Sample], tod: str) -> list[Sample]:
    return [s for s in samples if s.attrs.get("timeofday") == tod]


def make_splits(all_samples: list[Sample], cfg: DataConfig, lock_path: Path | None = None) -> dict[str, list[Sample]]:
    lock_path = lock_path or paths.manifests_dir() / LOCK_NAME
    usable = [s for s in all_samples if s.attrs.get("timeofday") in ("day", "night", "dawn")]
    by_id = {s.id: s for s in usable}

    # Golden: frozen.
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        missing = [i for i in lock["ids"] if i not in by_id]
        if missing:
            raise GoldenLockError(f"{len(missing)} golden images are missing from the data (e.g. {missing[:3]}). "
                                  f"The golden set is frozen by {lock_path}; restore the data or delete the lock "
                                  "deliberately (all previous scores become incomparable).")
        golden = [by_id[i] for i in lock["ids"]]
        if manifest_digest(golden) != lock["digest"]:
            raise GoldenLockError(f"golden labels changed since {lock['created']} (digest mismatch with {lock_path}).")
    else:
        golden = (stratified_take(_by_tod(usable, "day"), cfg.golden_day, cfg.seed)
                  + stratified_take(_by_tod(usable, "night"), cfg.golden_night, cfg.seed)
                  + stratified_take(_by_tod(usable, "dawn"), cfg.golden_dawn, cfg.seed))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "source": cfg.hf_repo, "seed": cfg.seed,
            "n": len(golden), "digest": manifest_digest(golden), "ids": sorted(s.id for s in golden),
        }, indent=1), encoding="utf-8")
        log.info("froze a new golden set: %d images -> %s", len(golden), lock_path)
    golden = sorted(golden, key=lambda s: s.id)  # canonical order: paired comparisons line up across runs

    taken_groups = {group_key(s) for s in golden}

    def rest(tod: str) -> list[Sample]:
        return [s for s in _by_tod(usable, tod) if group_key(s) not in taken_groups]

    probe: list[Sample] = []
    for tod, n in (("day", cfg.probe_day), ("night", cfg.probe_night), ("dawn", cfg.probe_dawn)):
        probe += stratified_take(rest(tod), n, cfg.seed + 10)
    taken_groups |= {group_key(s) for s in probe}

    val = stratified_take(rest("day"), cfg.val_day, cfg.seed + 20)
    taken_groups |= {group_key(s) for s in val}
    day_left = rest("day")
    train = stratified_take(day_left, cfg.train_day, cfg.seed + 30) if cfg.train_day > 0 else day_left
    taken_groups |= {group_key(s) for s in train}
    target_pool = rest("night") + rest("dawn")
    target_pool.sort(key=lambda s: _h(cfg.seed + 40, s.id))

    splits = {"golden": golden, "probe": probe, "val_day": val, "train_day": train, "target_pool": target_pool}
    check_disjoint(splits)
    return splits


def check_disjoint(splits: dict[str, list[Sample]]) -> None:
    owner: dict[str, str] = {}
    for name, ss in splits.items():
        for s in ss:
            g = group_key(s)
            if owner.setdefault(g, name) != name:
                raise AssertionError(f"leakage: group {g} is in both {owner[g]} and {name}")


def write_splits(splits: dict[str, list[Sample]], out_dir: Path | None = None) -> dict[str, dict]:
    out_dir = out_dir or paths.manifests_dir()
    summary = {}
    for name, ss in splits.items():
        write_manifest(out_dir / f"{name}.jsonl", ss)
        summary[name] = describe(ss)
    (out_dir / "splits_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def describe(ss: list[Sample]) -> dict:
    tod: dict[str, int] = defaultdict(int)
    weather: dict[str, int] = defaultdict(int)
    for s in ss:
        tod[s.attrs.get("timeofday", "?")] += 1
        weather[s.attrs.get("weather", "?")] += 1
    return {"images": len(ss), "boxes": sum(len(s.boxes) for s in ss), "timeofday": dict(tod), "weather": dict(weather)}


def load_split(name: str, limit: int = 0, mdir: Path | None = None) -> list[Sample]:
    p = (mdir or paths.manifests_dir()) / f"{name}.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found - run `failforge data prepare` first")
    ss = read_manifest(p)
    if limit and len(ss) > limit:
        # Deterministic, still stratified subsample (CI / smoke runs).
        ss = stratified_take(ss, limit, 7)[:limit]
    return ss
