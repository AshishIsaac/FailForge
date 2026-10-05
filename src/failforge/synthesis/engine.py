"""Targeted data synthesis: failure modes -> accepted, labeled training images.

Budget allocation: the image budget N is split across the addressable failure modes in proportion
to their *excess errors*. For each mode, source images are daytime training photos ranked by how
many objects of the mode's dominant classes they contain (so a "night pedestrian misses" mode gets
pedestrian-rich streets re-rendered at night). Each candidate is generated, quality-checked and
either accepted (written with the source's labels) or rejected with a reason, until each mode's
quota is filled or its candidate budget (quota x ``oversample``) is spent.

Every decision is appended to ``qc.jsonl`` as it happens, so a long generation run can be
interrupted and resumed exactly where it stopped.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from failforge.config import CLASSES, SynthesisConfig
from failforge.data.schema import Sample, write_manifest
from failforge.synthesis import augment
from failforge.synthesis.prompts import CONDITION_ATTRS, make_prompt

log = logging.getLogger(__name__)


def allocate(targets: list[dict], count: int) -> list[dict]:
    """Split ``count`` across targets proportionally to excess errors (largest remainder)."""
    if not targets or count <= 0:
        return []
    w = np.array([max(t.get("excess_errors", 1.0), 1e-6) for t in targets], dtype=np.float64)
    raw = w / w.sum() * count
    q = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - q))[: count - q.sum()]:
        q[i] += 1
    return [{**t, "quota": int(n)} for t, n in zip(targets, q) if n > 0]


def rank_sources(train: list[Sample], classes: list[str], seed: int) -> list[Sample]:
    want = {CLASSES.index(c) for c in classes if c in CLASSES}
    rng = random.Random(seed)
    scored = [(sum(b.cls in want for b in s.boxes) + rng.random(), s) for s in train if s.boxes]
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored]


def plan_candidates(alloc: list[dict], train: list[Sample], cfg: SynthesisConfig, oversample: float) -> list[dict]:
    plan = []
    for ti, t in enumerate(alloc):
        n_cand = int(math.ceil(t["quota"] * oversample))
        srcs = rank_sources(train, t["classes"], cfg.seed + ti)
        rng = random.Random(cfg.seed * 7919 + ti)
        for k in range(n_cand):
            s = srcs[k % len(srcs)]
            pos, neg = make_prompt(t["condition"], s.attrs.get("scene", "city_street"), rng)
            plan.append({"target": ti, "condition": t["condition"], "source": s.id, "prompt": pos, "negative": neg,
                         "seed": cfg.seed + 100003 * ti + k})
    return plan


def _gallery_tile(src: np.ndarray, ctrl: list[np.ndarray], gen: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    w, h = 384, 216
    tiles = [cv2.resize(src, (w, h))] + [cv2.resize(c, (w, h)) for c in ctrl[:2]]
    g = gen.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(g, (int(x1), int(y1)), (int(x2), int(y2)), (80, 255, 80), 2)
    tiles.append(cv2.resize(g, (w, h)))
    return cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR)


def run(backend: str, targets: list[dict], train: list[Sample], out_dir: Path, cfg: SynthesisConfig, count: int,
        generator=None, qfilter=None, embedder=None, id_prefix: str = "syn",
        progress: Callable[[int, int, dict], None] | None = None, gallery: int = 16) -> dict:
    """Produce ``count`` accepted images. ``backend``: 'controlnet' (needs generator + qfilter) or 'augment'."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "gallery").mkdir(exist_ok=True)
    alloc = allocate(targets, count)
    oversample = cfg.oversample if backend == "controlnet" else 1.0
    plan = plan_candidates(alloc, train, cfg, oversample)
    (out_dir / "plan.json").write_text(json.dumps({"alloc": alloc, "n_candidates": len(plan)}, indent=1), encoding="utf-8")
    by_id = {s.id: s for s in train}

    qc_path = out_dir / "qc.jsonl"
    done: dict[int, dict] = {}
    if qc_path.is_file():
        for line in qc_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["k"]] = r
    accepted = {i: sum(1 for r in done.values() if r["target"] == i and r["accepted"]) for i in range(len(alloc))}
    emb_path = out_dir / "embeddings.npy"
    embs: dict[int, np.ndarray] = {}
    if emb_path.is_file() and (out_dir / "embeddings_k.json").is_file():
        ks = json.loads((out_dir / "embeddings_k.json").read_text())
        for k, e in zip(ks, np.load(emb_path)):
            embs[int(k)] = e
    n_gallery = len(list((out_dir / "gallery").glob("*.jpg")))
    total_q = sum(a["quota"] for a in alloc)

    with qc_path.open("a", encoding="utf-8") as qf:
        for k, c in enumerate(plan):
            ti = c["target"]
            if k in done or accepted[ti] >= alloc[ti]["quota"]:
                continue
            s = by_id[c["source"]]
            bgr = cv2.imread(s.image, cv2.IMREAD_COLOR)
            src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes = [b.xyxy() for b in s.boxes]
            if backend == "controlnet":
                gen, ctrl = generator.generate(src, c["prompt"], c["negative"], c["seed"], c["condition"])
                q = qfilter.check(src, gen, boxes, c["prompt"], c["condition"])
            else:
                gen = augment.relight(src, c["condition"], np.random.default_rng(c["seed"]))
                ctrl = []
                q = {"accepted": True, "reasons": []}
                if qfilter is not None:  # score it the same way (reported, but the augment arm is never filtered)
                    q = {**qfilter.check(src, gen, boxes, c["prompt"], c["condition"]), "accepted": True, "reasons": []}
            rec = {"k": k, "target": ti, "condition": c["condition"], "source": s.id, "prompt": c["prompt"],
                   "accepted": bool(q["accepted"]), "reasons": q.get("reasons", []),
                   **{m: q[m] for m in ("clip_score", "p_target", "box_pass", "boxes_judged") if m in q}}
            if q["accepted"]:
                sid = f"{id_prefix}_{k:05d}_{s.id}"
                cv2.imwrite(str(out_dir / "images" / f"{sid}.jpg"), cv2.cvtColor(gen, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                rec["id"] = sid
                accepted[ti] += 1
                if "embedding" in q:
                    embs[k] = q["embedding"]
                if n_gallery < gallery:
                    cv2.imwrite(str(out_dir / "gallery" / f"{n_gallery:02d}_{c['condition']}.jpg"),
                                _gallery_tile(src, ctrl, gen, boxes), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    n_gallery += 1
            elif n_gallery < gallery and not any(r.get("reject_shown") for r in done.values()) and k % 9 == 0:
                # keep one rejected example around for the report, to show what the filters catch
                cv2.imwrite(str(out_dir / "gallery" / f"rejected_{k:05d}_{'-'.join(rec['reasons'])}.jpg"),
                            _gallery_tile(src, ctrl, gen, boxes), [cv2.IMWRITE_JPEG_QUALITY, 80])
                rec["reject_shown"] = True
            qf.write(json.dumps(rec) + "\n")
            qf.flush()
            done[k] = rec
            if embs and len(embs) % 25 == 0:
                _save_embs(out_dir, embs)
            if progress:
                progress(sum(accepted.values()), total_q, rec)
    _save_embs(out_dir, embs)
    return finalize(out_dir, alloc, done, by_id, id_prefix)


def _save_embs(out_dir: Path, embs: dict[int, np.ndarray]) -> None:
    if not embs:
        return
    ks = sorted(embs)
    np.save(out_dir / "embeddings.npy", np.stack([embs[k] for k in ks]).astype(np.float32))
    (out_dir / "embeddings_k.json").write_text(json.dumps(ks))


def finalize(out_dir: Path, alloc: list[dict], done: dict[int, dict], by_id: dict[str, Sample], id_prefix: str) -> dict:
    samples = []
    for k in sorted(done):
        r = done[k]
        if not r["accepted"]:
            continue
        s = by_id[r["source"]]
        attrs = {**s.attrs, **CONDITION_ATTRS.get(r["condition"], {}), "source": id_prefix, "parent": s.id,
                 "condition": r["condition"]}
        samples.append(Sample(id=r["id"], image=str((out_dir / "images" / f"{r['id']}.jpg").resolve()),
                              width=s.width, height=s.height, boxes=list(s.boxes), attrs=attrs))
    write_manifest(out_dir / "manifest.jsonl", samples)
    recs = list(done.values())
    rej: dict[str, int] = {}
    for r in recs:
        for why in r.get("reasons", []):
            rej[why] = rej.get(why, 0) + 1

    def avg(key: str, acc: bool | None = None) -> float | None:
        v = [r[key] for r in recs if key in r and (acc is None or r["accepted"] == acc)]
        return float(np.mean(v)) if v else None

    summary = {"candidates": len(recs), "accepted": len(samples),
               "acceptance_rate": len(samples) / len(recs) if recs else 0.0, "rejections": rej,
               "alloc": alloc, "mean_clip_score": avg("clip_score", True), "mean_p_target": avg("p_target", True),
               "mean_box_pass": avg("box_pass", True), "mean_p_target_rejected": avg("p_target", False)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
