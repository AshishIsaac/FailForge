"""From mined errors to named, ranked failure modes.

1. Each mined record (errors + sampled true positives) becomes a feature vector: the CLIP embedding
   of a context crop around the box, plus standardized photometric descriptors.
2. UMAP + HDBSCAN group the records (errors and TPs together, so each cluster has an error *rate*).
3. Every cluster is described without looking at the dataset's tags: CLIP zero-shot scores of the
   whole images against a vocabulary of scene conditions ("at night", "in the rain", "headlight
   glare", ...) and of the crops against object conditions ("tiny and distant", "blurry", ...),
   reported as enrichment over the mined population; plus photometric deltas and class mix.
4. Clusters are ranked by *excess errors* (errors beyond what the population error rate predicts).
   A cluster with lift >= ``min_lift`` is a failure mode. If its top enriched scene condition is one
   we can synthesize (night, dusk, rain, snow, fog, glare), it gets a synthesis recipe; otherwise it
   is reported as not addressable by synthesis (e.g. tiny distant objects: label more / raise imgsz).

Only *after* that, as a check on the unsupervised analysis, each cluster's real tag composition
(BDD time of day / weather) is attached to the report: did the clustering rediscover "night"?
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from failforge.analysis.cluster import cluster
from failforge.analysis.photometric import ATTRS, context_box, describe_crop, standardize
from failforge.config import CLASSES, AnalysisConfig, MiningConfig
from failforge.mining.mine import MinedSet

log = logging.getLogger(__name__)

# Scene conditions, scored on the whole image. Keys marked synthesizable map to a prompt recipe.
SCENE_CONCEPTS = {
    "night": "a dashcam photo of a road at night, dark sky, artificial lights",
    "dusk": "a dashcam photo at dusk or dawn, low sun, twilight sky",
    "daylight": "a dashcam photo in bright daylight, blue sky",
    "overcast": "a dashcam photo on an overcast grey day",
    "rain": "a dashcam photo in the rain, wet road, raindrops on the windshield",
    "snow": "a dashcam photo of a snowy road, snow on the ground",
    "fog": "a dashcam photo in dense fog, hazy, low visibility",
    "glare": "a dashcam photo with strong headlight glare and lens flare",
    "tunnel": "a dashcam photo inside a tunnel",
}
OBJECT_CONCEPTS = {
    "dark_object": "a barely visible object in the dark",
    "tiny": "a tiny, distant object far away",
    "blurry": "a blurry, motion-blurred object",
    "occluded": "a partially hidden, occluded object",
    "light_source": "a bright light, a lamp or a reflection",
    "crowd": "many objects crowded together",
    "clear_object": "a clearly visible, well-lit object",
}
SYNTHESIZABLE = ("night", "dusk", "rain", "snow", "fog", "glare")


def build_features(items: list[dict], embedder, mcfg: MiningConfig, acfg: AnalysisConfig,
                   progress=None) -> dict:
    """CLIP crop + image embeddings and photometric descriptors for every record."""
    by_img: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_img[it["image"]].append(i)
    crop_emb = np.zeros((len(items), embedder.dim), np.float32)
    img_emb_of: dict[str, np.ndarray] = {}
    photo = np.zeros((len(items), len(ATTRS)), np.float32)
    paths = list(by_img)
    done = 0
    for k in range(0, len(paths), 16):  # 16 images at a time keeps memory flat
        chunk = paths[k:k + 16]
        crops, idxs, fulls = [], [], []
        for p in chunk:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gmean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
            fulls.append(cv2.resize(rgb, (398, 224)))
            h, w = rgb.shape[:2]
            for i in by_img[p]:
                a, b, c, d = context_box(items[i]["box"], w, h, mcfg.context)
                crops.append(rgb[b:d, a:c])
                idxs.append(i)
                photo[i] = [describe_crop(bgr, items[i]["box"], gmean)[n] for n in ATTRS]
        e = embedder.images(crops, batch=acfg.batch)
        crop_emb[idxs] = e
        for p, f in zip(chunk, embedder.images(fulls, batch=acfg.batch)):
            img_emb_of[p] = f
        done += len(chunk)
        if progress:
            progress(done, len(paths))
    img_emb = np.stack([img_emb_of[it["image"]] for it in items]) if items else np.zeros((0, embedder.dim), np.float32)
    return {"crop": crop_emb, "image": img_emb, "photo": photo}


def analyze(mined: MinedSet, feats: dict, embedder, acfg: AnalysisConfig) -> dict:
    items = mined.errors + mined.tps
    n_err = len(mined.errors)
    is_err = np.zeros(len(items), bool)
    is_err[:n_err] = True
    # TPs were subsampled: each kept TP stands for 1/tp_rate true positives.
    weight = np.where(is_err, 1.0, 1.0 / max(mined.tp_rate, 1e-9))

    z_photo, mu, sd = standardize(feats["photo"])
    x = np.concatenate([feats["crop"], acfg.photometric_weight * z_photo / np.sqrt(z_photo.shape[1])], axis=1)
    labels, xy = cluster(x, acfg.reducer, acfg.n_components, acfg.n_neighbors, acfg.min_dist,
                         acfg.min_cluster_size, acfg.min_samples, acfg.seed)

    scene_names, obj_names = list(SCENE_CONCEPTS), list(OBJECT_CONCEPTS)
    p_scene = embedder.zero_shot(feats["image"], embedder.texts(list(SCENE_CONCEPTS.values())))
    p_obj = embedder.zero_shot(feats["crop"], embedder.texts(list(OBJECT_CONCEPTS.values())))
    base_scene = p_scene.mean(axis=0) + 1e-9
    base_obj = p_obj.mean(axis=0) + 1e-9

    total_err = float(is_err.sum())
    total_w = float(weight.sum())
    global_rate = total_err / total_w if total_w else 0.0

    clusters = []
    for c in sorted(set(labels.tolist()) - {-1}):
        m = labels == c
        n_e = float(is_err[m].sum())
        w = float(weight[m].sum())
        rate = n_e / w if w else 0.0
        lift = rate / global_rate if global_rate else 0.0
        excess = n_e - global_rate * w
        enr_scene = p_scene[m].mean(axis=0) / base_scene
        enr_obj = p_obj[m].mean(axis=0) / base_obj
        top_scene = [scene_names[i] for i in np.argsort(-enr_scene)[:3] if enr_scene[i] > 1.05]
        top_obj = [obj_names[i] for i in np.argsort(-enr_obj)[:2] if enr_obj[i] > 1.05]
        member = [items[i] for i in np.flatnonzero(m)]
        kinds = Counter(it["kind"] for it in member if it["kind"] != "tp")
        cls_counts = Counter(CLASSES[it["gt_cls"] if it["gt_cls"] >= 0 else it["cls"]] for it in member)
        photo_delta = {a: float(v) for a, v in zip(ATTRS, z_photo[m].mean(axis=0))}
        target = next((s for s in top_scene if s in SYNTHESIZABLE), None)
        tags = Counter(it["attrs"].get("timeofday", "?") for it in member)
        wx = Counter(it["attrs"].get("weather", "?") for it in member)
        name_bits = [*top_scene[:2], *top_obj[:1]]
        dominant_kind = kinds.most_common(1)[0][0] if kinds else "tp"
        clusters.append({
            "id": int(c),
            "name": " / ".join(name_bits) if name_bits else "generic",
            "size": int(m.sum()), "errors": int(n_e), "error_rate": rate, "lift": lift, "excess_errors": excess,
            "kinds": dict(kinds), "dominant_kind": dominant_kind,
            "classes": dict(cls_counts.most_common()),
            "scene_concepts": {scene_names[i]: float(enr_scene[i]) for i in range(len(scene_names))},
            "object_concepts": {obj_names[i]: float(enr_obj[i]) for i in range(len(obj_names))},
            "top_scene": top_scene, "top_object": top_obj,
            "photometric_z": photo_delta,
            "synthesis_target": target,
            # Validation only - never used for the decisions above:
            "true_timeofday": {k: v / m.sum() for k, v in tags.items()},
            "true_weather": {k: v / m.sum() for k, v in wx.most_common(4)},
            "examples": [int(i) for i in np.flatnonzero(m & is_err)[:12]],
        })
    clusters.sort(key=lambda d: -d["excess_errors"])
    modes = [c for c in clusters if c["lift"] >= acfg.min_lift and c["excess_errors"] > 0]
    for rank, c in enumerate(modes, 1):
        c["rank"] = rank
    addressable = [c for c in modes if c["synthesis_target"]][: acfg.top_modes]

    pts = [{"x": float(xy[i, 0]), "y": float(xy[i, 1]), "c": int(labels[i]), "k": items[i]["kind"],
            "cls": int(items[i]["gt_cls"] if items[i]["gt_cls"] >= 0 else items[i]["cls"]),
            "img": Path(items[i]["image"]).stem, "tod": items[i]["attrs"].get("timeofday", "?")}
           for i in range(len(items))]
    noise = float((labels == -1).mean()) if len(labels) else 0.0
    return {"global_error_rate": global_rate, "n_items": len(items), "n_errors": n_err, "noise_fraction": noise,
            "clusters": clusters, "modes": [c["id"] for c in modes],
            "targets": [{"cluster": c["id"], "condition": c["synthesis_target"], "classes": list(c["classes"])[:4],
                         "excess_errors": c["excess_errors"], "name": c["name"]} for c in addressable],
            "points": pts, "photometric_mu": mu.tolist(), "photometric_sd": sd.tolist(),
            "vocab": {"scene": SCENE_CONCEPTS, "object": OBJECT_CONCEPTS}}


def save_thumbnails(result: dict, items: list[dict], out_dir: Path, context: float, per_cluster: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in result["clusters"]:
        thumbs = []
        for k, i in enumerate(c["examples"][:per_cluster]):
            it = items[i]
            im = cv2.imread(it["image"], cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]
            a, b, cc, d = context_box(it["box"], w, h, max(context, 1.0))
            x1, y1, x2, y2 = (int(v) for v in it["box"])
            color = (60, 60, 230) if it["kind"] == "miss" else (0, 170, 255)
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
            crop = cv2.resize(im[b:d, a:cc], (160, 160), interpolation=cv2.INTER_AREA)
            name = f"c{c['id']}_{k}.jpg"
            cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumbs.append({"file": name, "kind": it["kind"], "cls": CLASSES[it["gt_cls"] if it["gt_cls"] >= 0 else it["cls"]],
                           "image_id": Path(it["image"]).stem})
        c["thumbs"] = thumbs


def write(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
