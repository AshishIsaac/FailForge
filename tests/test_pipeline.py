"""The full closed loop with a fake detector and a fake CLIP: checks orchestration, not ML.

The fake detector "learns" night: its recall on dark images grows with the number of dark images
in its training list. So arms that add night-like data (augment, synthetic, real) must beat the
baseline on the night slice, and the gate / registry / report must reflect that.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from failforge.config import load_config
from failforge.detections import Detections


def _dark(path: str) -> bool:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im is not None and im.mean() < 70


class FakeDetector:
    def __init__(self, weights, cfg):
        self.cfg = cfg
        self.device = "cpu"
        self.skill = json.loads(Path(weights).read_text())["night_skill"] if Path(weights).is_file() else 0.0

    @staticmethod
    def train(data_yaml, out_dir, init_weights, dcfg, tcfg, epochs, lr0, name="train", warmup_epochs=3.0, progress=None):
        imgs = (Path(data_yaml).parent / "train.txt").read_text().split()
        n_dark = sum(_dark(p) for p in imgs)
        base = json.loads(Path(init_weights).read_text())["night_skill"] if Path(init_weights).is_file() else 0.0
        out_dir.mkdir(parents=True, exist_ok=True)
        w = out_dir / "weights.pt"
        w.write_text(json.dumps({"night_skill": min(1.0, base + n_dark / 8.0)}))
        for e in range(epochs):
            if progress:
                progress(e + 1, epochs, {"metrics/mAP50-95(B)": 0.5})
        return w

    def predict(self, samples, conf=None, batch=16, progress=None):
        out = {}
        for s in samples:
            rng = np.random.default_rng(int(hashlib.md5(s.id.encode()).hexdigest()[:8], 16))
            keep_p = 0.95 if not _dark(s.image) else 0.15 + 0.8 * self.skill
            bx, sc, cl = [], [], []
            for b in s.boxes:
                if rng.random() < keep_p:
                    j = rng.normal(0, 1.0, 4)
                    bx.append(np.array(b.xyxy()) + j)
                    sc.append(rng.uniform(0.5, 1.0))
                    cl.append(b.cls)
            if rng.random() < 0.3:
                bx.append([5, 5, 40, 40])
                sc.append(0.3)
                cl.append(0)
            out[s.id] = Detections(np.array(bx, np.float32).reshape(-1, 4), np.array(sc, np.float32), np.array(cl, np.int64))
        return out


class FakeClip:
    dim = 16
    logit_scale = 30.0

    def __init__(self, *a, **k):
        self.rng = np.random.default_rng(0)

    def images(self, imgs, batch=64):
        out = []
        for im in imgs:
            b = float(np.asarray(im).mean()) / 255
            v = np.r_[1 - b, b, np.asarray(im).reshape(-1, 3).mean(0) / 255, np.zeros(11)]
            v[5:] = np.random.default_rng(int(b * 1e6)).normal(0, 0.05, 11)
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32).reshape(-1, self.dim)

    def texts(self, prompts):
        out = []
        for p in prompts:
            if any(w in p for w in ("night", "dark", "barely")):
                v = np.r_[1.0, 0.0, np.zeros(14)]
            elif any(w in p for w in ("daylight", "well-lit", "overcast")):
                v = np.r_[0.0, 1.0, np.zeros(14)]
            else:
                v = np.r_[0.4, 0.4, np.random.default_rng(len(p)).normal(0, 0.3, 14)]
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32)

    def zero_shot(self, img_emb, text_emb):
        z = self.logit_scale * img_emb @ text_emb.T
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)


@pytest.fixture
def fakes(monkeypatch):
    import failforge.analysis.embed as emb
    import failforge.detector as det

    monkeypatch.setattr(det, "Detector", FakeDetector)
    monkeypatch.setattr(emb, "ClipEmbedder", FakeClip)


def _cfg():
    return load_config("smoke", ["limit_images=0", "analysis.reducer=pca", "analysis.min_cluster_size=4",
                                 "analysis.min_samples=2", "drift.permutations=30", "synthesis.count=12",
                                 "gate.bootstrap=40", "gate.min_target_gain=0.01", "gate.max_slice_drop=0.2",
                                 "gate.max_overall_drop=0.2", "trigger.force=false", "synthesis.kid_reference=12",
                                 "mining.tp_sample=100"])


def test_full_loop_with_fakes(tiny_data, workspace, fakes):
    from failforge.loop.pipeline import STAGE_NAMES, run_loop
    from failforge.loop.registry import Registry

    run_dir, outs = run_loop(_cfg(), "t1")
    st = json.loads((run_dir / "status.json").read_text())
    assert st["state"] == "done", st
    assert [s["state"] for s in st["stages"]] == ["done"] * len(STAGE_NAMES)
    mon = outs["monitor"]
    assert mon["triggered"] and ("drift" in mon["reasons"] or any(r.startswith("SLO") for r in mon["reasons"]))
    ab = json.loads((run_dir / "eval" / "ablation.json").read_text())
    assert set(ab) == {"baseline", "baseline_ft", "augment", "synthetic", "real"}
    # Data that looks like night helps the fake model at night; more epochs alone do not.
    assert ab["synthetic"]["night"] > ab["baseline"]["night"] + 0.02
    assert ab["real"]["night"] > ab["baseline"]["night"] + 0.02
    assert abs(ab["baseline_ft"]["night"] - ab["baseline"]["night"]) < 1e-9
    gate = json.loads((run_dir / "gate" / "gate.json").read_text())
    assert gate["candidate"] == "synthetic" and gate["decision"]["promote"], gate["decision"]
    reg = Registry(workspace / "registry" / "smoke")
    assert reg.champion()["version"] == 2 and reg.champion()["arm"] == "synthetic"
    assert [e["event"] for e in reg.history()] == ["promoted", "promoted"]
    html = (run_dir / "report" / "report.html").read_text(encoding="utf-8")
    assert "PROMOTED" in html and "Ablation" in html
    a = json.loads((run_dir / "analyze" / "analysis.json").read_text())
    assert a["targets"] and a["targets"][0]["condition"] == "night"

    # Re-running the same run is fully cached ...
    t_before = {p.name: p.stat().st_mtime for p in (run_dir / "stages").glob("*.json")}
    run_loop(_cfg(), "t1")
    assert {p.name: p.stat().st_mtime for p in (run_dir / "stages").glob("*.json")} == t_before
    # ... and changing a gate threshold re-runs only the gate (+ report).
    _, outs2 = run_loop(load_config("smoke", [*_overrides(), "gate.min_target_gain=0.99"]), "t1")
    assert not outs2["gate"]["promote"]
    changed = {p.name for p in (run_dir / "stages").glob("*.json") if p.stat().st_mtime != t_before.get(p.name)}
    assert changed == {"gate.json", "report.json"}


def _overrides():
    return ["limit_images=0", "analysis.reducer=pca", "analysis.min_cluster_size=4", "analysis.min_samples=2",
            "drift.permutations=30", "synthesis.count=12", "gate.bootstrap=40", "gate.max_slice_drop=0.2",
            "gate.max_overall_drop=0.2", "trigger.force=false", "synthesis.kid_reference=12", "mining.tp_sample=100"]


def test_stop_request_is_resumable(tiny_data, workspace, fakes):
    from failforge.loop.pipeline import run_loop

    d = workspace / "runs" / "t2"
    d.mkdir(parents=True)
    (d / "STOP").write_text("x")
    run_dir, _ = run_loop(_cfg(), "t2")
    assert json.loads((run_dir / "status.json").read_text())["state"] == "stopped"
    run_dir, _ = run_loop(_cfg(), "t2")  # STOP was consumed: resumes and finishes
    assert json.loads((run_dir / "status.json").read_text())["state"] == "done"


def test_healthy_model_skips_the_loop(tiny_data, workspace, fakes, monkeypatch):
    from failforge.loop.pipeline import run_loop

    cfg = load_config("smoke", [*_overrides(), "trigger.slo={}", "drift.alpha=0.0"])
    run_dir, outs = run_loop(cfg, "t3")
    st = json.loads((run_dir / "status.json").read_text())
    assert not outs["monitor"]["triggered"]
    assert {s["name"]: s["state"] for s in st["stages"]}["synthesize"] == "skipped"
    assert (run_dir / "report" / "report.html").is_file()
