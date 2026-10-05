"""The closed loop, as a staged, resumable pipeline.

    baseline -> golden eval -> monitor (drift + SLOs) -> mine -> analyze -> synthesize
             -> retrain (ablation arms) -> evaluate arms -> gate (+ registry) -> report

Each stage writes its outputs under ``runs/<run_id>/<stage>/`` and a completion record in
``runs/<run_id>/stages/<stage>.json`` holding a fingerprint of the configuration it depends on
(chained with its upstream stages). Re-running a run skips finished stages whose fingerprint still
matches, so an interrupted multi-hour loop resumes where it stopped, and changing, say, the gate
thresholds re-runs only the gate and the report. ``status.json`` is rewritten as work progresses;
the dashboard polls it.

The baseline is the registry's current champion when there is one (production semantics: the loop
improves what is deployed). On the very first run there is none, so the baseline is trained on
daytime data and registered as v1.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import random
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from failforge import paths
from failforge.config import Config, to_dict
from failforge.data import splits as splits_mod
from failforge.data import yolo
from failforge.data.schema import Sample, read_manifest
from failforge.detections import load_predictions, save_predictions
from failforge.eval.evaluate import compare, evaluate
from failforge.eval.metrics import EvalIndex
from failforge.loop import gate as gate_mod
from failforge.loop.registry import Registry

log = logging.getLogger("failforge.loop")

STAGES = [
    ("baseline", "Baseline detector (champion or day-only training)", ("data", "detector", "train")),
    ("golden_baseline", "Evaluate baseline on the frozen golden set", ("detector",)),
    ("monitor", "Monitor: drift test and SLO check (trigger)", ("drift", "trigger", "analysis")),
    ("mine", "Mine failures on the production probe (TIDE typing)", ("mining",)),
    ("analyze", "Cluster failures (CLIP + UMAP + HDBSCAN) and name them", ("analysis",)),
    ("synthesize", "Synthesize targeted data + quality filters", ("synthesis", "arms")),
    ("retrain", "Fine-tune the ablation arms (equal budgets)", ("train", "arms")),
    ("golden_arms", "Evaluate every arm on the golden set (+ bootstrap CIs)", ("gate.bootstrap",)),
    ("gate", "Promotion gate and model registry", ("gate",)),
    ("report", "Evaluation report", ()),
]
STAGE_NAMES = [s[0] for s in STAGES]


class StopRequested(Exception):
    pass


def new_run_id(profile: str) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{profile}"


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Pipeline:
    def __init__(self, cfg: Config, run_dir: Path, registry: Registry | None = None) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_id = run_dir.name
        # One registry per profile: a smoke or quick run must never become the full profile's champion.
        self.registry = registry or Registry(paths.registry_dir() / cfg.profile)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stages").mkdir(exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=1), encoding="utf-8")
        self._status = self._load_status()
        self._last_write = 0.0
        self._fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        self._fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(self._fh)
        self._splits: dict[str, list[Sample]] = {}
        self._idx: dict[str, EvalIndex] = {}

    # ------------------------------------------------------------------ status

    def _load_status(self) -> dict:
        p = self.run_dir / "status.json"
        if p.is_file():
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
                if [s["name"] for s in st.get("stages", [])] == STAGE_NAMES:
                    return st
            except (OSError, ValueError, KeyError):
                pass
        return {"run_id": self.run_id, "profile": self.cfg.profile, "state": "pending", "stage": None,
                "stages": [{"name": n, "title": t, "state": "pending", "detail": "", "progress": None,
                            "started": None, "finished": None} for n, t, _ in STAGES]}

    def _stage(self, name: str) -> dict:
        return next(s for s in self._status["stages"] if s["name"] == name)

    def write_status(self, force: bool = True) -> None:
        now = time.time()
        if not force and now - self._last_write < 1.0:
            return
        self._last_write = now
        self._status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._status["pid"] = os.getpid()
        p = self.run_dir / "status.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._status, indent=1), encoding="utf-8")
        tmp.replace(p)
        if (self.run_dir / "STOP").exists():
            raise StopRequested()

    def progress(self, name: str, done: float, total: float, detail: str = "") -> None:
        s = self._stage(name)
        s["progress"] = [done, total]
        if detail:
            s["detail"] = detail
        self.write_status(force=False)

    def track(self, key: str, value, step: int | None = None) -> None:
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "key": key, "value": value, "step": step}) + "\n")
        if os.environ.get("FAILFORGE_MLFLOW"):
            try:
                import mlflow

                if isinstance(value, (int, float)):
                    mlflow.log_metric(key.replace(":", "_"), float(value), step=step)
            except Exception:  # tracking must never break the loop
                pass

    # ------------------------------------------------------------------ caching

    def _fingerprint(self, name: str) -> str:
        i = STAGE_NAMES.index(name)
        parts = [self.cfg.fingerprint(*STAGES[j][2]) if STAGES[j][2] else "" for j in range(i + 1)]
        parts.append(str(self.cfg.limit_images))
        return "-".join(p[:6] for p in parts if p)

    def _done(self, name: str) -> dict | None:
        p = self.run_dir / "stages" / f"{name}.json"
        if not p.is_file():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        return rec if rec.get("fingerprint") == self._fingerprint(name) else None

    def _finish(self, name: str, outputs: dict) -> None:
        (self.run_dir / "stages" / f"{name}.json").write_text(json.dumps({
            "fingerprint": self._fingerprint(name), "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "outputs": outputs,
        }, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ data

    def split(self, name: str) -> list[Sample]:
        if name not in self._splits:
            self._splits[name] = splits_mod.load_split(name, self.cfg.limit_images)
        return self._splits[name]

    def stage_dir(self, name: str) -> Path:
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ run

    def run(self, until: str | None = None, only: list[str] | None = None) -> dict:
        self._status["state"] = "running"
        self._status.pop("error", None)
        outputs: dict[str, dict] = {}
        try:
            self.write_status()
            for name, title, _ in STAGES:
                if only and name not in only:
                    rec = self._done(name)
                    if rec:
                        outputs[name] = rec["outputs"]
                    continue
                st = self._stage(name)
                rec = self._done(name)
                if rec is not None:
                    outputs[name] = rec["outputs"]
                    if st["state"] != "done":
                        st.update(state="done", detail=st.get("detail") or "cached")
                    self.write_status()
                    if name == "monitor" and not rec["outputs"].get("triggered", True):
                        self._skip_rest("monitor")
                        break
                    if until == name:
                        break
                    continue
                st.update(state="active", started=time.strftime("%Y-%m-%d %H:%M:%S"), progress=None, detail="")
                self._status["stage"] = name
                self.write_status()
                log.info("=== stage %s: %s", name, title)
                t0 = time.time()
                out = getattr(self, f"stage_{name}")(outputs)
                outputs[name] = out
                self._finish(name, out)
                st.update(state="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed=time.time() - t0,
                          detail=out.get("_detail", st.get("detail", "")))
                self.write_status()
                _free_gpu()
                if name == "monitor" and not out.get("triggered", True):
                    self._skip_rest("monitor")
                    break
                if until == name:
                    break
            self._status["state"] = "done"
            self._status["stage"] = None
        except StopRequested:
            self._status["state"] = "stopped"
            for s in self._status["stages"]:
                if s["state"] == "active":
                    s["state"] = "pending"
                    s["detail"] = "stopped - run again to resume"
            log.warning("stop requested: the run can be resumed")
        except Exception as e:
            self._status["state"] = "failed"
            self._status["error"] = f"{type(e).__name__}: {e}"
            for s in self._status["stages"]:
                if s["state"] == "active":
                    s["state"] = "failed"
                    s["detail"] = str(e)[:300]
            log.error("stage failed:\n%s", traceback.format_exc())
            raise
        finally:
            (self.run_dir / "STOP").unlink(missing_ok=True)
            with contextlib.suppress(StopRequested):
                self.write_status()
            logging.getLogger().removeHandler(self._fh)
            self._fh.close()
        return outputs

    def _skip_rest(self, after: str) -> None:
        for s in self._status["stages"][STAGE_NAMES.index(after) + 1:]:
            s.update(state="skipped", detail="no drift and all SLOs met - nothing to fix")
        if "report" in STAGE_NAMES:
            self.stage_report({})
            self._stage("report").update(state="done", detail="health report")
        self.write_status()

    # ================================================================== stages

    def stage_baseline(self, _: dict) -> dict:
        from failforge.detector import Detector

        champ = self.registry.champion()
        if champ is not None:
            log.info("baseline = registry champion v%s (%s)", champ["version"], champ["weights"])
            return {"weights": champ["weights"], "version": champ["version"], "trained": False,
                    "_detail": f"champion v{champ['version']}"}
        d = self.stage_dir("baseline")
        data_yaml = yolo.build_dataset(d / "dataset", self.split("train_day"), self.split("val_day"), self.cfg.detector.imgsz)
        n = len(self.split("train_day"))
        ep = self.cfg.train.baseline_epochs

        def prog(e: int, total: int, m: dict) -> None:
            self.progress("baseline", e, total, f"epoch {e}/{total} - day-only val mAP50-95 {m.get('metrics/mAP50-95(B)', 0):.3f}")
            self.track("baseline:val_map", m.get("metrics/mAP50-95(B)"), e)

        self.progress("baseline", 0, ep, f"training on {n} daytime images")
        w = Detector.train(data_yaml, d, self.cfg.detector.weights, self.cfg.detector, self.cfg.train,
                           epochs=ep, lr0=self.cfg.train.lr0, progress=prog)
        return {"weights": str(w), "version": None, "trained": True, "train_images": n,
                "_detail": f"trained {ep} epochs on {n} day images"}

    def _predict(self, weights: str, samples: list[Sample], out: Path, stage: str, label: str) -> dict:
        from failforge.detector import Detector

        if out.is_file():
            preds = load_predictions(out)
            if all(s.id in preds for s in samples):
                return preds
        det = Detector(weights, self.cfg.detector)
        preds = det.predict(samples, batch=self.cfg.detector.batch,
                            progress=lambda a, b: self.progress(stage, a, b, f"{label}: {a}/{b} images"))
        save_predictions(out, preds)
        del det
        _free_gpu()
        return preds

    def _evaluate(self, name: str, weights: str, stage: str) -> dict:
        d = self.stage_dir("eval")
        golden = self.split("golden")
        preds = self._predict(weights, golden, d / f"{name}_preds.npz", stage, f"golden / {name}")
        self.progress(stage, 0, 1, f"scoring {name}")
        res, idx = evaluate(golden, preds, self.cfg.detector.mine_conf)
        res["model"] = {"name": name, "weights": weights}
        (d / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        self._idx[name] = idx
        self.track(f"golden:{name}:map", res["overall"]["map"])
        for s, v in res["slices"].items():
            self.track(f"golden:{name}:{s}:map", v["map"])
        return res

    def _index(self, name: str) -> EvalIndex:
        if name not in self._idx:
            preds = load_predictions(self.run_dir / "eval" / f"{name}_preds.npz")
            self._idx[name] = EvalIndex(self.split("golden"), preds)
        return self._idx[name]

    def stage_golden_baseline(self, outs: dict) -> dict:
        res = self._evaluate("baseline", outs["baseline"]["weights"], "golden_baseline")
        b = outs["baseline"]
        if b.get("trained"):
            entry = self.registry.register(Path(b["weights"]), run_id=self.run_id, arm="baseline",
                                           golden=_headline(res), parent=None, promote=True,
                                           note="initial day-only baseline")
            log.info("registered baseline as champion v%d", entry["version"])
        sl = res["slices"]
        return {"map": res["overall"]["map"], "slices": {k: v["map"] for k, v in sl.items()},
                "_detail": f"overall {res['overall']['map']:.3f} | day {sl.get('day', {}).get('map') or 0:.3f} | "
                           f"night {sl.get('night', {}).get('map') or 0:.3f}"}

    def stage_monitor(self, outs: dict) -> dict:
        import cv2

        from failforge.analysis.drift import drift_report
        from failforge.analysis.embed import ClipEmbedder

        c = self.cfg.drift
        rng = random.Random(0)
        ref = self.split("train_day")
        ref = rng.sample(ref, min(c.reference_samples, len(ref)))
        cur = self.split("probe")[: c.window]
        emb = ClipEmbedder(self.cfg.analysis.embedder)

        def load(ss: list[Sample], tag: str) -> tuple[np.ndarray, np.ndarray]:
            ims, br = [], []
            for i, s in enumerate(ss):
                im = cv2.imread(s.image)
                br.append(float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()) / 255)
                ims.append(cv2.cvtColor(cv2.resize(im, (398, 224)), cv2.COLOR_BGR2RGB))
                if i % 50 == 0:
                    self.progress("monitor", i, len(ss), f"embedding {tag} images")
            return emb.images(ims, self.cfg.analysis.batch), np.array(br)

        er, br = load(ref, "reference (train)")
        ec, bc = load(cur, "production window")
        rep = drift_report(er, ec, br, bc, c.permutations, c.alpha)
        slo_viol = {}
        g = outs["golden_baseline"]
        for k, thr in self.cfg.trigger.slo.items():
            v = g["map"] if k == "overall" else g["slices"].get(k)
            if v is not None and v < thr:
                slo_viol[k] = {"map": v, "slo": thr}
        triggered = bool(rep["drift"] or slo_viol or self.cfg.trigger.force)
        out = {"drift": rep, "slo_violations": slo_viol, "triggered": triggered,
               "reasons": (["drift"] if rep["drift"] else []) + [f"SLO:{k}" for k in slo_viol]
                          + (["forced"] if self.cfg.trigger.force else []),
               "_detail": (f"drift p={rep['p_value']:.3g}, domain AUC {rep['domain_auc']:.2f}; "
                           + (f"SLO violated: {', '.join(slo_viol)}" if slo_viol else "SLOs met")
                           + (" -> loop TRIGGERED" if triggered else " -> healthy"))}
        (self.stage_dir("monitor") / "monitor.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        for k in ("mmd2", "p_value", "domain_auc", "psi_brightness"):
            self.track(f"drift:{k}", rep[k])
        return out

    def stage_mine(self, outs: dict) -> dict:
        from failforge.mining.mine import mine

        d = self.stage_dir("mine")
        probe = self.split("probe")
        preds = self._predict(outs["baseline"]["weights"], probe, d / "probe_preds.npz", "mine", "probe")
        res, _ = evaluate(probe, preds, self.cfg.detector.mine_conf)
        (d / "probe_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        m = mine(probe, preds, self.cfg.detector.mine_conf, self.cfg.mining)
        m.save(d / "mined.json")
        s = m.summary
        return {"errors": len(m.errors), "tps": len(m.tps), "tide": s,
                "_detail": f"{len(m.errors)} errors on {len(probe)} probe images "
                           f"(miss {s['miss']}, bkg {s['bkg']}, loc {s['loc']}, cls {s['cls']})"}

    def stage_analyze(self, outs: dict) -> dict:
        from failforge.analysis import failure_modes as fm
        from failforge.analysis.embed import ClipEmbedder
        from failforge.mining.mine import MinedSet

        d = self.stage_dir("analyze")
        mined = MinedSet.load(self.run_dir / "mine" / "mined.json")
        items = mined.errors + mined.tps
        emb = ClipEmbedder(self.cfg.analysis.embedder)
        feats = fm.build_features(items, emb, self.cfg.mining, self.cfg.analysis,
                                  progress=lambda a, b: self.progress("analyze", a, b, f"embedding crops: {a}/{b} images"))
        np.savez_compressed(d / "features.npz", **feats)
        self.progress("analyze", 0, 1, "UMAP + HDBSCAN")
        res = fm.analyze(mined, feats, emb, self.cfg.analysis)
        if not res["targets"]:
            # Nothing addressable found by clustering: fall back to the SLO-violating condition slices.
            viol = outs.get("monitor", {}).get("slo_violations", {})
            fallback = [k for k in viol if k in ("night", "dawn", "rainy", "snowy")]
            cond = {"dawn": "dusk", "rainy": "rain", "snowy": "snow"}
            res["targets"] = [{"cluster": None, "condition": cond.get(k, k), "classes": ["car", "pedestrian"],
                               "excess_errors": 1.0, "name": f"SLO slice {k} (fallback)"} for k in fallback]
            res["fallback"] = True
        fm.save_thumbnails(res, items, d / "thumbs", self.cfg.mining.context)
        fm.write(res, d / "analysis.json")
        tg = ", ".join(f"{t['name']} -> {t['condition']}" for t in res["targets"]) or "none"
        return {"clusters": len(res["clusters"]), "modes": len(res["modes"]), "targets": res["targets"],
                "_detail": f"{len(res['clusters'])} clusters, {len(res['modes'])} failure modes; targets: {tg}"}

    def stage_synthesize(self, outs: dict) -> dict:
        import cv2

        from failforge.analysis.embed import ClipEmbedder
        from failforge.synthesis import engine
        from failforge.synthesis.quality import QualityFilter, kid

        d = self.stage_dir("synthesize")
        targets = outs["analyze"]["targets"]
        if not targets:
            raise RuntimeError("no synthesis targets (no addressable failure mode and no SLO violation)")
        train = self.split("train_day")
        c = self.cfg.synthesis
        emb = ClipEmbedder(self.cfg.analysis.embedder)
        teacher = None
        if c.teacher and c.backend == "controlnet":
            from failforge.synthesis.quality import Teacher

            teacher = Teacher(c.teacher, imgsz=self.cfg.detector.imgsz)
        qf = QualityFilter(emb, c, teacher)
        result: dict = {}
        n_syn = c.count
        if "synthetic" in self.cfg.arms:
            gen = None
            if c.backend == "controlnet":
                from failforge.synthesis.controlnet import ControlNetGenerator

                self.progress("synthesize", 0, c.count, "loading Stable Diffusion 1.5 + ControlNet (canny, depth)")
                gen = ControlNetGenerator(c)
            t0 = time.time()

            def prog(a: int, b: int, rec: dict) -> None:
                rate = (time.time() - t0) / max(a, 1)
                self.progress("synthesize", a, b, f"synthetic: {a}/{b} accepted ({rec['condition']}, "
                                                  f"{'ok' if rec['accepted'] else 'rejected: ' + ','.join(rec['reasons'])})"
                                                  f" - {rate:.1f} s/img")

            s = engine.run(c.backend, targets, train, d / "synthetic", c, c.count, generator=gen, qfilter=qf,
                           id_prefix="syn", progress=prog)
            s["seconds"] = time.time() - t0
            result["synthetic"] = s
            n_syn = s["accepted"]
            del gen
            _free_gpu()
        if "augment" in self.cfg.arms:
            s = engine.run("augment", targets, train, d / "augment", c, n_syn, qfilter=qf, id_prefix="aug",
                           progress=lambda a, b, r: self.progress("synthesize", a, b, f"augment: {a}/{b}"))
            result["augment"] = s
        if "real" in self.cfg.arms:
            result["real"] = self._select_real(targets, n_syn, d / "real")
        # KID diagnostic (CLIP features): which set looks most like real target-condition images?
        pool = self.split("target_pool")
        real_used = {s.id for s in read_manifest(d / "real" / "manifest.jsonl")} if "real" in result else set()
        ref = [s for s in pool if s.id not in real_used][: c.kid_reference]

        def embed(ss: list[Sample]) -> np.ndarray:
            return emb.images([cv2.cvtColor(cv2.resize(cv2.imread(s.image), (398, 224)), cv2.COLOR_BGR2RGB) for s in ss])

        kids = {}
        if len(ref) >= 10:
            e_ref = embed(ref)
            sets = {"day_sources": [s for s in train[: c.kid_reference]]}
            for arm in ("synthetic", "augment"):
                if arm in result and (d / arm / "manifest.jsonl").is_file():
                    sets[arm] = read_manifest(d / arm / "manifest.jsonl")[: c.kid_reference]
            for k, ss in sets.items():
                if len(ss) >= 10:
                    m, sd = kid(embed(ss), e_ref)
                    kids[k] = {"kid": m, "std": sd, "n": len(ss)}
        result["kid_vs_real_target"] = kids
        (d / "synthesis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        acc = result.get("synthetic", {})
        return {"n": n_syn, "summary": {k: {kk: vv for kk, vv in v.items() if kk != "alloc"} if isinstance(v, dict) else v
                                        for k, v in result.items()},
                "_detail": (f"{acc.get('accepted', 0)}/{acc.get('candidates', 0)} synthetic accepted"
                            f" ({100 * acc.get('acceptance_rate', 0):.0f}%), N={n_syn} per arm") if acc else f"N={n_syn}"}

    def _select_real(self, targets: list[dict], n: int, out: Path) -> dict:
        from failforge.data.schema import write_manifest
        from failforge.synthesis.engine import allocate

        pool = self.split("target_pool")
        match = {"night": lambda a: a.get("timeofday") == "night", "glare": lambda a: a.get("timeofday") == "night",
                 "dusk": lambda a: a.get("timeofday") == "dawn", "rain": lambda a: a.get("weather") == "rainy",
                 "snow": lambda a: a.get("weather") == "snowy", "fog": lambda a: a.get("weather") == "foggy"}
        chosen: list[Sample] = []
        used: set[str] = set()
        for t in allocate(targets, n):
            cand = [s for s in pool if match.get(t["condition"], lambda a: True)(s.attrs) and s.id not in used]
            take = cand[: t["quota"]]
            used |= {s.id for s in take}
            chosen += take
        if len(chosen) < n:  # not enough matching real images: top up from the rest of the pool
            chosen += [s for s in pool if s.id not in used][: n - len(chosen)]
        write_manifest(out / "manifest.jsonl", chosen)
        return {"accepted": len(chosen), "requested": n}

    def _arm_extra(self, arm: str) -> list[Sample]:
        d = self.run_dir / "synthesize"
        f = {"synthetic": d / "synthetic" / "manifest.jsonl", "augment": d / "augment" / "manifest.jsonl",
             "real": d / "real" / "manifest.jsonl"}.get(arm)
        return read_manifest(f) if f is not None and f.is_file() else []

    def stage_retrain(self, outs: dict) -> dict:
        from failforge.detector import Detector

        base_w = outs["baseline"]["weights"]
        train = self.split("train_day")
        rng = random.Random(self.cfg.train.seed)
        replay = train if self.cfg.train.replay_fraction >= 1 else rng.sample(
            train, int(len(train) * self.cfg.train.replay_fraction))
        res = {}
        arms = self.cfg.arms
        for i, arm in enumerate(arms):
            d = self.stage_dir("retrain") / arm
            w = d / "weights.pt"
            marker = d / "done.json"
            extra = self._arm_extra(arm)
            if w.is_file() and marker.is_file() and json.loads(marker.read_text())["fp"] == self._fingerprint("retrain"):
                res[arm] = {"weights": str(w), "train_images": len(replay) + len(extra), "extra": len(extra)}
                continue
            data_yaml = yolo.build_dataset(d / "dataset", replay + extra, self.split("val_day"), self.cfg.detector.imgsz)
            ep = self.cfg.train.finetune_epochs
            n_extra = len(extra)

            def prog(e: int, total: int, m: dict, arm=arm, i=i, n_extra=n_extra) -> None:
                self.progress("retrain", i * total + e, len(arms) * total,
                              f"arm {i + 1}/{len(arms)} '{arm}': epoch {e}/{total} (+{n_extra} images)")
                self.track(f"retrain:{arm}:val_map", m.get("metrics/mAP50-95(B)"), e)

            self.progress("retrain", i * ep, len(arms) * ep, f"arm {i + 1}/{len(arms)} '{arm}' (+{len(extra)} images)")
            Detector.train(data_yaml, d, base_w, self.cfg.detector, self.cfg.train, epochs=ep,
                           lr0=self.cfg.train.finetune_lr0, warmup_epochs=1.0, progress=prog)
            marker.write_text(json.dumps({"fp": self._fingerprint("retrain")}))
            res[arm] = {"weights": str(w), "train_images": len(replay) + len(extra), "extra": len(extra)}
            _free_gpu()
        return {"arms": res, "_detail": ", ".join(f"{a} (+{r['extra']})" for a, r in res.items())}

    def stage_golden_arms(self, outs: dict) -> dict:

        evals = {"baseline": json.loads((self.run_dir / "eval" / "baseline.json").read_text(encoding="utf-8"))}
        for i, (arm, r) in enumerate(outs["retrain"]["arms"].items()):
            self.progress("golden_arms", i, len(outs["retrain"]["arms"]), f"evaluating {arm}")
            evals[arm] = self._evaluate(arm, r["weights"], "golden_arms")
        base = self._index("baseline")
        deltas = {}
        # CIs where they are used: the gate's target slices + the report's forest plot. The gate's
        # per-slice no-regression checks use point deltas over every gated slice.
        want = ["overall", *self.cfg.gate.target_slices, "day", "dawn"]
        names = [s for s in dict.fromkeys(want) if s == "overall" or s in evals["baseline"]["slices"]]
        for i, arm in enumerate(outs["retrain"]["arms"]):
            self.progress("golden_arms", i, len(outs["retrain"]["arms"]), f"bootstrap CIs: {arm}")
            deltas[arm] = compare(base, self._index(arm), names, n_boot=self.cfg.gate.bootstrap)
        (self.run_dir / "eval" / "deltas.json").write_text(json.dumps(deltas, indent=1), encoding="utf-8")
        table = {arm: {"overall": e["overall"]["map"], **{k: v["map"] for k, v in e["slices"].items()}}
                 for arm, e in evals.items()}
        (self.run_dir / "eval" / "ablation.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
        tgt = self.cfg.gate.target_slices[0] if self.cfg.gate.target_slices else "overall"
        det = ", ".join(f"{a} {deltas[a].get(tgt, {}).get('delta', 0):+.3f}" for a in deltas)
        return {"table": table, "_detail": f"Delta {tgt} mAP@50:95: {det}"}

    def stage_gate(self, outs: dict) -> dict:
        evals = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (self.run_dir / "eval").glob("*.json")
                 if p.stem not in ("deltas", "ablation")}
        deltas = json.loads((self.run_dir / "eval" / "deltas.json").read_text(encoding="utf-8"))
        cand = "synthetic" if "synthetic" in deltas else next(iter(deltas))
        decisions = {a: gate_mod.decide(evals["baseline"], evals[a], deltas[a], self.cfg.gate) for a in deltas}
        dec = decisions[cand]
        champ = self.registry.champion()
        w = outs["retrain"]["arms"][cand]["weights"]
        if dec["promote"]:
            entry = self.registry.register(Path(w), run_id=self.run_id, arm=cand, golden=_headline(evals[cand]),
                                           parent=champ["version"] if champ else None, promote=True,
                                           note="passed the promotion gate")
            verdict = f"PROMOTED '{cand}' as champion v{entry['version']}"
        else:
            failed = [c["name"] for c in dec["checks"] if not c["passed"]]
            self.registry.reject(run_id=self.run_id, arm=cand, reason="; ".join(failed))
            verdict = f"REJECTED '{cand}': failed {', '.join(failed)}"
        out = {"candidate": cand, "decision": dec, "all_arms": decisions, "verdict": verdict}
        (self.stage_dir("gate") / "gate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return {"promote": dec["promote"], "candidate": cand, "_detail": verdict}

    def stage_report(self, outs: dict) -> dict:
        from failforge.report.report import build_report

        p = build_report(self.run_dir)
        return {"html": str(p), "_detail": "report.html written"}


def _headline(res: dict) -> dict:
    return {"map": res["overall"]["map"], "map50": res["overall"]["map50"],
            **{f"{k}_map": v["map"] for k, v in res["slices"].items() if k in ("day", "night", "dawn", "rainy", "snowy")}}


def run_loop(cfg: Config, run_id: str | None = None, until: str | None = None, only: list[str] | None = None,
             on_start: Callable[[Path], None] | None = None) -> tuple[Path, dict]:
    paths.configure_caches()
    run_dir = paths.runs_dir() / (run_id or new_run_id(cfg.profile))
    pipe = Pipeline(cfg, run_dir)
    if on_start:
        on_start(run_dir)
    return run_dir, pipe.run(until=until, only=only)
