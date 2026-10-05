"""``failforge`` command-line interface.

    failforge data download|prepare|validate      BDD100K mirror -> manifests -> frozen golden set + splits
    failforge models download [--no-diffusion]    YOLO11n, CLIP, Stable Diffusion 1.5 + ControlNets, depth
    failforge loop -c quick|full|smoke            run (or resume) the closed loop end to end
    failforge eval --weights W [--split golden]   score any checkpoint on a split (slices, TIDE, sizes)
    failforge gate --check DIR                    re-run the promotion decision on saved artifacts (CI)
    failforge report RUN_DIR                      rebuild a run's HTML/Markdown report
    failforge bench [--weights W]                 latency / throughput / memory: PyTorch vs ONNX Runtime
    failforge serve                               the dashboard (what `python run.py` opens)
    failforge registry                            champion + promotion history
    failforge publish RUN_DIR                     copy a run's compact results to reports/reference/
    failforge doctor                              environment check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from failforge import __version__, paths

log = logging.getLogger("failforge")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "PIL", "matplotlib", "numba", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cfg(a: argparse.Namespace):
    from failforge.config import load_config

    return load_config(getattr(a, "config", None) or "full", list(getattr(a, "set", None) or []))


# ------------------------------------------------------------------ data / models


def cmd_data(a: argparse.Namespace) -> int:
    from failforge.data import bdd, splits
    from failforge.data.schema import read_manifest, validate_samples

    cfg = _cfg(a)
    if a.action in ("download", "all"):
        bdd.download(cfg.data)
    if a.action in ("prepare", "all"):
        all_p = paths.manifests_dir() / "all.jsonl"
        if a.rebuild or not all_p.is_file():
            bdd.build_manifest(cfg.data)
        sp = splits.make_splits(read_manifest(all_p), cfg.data)
        summ = splits.write_splits(sp)
        for k, v in summ.items():
            print(f"  {k:<12} {v['images']:>5} images {v['boxes']:>6} boxes  {v['timeofday']}")
    if a.action in ("validate", "prepare", "all"):
        bad = 0
        names = splits.SPLITS
        for n in names:
            p = paths.manifests_dir() / f"{n}.jsonl"
            if not p.is_file():
                print(f"  {n}: missing")
                bad += 1
                continue
            ss = read_manifest(p)
            probs = validate_samples(ss, check_files=a.check_files)
            print(f"  {n:<12} {'OK' if not probs else f'{len(probs)} problem(s)'}")
            for x in probs[:10]:
                print("     ", x)
            bad += bool(probs)
        sp = {n: read_manifest(paths.manifests_dir() / f"{n}.jsonl") for n in names
              if (paths.manifests_dir() / f"{n}.jsonl").is_file()}
        try:
            splits.check_disjoint(sp)
            print("  leakage check: OK (no drive appears in two splits)")
        except AssertionError as e:
            print(f"  leakage check: FAILED - {e}")
            bad += 1
        return 1 if bad else 0
    return 0


def cmd_models(a: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    from failforge.detector import pretrained_weights

    paths.configure_caches()
    cfg = _cfg(a)
    print("YOLO:", pretrained_weights(cfg.detector.weights))
    if cfg.synthesis.teacher:
        print("QC teacher:", pretrained_weights(cfg.synthesis.teacher))
    snapshot_download(cfg.analysis.embedder, allow_patterns=["*.json", "*.txt", "model.safetensors", "pytorch_model.bin"])
    print("CLIP:", cfg.analysis.embedder)
    if not a.no_diffusion:
        from failforge.synthesis.controlnet import download_models

        download_models(cfg.synthesis)
        print("Diffusion:", cfg.synthesis.base_model, *cfg.synthesis.controlnets, cfg.synthesis.depth_model)
    return 0


# ------------------------------------------------------------------ loop / eval / gate


def cmd_loop(a: argparse.Namespace) -> int:
    from failforge.loop.pipeline import STAGE_NAMES, run_loop

    cfg = _cfg(a)
    setup_logging(a.log_level or cfg.log_level)
    if a.until and a.until not in STAGE_NAMES:
        print(f"--until must be one of {STAGE_NAMES}")
        return 2
    only = a.only.split(",") if a.only else None
    run_id = a.run_id
    if a.resume and not run_id:
        runs = sorted((p for p in paths.runs_dir().glob(f"*-{cfg.profile}") if p.is_dir()), key=lambda p: p.name)
        run_id = runs[-1].name if runs else None
    run_dir, outs = run_loop(cfg, run_id, until=a.until, only=only,
                             on_start=lambda d: print(f"run directory: {d}", flush=True))
    g = outs.get("gate")
    if g:
        print(f"\n{g['_detail']}")
    rep = run_dir / "report" / "report.html"
    if rep.is_file():
        print(f"report: {rep}")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from failforge.data.splits import load_split
    from failforge.detector import Detector
    from failforge.eval.evaluate import evaluate

    cfg = _cfg(a)
    setup_logging(cfg.log_level)
    paths.configure_caches()
    ss = load_split(a.split, cfg.limit_images)
    preds = Detector(a.weights, cfg.detector).predict(ss, batch=cfg.detector.batch)
    res, _ = evaluate(ss, preds, cfg.detector.mine_conf)
    o = res["overall"]
    print(f"{a.split}: {o['n_images']} images  mAP@50:95 {o['map']:.4f}  mAP50 {o['map50']:.4f}")
    for k, v in res["slices"].items():
        print(f"  {k:<12} {v['n_images']:>5} img  mAP@50:95 {v['map'] if v['map'] is not None else float('nan'):.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    """Recompute the promotion decision from saved evaluation artifacts (no GPU, no data needed)."""
    from failforge.loop.gate import decide

    cfg = _cfg(a)
    d = Path(a.check)
    ev = d / "eval"
    need = [ev / "baseline.json", ev / "deltas.json"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        print("missing artifacts:", *missing, sep="\n  ")
        return 2
    base = json.loads((ev / "baseline.json").read_text(encoding="utf-8"))
    deltas = json.loads((ev / "deltas.json").read_text(encoding="utf-8"))
    arm = a.arm or ("synthetic" if "synthetic" in deltas else next(iter(deltas)))
    cand = json.loads((ev / f"{arm}.json").read_text(encoding="utf-8"))
    for name, r in (("baseline", base), (arm, cand)):
        for key in ("overall", "slices", "tide"):
            if key not in r:
                print(f"schema error: {name}.json has no '{key}'")
                return 2
    dec = decide(base, cand, deltas[arm], cfg.gate)
    print(f"candidate: {arm}")
    for c in dec["checks"]:
        v = "n/a" if c["value"] is None else f"{c['value']:+.4f}"
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']:<34} {v:>9}  (threshold {c['threshold']:+.4f}) {c['detail']}")
    print("decision:", "PROMOTE" if dec["promote"] else "REJECT")
    if a.expect:
        want = a.expect == "promote"
        return 0 if dec["promote"] == want else 1
    return 0 if dec["promote"] else 1


def cmd_report(a: argparse.Namespace) -> int:
    from failforge.report.report import build_report

    print(build_report(Path(a.run_dir)))
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    from failforge.bench.bench import run_bench

    cfg = _cfg(a)
    setup_logging(cfg.log_level)
    paths.configure_caches()
    out = Path(a.out) if a.out else paths.REPO / "benchmarks" / "results"
    res = run_bench(cfg, a.weights, out, synth=a.synth)
    print(json.dumps(res.get("summary", res), indent=1))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from failforge.app.server import serve

    setup_logging("INFO")
    return serve(host=a.host, port=a.port, open_browser=not a.no_browser, autostart=a.start)


def cmd_registry(a: argparse.Namespace) -> int:
    from failforge.loop.registry import Registry

    root = paths.registry_dir() / (a.profile or "full")
    r = Registry(root)
    c = r.champion()
    print("champion:", "none" if not c else f"v{c['version']} ({c['arm']}, run {c['run_id']}) golden {c['golden']}")
    for e in r.history():
        print(f"  {e['time']}  {e['event']:<10} {('v' + str(e['version'])) if 'version' in e else '':<4} "
              f"{e.get('arm', '')} {e.get('note', '') or e.get('reason', '')}")
    return 0


def cmd_publish(a: argparse.Namespace) -> int:
    from failforge.report.publish import publish

    print(publish(Path(a.run_dir), Path(a.dest) if a.dest else paths.reference_dir()))
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    from failforge.doctor import doctor

    return doctor()


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="failforge", description="Automated model QA and targeted synthetic data.")
    p.add_argument("--version", action="version", version=f"failforge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, config: bool = True) -> None:
        if config:
            sp.add_argument("-c", "--config", default=None, help="profile name (quick, full, smoke) or YAML path")
            sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config field")

    s = sub.add_parser("data", help="download / prepare / validate the dataset")
    s.add_argument("action", choices=["download", "prepare", "validate", "all"])
    s.add_argument("--rebuild", action="store_true", help="re-parse the raw labels")
    s.add_argument("--check-files", action="store_true", help="validate: also check every image exists")
    common(s)
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("models", help="download model weights")
    s.add_argument("action", choices=["download"])
    s.add_argument("--no-diffusion", action="store_true", help="skip Stable Diffusion / ControlNet (3.6 GB)")
    common(s)
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("loop", help="run or resume the closed loop")
    s.add_argument("--run-id", default=None, help="resume / name a run (folder under workspace/runs)")
    s.add_argument("--resume", action="store_true", help="resume the latest run of this profile")
    s.add_argument("--until", default=None, help="stop after this stage")
    s.add_argument("--only", default=None, help="comma-separated stages to (re)run, others are taken from cache")
    s.add_argument("--log-level", default=None)
    common(s)
    s.set_defaults(fn=cmd_loop)

    s = sub.add_parser("eval", help="evaluate a checkpoint")
    s.add_argument("--weights", required=True)
    s.add_argument("--split", default="golden")
    s.add_argument("--out", default=None)
    common(s)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("gate", help="promotion decision from saved artifacts (CI)")
    s.add_argument("--check", required=True, help="a run directory or reports/reference")
    s.add_argument("--arm", default=None)
    s.add_argument("--expect", choices=["promote", "reject"], default=None,
                   help="exit 0 when the decision matches (regression test of committed results)")
    common(s)
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("report", help="rebuild a run's report")
    s.add_argument("run_dir")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("bench", help="inference / generation benchmarks")
    s.add_argument("--weights", default=None, help="checkpoint (default: the full profile's champion)")
    s.add_argument("--out", default=None)
    s.add_argument("--synth", action="store_true", help="also time ControlNet generation")
    common(s)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("serve", help="dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--start", default=None, metavar="PROFILE", help="start a loop with this profile right away")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("registry", help="champion and promotion history")
    s.add_argument("--profile", default=None)
    s.set_defaults(fn=cmd_registry)

    s = sub.add_parser("publish", help="copy a run's compact results to reports/reference")
    s.add_argument("run_dir")
    s.add_argument("--dest", default=None)
    s.set_defaults(fn=cmd_publish)

    s = sub.add_parser("doctor", help="environment check")
    s.set_defaults(fn=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.command not in ("loop", "eval", "bench", "serve"):
        setup_logging("INFO")
    paths.configure_caches()
    try:
        return int(a.fn(a) or 0)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
