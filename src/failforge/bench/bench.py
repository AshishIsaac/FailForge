"""Inference and generation benchmarks: latency, throughput, memory.

Detector (the model FailForge promotes), at the training resolution:

* PyTorch eager, fp32 and fp16 (GPU), model forward only, batch 1 and 8
* ONNX Runtime (exported by Ultralytics, dynamic batch), CUDA / TensorRT / CPU execution providers,
  whichever this machine has
* end to end ``predict`` on real golden images (decode + letterbox + forward + NMS), batch 1

Synthesis (``--synth``): seconds per ControlNet image and peak GPU memory, the number that sets
the cost of a synthetic mAP point.

Latency is reported as p50 / p90 / p99 over ``bench.runs`` timed iterations after ``bench.warmup``
warm-up iterations, with CUDA synchronization around every timed call.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np

from failforge import paths
from failforge.config import Config

log = logging.getLogger(__name__)


def _stats(ms: list[float], batch: int) -> dict:
    a = np.array(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean()),
            "throughput_img_s": float(batch * 1000.0 / a.mean()), "batch": batch, "runs": len(ms)}


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        return float("nan")


def _default_weights() -> str:
    from failforge.loop.registry import Registry

    c = Registry(paths.registry_dir() / "full").champion()
    if c:
        return c["weights"]
    ref = paths.reference_dir() / "weights"
    for n in ("synthetic.pt", "baseline.pt"):
        if (ref / n).is_file():
            return str(ref / n)
    raise FileNotFoundError("no checkpoint: pass --weights, or run the loop once")


def bench_torch(weights: str, cfg: Config) -> list[dict]:
    import torch
    from ultralytics import YOLO

    out = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    precisions = ["fp32", "fp16"] if dev == "cuda" else ["fp32"]
    for prec in precisions:
        model = YOLO(weights).model.to(dev).eval()
        if prec == "fp16":
            model = model.half()
        for bs in cfg.bench.batch_sizes:
            x = torch.rand(bs, 3, cfg.detector.imgsz, cfg.detector.imgsz, device=dev,
                           dtype=torch.float16 if prec == "fp16" else torch.float32)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ts = []
            with torch.inference_mode():
                for i in range(cfg.bench.warmup + cfg.bench.runs):
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(x)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    if i >= cfg.bench.warmup:
                        ts.append((time.perf_counter() - t0) * 1000)
            r = {"engine": f"PyTorch {dev.upper()} {prec}", **_stats(ts, bs)}
            if dev == "cuda":
                r["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
            r["rss_mb"] = _rss_mb()
            out.append(r)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    return out


def bench_onnx(weights: str, cfg: Config, work: Path) -> list[dict]:
    try:
        import onnxruntime as ort
    except ImportError:
        return [{"engine": "ONNX Runtime", "skipped": "onnxruntime not installed"}]
    from ultralytics import YOLO

    work.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(YOLO(weights).export(format="onnx", imgsz=cfg.detector.imgsz, dynamic=True, simplify=True,
                                          opset=17, verbose=False))
    avail = ort.get_available_providers()
    out = []
    candidates = [("TensorrtExecutionProvider", "ONNX Runtime TensorRT fp16",
                   [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                   "trt_engine_cache_path": str(work / "trt")}), "CUDAExecutionProvider"]),
                  ("CUDAExecutionProvider", "ONNX Runtime CUDA fp32", ["CUDAExecutionProvider"]),
                  ("CPUExecutionProvider", "ONNX Runtime CPU fp32", ["CPUExecutionProvider"])]
    for key, name, providers in candidates:
        if key not in avail:
            out.append({"engine": name, "skipped": "provider not available on this machine"})
            continue
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception as e:  # noqa: BLE001 - e.g. TensorRT libraries missing at runtime
            out.append({"engine": name, "skipped": f"session failed: {str(e)[:120]}"})
            continue
        if sess.get_providers()[0] != key:
            out.append({"engine": name, "skipped": f"fell back to {sess.get_providers()[0]}"})
            continue
        inp = sess.get_inputs()[0].name
        for bs in cfg.bench.batch_sizes:
            if key == "CPUExecutionProvider" and bs > 1:
                continue
            x = np.random.default_rng(0).random((bs, 3, cfg.detector.imgsz, cfg.detector.imgsz), dtype=np.float32)
            runs = cfg.bench.runs if key != "CPUExecutionProvider" else max(cfg.bench.runs // 4, 5)
            ts = []
            for i in range(cfg.bench.warmup + runs):
                t0 = time.perf_counter()
                sess.run(None, {inp: x})
                if i >= cfg.bench.warmup:
                    ts.append((time.perf_counter() - t0) * 1000)
            out.append({"engine": name, **_stats(ts, bs), "rss_mb": _rss_mb()})
        del sess
    return out


def bench_end_to_end(weights: str, cfg: Config) -> dict:
    from failforge.data.splits import load_split
    from failforge.detector import Detector

    try:
        ss = load_split("golden")[: cfg.bench.warmup + cfg.bench.runs]
    except FileNotFoundError:
        return {"engine": "end-to-end predict", "skipped": "no golden set on this machine"}
    det = Detector(weights, cfg.detector)
    ts = []
    for i, s in enumerate(ss):
        t0 = time.perf_counter()
        det.predict([s], conf=0.25, batch=1)
        if i >= cfg.bench.warmup:
            ts.append((time.perf_counter() - t0) * 1000)
    return {"engine": f"end-to-end predict ({det.device}, fp16={det.device != 'cpu'})", **_stats(ts, 1)}


def bench_synthesis(cfg: Config, n: int = 6) -> dict:
    import cv2
    import torch

    from failforge.data.splits import load_split
    from failforge.synthesis.controlnet import ControlNetGenerator

    src = load_split("train_day")[:n + 1]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    gen = ControlNetGenerator(cfg.synthesis)
    t_load = time.perf_counter() - t_load
    ts = []
    for i, s in enumerate(src):
        rgb = cv2.cvtColor(cv2.imread(s.image), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        gen.generate(rgb, "a realistic dashcam photo of a city street at night", "cartoon", 1000 + i)
        if i > 0:
            ts.append(time.perf_counter() - t0)
    c = cfg.synthesis
    return {"engine": f"SD1.5 + {len(c.controlnets)} ControlNets img2img {c.width}x{c.height}, {c.steps} steps"
                      f"{', CPU offload' if c.cpu_offload else ''}",
            "s_per_image": float(np.mean(ts)), "load_s": t_load, "images": len(ts),
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None}


def run_bench(cfg: Config, weights: str | None, out_dir: Path, synth: bool = False) -> dict:
    import torch

    weights = weights or _default_weights()
    env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu": platform.processor(), "imgsz": cfg.detector.imgsz, "weights": Path(weights).name}
    res = {"env": env, "detector": [], "synthesis": None}
    log.info("PyTorch ...")
    res["detector"] += bench_torch(weights, cfg)
    if cfg.bench.onnx:
        log.info("ONNX Runtime ...")
        res["detector"] += bench_onnx(weights, cfg, paths.models_dir() / "bench")
    log.info("end to end ...")
    res["detector"].append(bench_end_to_end(weights, cfg))
    if synth:
        log.info("ControlNet generation ...")
        res["synthesis"] = bench_synthesis(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "bench.md").write_text(to_markdown(res), encoding="utf-8")
    res["summary"] = [f"{r['engine']} b{r.get('batch', '')}: " + (f"{r['p50_ms']:.2f} ms p50, {r['throughput_img_s']:.0f} img/s"
                                                                  if "p50_ms" in r else r.get("skipped", ""))
                      for r in res["detector"]]
    return res


def to_markdown(res: dict) -> str:
    e = res["env"]
    L = [f"Machine: {e['gpu'] or 'CPU only'} · {e['platform']} · Python {e['python']} · torch {e['torch']} · "
         f"model `{e['weights']}` at {e['imgsz']}x{e['imgsz']}", "",
         "| engine | batch | p50 ms | p90 ms | p99 ms | throughput (img/s) | GPU peak MB | process RSS MB |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["detector"]:
        if "skipped" in r:
            L.append(f"| {r['engine']} | | | | | *{r['skipped']}* | | |")
            continue
        gp = f"{r['gpu_peak_mb']:.0f}" if r.get("gpu_peak_mb") else ""
        rs = f"{r['rss_mb']:.0f}" if r.get("rss_mb") == r.get("rss_mb") and r.get("rss_mb") else ""
        L.append(f"| {r['engine']} | {r['batch']} | {r['p50_ms']:.2f} | {r['p90_ms']:.2f} | {r['p99_ms']:.2f} | "
                 f"{r['throughput_img_s']:.0f} | {gp} | {rs} |")
    s = res.get("synthesis")
    if s:
        L += ["", f"Synthesis: {s['engine']}: **{s['s_per_image']:.2f} s/image**, peak GPU memory "
                  f"{(s['gpu_peak_mb'] or 0):.0f} MB, pipeline load {s['load_s']:.0f} s."]
    return "\n".join(L) + "\n"
