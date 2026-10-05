"""Copy a finished run's compact, shareable results into ``reports/reference/`` (committed to git).

What travels: evaluation JSON (per arm, deltas, ablation), monitor / mining / analysis summaries
(UMAP points included, but not the feature matrices), synthesis QC summaries and the gallery, the
gate decision, the report (HTML + Markdown + figures), the run config, and the baseline and
champion checkpoints (YOLO11n is ~5.5 MB each). What does not: images, predictions, datasets.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def publish(run_dir: Path, dest: Path) -> Path:
    run_dir, dest = Path(run_dir), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    def cp(rel: str) -> None:
        src = run_dir / rel
        if src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
        elif src.is_dir():
            shutil.copytree(src, dest / rel, dirs_exist_ok=True)

    for rel in ("config.json", "status.json", "eval/baseline.json", "eval/deltas.json", "eval/ablation.json",
                "monitor/monitor.json", "mine/probe_eval.json", "analyze/analysis.json", "analyze/thumbs",
                "synthesize/synthesis.json", "synthesize/synthetic/summary.json", "synthesize/synthetic/plan.json",
                "synthesize/synthetic/gallery", "synthesize/augment/summary.json", "synthesize/augment/gallery",
                "gate/gate.json", "report", "metrics.jsonl"):
        cp(rel)
    for p in (run_dir / "eval").glob("*.json"):
        cp(f"eval/{p.name}")
    for p in (run_dir / "stages").glob("*.json"):
        cp(f"stages/{p.name}")
    # Mined errors are large; keep only the summary.
    mined = run_dir / "mine" / "mined.json"
    if mined.is_file():
        d = json.loads(mined.read_text(encoding="utf-8"))
        (dest / "mine").mkdir(exist_ok=True)
        (dest / "mine" / "mined_summary.json").write_text(json.dumps(d["summary"], indent=1), encoding="utf-8")
    w = dest / "weights"
    w.mkdir()
    base = run_dir / "baseline" / "weights.pt"
    if base.is_file():
        shutil.copy2(base, w / "baseline.pt")
    gate = run_dir / "gate" / "gate.json"
    if gate.is_file():
        g = json.loads(gate.read_text(encoding="utf-8"))
        cand = run_dir / "retrain" / g["candidate"] / "weights.pt"
        if cand.is_file():
            shutil.copy2(cand, w / f"{g['candidate']}.pt")
    # Paths inside the copied status file point at the author's machine; strip them.
    st = dest / "status.json"
    if st.is_file():
        s = json.loads(st.read_text(encoding="utf-8"))
        s.pop("pid", None)
        st.write_text(json.dumps(s, indent=1), encoding="utf-8")
    for p in (dest / "eval").glob("*.json"):
        e = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(e, dict) and "model" in e:
            e["model"]["weights"] = Path(e["model"]["weights"]).name
            p.write_text(json.dumps(e, indent=1), encoding="utf-8")
    scrub(dest)
    return dest


def scrub(dest: Path) -> None:
    """Replace absolute machine paths (workspace, repository) in text artifacts with placeholders."""
    from failforge import paths

    roots = {str(paths.home()): "<workspace>", str(paths.REPO): "<repo>"}
    variants = {}
    for r, tag in roots.items():
        for v in (r, r.replace("\\", "/"), r.replace("\\", "\\\\"), json.dumps(r)[1:-1]):
            variants[v] = tag
    for p in dest.rglob("*"):
        if p.suffix not in (".json", ".jsonl", ".md", ".txt", ".yaml"):
            continue
        t = p.read_text(encoding="utf-8")
        for v in sorted(variants, key=len, reverse=True):
            t = t.replace(v, variants[v])
        p.write_text(t, encoding="utf-8")
