"""Evaluation report for one loop run: ``report/report.html`` (self-contained) + ``report/report.md``.

The report answers, in order: Is the model failing, and where? (golden slices, drift, TIDE) What
are the failure modes? (clusters) What did we synthesize, and was it any good? (gallery, QC, KID)
Did it help, compared with the honest alternatives? (ablation with paired-bootstrap CIs) And was the
result promoted? (gate checks). It reads only the run's artifacts, so it can be rebuilt any time.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

ARM_LABEL = {"baseline": "Baseline (day only)", "baseline_ft": "+ more epochs (no new data)",
             "augment": "+ classical augmentation", "synthetic": "+ ControlNet synthetic", "real": "+ real night (upper bound)"}
ARM_COLOR = {"baseline": "#8a8f98", "baseline_ft": "#5b6472", "augment": "#e08a2b", "synthetic": "#2f6fd6", "real": "#2a9d6a"}
SLICE_ORDER = ["overall", "day", "night", "dawn", "rainy", "snowy", "night_rainy", "clear", "overcast", "highway", "city_street"]


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=False)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.facecolor": "white"})
    return plt


def fig_slices(ablation: dict, slices: list[str]) -> bytes:
    plt = _style()
    arms = [a for a in ARM_LABEL if a in ablation]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    w = 0.8 / max(len(arms), 1)
    for i, a in enumerate(arms):
        ys = [ablation[a].get(s) or 0 for s in slices]
        xs = [k + (i - (len(arms) - 1) / 2) * w for k in range(len(slices))]
        ax.bar(xs, ys, w * 0.92, color=ARM_COLOR[a], label=ARM_LABEL[a])
    ax.set_xticks(range(len(slices)), slices)
    ax.set_ylabel("mAP@50:95 (golden set)")
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    return _fig_png(fig)


def fig_deltas(deltas: dict, slices: list[str]) -> bytes:
    plt = _style()
    arms = [a for a in ARM_LABEL if a in deltas]
    fig, axes = plt.subplots(1, len(slices), figsize=(2.6 * len(slices), 2.6), sharey=True)
    axes = [axes] if len(slices) == 1 else axes
    for ax, s in zip(axes, slices):
        for i, a in enumerate(arms):
            d = deltas[a].get(s)
            if not d:
                continue
            ax.errorbar(d["delta"], i, xerr=[[d["delta"] - d["ci_low"]], [d["ci_high"] - d["delta"]]], fmt="o",
                        color=ARM_COLOR[a], capsize=3, ms=5)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("Δ mAP@50:95 vs baseline")
        ax.set_yticks(range(len(arms)), [ARM_LABEL[a] for a in arms])
        ax.invert_yaxis()
    fig.suptitle("Paired bootstrap 95% CI (golden set)", fontsize=9, y=1.03)
    return _fig_png(fig)


def fig_umap(analysis: dict) -> bytes:
    plt = _style()
    import numpy as np

    pts = analysis["points"]
    x = np.array([p["x"] for p in pts])
    y = np.array([p["y"] for p in pts])
    c = np.array([p["c"] for p in pts])
    tp = np.array([p["k"] == "tp" for p in pts])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    ax.scatter(x[tp], y[tp], s=3, c="#c9ccd1", label="true positives", linewidths=0)
    modes = analysis["modes"]
    cmap = plt.get_cmap("tab10")
    for k, cid in enumerate(modes[:10]):
        m = (c == cid) & ~tp
        name = next(cl["name"] for cl in analysis["clusters"] if cl["id"] == cid)
        ax.scatter(x[m], y[m], s=5, color=cmap(k % 10), label=f"#{k + 1} {name}"[:38], linewidths=0)
    other = ~tp & ~np.isin(c, modes)
    ax.scatter(x[other], y[other], s=3, c="#7d828a", alpha=0.5, label="other errors", linewidths=0)
    ax.set_title("Failure clusters (UMAP of CLIP crops + photometrics)", fontsize=9)
    ax.legend(fontsize=6.5, markerscale=2.5, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[1]
    tod = np.array([p["tod"] for p in pts])
    for t, col in (("day", "#e0b63a"), ("dawn", "#d9733b"), ("night", "#273a8c")):
        m = (tod == t) & ~tp
        ax.scatter(x[m], y[m], s=4, color=col, label=t, linewidths=0)
    ax.set_title("Same errors, coloured by the true BDD time-of-day tag", fontsize=9)
    ax.legend(fontsize=7, markerscale=3, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    return _fig_png(fig)


def fig_tide(evals: dict) -> bytes:
    plt = _style()
    kinds = ["miss", "bkg", "loc", "cls", "both", "dupe"]
    cols = ["#d1495b", "#edae49", "#00798c", "#30638e", "#8d6a9f", "#9aa0a6"]
    rows = []
    for a in ("baseline", "synthetic"):
        e = evals.get(a)
        if not e:
            continue
        for s, t in e.get("tide_by_slice", {}).items():
            rows.append((f"{'base' if a == 'baseline' else 'synth'} / {s}", t))
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * max(len(rows), 1) + 0.9))
    for i, (_lab, t) in enumerate(rows):
        left = 0
        for k, col in zip(kinds, cols):
            v = t.get(k, 0)
            ax.barh(i, v, left=left, color=col, label=k if i == 0 else None)
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("errors at the operating point (golden set, TIDE types)")
    ax.legend(ncol=6, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    return _fig_png(fig)


def _b64(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _f(v, nd: int = 3, sign: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


CSS = """
:root{--bg:#ffffff;--fg:#1d2026;--muted:#5f6670;--line:#e3e6ea;--card:#f6f7f9;--ok:#1f8a5b;--bad:#c23b3b;--acc:#2f6fd6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121418;--fg:#e7e9ec;--muted:#9aa1ab;--line:#2a2e35;--card:#1a1d22;--ok:#3cc387;--bad:#ef6b6b;--acc:#6aa0ff}}
:root[data-theme="dark"]{--bg:#121418;--fg:#e7e9ec;--muted:#9aa1ab;--line:#2a2e35;--card:#1a1d22;--ok:#3cc387;--bad:#ef6b6b;--acc:#6aa0ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.kpi b{display:block;font-size:22px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.pass{color:var(--ok);font-weight:600}.fail{color:var(--bad);font-weight:600}img.fig{width:100%;background:#fff;border-radius:8px;border:1px solid var(--line)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.thumbs{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}.thumbs img{width:64px;height:64px;border-radius:4px}
.gallery img{width:100%;border-radius:6px;margin-bottom:8px}.tag{display:inline-block;padding:1px 7px;border-radius:99px;background:var(--line);font-size:12px;margin:1px}
.verdict{font-size:18px;font-weight:700;padding:12px 14px;border-radius:10px;border:1px solid var(--line);background:var(--card)}
.scroll{overflow-x:auto}code{font-size:12px}
"""


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "report"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    ev = run_dir / "eval"
    evals = {p.stem: _j(p) for p in sorted(ev.glob("*.json")) if p.stem not in ("deltas", "ablation")} if ev.is_dir() else {}
    deltas = _j(ev / "deltas.json") or {}
    ablation = _j(ev / "ablation.json") or {}
    if not ablation and "baseline" in evals:
        b = evals["baseline"]
        ablation = {"baseline": {"overall": b["overall"]["map"], **{k: v["map"] for k, v in b["slices"].items()}}}
    monitor = _j(run_dir / "monitor" / "monitor.json")
    mine = _j(run_dir / "mine" / "probe_eval.json")
    analysis = _j(run_dir / "analyze" / "analysis.json")
    synth = _j(run_dir / "synthesize" / "synthesis.json")
    gate = _j(run_dir / "gate" / "gate.json")
    cfg = _j(run_dir / "config.json") or {}
    status = _j(run_dir / "status.json") or {}
    slices = [s for s in SLICE_ORDER if any(s in a for a in ablation.values())][:7]
    tgt = (cfg.get("gate", {}).get("target_slices") or ["night"])[0]

    images: dict[str, bytes] = {}
    if ablation:
        images["slices"] = fig_slices(ablation, slices)
    if deltas:
        images["deltas"] = fig_deltas(deltas, [s for s in ("overall", "day", tgt, "dawn") if s in slices or s == "overall"])
    if analysis and analysis.get("points"):
        images["umap"] = fig_umap(analysis)
    if evals:
        images["tide"] = fig_tide(evals)
    for k, v in images.items():
        (figs / f"{k}.png").write_bytes(v)

    E = html.escape
    h: list[str] = []
    h.append(f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>FailForge Report</title><style>{CSS}</style></head><body><main>")
    h.append(f"<h1>FailForge evaluation report</h1><div class=muted>run <code>{E(run_dir.name)}</code> · profile "
             f"<b>{E(str(cfg.get('profile', '?')))}</b> · state {E(str(status.get('state', '?')))}</div>")
    if gate:
        ok = gate["decision"]["promote"]
        h.append(f"<p class='verdict {'pass' if ok else 'fail'}'>{E(gate['verdict'])}</p>")

    # Headline KPIs
    b = evals.get("baseline")
    if b:
        k = []
        cand = gate["candidate"] if gate else ("synthetic" if "synthetic" in evals else None)
        k.append(("Baseline overall", _f(b["overall"]["map"])))
        k.append((f"Baseline {tgt}", _f(b["slices"].get(tgt, {}).get("map"))))
        k.append(("Baseline day", _f(b["slices"].get("day", {}).get("map"))))
        if cand and cand in deltas:
            d = deltas[cand].get(tgt, {})
            k.append((f"Δ {tgt} ({cand})", f"{_f(d.get('delta'), sign=True)}"))
            k.append(("95% CI", f"{_f(d.get('ci_low'), sign=True)} … {_f(d.get('ci_high'), sign=True)}"))
            k.append(("Δ overall", _f(deltas[cand].get("overall", {}).get("delta"), sign=True)))
        h.append("<div class=grid>" + "".join(f"<div class=kpi><span class=muted>{E(a)}</span><b>{E(v)}</b></div>" for a, v in k)
                 + "</div><p class=muted>mAP@50:95, COCO protocol, on the frozen golden set "
                 f"({b['overall']['n_images']} images, {b['overall']['n_gt']} boxes).</p>")

    # 1. Health
    h.append("<h2>1. Is the model failing? Monitoring and golden slices</h2>")
    if monitor:
        dr = monitor["drift"]
        viol = ", ".join("{} {:.3f} &lt; {:.2f}".format(k, v["map"], v["slo"]) for k, v in monitor["slo_violations"].items())
        h.append(f"<p>Drift test, CLIP embeddings of {dr['n_reference']} training images vs {dr['n_current']} production images: "
                 f"MMD² = {dr['mmd2']:.4f} (permutation p = {dr['p_value']:.3g}), domain-classifier AUC = {dr['domain_auc']:.2f}, "
                 f"brightness PSI = {dr['psi_brightness']:.2f} (mean brightness {dr['brightness_ref_mean']:.2f} → {dr['brightness_cur_mean']:.2f}). "
                 f"SLO violations: {viol or 'none'}. "
                 f"<b>Loop {'triggered' if monitor['triggered'] else 'not triggered'}</b> ({', '.join(monitor['reasons']) or 'healthy'}).</p>")
    if b:
        h.append("<div class=scroll><table><tr><th>slice</th><th>images</th><th>mAP@50:95</th><th>mAP50</th><th>small</th><th>medium</th><th>large</th></tr>")
        rows = [("overall", b["overall"])] + [(s, b["slices"][s]) for s in SLICE_ORDER if s in b["slices"]]
        for s, r in rows:
            h.append(f"<tr><td>{E(s)}</td><td>{r['n_images']}</td><td>{_f(r['map'])}</td><td>{_f(r['map50'])}</td>"
                     f"<td>{_f(r.get('map_small'))}</td><td>{_f(r.get('map_medium'))}</td><td>{_f(r.get('map_large'))}</td></tr>")
        h.append("</table></div>")
    if "tide" in images:
        h.append(f"<img class=fig alt='TIDE error types' src='{_b64(images['tide'])}'>")

    # 2. Failure modes
    if analysis:
        h.append("<h2>2. Failure modes found on the production probe</h2>")
        if mine:
            t = mine["tide"]
            h.append(f"<p>Mined at the deployment operating point (conf ≥ {t['operating_conf']}): {t['miss']} missed objects, {t['bkg']} background "
                     f"false positives, {t['loc']} localization, {t['cls']} classification, {t['both']} both, {t['dupe']} duplicates "
                     f"(precision {t['precision']:.2f}, recall {t['recall']:.2f}). {analysis['n_errors']} errors and "
                     f"{analysis['n_items'] - analysis['n_errors']} sampled true positives were clustered; "
                     f"{100 * analysis['noise_fraction']:.0f}% left unassigned by HDBSCAN.</p>")
        if "umap" in images:
            h.append(f"<img class=fig alt='UMAP of failures' src='{_b64(images['umap'])}'>")
        h.append("<p class=muted>Cluster names come from CLIP zero-shot enrichment against a fixed vocabulary of conditions; "
                 "the true BDD tags (right plot, and 'true time of day' below) are shown only to validate the unsupervised result.</p>")
        h.append("<div class=cards>")
        thumbs_dir = run_dir / "analyze" / "thumbs"
        for cl in [c for c in analysis["clusters"] if c["id"] in analysis["modes"]][:6]:
            tod = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(cl["true_timeofday"].items(), key=lambda x: -x[1]))
            th = "".join(f"<img alt='' src='{_b64((thumbs_dir / t['file']).read_bytes(), 'image/jpeg')}'>"
                         for t in cl.get("thumbs", [])[:8] if (thumbs_dir / t["file"]).is_file())
            kinds = ", ".join(f"{k} {v}" for k, v in cl["kinds"].items())
            h.append(f"<div class=card><b>#{cl['rank']} {E(cl['name'])}</b><div class=muted>{cl['errors']} errors · error rate "
                     f"{100 * cl['error_rate']:.0f}% · lift ×{cl['lift']:.2f} · excess {cl['excess_errors']:.0f}</div>"
                     f"<div>{''.join(f'<span class=tag>{E(c)}</span>' for c in list(cl['classes'])[:4])}</div>"
                     f"<div class=muted>{E(kinds)}</div><div class=muted>true time of day: {E(tod)}</div>"
                     f"<div>synthesis target: <b>{E(cl['synthesis_target'] or 'not addressable by synthesis')}</b></div>"
                     f"<div class=thumbs>{th}</div></div>")
        h.append("</div>")

    # 3. Synthesis
    if synth:
        h.append("<h2>3. Targeted synthesis and quality control</h2>")
        s = synth.get("synthetic")
        if s:
            rej = ", ".join(f"{k} {v}" for k, v in s.get("rejections", {}).items()) or "none"
            sec = s.get("seconds")
            h.append(f"<p>{s['accepted']} of {s['candidates']} candidates accepted ({100 * s['acceptance_rate']:.0f}%); rejected by: {E(rej)}. "
                     f"Accepted images: mean CLIP prompt score {_f(s.get('mean_clip_score'))}, P(target condition) {_f(s.get('mean_p_target'), 2)}, "
                     f"boxes keeping their structure {_f(s.get('mean_box_pass'), 2)}"
                     + (f"; {sec / max(s['candidates'], 1):.1f} s per candidate ({sec / 3600:.2f} GPU-hours)" if sec else "") + ".</p>")
        kid = synth.get("kid_vs_real_target") or {}
        if kid:
            h.append("<table><tr><th>set</th><th>KID vs real target-condition images (CLIP features) ↓</th><th>n</th></tr>"
                     + "".join(f"<tr><td>{E(k)}</td><td>{v['kid']:.4f} ± {v['std']:.4f}</td><td>{v['n']}</td></tr>" for k, v in kid.items())
                     + "</table>")
        gal = run_dir / "synthesize" / "synthetic" / "gallery"
        tiles = sorted(gal.glob("[0-9]*.jpg"))[:8] if gal.is_dir() else []
        rejected = sorted(gal.glob("rejected_*.jpg"))[:1] if gal.is_dir() else []
        if tiles:
            h.append("<p class=muted>Each row: source day photo · Canny control · depth control · generated image with the inherited boxes.</p><div class=gallery>")
            h += [f"<img alt='synthetic example' src='{_b64(t.read_bytes(), 'image/jpeg')}'>" for t in tiles]
            for r in rejected:
                h.append(f"<p class=muted>Rejected by the filters ({E(r.stem.split('_', 2)[-1])}):</p><img alt='rejected example' src='{_b64(r.read_bytes(), 'image/jpeg')}'>")
            h.append("</div>")

    # 4. Ablation
    if len(ablation) > 1:
        h.append("<h2>4. Did it help? Ablation on the golden set</h2>")
        h.append("<p>Every arm fine-tunes the same baseline for the same number of epochs on the same day images plus N extra images "
                 "(N identical across arms). <i>More epochs</i> isolates the extra training; <i>classical augmentation</i> is the cheap "
                 "alternative; <i>real night</i> is the upper bound if we had paid for labels.</p>")
        h.append(f"<img class=fig alt='ablation by slice' src='{_b64(images['slices'])}'>")
        if "deltas" in images:
            h.append(f"<img class=fig alt='bootstrap deltas' src='{_b64(images['deltas'])}'>")
        h.append("<div class=scroll><table><tr><th>arm</th>" + "".join(f"<th>{E(s)}</th>" for s in slices) + f"<th>Δ {E(tgt)} [95% CI]</th></tr>")
        for a in [x for x in ARM_LABEL if x in ablation]:
            d = deltas.get(a, {}).get(tgt)
            ci = f"{_f(d['delta'], sign=True)} [{_f(d['ci_low'], sign=True)}, {_f(d['ci_high'], sign=True)}]" if d else "–"
            h.append(f"<tr><td>{E(ARM_LABEL[a])}</td>" + "".join(f"<td>{_f(ablation[a].get(s))}</td>" for s in slices) + f"<td>{ci}</td></tr>")
        h.append("</table></div>")
        if synth and "synthetic" in synth and "synthetic" in deltas and synth["synthetic"].get("seconds"):
            d = deltas["synthetic"].get(tgt, {}).get("delta")
            gh = synth["synthetic"]["seconds"] / 3600
            if d and d > 0:
                h.append(f"<p>Cost: {gh:.2f} GPU-hours of generation for {100 * d:.1f} mAP points on <i>{E(tgt)}</i> → "
                         f"{gh / (100 * d):.2f} GPU-hours per mAP point.</p>")

    # 5. Gate
    if gate:
        h.append("<h2>5. Promotion gate</h2><table><tr><th>check</th><th>value</th><th>threshold</th><th>result</th></tr>")
        for c in gate["decision"]["checks"]:
            h.append(f"<tr><td>{E(c['name'])}</td><td>{_f(c['value'], 4, True)}</td><td>{_f(c['threshold'], 4, True)}</td>"
                     f"<td class={'pass' if c['passed'] else 'fail'}>{'PASS' if c['passed'] else 'FAIL'}</td></tr>")
        h.append("</table>")
        other = ", ".join(a + (": pass" if d["promote"] else ": fail") for a, d in gate.get("all_arms", {}).items())
        h.append(f"<p class=muted>Gate outcome per arm (same rules): {other}.</p>")
    h.append("<p class=muted>Generated by FailForge. Data: BDD100K (Berkeley DeepDrive, non-commercial research license).</p></main></body></html>")
    (out / "report.html").write_text("\n".join(h), encoding="utf-8")
    (out / "report.md").write_text(markdown(run_dir, evals, deltas, ablation, monitor, analysis, synth, gate, slices, tgt),
                                   encoding="utf-8")
    return out / "report.html"


def markdown(run_dir, evals, deltas, ablation, monitor, analysis, synth, gate, slices, tgt) -> str:
    L = [f"# FailForge report: `{run_dir.name}`", ""]
    if gate:
        L += [f"**{gate['verdict']}**", ""]
    if ablation:
        L += ["## Golden-set mAP@50:95 by slice", "", "| arm | " + " | ".join(slices) + f" | Δ {tgt} [95% CI] |",
              "|---|" + "---:|" * (len(slices) + 1)]
        for a in [x for x in ARM_LABEL if x in ablation]:
            d = deltas.get(a, {}).get(tgt)
            ci = f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "–"
            L.append(f"| {ARM_LABEL[a]} | " + " | ".join(_f(ablation[a].get(s)) for s in slices) + f" | {ci} |")
        L += ["", "![slices](figures/slices.png)", ""]
        if deltas:
            L += ["![deltas](figures/deltas.png)", ""]
    if monitor:
        d = monitor["drift"]
        L += ["## Monitoring", "", f"- MMD² {d['mmd2']:.4f}, p = {d['p_value']:.3g}; domain AUC {d['domain_auc']:.2f}; brightness PSI {d['psi_brightness']:.2f}",
              f"- SLO violations: {', '.join(monitor['slo_violations']) or 'none'}; triggered: {monitor['triggered']}", ""]
    if analysis:
        L += ["## Failure modes", "", "| # | cluster | errors | lift | excess | true time of day | synthesis target |", "|---|---|---:|---:|---:|---|---|"]
        for cl in [c for c in analysis["clusters"] if c["id"] in analysis["modes"]][:8]:
            tod = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(cl["true_timeofday"].items(), key=lambda x: -x[1])[:2])
            L.append(f"| {cl['rank']} | {cl['name']} | {cl['errors']} | {cl['lift']:.2f} | {cl['excess_errors']:.0f} | {tod} | {cl['synthesis_target'] or '–'} |")
        L += ["", "![umap](figures/umap.png)", ""]
    if synth and synth.get("synthetic"):
        s = synth["synthetic"]
        L += ["## Synthesis", "", f"- accepted {s['accepted']}/{s['candidates']} ({100 * s['acceptance_rate']:.0f}%), rejections {s.get('rejections')}"]
        for k, v in (synth.get("kid_vs_real_target") or {}).items():
            L.append(f"- KID vs real target images, {k}: {v['kid']:.4f} ± {v['std']:.4f}")
        L.append("")
    if gate:
        L += ["## Promotion gate", "", "| check | value | threshold | result |", "|---|---:|---:|---|"]
        for c in gate["decision"]["checks"]:
            L.append(f"| {c['name']} | {_f(c['value'], 4, True)} | {_f(c['threshold'], 4, True)} | {'PASS' if c['passed'] else 'FAIL'} |")
    return "\n".join(L) + "\n"
