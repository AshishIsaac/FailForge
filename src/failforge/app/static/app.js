/* FailForge dashboard (no build step, no dependencies). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f3 = (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(3));
const sgn = (v, n = 3) => (v === null || v === undefined ? "–" : (v >= 0 ? "+" : "") + Number(v).toFixed(n));
const ARM = { baseline: "Baseline (day only)", baseline_ft: "+ more epochs", augment: "+ classical augmentation", synthetic: "+ ControlNet synthetic", real: "+ real night (upper bound)" };
const ARMS = Object.keys(ARM);
const PALETTE = ["#2f6fd6", "#e08a2b", "#2a9d6a", "#c23b3b", "#8d5fd3", "#1aa3b8", "#b7791f", "#d6457f", "#5b6472", "#6b8e23"];
const TOD = { day: "#e0b63a", dawn: "#d9733b", night: "#273a8c" };

const S = { overview: null, runs: [], rid: null, run: null, analysis: null, analysisFor: null, profile: "quick", tab: "pipeline",
  highlight: null, colorBy: "cluster" };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
}
const fileUrl = (rel) => `/api/runs/${encodeURIComponent(S.rid)}/file/${rel}`;

/* ---------------------------------------------------------------- theme */
try { const t = localStorage.getItem("ff-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }
$("#theme").onclick = () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("ff-theme", next); } catch (e) { /* ignore */ }
  drawScatter();
};

/* ---------------------------------------------------------------- sidebar */
function renderProfiles() {
  const o = S.overview;
  $("#env").textContent = `${o.gpu || "CPU only"} · workspace ${o.workspace}` + (o.golden ? ` · golden set ${o.golden.n} images (frozen ${o.golden.created})` : "");
  $("#profiles").innerHTML = Object.entries(o.profiles).map(([k, p]) =>
    `<div class="profile ${k === S.profile ? "on" : ""}" data-p="${k}"><b>${esc(p.title)}</b> <span class="muted small">· ${esc(o.gpu ? p.gpu : p.cpu)}</span><div>${esc(p.detail)}</div></div>`).join("");
  for (const el of document.querySelectorAll(".profile")) el.onclick = () => { S.profile = el.dataset.p; renderProfiles(); };
  const busy = o.job.running;
  $("#start").disabled = busy || !o.data_ready;
  $("#start").textContent = busy ? "A loop is running…" : "Start a new loop";
  $("#start-msg").textContent = !o.data_ready ? "Data not prepared yet: run python run.py (first-run setup) or `failforge data prepare`." : "";
  const ch = Object.entries(o.champions);
  $("#champions").innerHTML = ch.length ? ch.map(([p, c]) => `<div><b>${esc(p)}</b>: v${c.version} (${esc(c.arm)}) · night ${f3(c.golden.night_map)} · overall ${f3(c.golden.map)}</div>`).join("")
    : `<span class="muted">none yet - the first loop registers its baseline as v1</span>`;
}

function renderRuns() {
  $("#runs").innerHTML = S.runs.map((r) => `<div class="run ${r.id === S.rid ? "on" : ""}" data-id="${esc(r.id)}">
      <div class="id"><span class="dot ${esc(r.state)}"></span>${r.reference ? "Reference run (in the repo)" : esc(r.id)}</div>
      <div class="st">${esc(r.profile || "")} · ${esc(r.state)}${r.stage ? " · " + esc(r.stage) : ""}${r.verdict ? " · " + (r.promote ? "promoted" : "rejected") : ""}</div></div>`).join("")
    || `<div class="muted small">No runs yet.</div>`;
  for (const el of document.querySelectorAll(".run")) el.onclick = () => selectRun(el.dataset.id);
}

$("#start").onclick = async () => {
  $("#start").disabled = true;
  try {
    const r = await api("/api/loop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profile: S.profile }) });
    $("#start-msg").textContent = `Started ${r.run_id}`;
    await refresh();
    selectRun(r.run_id);
  } catch (e) { $("#start-msg").textContent = e.message; }
};

/* ---------------------------------------------------------------- tabs */
for (const b of document.querySelectorAll("#tabs button")) b.onclick = () => {
  S.tab = b.dataset.tab;
  for (const x of document.querySelectorAll("#tabs button")) x.classList.toggle("on", x === b);
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("on", t.id === "tab-" + S.tab);
  renderTab();
};

async function selectRun(id) {
  S.rid = id;
  S.analysis = null;
  S.highlight = null;
  renderRuns();
  await loadRun();
}

async function loadRun() {
  if (!S.rid) return;
  try { S.run = await api(`/api/runs/${encodeURIComponent(S.rid)}`); } catch (e) { S.run = null; }
  renderHead();
  renderTab();
}

/* ---------------------------------------------------------------- head */
function renderHead() {
  const r = S.run;
  if (!r) { $("#run-head").innerHTML = `<div class="empty">Select a run.</div>`; return; }
  const s = r.summary, b = r.baseline, cfgT = (r.config?.gate?.target_slices || ["night"])[0];
  const cand = r.gate?.candidate || "synthetic";
  const d = r.deltas?.[cand]?.[cfgT];
  const job = S.overview?.job;
  const live = job?.running && job.run_id === S.rid;
  const canResume = !s.reference && ["stopped", "failed", "interrupted"].includes(s.state) && !S.overview?.job.running;
  let kp = "";
  if (b) {
    kp = `<div class="kpis">
      <div class="kpi"><span>Baseline overall</span><b>${f3(b.overall.map)}</b></div>
      <div class="kpi"><span>Baseline day</span><b>${f3(b.slices.day?.map)}</b></div>
      <div class="kpi"><span>Baseline ${esc(cfgT)}</span><b>${f3(b.slices[cfgT]?.map)}</b></div>
      ${d ? `<div class="kpi"><span>Δ ${esc(cfgT)} (${esc(cand)})</span><b>${sgn(d.delta)}</b></div>
      <div class="kpi"><span>95% CI</span><b style="font-size:15px">${sgn(d.ci_low)} … ${sgn(d.ci_high)}</b></div>` : ""}
    </div>`;
  }
  $("#run-head").innerHTML = `<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <h2>${s.reference ? "Reference run" : esc(s.id)}</h2><span class="muted">${esc(s.profile || "")} · <span class="dot ${esc(s.state)}"></span>${esc(s.state)}</span>
      <span style="flex:1"></span>
      ${r.has_report ? `<a href="${fileUrl("report/report.html")}" target="_blank"><button>Open full report</button></a>` : ""}
      ${live ? `<button class="danger" id="stop">Stop (resumable)</button>` : ""}
      ${canResume ? `<button id="resume">Resume</button>` : ""}
    </div>
    ${s.reference ? `<div class="muted small">Results of the full loop on the author's machine, committed in reports/reference. Start a loop to reproduce them here.</div>` : ""}
    ${r.gate ? `<div class="verdict ${r.gate.decision.promote ? "pass" : "fail"}">${esc(r.gate.verdict)}</div>` : ""}${kp}`;
  const st = $("#stop");
  if (st) st.onclick = async () => { st.disabled = true; await api(`/api/runs/${encodeURIComponent(S.rid)}/stop`, { method: "POST" }); };
  const rs = $("#resume");
  if (rs) rs.onclick = async () => {
    rs.disabled = true;
    await api("/api/loop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profile: s.profile, run_id: s.id }) });
    refresh();
  };
}

/* ---------------------------------------------------------------- tabs */
function renderTab() {
  if (!S.run) return;
  if (S.tab === "pipeline") renderPipeline();
  else if (S.tab === "failures") renderFailures();
  else if (S.tab === "synthesis") renderSynthesis();
  else if (S.tab === "results") renderResults();
  else if (S.tab === "log") loadLog();
}

function fmtDur(s) { if (!s && s !== 0) return ""; s = Math.round(s); return s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`; }

function renderPipeline() {
  const st = S.run.status;
  if (!st) { $("#tab-pipeline").innerHTML = `<div class="card empty">No status yet.</div>`; return; }
  const rows = st.stages.map((s) => {
    const p = s.progress && s.progress[1] ? Math.min(100, (100 * s.progress[0]) / s.progress[1]) : null;
    return `<div class="stage"><div class="ic ${esc(s.state)}"></div>
      <div><div class="t">${esc(s.title)}</div><div class="d">${esc(s.detail || "")}</div>${s.state === "active" && p !== null ? `<div class="bar"><i style="width:${p}%"></i></div>` : ""}</div>
      <div class="muted small">${s.state === "done" ? fmtDur(s.elapsed) : s.state === "active" && p !== null ? Math.round(p) + "%" : esc(s.state)}</div></div>`;
  }).join("");
  const err = st.error ? `<div class="card fail">${esc(st.error)}</div>` : "";
  const mon = S.run.monitor;
  const monCard = mon ? `<div class="card"><h4>Monitoring: why the loop ran</h4><div class="small">
     Drift (CLIP embeddings, train vs production): MMD² ${mon.drift.mmd2.toFixed(4)}, permutation p = ${mon.drift.p_value.toPrecision(2)},
     domain-classifier AUC ${mon.drift.domain_auc.toFixed(2)}, brightness PSI ${mon.drift.psi_brightness.toFixed(2)}.<br>
     SLO violations: ${Object.entries(mon.slo_violations).map(([k, v]) => `${esc(k)} ${f3(v.map)} &lt; ${v.slo}`).join(", ") || "none"} →
     <b>${mon.triggered ? "triggered" : "healthy, nothing to do"}</b> (${esc(mon.reasons.join(", "))})</div></div>` : "";
  $("#tab-pipeline").innerHTML = `${err}<div class="card">${rows}</div>${monCard}`;
}

/* ---------------------------------------------------------------- failures */
async function renderFailures() {
  const el = $("#tab-failures");
  if (S.analysisFor !== S.rid) {
    el.innerHTML = `<div class="card empty">Loading…</div>`;
    try { S.analysis = await api(`/api/runs/${encodeURIComponent(S.rid)}/analysis`); } catch (e) { S.analysis = null; }
    S.analysisFor = S.rid;
  }
  const a = S.analysis;
  if (!a) { el.innerHTML = `<div class="card empty">Failure analysis not available yet (stage "analyze").</div>`; return; }
  const modes = a.clusters.filter((c) => a.modes.includes(c.id));
  const probe = S.run.probe?.tide;
  el.innerHTML = `<div class="split"><div class="card"><h4>Failure map: UMAP of ${a.n_errors} errors + ${a.n_items - a.n_errors} sampled true positives</h4>
      <canvas id="sc" height="460"></canvas>
      <div class="legend" id="lg"></div>
      <div class="small muted" style="margin-top:6px">Colour by: <button id="cb-cl" class="${S.colorBy === "cluster" ? "on" : ""}">failure mode</button>
      <button id="cb-tod">true time of day (validation)</button> · hover a point · click a card to highlight</div></div>
    <div class="card"><h4>What went wrong (TIDE error types, probe set)</h4>${probe ? tideTable(probe) : ""}
      <p class="small muted">Clusters are found without the dataset's tags: CLIP crop embeddings + photometric descriptors → UMAP → HDBSCAN,
      named by CLIP zero-shot enrichment. ${(100 * a.noise_fraction).toFixed(0)}% of points are unassigned (HDBSCAN noise).</p>
      ${a.fallback ? `<p class="small fail">No addressable cluster: fell back to the SLO-violating slices.</p>` : ""}
      <h4 style="margin-top:12px">Synthesis targets</h4>${a.targets.map((t) => `<div class="small">• ${esc(t.name)} → <b>${esc(t.condition)}</b> (${esc(t.classes.join(", "))})</div>`).join("") || "<div class=small>none</div>"}</div></div>
    <div class="clusters">${modes.map((c, i) => clusterCard(c, i)).join("")}</div>`;
  $("#cb-cl").onclick = () => { S.colorBy = "cluster"; drawScatter(); };
  $("#cb-tod").onclick = () => { S.colorBy = "tod"; drawScatter(); };
  for (const c of document.querySelectorAll(".cl")) c.onclick = () => {
    const id = Number(c.dataset.id);
    S.highlight = S.highlight === id ? null : id;
    for (const x of document.querySelectorAll(".cl")) x.classList.toggle("on", Number(x.dataset.id) === S.highlight);
    drawScatter();
  };
  drawScatter();
}

function tideTable(t) {
  const rows = [["missed objects", t.miss], ["background FPs", t.bkg], ["localization", t.loc], ["wrong class", t.cls], ["class + loc", t.both], ["duplicates", t.dupe]];
  return `<table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}<tr><td>precision / recall @ conf ${t.operating_conf}</td><td>${t.precision.toFixed(2)} / ${t.recall.toFixed(2)}</td></tr></table>`;
}

function clusterCard(c, i) {
  const tod = Object.entries(c.true_timeofday).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${(100 * v).toFixed(0)}%`).join(", ");
  const kinds = Object.entries(c.kinds).map(([k, v]) => `${k} ${v}`).join(" · ");
  const th = (c.thumbs || []).slice(0, 8).map((t) => `<img loading="lazy" alt="${esc(t.kind)} ${esc(t.cls)}" title="${esc(t.kind)} · ${esc(t.cls)} · ${esc(t.image_id)}" src="${fileUrl("analyze/thumbs/" + t.file)}">`).join("");
  return `<div class="cl ${S.highlight === c.id ? "on" : ""}" data-id="${c.id}"><div><i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${PALETTE[i % 10]}"></i>
      <b> #${c.rank} ${esc(c.name)}</b></div>
    <div class="m">${c.errors} errors · rate ${(100 * c.error_rate).toFixed(0)}% · lift ×${c.lift.toFixed(2)} · excess ${c.excess_errors.toFixed(0)}</div>
    <div>${Object.keys(c.classes).slice(0, 4).map((k) => `<span class="tag">${esc(k)}</span>`).join("")}</div>
    <div class="m">${esc(kinds)}</div><div class="m">true time of day: ${esc(tod)}</div>
    <div class="m">synthesis: <b>${esc(c.synthesis_target || "not addressable")}</b></div><div class="thumbs">${th}</div></div>`;
}

function drawScatter() {
  const cv = $("#sc"), a = S.analysis;
  if (!cv || !a) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = 460;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const pts = a.points;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of pts) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
  const pad = 14, sx = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * pad), sy = (v) => pad + ((y1 - v) / (y1 - y0 || 1)) * (H - 2 * pad);
  const modeIdx = new Map(a.modes.map((id, i) => [id, i]));
  const tpCol = getComputedStyle(document.documentElement).getPropertyValue("--track").trim() || "#ddd";
  const col = (p) => {
    if (S.colorBy === "tod") return p.k === "tp" ? tpCol : TOD[p.tod] || "#888";
    if (p.k === "tp") return tpCol;
    return modeIdx.has(p.c) ? PALETTE[modeIdx.get(p.c) % 10] : "#8a8f98";
  };
  const order = [...pts].sort((p, q) => (p.k === "tp") - (q.k === "tp")).reverse();
  for (const p of order) {
    const dim = S.highlight !== null && p.c !== S.highlight;
    g.globalAlpha = dim ? 0.12 : p.k === "tp" ? 0.6 : 0.85;
    g.fillStyle = col(p);
    g.beginPath(); g.arc(sx(p.x), sy(p.y), p.k === "tp" ? 1.6 : 2.3, 0, 6.283); g.fill();
  }
  g.globalAlpha = 1;
  const lg = $("#lg");
  if (S.colorBy === "tod") lg.innerHTML = Object.entries(TOD).map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("") + `<span><i style="background:${tpCol}"></i>true positives</span>`;
  else lg.innerHTML = a.modes.slice(0, 10).map((id, i) => { const c = a.clusters.find((x) => x.id === id); return `<span><i style="background:${PALETTE[i % 10]}"></i>#${i + 1} ${esc(c.name)}</span>`; }).join("")
    + `<span><i style="background:#8a8f98"></i>other errors</span><span><i style="background:${tpCol}"></i>true positives</span>`;
  cv.onmousemove = (e) => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 64;
    for (const p of pts) { const d = (sx(p.x) - mx) ** 2 + (sy(p.y) - my) ** 2; if (d < bd) { bd = d; best = p; } }
    const tip = $("#tip");
    if (!best) { tip.style.display = "none"; return; }
    const cl = a.clusters.find((c) => c.id === best.c);
    const cls = (S.run.baseline?.classes || [])[best.cls] || best.cls;
    tip.innerHTML = `<b>${esc(best.k)}</b> · ${esc(cls)}<br>image ${esc(best.img)} (${esc(best.tod)})<br>${cl ? "cluster: " + esc(cl.name) : "unclustered"}`;
    tip.style.display = "block"; tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 12 + "px";
  };
  cv.onmouseleave = () => { $("#tip").style.display = "none"; };
}
window.addEventListener("resize", () => { if (S.tab === "failures") drawScatter(); });

/* ---------------------------------------------------------------- synthesis */
function renderSynthesis() {
  const el = $("#tab-synthesis"), sy = S.run.synthesis;
  if (!sy) { el.innerHTML = `<div class="card empty">No synthesis yet (stage "synthesize").</div>`; return; }
  const s = sy.synthetic;
  const rej = s ? Object.entries(s.rejections || {}).map(([k, v]) => `${k} ${v}`).join(", ") || "none" : "";
  const kid = Object.entries(sy.kid_vs_real_target || {});
  const tiles = S.run.gallery.filter((g) => !g.startsWith("rejected"));
  const rejected = S.run.gallery.filter((g) => g.startsWith("rejected"));
  el.innerHTML = `<div class="split"><div class="card"><h4>Quality control</h4>${s ? `<table>
      <tr><td>candidates generated</td><td>${s.candidates}</td></tr><tr><td>accepted</td><td>${s.accepted} (${(100 * s.acceptance_rate).toFixed(0)}%)</td></tr>
      <tr><td>rejected by</td><td>${esc(rej)}</td></tr><tr><td>mean CLIP prompt score</td><td>${f3(s.mean_clip_score)}</td></tr>
      <tr><td>mean P(target condition)</td><td>${f3(s.mean_p_target)}</td></tr><tr><td>boxes keeping their structure</td><td>${f3(s.mean_box_pass)}</td></tr>
      ${s.seconds ? `<tr><td>generation time</td><td>${fmtDur(s.seconds)} (${(s.seconds / Math.max(s.candidates, 1)).toFixed(1)} s / candidate)</td></tr>` : ""}</table>` : "<p>no synthetic arm</p>"}
      ${kid.length ? `<h4 style="margin-top:12px">Closer to real night? KID vs real target-condition images (lower is closer)</h4><table>${kid.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.kid.toFixed(4)} ± ${v.std.toFixed(4)}</td></tr>`).join("")}</table>` : ""}
    </div><div class="card"><h4>How the labels survive</h4><p class="small">Each tile: <b>source</b> day photo · <b>Canny</b> control · <b>depth</b> control · <b>generated</b> image with the source's boxes.
      ControlNet pins edges and layout while img2img re-renders the lighting, so the boxes still fit; the structure filter checks that per box.</p>
      ${S.run.gallery_augment.length ? `<p class="small muted">Classical augmentation (the cheap baseline arm), for comparison:</p>${S.run.gallery_augment.slice(0, 2).map((g) => `<img style="width:100%;border-radius:8px;margin-bottom:6px" src="${fileUrl("synthesize/augment/gallery/" + g)}">`).join("")}` : ""}
    </div></div>
    <div class="card gallery"><h4>Accepted synthetic examples</h4>${tiles.map((g) => `<img loading="lazy" alt="synthetic example" src="${fileUrl("synthesize/synthetic/gallery/" + g)}">`).join("") || "<p class=muted>none</p>"}
    ${rejected.map((g) => `<p class="small fail">Rejected (${esc(g.replace(/^rejected_\d+_|\.jpg$/g, ""))}):</p><img alt="rejected" src="${fileUrl("synthesize/synthetic/gallery/" + g)}">`).join("")}</div>`;
}

/* ---------------------------------------------------------------- results */
function renderResults() {
  const el = $("#tab-results"), r = S.run;
  const ab = r.ablation;
  if (!ab) { el.innerHTML = `<div class="card empty">No ablation results yet (stage "golden_arms").</div>`; return; }
  const slices = ["overall", "day", "night", "dawn", "rainy", "snowy"].filter((s) => Object.values(ab).some((a) => a[s] !== undefined));
  const tgt = (r.config?.gate?.target_slices || ["night"])[0];
  const arms = ARMS.filter((a) => ab[a]);
  const table = `<div class="scroll"><table><tr><th>arm</th>${slices.map((s) => `<th>${s}</th>`).join("")}<th>Δ ${esc(tgt)} [95% CI]</th></tr>
    ${arms.map((a) => { const d = r.deltas?.[a]?.[tgt]; return `<tr><td><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:var(--a-${a})"></i> ${ARM[a]}</td>
      ${slices.map((s) => `<td>${f3(ab[a][s])}</td>`).join("")}<td>${d ? `${sgn(d.delta)} [${sgn(d.ci_low)}, ${sgn(d.ci_high)}]` : "–"}</td></tr>`; }).join("")}</table></div>`;
  const gate = r.gate ? `<div class="card"><h4>Promotion gate (candidate: ${esc(r.gate.candidate)})</h4><table><tr><th>check</th><th>value</th><th>threshold</th><th>result</th></tr>
    ${r.gate.decision.checks.map((c) => `<tr><td>${esc(c.name)}</td><td>${sgn(c.value, 4)}</td><td>${sgn(c.threshold, 4)}</td><td class="${c.passed ? "pass" : "fail"}">${c.passed ? "PASS" : "FAIL"}</td></tr>`).join("")}</table></div>` : "";
  el.innerHTML = `<div class="card"><h4>Golden-set mAP@50:95 per slice (every arm: same baseline, same epochs, same N extra images)</h4>${table}</div>
    <div class="card"><h4>Δ mAP@50:95 vs baseline with paired-bootstrap 95% CIs</h4>${forest(r.deltas, arms.filter((a) => a !== "baseline"), ["overall", "day", tgt])}</div>${gate}`;
}

function forest(deltas, arms, slices) {
  if (!deltas) return "";
  const W = 760, rowH = 22, padL = 190, colW = (W - padL - 10) / slices.length, H = arms.length * rowH + 40;
  let lo = 0, hi = 0;
  for (const a of arms) for (const s of slices) { const d = deltas[a]?.[s]; if (d) { lo = Math.min(lo, d.ci_low); hi = Math.max(hi, d.ci_high); } }
  const span = Math.max(hi - lo, 0.01); lo -= span * 0.1; hi += span * 0.1;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="bootstrap deltas">`;
  slices.forEach((s, j) => {
    const x0 = padL + j * colW, X = (v) => x0 + 8 + ((v - lo) / (hi - lo)) * (colW - 16);
    svg += `<text x="${x0 + colW / 2}" y="14" text-anchor="middle">${esc(s)}</text><line class="axis" x1="${X(0)}" x2="${X(0)}" y1="22" y2="${H - 14}" stroke-dasharray="3 3"/>`;
    svg += `<text x="${X(lo + span * 0.1)}" y="${H - 2}" text-anchor="middle">${sgn(lo + span * 0.1, 2)}</text><text x="${X(hi - span * 0.1)}" y="${H - 2}" text-anchor="middle">${sgn(hi - span * 0.1, 2)}</text>`;
    arms.forEach((a, i) => {
      const d = deltas[a]?.[s]; if (!d) return;
      const y = 32 + i * rowH;
      svg += `<line x1="${X(d.ci_low)}" x2="${X(d.ci_high)}" y1="${y}" y2="${y}" stroke="var(--a-${a})" stroke-width="2.5"/><circle cx="${X(d.delta)}" cy="${y}" r="4.5" fill="var(--a-${a})"><title>${ARM[a]} ${s}: ${sgn(d.delta)} [${sgn(d.ci_low)}, ${sgn(d.ci_high)}]</title></circle>`;
    });
  });
  arms.forEach((a, i) => { svg += `<text x="4" y="${36 + i * rowH}">${esc(ARM[a])}</text>`; });
  return svg + "</svg>";
}

/* ---------------------------------------------------------------- log */
async function loadLog() {
  try { $("#log").textContent = (await api(`/api/runs/${encodeURIComponent(S.rid)}/log?tail=300`)) || "(empty)"; } catch (e) { $("#log").textContent = e.message; }
  const pre = $("#log"); pre.scrollTop = pre.scrollHeight;
}

/* ---------------------------------------------------------------- polling */
async function refresh() {
  try {
    [S.overview, S.runs] = await Promise.all([api("/api/overview"), api("/api/runs")]);
  } catch (e) { $("#env").textContent = "dashboard server not reachable"; return; }
  renderProfiles();
  if (!S.rid) {
    const job = S.overview.job;
    S.rid = job.running ? job.run_id : (S.runs.find((r) => !r.reference)?.id || S.runs[0]?.id || null);
  }
  renderRuns();
}

async function tick() {
  await refresh();
  const cur = S.runs.find((r) => r.id === S.rid);
  if (cur && (cur.state === "running" || !S.run || S.run.summary.state !== cur.state)) {
    const before = S.run?.summary.stage;
    await loadRun();
    if (before !== S.run?.summary.stage) S.analysisFor = null;
  }
  setTimeout(tick, cur && cur.state === "running" ? 2000 : 6000);
}

(async () => {
  const q = new URLSearchParams(location.search);  // deep links: ?run=<id>&tab=<pipeline|failures|synthesis|results|log>
  if (q.get("run")) S.rid = q.get("run");
  await refresh();
  const tb = q.get("tab") && document.querySelector(`#tabs button[data-tab="${q.get("tab")}"]`);
  if (tb) tb.click();
  await loadRun();
  tick();
})();
