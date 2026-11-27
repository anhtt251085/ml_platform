/* Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE. */
/* SAI Platform frontend: a dependency-free single-page app (hash routing). */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 0) => (n === null || n === undefined || Number.isNaN(n)) ? "–" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
const bytes = (n) => n === null || n === undefined ? "–" : n < 1024 ? `${fmt(n)} B` : `${fmt(n / 1024, n < 10240 ? 2 : 1)} KB`;
const pct = (v) => v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(1)}%`;
const ms = (v) => v === null || v === undefined ? "–" : v < 1 ? `${(v * 1000).toFixed(0)} µs` : `${fmt(v, 2)} ms`;
const ago = (t) => { const s = Date.now() / 1000 - t; return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago` : s < 86400 ? `${Math.floor(s / 3600)} h ago` : `${Math.floor(s / 86400)} d ago`; };

/* Every page render bumps GEN. A request started under an older GEN resolves into a Stale error instead of
   data, so a slow response or a poll can never write into a page the user has already left. */
let GEN = 0;
class Stale extends Error {}
window.addEventListener("unhandledrejection", (e) => { if (e.reason instanceof Stale) e.preventDefault(); });
function errText(body, status) {
  const d = body && body.detail;
  if (Array.isArray(d)) return d.map((x) => `${(x.loc || []).slice(1).join(".")}: ${x.msg}`).join("; ");
  return d || `HTTP ${status}`;
}
async function api(path, opts = {}) {
  const gen = GEN;
  const init = { ...opts };
  if (opts.json !== undefined) { init.method = init.method || "POST"; init.headers = { "Content-Type": "application/json" }; init.body = JSON.stringify(opts.json); }
  const r = await fetch(path, init);
  if (gen !== GEN && !opts.anyPage) throw new Stale("stale");
  if (r.status === 401 && !path.startsWith("/api/auth/")) { showLogin(); throw new Stale("signed out"); }
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error(errText(body, r.status));
  return body;
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 3600); }

const EVIDENCE = {
  static_exact: ["exact", "b-ok", "Computed exactly by SAI from the static plan"],
  linked_exact: ["linked", "b-ok", "Measured from the linked ELF built with the target toolchain"],
  proven_upper_bound: ["proven bound", "b-ok", "SAI proved this stack bound by call-graph analysis"],
  measured: ["measured", "b-ok", "Measured on a board for this exact program"],
  calibrated: ["calibrated", "b-info", "Fit to board-farm measurements of similar programs"],
  sai_cost_model: ["SAI model", "b-info", "SAI kernel cost model, uncalibrated: a planning number"],
  estimate: ["estimate", "b-warn", "Kernel-frame estimate, not a bound"],
  heuristic: ["heuristic", "b-warn", "MACs × cycles/MAC planning heuristic"],
  assumed: ["assumed", "b-warn", "Assumed value: no estimate available"],
  unavailable: ["n/a", "", "Not available without a toolchain or measurement"],
};
const ev = (e) => { const [t, c, tip] = EVIDENCE[e] || [e || "n/a", "", ""]; return `<span class="badge ${c}" title="${esc(tip)}">${esc(t)}</span>`; };
const TARGET_COLOR = { "md6605": "var(--c2)", "cortex-m7": "var(--c1)" };
const GOAL_LABEL = { balanced: "Balanced", fastest: "Fastest", "min-ram": "Least RAM", "min-flash": "Smallest flash", accuracy: "Best accuracy" };
const TASK_BADGE = { classification: "b-acc", regression: "b-info", anomaly: "b-warn" };

let SYS = null;
async function system() { if (!SYS) SYS = await api("/api/system"); return SYS; }

/* ------------------------------------------------------------------------------------------------ router */
const routes = [];
const route = (re, fn) => routes.push([re, fn]);
let cleanup = [];
const NAV_ALIAS = { "/recordings": "/data", "/recording": "/data", "/connectors": "/data", "/physics": "/data", "/devboard": "/projects" };
function onLeave(fn) { cleanup.push(fn); }
function leave() { GEN++; cleanup.forEach((f) => f()); cleanup = []; }
async function render() {
  leave();
  const hash = location.hash.slice(1) || "/";
  const base = hash.split("?")[0];
  const main = $("#main");
  for (const [re, fn] of routes) {
    const m = hash.match(re);
    if (m) {
      $$(".side nav a").forEach((a) => { const seg = base.split("/").slice(0, 2).join("/"); const on = a.getAttribute("href") === "#" + (NAV_ALIAS[seg] || seg || "/"); a.classList.toggle("active", on); if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
      main.innerHTML = `<div class="empty">Loading…</div>`;
      try { await fn(main, ...m.slice(1)); } catch (e) { if (!(e instanceof Stale)) main.innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; }
      window.scrollTo(0, 0);
      return;
    }
  }
  main.innerHTML = `<div class="empty">Page not found</div>`;
}
window.addEventListener("hashchange", render);
document.addEventListener("keydown", (e) => {
  const el = e.target;
  if ((e.key === "Enter" || e.key === " ") && el && el.getAttribute && el.getAttribute("role") === "button" && el.tagName !== "BUTTON") {
    e.preventDefault();
    if (el.id === "drop") { const f = el.querySelector('input[type="file"]') || $("#file"); if (f) f.click(); } else el.click();
  }
});

/* ------------------------------------------------------------------------------------------------ charts */
function scatter(points, { w = 560, h = 260, xLabel = "", yLabel = "", logX = false, logY = false } = {}) {
  if (!points.length) return `<div class="empty small">No data</div>`;
  const pad = { l: 58, r: 14, t: 12, b: 36 };
  const tx = (v) => logX ? Math.log10(Math.max(v, 1e-9)) : v, ty = (v) => logY ? Math.log10(Math.max(v, 1e-9)) : v;
  const xs = points.map((p) => tx(p.x)), ys = points.map((p) => ty(p.y));
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (x0 === x1) { x0 -= 1; x1 += 1; } if (y0 === y1) { y0 -= 1; y1 += 1; }
  const mx = (x1 - x0) * 0.08, my = (y1 - y0) * 0.1; x0 -= mx; x1 += mx; y0 -= my; y1 += my;
  if (!logX) x0 = Math.max(x0, 0); if (!logY) y0 = Math.max(y0, 0);
  const X = (v) => pad.l + (tx(v) - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - (ty(v) - y0) / (y1 - y0) * (h - pad.t - pad.b);
  const ticks = (a, b, n = 4) => Array.from({ length: n + 1 }, (_, i) => a + (b - a) * i / n);
  const lab = (v, log) => { const r = log ? 10 ** v : v; return r >= 1e6 ? (r / 1e6).toFixed(1) + "M" : r >= 1e3 ? (r / 1e3).toFixed(r >= 1e4 ? 0 : 1) + "k" : r >= 10 ? r.toFixed(0) : r.toFixed(2); };
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="${esc(yLabel)} vs ${esc(xLabel)}">`;
  for (const t of ticks(y0, y1)) { const yy = h - pad.b - (t - y0) / (y1 - y0) * (h - pad.t - pad.b); s += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${yy}" y2="${yy}" stroke="var(--line)"/><text x="${pad.l - 6}" y="${yy + 4}" text-anchor="end">${lab(t, logY)}</text>`; }
  for (const t of ticks(x0, x1)) { const xx = pad.l + (t - x0) / (x1 - x0) * (w - pad.l - pad.r); s += `<text x="${xx}" y="${h - pad.b + 16}" text-anchor="middle">${lab(t, logX)}</text>`; }
  s += `<text x="${(w + pad.l) / 2}" y="${h - 4}" text-anchor="middle">${esc(xLabel)}</text><text x="12" y="${(h - pad.b) / 2}" transform="rotate(-90 12 ${(h - pad.b) / 2})" text-anchor="middle">${esc(yLabel)}</text>`;
  for (const p of points) {
    const cx = X(p.x), cy = Y(p.y), c = p.color || "var(--c1)";
    const shape = p.hollow ? `fill="none" stroke="${c}" stroke-width="1.6"` : `fill="${c}"`;
    s += p.star ? `<circle cx="${cx}" cy="${cy}" r="9" fill="none" stroke="${c}" stroke-width="2"/>` : "";
    s += `<circle cx="${cx}" cy="${cy}" r="${p.r || 5}" ${shape} opacity="${p.dim ? 0.35 : 0.95}"><title>${esc(p.title || "")}</title></circle>`;
  }
  return s + "</svg>";
}

function lineChart(series, { w = 560, h = 200, xLabel = "", yLabel = "" } = {}) {
  const all = series.flatMap((s) => s.values);
  if (!all.length) return "";
  const pad = { l: 46, r: 10, t: 10, b: 30 }, n = Math.max(...series.map((s) => s.values.length));
  let y0 = Math.min(...all), y1 = Math.max(...all); if (y0 === y1) y1 = y0 + 1;
  const X = (i) => pad.l + i / Math.max(n - 1, 1) * (w - pad.l - pad.r), Y = (v) => h - pad.b - (v - y0) / (y1 - y0) * (h - pad.t - pad.b);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="${esc(yLabel)}">`;
  for (let i = 0; i <= 3; i++) { const v = y0 + (y1 - y0) * i / 3; s += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${pad.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`; }
  s += `<text x="${(w + pad.l) / 2}" y="${h - 4}" text-anchor="middle">${esc(xLabel)}</text>`;
  for (const se of series) s += `<polyline fill="none" stroke="${se.color}" stroke-width="2" points="${se.values.map((v, i) => `${X(i)},${Y(v)}`).join(" ")}"/>`;
  return s + "</svg>";
}

function sparkSeries(series, w = 220, h = 54) {
  const cols = ["var(--c1)", "var(--c2)", "var(--c3)"];
  const all = series.flat(); const lo = Math.min(...all), hi = Math.max(...all) || 1;
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" aria-hidden="true">`;
  series.forEach((vals, k) => { s += `<polyline fill="none" stroke="${cols[k % 3]}" stroke-width="1.3" points="${vals.map((v, i) => `${i / Math.max(vals.length - 1, 1) * w},${h - 3 - (v - lo) / ((hi - lo) || 1) * (h - 6)}`).join(" ")}"/>`; });
  return s + "</svg>";
}
function heat(img, w = 120) {
  const H = img.length, W = img[0].length, all = img.flat(), lo = Math.min(...all), hi = Math.max(...all), cw = w / W;
  let s = `<svg viewBox="0 0 ${w} ${cw * H}" width="${w}" aria-hidden="true">`;
  img.forEach((row, y) => row.forEach((v, x) => { const t = (v - lo) / ((hi - lo) || 1); s += `<rect x="${x * cw}" y="${y * cw}" width="${cw + .3}" height="${cw + .3}" fill="hsl(${240 - t * 240} 80% ${30 + t * 30}%)"/>`; }));
  return s + "</svg>";
}
function bars(obj) {
  const max = Math.max(...Object.values(obj), 1);
  return `<div class="stack">${Object.entries(obj).map(([k, v]) => `<div class="row small" style="gap:8px"><span style="width:110px" class="mono">${esc(k)}</span><div class="progress" style="flex:1;height:10px"><i style="width:${v / max * 100}%"></i></div><span class="mono" style="width:44px;text-align:right">${fmt(v)}</span></div>`).join("")}</div>`;
}

/* ------------------------------------------------------------------------------------------------ dashboard */
route(/^\/$/, async (el) => {
  const [sys, projects, cat, b] = await Promise.all([system(), api("/api/projects"), api("/api/catalog"), api("/api/boards")]);
  const online = b.boards.filter((x) => x.online).length;
  el.innerHTML = `
  <div class="page-h"><div><h1>From sensor data to MCU firmware</h1>
    <p class="muted">Pick a model type, bring your data, choose a board. SAI trains, quantizes, picks the best execution plan and hands you a verified C package.</p></div>
    <a class="btn primary" href="#/new">+ New project</a></div>
  <div class="grid g4">
    <div class="card stat"><span class="v">${projects.length}</span><span class="l">Projects</span></div>
    <div class="card stat"><span class="v">${cat.length}</span><span class="l">Model types in the pool</span></div>
    <div class="card stat"><span class="v">${online}/${b.boards.length}</span><span class="l">Boards online${b.boards.every((x) => x.kind !== "physical") && b.boards.length ? ` (${esc(b.boards[0].kind)})` : ""}</span></div>
    <div class="card stat"><span class="v mono" style="font-size:16px">${esc(sys.sai.sai || "?")}</span><span class="l">SAI compiler · ABI ${esc(sys.sai.abi || "?")}</span></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h2>Recent projects</h2>${projectsTable(projects.slice(0, 8))}</div>
    <div class="card"><h2>How it works</h2>
      <ol class="small" style="padding-left:18px;margin:0;line-height:1.9">
        <li><b>Choose a model type</b> from the pool (bearing faults, arc faults, SoH, activity, keyword spotting…)</li>
        <li><b>Bring data</b>: CSV / NPZ upload, an HTTPS link, or synthetic demo data</li>
        <li><b>Pick boards and a goal</b>: fastest, least RAM, smallest flash, best accuracy or balanced</li>
        <li>The platform <b>trains several sizes</b>, quantizes to INT8 (QAT when needed)</li>
        <li><b>SAI searches execution plans</b> per board: exact RAM, linked flash, proven stack, latency estimate</li>
        <li>It <b>compiles, verifies and measures bit-exact accuracy</b> of the deployed C program</li>
        <li>Download the package, or <b>run it on a board</b> in the farm</li>
      </ol></div>
  </div>`;
});

function projectsTable(rows) {
  if (!rows.length) return `<div class="empty">No projects yet. <a href="#/new">Start one</a>.</div>`;
  return `<div class="tbl-wrap"><table class="t"><thead><tr><th>Name</th><th>Model type</th><th>Goal</th><th>Status</th><th>Created</th></tr></thead><tbody>
  ${rows.map((p) => `<tr><td><a href="#/job/${esc(p.job_id)}">${esc(p.name)}</a></td><td>${esc(p.template_id)}</td><td>${esc(GOAL_LABEL[p.config.goal] || p.config.goal)}</td>
    <td>${statusBadge(p.job_status, p.job_progress)}</td><td class="muted">${ago(p.created)}</td></tr>`).join("")}</tbody></table></div>`;
}
const GATED = ["package.zip", "model.onnx", "report.json"];
const LAYER_PARAMS = { conv: ["out", "k", "stride", "dilation", "relu"], dwconv: ["k", "stride", "dilation", "relu"], pwconv: ["out", "relu"], maxpool: ["k"], avgpool: ["k"], gap: [], fc: ["out", "relu"], residual: ["k"], dropout: ["p"],
  lstm: ["units", "bidirectional", "sequences"], gru: ["units", "bidirectional", "sequences"], rnn: ["units", "bidirectional", "sequences"], attention: ["heads"], layernorm: [] };
const LAYER_DEF = { conv: { out: 16, k: 3, stride: 1, relu: true }, dwconv: { k: 3, stride: 1, relu: true }, pwconv: { out: 16, relu: true }, maxpool: { k: 2 }, avgpool: { k: 2 }, gap: {}, fc: { out: 16, relu: true }, residual: { k: 3 }, dropout: { p: 0.1 },
  lstm: { units: 16, bidirectional: false, sequences: false }, gru: { units: 16, bidirectional: false, sequences: false }, rnn: { units: 16, bidirectional: false, sequences: false }, attention: { heads: 2 }, layernorm: {} };
const PARAM_RANGE = { out: [1, 1024, 1, "output channels / units"], k: [1, 31, 1, "kernel / pool size"], stride: [1, 8, 1, "stride"], dilation: [1, 32, 1, "dilation"], p: [0, 0.9, 0.05, "dropout rate (training only)"], units: [1, 512, 1, "recurrent units (state size)"], heads: [1, 16, 1, "attention heads (must divide the features)"] };
const BOOL_PARAMS = { relu: ["ReLU", true], bidirectional: ["both directions", false], sequences: ["pass every step on", false] };
const SEQ_OPS = ["lstm", "gru", "rnn", "attention", "layernorm"];
const paramInput = (l, p, w = 78) => BOOL_PARAMS[p] ? `<label class="check small" title="${p}"><input type="checkbox" data-f="${p}" ${(l[p] ?? BOOL_PARAMS[p][1]) ? "checked" : ""}>${BOOL_PARAMS[p][0]}</label>` : `<label class="f" title="${PARAM_RANGE[p][3]}" style="width:${w}px">${p}<input data-f="${p}" type="number" min="${PARAM_RANGE[p][0]}" max="${PARAM_RANGE[p][1]}" step="${PARAM_RANGE[p][2]}" value="${l[p] ?? (p === "stride" || p === "dilation" ? 1 : "")}"></label>`;
// Model editor: start from a baseline, add / remove / reorder layers and change their parameters; every change is
// checked on the user's data (shapes, MACs, parameters, receptive field) and can be compiled with SAI.
function modelEditor(box, o) {
  let layers = JSON.parse(JSON.stringify(o.layers)), vt = null, valid = false, ft = null;
  const ops = Object.keys(o.ops);
  onLeave(() => { clearTimeout(vt); clearTimeout(ft); });
  const draw = () => {
    box.innerHTML = `<div class="card" style="margin-top:16px;border-color:var(--accent)"><div class="row"><h2 style="margin:0">Customize · ${esc(o.baseName)}</h2><span class="spacer"></span><label class="f" style="min-width:260px">Name<input id="ed_name" value="${esc(o.label)}"></label></div>
      <p class="small muted" style="margin-top:6px">Your baseline is <b>${esc(o.baseName)}</b>. Add, remove or reorder layers and change their parameters; every change is checked on your data. Only SAI operators are offered, and the output head is added for you.</p>
      <div class="tbl-wrap"><table class="t layers"><thead><tr><th>#</th><th>Layer</th><th>Parameters</th><th>Output</th><th class="n">MACs</th><th class="n">Params</th><th></th></tr></thead><tbody id="ed_rows">
      ${layers.map((l, i) => `<tr data-i="${i}"><td class="muted">${i + 1}</td><td><select data-f="op">${ops.map((x) => `<option ${x === l.op ? "selected" : ""}>${x}</option>`).join("")}</select><div class="small muted">${esc(o.ops[l.op] || "")}</div></td>
        <td><div class="row" style="gap:6px;flex-wrap:wrap">${LAYER_PARAMS[l.op].map((p) => paramInput(l, p)).join("") || `<span class="small muted">no parameters</span>`}</div></td>
        <td class="mono small" data-shape="${i}"></td><td class="n mono small" data-macs="${i}"></td><td class="n mono small" data-par="${i}"></td>
        <td style="white-space:nowrap"><button class="btn ghost sm" data-a="up" title="move up">↑</button><button class="btn ghost sm" data-a="dn" title="move down">↓</button><button class="btn ghost sm" data-a="dup" title="duplicate">⧉</button><button class="btn ghost sm" data-a="rm" title="remove">✕</button></td></tr>`).join("")}
      <tr class="muted"><td></td><td>head</td><td class="small">output layer, automatic</td><td class="mono small" data-shape="head"></td><td class="n mono small" data-macs="head"></td><td class="n mono small" data-par="head"></td><td></td></tr></tbody></table></div>
      <div class="row" style="margin-top:10px;gap:8px"><select id="ed_op">${ops.map((x) => `<option>${x}</option>`).join("")}</select><select id="ed_pos"><option value="end">at the end</option>${layers.map((l, i) => `<option value="${i}">after layer ${i + 1} (${esc(l.op)})</option>`).join("")}</select><button class="btn sm" id="ed_add">+ Add layer</button><span class="spacer"></span><button class="btn sm ghost" id="ed_reset">Reset to baseline</button></div>
      <div id="ed_sum" style="margin-top:12px"></div>
      <div class="row" style="margin-top:12px;gap:8px"><button class="btn" id="ed_fit">Check with SAI</button>${o.canSaveToLibrary ? `<button class="btn ghost" id="ed_lib">Save to library</button>` : ""}<span class="spacer"></span><button class="btn ghost" id="ed_cancel">Cancel</button><button class="btn primary" id="ed_ok">${o.okText || "Use this model"}</button></div>
      <div id="ed_fitres"></div></div>`;
    $("#ed_rows").onchange = (e) => {
      const tr = e.target.closest("tr[data-i]"); if (!tr) return; const i = +tr.dataset.i, f = e.target.dataset.f;
      if (f === "op") { layers[i] = { op: e.target.value, ...LAYER_DEF[e.target.value] }; return draw(); }
      layers[i][f] = BOOL_PARAMS[f] ? e.target.checked : e.target.value === "" ? undefined : +e.target.value; check();
    };
    $("#ed_rows").onclick = (e) => {
      const a = e.target.dataset.a; if (!a) return; const i = +e.target.closest("tr").dataset.i;
      if (a === "rm") layers.splice(i, 1);
      if (a === "dup") layers.splice(i + 1, 0, JSON.parse(JSON.stringify(layers[i])));
      if (a === "up" && i > 0) [layers[i - 1], layers[i]] = [layers[i], layers[i - 1]];
      if (a === "dn" && i < layers.length - 1) [layers[i + 1], layers[i]] = [layers[i], layers[i + 1]];
      draw();
    };
    $("#ed_add").onclick = () => { const op = $("#ed_op").value, pos = $("#ed_pos").value; const l = { op, ...LAYER_DEF[op] }; if (pos === "end") layers.push(l); else layers.splice(+pos + 1, 0, l); draw(); };
    $("#ed_reset").onclick = () => { layers = JSON.parse(JSON.stringify(o.baseline)); draw(); };
    $("#ed_cancel").onclick = () => { box.innerHTML = ""; o.onCancel && o.onCancel(); };
    $("#ed_ok").onclick = () => { if (!valid) return toast("Fix the model first: see the message under the table"); o.onSave(JSON.parse(JSON.stringify(layers)), $("#ed_name").value.trim() || o.label); box.innerHTML = ""; };
    if ($("#ed_lib")) $("#ed_lib").onclick = () => {
      if (!valid) return toast("Fix the model first");
      return busy($("#ed_lib"), "Saving…", async () => { const a = await api("/api/architectures", { json: { name: $("#ed_name").value.trim() || o.label, family: "Custom", inputs: [o.inputKind], tasks: [o.task], layers, summary: `Customized from ${o.baseName}` } }); toast(`Saved to your library as ${a.name}`); });
    };
    $("#ed_fit").onclick = async () => {
      if (!valid) return toast("Fix the model first");
      if ($("#ed_fit").disabled) return;
      $("#ed_fit").disabled = true; $("#ed_fit").textContent = "Checking with SAI…";
      const fitDone = () => { if ($("#ed_fit")) { $("#ed_fit").disabled = false; $("#ed_fit").textContent = "Check with SAI"; } };
      $("#ed_fitres").innerHTML = `<div class="callout info" style="margin-top:10px">Compiling with SAI on ${o.targets().map(esc).join(", ")}…</div>`;
      try {
        const job = await api("/api/models/fitcheck", { json: { dataset_id: o.datasetId, layers, preprocess: o.prep(), targets: o.targets(), goal: o.goal() } });
        const wait = async () => {
          let j; try { j = await api(`/api/jobs/${job.id}`); } catch (e) { if (!(e instanceof Stale)) $("#ed_fitres").innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; return; }
          if (!$("#ed_fitres")) return;
          if (["succeeded", "failed", "cancelled"].includes(j.status)) fitDone();
          if (j.status === "succeeded") { const p = j.result.plans; $("#ed_fitres").innerHTML = `<div style="margin-top:12px">${p.recommended ? `<div class="callout ok">Fits: best plan ${esc(p.recommended.target)} · ${bytes(p.recommended.ram)} RAM · ${bytes(p.recommended.flash)} flash · ${ms(p.recommended.latency.ms)}</div>` : `<div class="callout bad">${esc(p.advice || "no plan fits")}</div>`}${planTable(p.candidates)}</div>`; return; }
          if (j.status === "failed" || j.status === "cancelled") { $("#ed_fitres").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(j.error || j.status)}</div>`; return; }
          ft = setTimeout(wait, 1000);
        };
        wait();
      } catch (e) { fitDone(); $("#ed_fitres").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(e.message)}</div>`; }
    };
    check();
  };
  const check = () => {
    clearTimeout(vt);
    vt = setTimeout(async () => {
      let r; try { r = await api("/api/models/preview", { json: { dataset_id: o.datasetId, layers, preprocess: o.prep() } }); } catch (e) { if (!(e instanceof Stale)) toast(e.message); return; }
      if (!$("#ed_sum")) return;
      valid = r.ok;
      $("#ed_ok").disabled = !r.ok;
      if (!r.ok) { $$("[data-shape],[data-macs],[data-par]", box).forEach((x) => (x.textContent = "")); $("#ed_sum").innerHTML = `<div class="callout bad">${esc(r.error)}</div>`; return; }
      r.summary.layers.forEach((row, i) => { const k = row.head ? "head" : i; const set = (a, v) => { const x = $(`[data-${a}="${k}"]`, box); if (x) x.textContent = v; }; set("shape", row.shape.join("×")); set("macs", fmt(row.macs)); set("par", row.params ? fmt(row.params) : ""); });
      const s = r.summary;
      $("#ed_sum").innerHTML = `<div class="kpis"><div class="kpi"><div class="l">Parameters</div><div class="v">${fmt(s.params)}</div></div><div class="kpi"><div class="l">MACs</div><div class="v">${fmt(s.macs)}</div></div><div class="kpi"><div class="l">Receptive field</div><div class="v">${r.receptive_field ? fmt(r.receptive_field) : "–"}</div></div><div class="kpi"><div class="l">Largest activation</div><div class="v">${bytes(s.peak_activation_bytes)}</div></div></div>
        <p class="small muted" style="margin-top:6px">Input ${esc(r.spec.input.channels + "×" + r.spec.input.height + "×" + r.spec.input.width)} after your preprocessing · ${r.spec.outputs} output(s) · weights ≈ ${bytes(s.weight_bytes_int8)} INT8. RAM and flash come from SAI: use “Check with SAI”.</p>
        ${r.backend === "xmodels" ? `<div class="callout info" style="margin-top:6px"><b>Sequence model</b> (LSTM / GRU / RNN / attention): compiled by SAI's front end through the XModels backend. The model reads the window as ${esc(r.spec.input.height === 1 ? r.spec.input.width + " steps of " + r.spec.input.channels + " features" : r.spec.input.height + " steps of " + (r.spec.input.channels * r.spec.input.width) + " features")}; its package is <b>evaluation-only</b> until the front end is qualified.</div>` : ""}`;
    }, 250);
  };
  draw();
}
const archChain = (layers) => layers.map((l) => (l.bidirectional ? "bi" : "") + l.op + (l.out ? l.out : "") + (l.units ? l.units : "") + (l.heads ? "h" + l.heads : "") + (l.sequences ? "·seq" : "") + (l.k ? "k" + l.k : "") + (l.stride > 1 ? "/s" + l.stride : "") + (l.dilation > 1 ? "d" + l.dilation : "")).join(" → ") + " → fc(head)";
function archChecks(a, targets) {
  const ts = targets || Object.keys(a.checks || {});
  if (!ts.length || !Object.keys(a.checks || {}).length) return `<span class="badge">not SAI-checked yet</span>`;
  return ts.map((t) => { const c = (a.checks || {})[t]; return !c ? `<span class="badge">${esc(t)}: not checked</span>` : c.fits ? `<span class="badge b-ok" title="SAI ${esc(c.sai_version)} · ${bytes(c.ram)} RAM · ${bytes(c.flash)} flash · ${fmt(c.latency_ms, 3)} ms at the reference input">✓ ${esc(t)}</span>` : `<span class="badge b-bad" title="${esc(((c.detail || {}).violations || []).join("; ") || (c.detail || {}).error || "")}">✗ ${esc(t)}</span>`; }).join(" ");
}   // downloads that need a manager's approval
// Disables `btn` while `fn` runs (double clicks, a second request while the first is still at work); the server
// also refuses a duplicate of a job that is queued or running, with the job id (409), which ends up in the toast.
async function busy(btn, label, fn) {
  if (!btn || btn.disabled) return undefined;
  const text = btn.innerHTML;
  btn.disabled = true;
  if (label) btn.textContent = label;
  try { return await fn(); }
  catch (e) { if (!(e instanceof Stale)) toast(e.message); return undefined; }
  finally { if (btn.isConnected) { btn.disabled = false; btn.innerHTML = text; } }
}
function dur(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  return sec < 60 ? `${sec} s` : sec < 3600 ? `${Math.floor(sec / 60)} min ${sec % 60} s` : `${Math.floor(sec / 3600)} h ${Math.floor(sec % 3600 / 60)} min`;
}
function jobTitle(job) {
  const p = job.params || {};
  return { train: p.name || "Project", fitcheck: "Studio fit check", rebuild: "Reproducibility rebuild", regression: "Pool regression",
    basetrain: "Pool base weights", recommend: "Architecture recommendation", archverify: "Architecture library SAI check", multimodel: p.name || "Multi-model build", ingest: `Import ${p.filename || "dataset"}` }[job.kind] || `Analyze ${p.filename || "model"}`;
}
function statusBadge(s, p) {
  const c = { succeeded: "b-ok", failed: "b-bad", running: "b-acc", queued: "", cancelled: "b-warn" }[s] || "";
  return `<span class="badge ${c}">${esc(s || "?")}${s === "running" && p !== undefined ? ` ${Math.round(p * 100)}%` : ""}</span>`;
}

route(/^\/projects$/, async (el) => {
  const projects = await api("/api/projects");
  el.innerHTML = `<div class="page-h"><div><h1>Projects</h1><p class="muted">Every project is one end-to-end pipeline run.</p></div><a class="btn primary" href="#/new">+ New project</a></div>
  <div class="card">${projectsTable(projects)}</div>`;
});

/* ------------------------------------------------------------------------------------------------ model pool */
function templateCard(t, sel) {
  const sizes = Object.entries(t.sizes || {}).map(([k, v]) => v.error ? "" : `<span class="badge" title="${fmt(v.params)} params">${esc(k)} · ${fmt(v.macs / 1000, 0)}k MACs</span>`).join(" ");
  return `<div class="card tile ${sel ? "sel" : ""}" data-tid="${esc(t.id)}" role="button" tabindex="0" aria-pressed="${sel ? "true" : "false"}" aria-label="${esc(t.name)}">
    <div class="row"><span class="ic">${esc(t.icon || "◆")}</span><div class="spacer"></div><span class="badge ${TASK_BADGE[t.task] || ""}">${esc(t.task)}</span>${t.source === "studio" ? `<span class="badge b-info">research</span>` : ""}</div>
    <div class="ttl">${esc(t.name)}</div><div class="muted small">${esc(t.domain)}</div>
    <div class="small">${esc(t.summary)}</div>
    <div class="small muted">Input ${t.input.channels}×${t.input.height > 1 ? t.input.height + "×" : ""}${t.input.width} · ${t.task === "classification" ? (t.labels || []).length + " classes" : esc(t.labels.join(", "))}</div>
    <div class="chips">${sizes}</div></div>`;
}

route(/^\/pool$/, async (el) => {
  const cat = await api("/api/catalog");
  await system();
  const domains = ["All", ...new Set(cat.map((t) => t.domain))];
  let dom = "All", q = "";
  const lb = await api("/api/pool/leaderboard");
  el.innerHTML = `<div class="page-h"><div><h1>Model pool</h1><p class="muted">Ready-to-train model types. Every architecture uses only SAI's operators, so it compiles for every supported MCU.</p></div><a class="btn" href="#/studio">Create a model type →</a></div>
  ${Object.keys(lb).length ? `<div class="card" style="margin-bottom:16px"><h2>Leaderboard: real-data results of the pool program</h2><div class="tbl-wrap"><table class="t"><thead><tr><th>Model type</th><th>Data</th><th>Split</th><th>Size</th><th>Board</th><th class="n">Bit-exact</th><th class="n">RAM</th><th class="n">Flash</th><th class="n">Latency</th><th></th></tr></thead><tbody>
    ${Object.entries(lb).map(([tid, rows]) => { const e = rows[0], t = cat.find((x) => x.id === tid) || { name: tid, icon: "" }; return `<tr><td>${esc(t.icon)} <b>${esc(t.name)}</b>${rows.length > 1 ? ` <span class="small muted">+${rows.length - 1} more</span>` : ""}</td><td class="small">${esc(e.dataset)}<br><span class="badge ${e.synthetic ? "b-warn" : "b-ok"}">${esc(e.licence)}</span> ${fmt(e.n)} windows</td><td class="small">${esc(e.split)}${e.recordings ? ` (${e.recordings} rec.)` : ""}</td><td>${esc(e.size)}</td><td>${esc(e.target)}</td><td class="n"><b>${e.metric === "accuracy" ? pct(e.value) : fmt(e.value, 3)}</b></td><td class="n">${bytes(e.ram)}</td><td class="n">${bytes(e.flash)}</td><td class="n">${ms(e.latency.ms)} ${ev(e.latency.evidence)}</td><td><a href="#/job/${e.job_id}">report</a></td></tr>`; }).join("")}
  </tbody></table></div></div>` : ""}
  <div class="row" style="margin-bottom:14px"><div class="chips" id="doms"></div><div class="spacer"></div><input id="q" placeholder="Search…" style="width:220px"></div>
  <div class="grid gauto" id="grid"></div><div id="detail"></div>`;
  const draw = () => {
    $("#doms").innerHTML = domains.map((d) => `<button class="chip ${d === dom ? "on" : ""}" data-d="${esc(d)}">${esc(d)}</button>`).join("");
    const list = cat.filter((t) => (dom === "All" || t.domain === dom) && (`${t.name} ${t.summary} ${t.domain}`).toLowerCase().includes(q));
    $("#grid").innerHTML = list.map((t) => templateCard(t)).join("") || `<div class="empty">No match</div>`;
  };
  draw();
  $("#doms").onclick = (e) => { if (e.target.dataset.d) { dom = e.target.dataset.d; draw(); } };
  $("#q").oninput = (e) => { q = e.target.value.toLowerCase(); draw(); };
  $("#grid").onclick = (e) => { const c = e.target.closest("[data-tid]"); if (c) showTemplate(cat.find((t) => t.id === c.dataset.tid)); };
  function showTemplate(t) {
    $("#detail").innerHTML = `<div class="card" style="margin-top:16px"><div class="row"><span class="ic" style="font-size:26px">${esc(t.icon)}</span><div><h2 style="margin:0">${esc(t.name)}</h2><div class="muted small">${esc(t.domain)} · by ${esc(t.author)}</div></div><div class="spacer"></div>
      ${ME && ME.perms.publish ? `<button class="btn" id="basetrain" data-t="${esc(t.id)}">Train base weights</button>` : ""}
      <a class="btn primary" href="#/new?template=${encodeURIComponent(t.id)}">Use this model type</a></div>
      ${(SYS.pool_weights || {})[t.id] ? `<p class="small" style="margin-top:8px"><span class="badge b-ok">base weights</span> sizes ${Object.keys(SYS.pool_weights[t.id].sizes).join(", ")}, licence ${esc(SYS.pool_weights[t.id].licence)}: customers can fine-tune from them.</p>` : ""}
      <div class="grid g3" style="margin-top:14px">
        <div><h3>Use cases</h3><ul class="small" style="padding-left:18px;margin:0">${(t.use_cases || []).map((u) => `<li>${esc(u)}</li>`).join("")}</ul></div>
        <div><h3>Input</h3><div class="small">${esc(t.input.kind)} · ${t.input.channels} channel(s) ${esc((t.input.channel_names || []).join(", "))}<br>${t.input.height > 1 ? `${t.input.height}×${t.input.width} frames` : `window ${t.input.width} samples`}${t.input.sample_rate_hz ? ` @ ${fmt(t.input.sample_rate_hz)} Hz` : ""}</div>
          <h3 style="margin-top:10px">Outputs</h3><div class="chips">${t.labels.map((l) => `<span class="badge">${esc(l)}</span>`).join("")}</div></div>
        <div><h3>Data</h3><p class="small">${esc(t.datasets_hint || "")}</p></div></div>
      <h3 style="margin-top:12px">Size variants</h3><div class="tbl-wrap"><table class="t"><thead><tr><th>Size</th><th class="n">Params</th><th class="n">MACs</th><th>Layers</th></tr></thead><tbody>
      ${Object.entries(t.variants).map(([k, layers]) => `<tr><td>${esc(k)}</td><td class="n">${fmt(t.sizes[k]?.params)}</td><td class="n">${fmt(t.sizes[k]?.macs)}</td><td class="mono small">${layers.map((l) => l.op + (l.out ? l.out : "") + (l.k ? "k" + l.k : "") + (l.stride > 1 ? "/s" + l.stride : "")).join(" → ")} → fc(head)</td></tr>`).join("")}
      </tbody></table></div></div>`;
    const bt = $("#basetrain");
    if (bt) bt.onclick = () => busy(bt, "Starting…", async () => { const j = await api("/api/studio/basetrain", { json: { template_id: bt.dataset.t } }); SYS = null; location.hash = `#/job/${j.id}`; });
    $("#detail").scrollIntoView({ behavior: "smooth" });
  }
});

/* ------------------------------------------------------------------------------------------------ wizard */
route(/^\/new(?:\?(.*))?$/, async (el, query) => {
  SYS = null;
  const qp = new URLSearchParams(query || "");
  const pre = qp.get("template");
  const [cat, sys] = await Promise.all([api("/api/catalog"), system()]);
  const st = { step: pre ? 1 : 0, tpl: pre ? cat.find((t) => t.id === pre) : null, ds: null, prep: {},
    targets: [], goal: "balanced", budgets: {}, metric: "", sizes: ["tiny", "small", "base"], epochs: 30, name: "", training: {}, archs: [], customs: [] };
  if (st.tpl) st.targets = [...st.tpl.targets];
  if (qp.get("dataset")) { try { st.ds = await api(`/api/datasets/${encodeURIComponent(qp.get("dataset"))}`); st.step = 2; } catch { /* stays on Data */ } }
  const STEPS = ["Model type", "Data", "Preprocess", "Board & goal", "Review"];
  const draw = () => {
    el.innerHTML = `<div class="page-h"><div><h1>New project</h1><p class="muted">Four steps. The platform does the rest.</p></div></div>
    <div class="steps">${STEPS.map((s, i) => `<div class="step ${i === st.step ? "on" : i < st.step ? "done" : ""}"><b>${i < st.step ? "✓" : i + 1}</b>${s}</div>`).join("")}</div>
    <div id="body"></div>`;
    [stepModel, stepData, stepPrep, stepTarget, stepReview][st.step]($("#body"));
  };
  const nav = (back, next, nextOk = true, nextLabel = "Continue") => `<div class="row" style="margin-top:18px">${back ? `<button class="btn" id="back">← Back</button>` : ""}<div class="spacer"></div>${next ? `<button class="btn primary" id="next" ${nextOk ? "" : "disabled"}>${nextLabel}</button>` : ""}</div>`;
  const wire = () => { const b = $("#back"), n = $("#next"); if (b) b.onclick = () => { st.step--; draw(); }; if (n) n.onclick = () => { st.step++; draw(); }; };

  function stepModel(b) {
    b.innerHTML = `<div class="grid gauto" id="g">${cat.map((t) => templateCard(t, st.tpl && st.tpl.id === t.id)).join("")}</div>${nav(false, true, !!st.tpl)}`;
    $("#g").onclick = (e) => { const c = e.target.closest("[data-tid]"); if (!c) return; st.tpl = cat.find((t) => t.id === c.dataset.tid); st.targets = [...st.tpl.targets]; st.ds = null; draw(); };
    wire();
  }

  function stepData(b) {
    const t = st.tpl, ts = t.input.height === 1;
    const csvCols = [...(t.input.channel_names || []), "label", "segment (optional)"].join(", ");
    b.innerHTML = `<div class="grid g2"><div class="card">
      <div class="tabs"><button class="tab on" data-tab="up">Upload file</button><button class="tab" data-tab="url">From URL</button><button class="tab" data-tab="demo">Demo data</button><button class="tab" data-tab="have">Existing dataset</button><button class="tab" data-tab="soon">Live, cloud &amp; physics</button></div>
      <div id="tab"></div></div>
      <div class="card"><h2>${esc(t.icon)} ${esc(t.name)} – data contract</h2>
        ${ts ? `<p class="small"><b>CSV</b>: one row per sample with columns <code>${esc(csvCols)}</code>. Windows of <b>${fmt(t.window)}</b> samples every <b>${fmt(t.hop)}</b> are cut per segment; the window label is its majority label.</p>` : ""}
        <p class="small"><b>NPZ</b>: arrays <code>X</code> ${ts ? `[N, ${t.input.channels}, T]` : `[N, ${t.input.channels}, ${t.input.height}, ${t.input.width}]`} and <code>y</code> [N] ${t.task === "regression" ? "(numeric target)" : "(class index or name)"}; optional <code>labels</code>.</p>
        <p class="small muted">${esc(t.datasets_hint || "")}</p>
        <div id="dsinfo"></div></div></div>
      ${nav(true, true, !!st.ds)}`;
    const tabs = {
      up: () => `${licSelect()}<div class="drop" id="drop" style="margin-top:10px" role="button" tabindex="0" aria-label="Choose a data file">Drop a <b>.csv</b>, <b>.npz</b>${t.frontend ? " or a <b>.zip</b> of label folders with WAV files" : ""} here, or click to choose<input type="file" id="file" accept=".csv,.npz,.txt,.zip" hidden></div><div id="mapbox"></div>
        ${ts ? `<div class="row" style="margin-top:10px"><label class="f">Window<input id="win" type="number" value="${+t.window || 128}" style="width:100px"></label><label class="f">Hop<input id="hop" type="number" value="${t.hop}" style="width:100px"></label></div>` : ""}`,
      url: () => `${licSelect()}<label class="f" style="margin-top:10px">HTTPS link to a .csv, .npz or .zip<input id="url" placeholder="https://…/recordings.csv"></label>
        ${ts ? `<div class="row" style="margin-top:10px"><label class="f">Window<input id="win" type="number" value="${+t.window || 128}" style="width:100px"></label><label class="f">Hop<input id="hop" type="number" value="${+t.hop || 64}" style="width:100px"></label></div>` : ""}<div class="row" style="margin-top:10px"><button class="btn" id="fetch">Fetch</button></div>`,
      demo: () => `<p class="small">Generate a <b>synthetic</b> dataset with the physics of this task, to try the whole pipeline now. Results on synthetic data say nothing about field accuracy.</p>
        <label class="f">Windows per class<input id="per" type="number" value="300" min="30" max="2000" style="width:120px"></label><div class="row" style="margin-top:10px"><button class="btn" id="gen" ${t.synth ? "" : "disabled"}>Generate demo data</button></div>`,
      have: () => `<div id="havelist" class="small muted">Loading…</div>`,
      soon: () => `<p class="small">Other ways to bring data in (each makes a dataset you can pick under <i>Existing dataset</i>):</p><ul class="small" style="padding-left:18px"><li><a href="#/recordings">Live capture and labelling</a>: stream from a sensor board, mark what happens, or label a long recording by time span</li><li><a href="#/connectors">Cloud and database</a>: S3 / GCS / Azure Blob, InfluxDB, PostgreSQL</li><li><a href="#/physics">Physics augmentation</a>: simulate fault signatures from your machine's parameters</li></ul>`,
    };
    const setTab = (k) => {
      $$(".tab", b).forEach((x) => x.classList.toggle("on", x.dataset.tab === k)); $("#tab").innerHTML = tabs[k]();
      if (k === "up") {
        const d = $("#drop"), f = $("#file");
        d.onclick = () => f.click(); f.onchange = () => f.files[0] && upload(f.files[0]);
        d.ondragover = (e) => { e.preventDefault(); d.classList.add("hover"); }; d.ondragleave = () => d.classList.remove("hover");
        d.ondrop = (e) => { e.preventDefault(); d.classList.remove("hover"); e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]); };
      }
      if (k === "url") $("#fetch").onclick = () => doIngest(() => api("/api/datasets/url", { json: { template_id: t.id, url: $("#url").value, licence: $("#lic").value, window: +($("#win")?.value || 0) || null, hop: +($("#hop")?.value || 0) || null } }));
      if (k === "demo") $("#gen").onclick = () => doIngest(() => api("/api/datasets/demo", { json: { template_id: t.id, per_class: +$("#per").value } }));
      if (k === "have") api(`/api/datasets?template_id=${encodeURIComponent(t.id)}`).then((list) => {
        $("#havelist").innerHTML = list.length ? `<table class="t"><tbody>${list.map((d) => `<tr><td><b>${esc(d.name)}</b><div class="muted">${fmt(d.n)} windows · ${esc((d.shape || []).join("×"))} · ${esc(d.licence)}</div></td><td class="muted">${ago(d.created)}</td><td><button class="btn sm" data-use="${esc(d.id)}">Use</button></td></tr>`).join("")}</tbody></table>` : `<p>No datasets for this model type yet. <a href="#/data">All datasets</a></p>`;
        $$("[data-use]", b).forEach((x) => (x.onclick = () => doIngest(() => api(`/api/datasets/${x.dataset.use}`))));
      });
    };
    $$(".tab", b).forEach((x) => (x.onclick = () => setTab(x.dataset.tab)));
    setTab("up");
    function licSelect() {
      return `<label class="f">Data licence<select id="lic">${Object.entries(LIC()).filter(([k]) => k !== "synthetic").map(([k, v]) => `<option value="${k}" ${k === "proprietary" ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select></label>`;
    }
    async function upload(file) {
      if (/\.(csv|txt)$/i.test(file.name)) return mapColumns(file);
      send(file, null);
    }
    function send(file, mapping) {
      const fd = new FormData(); fd.append("template_id", t.id); fd.append("file", file); fd.append("licence", $("#lic").value);
      if ($("#win")) { fd.append("window", $("#win").value); fd.append("hop", $("#hop").value); }
      if (mapping) fd.append("mapping", JSON.stringify(mapping));
      doIngest(() => api("/api/datasets/upload", { method: "POST", body: fd }));
    }
    async function mapColumns(file) {
      const fd = new FormData(); fd.append("file", file);
      let pv; try { pv = await api("/api/datasets/preview", { method: "POST", body: fd }); } catch (e) { return toast(e.message); }
      const opt = (sel) => `<option value="">—</option>${pv.header.map((h) => `<option value="${esc(h)}" ${h === sel ? "selected" : ""}>${esc(h)}</option>`).join("")}`;
      $("#mapbox").innerHTML = `<div class="card" style="margin-top:12px"><h3>Map the columns of ${esc(file.name)} <span class="muted small">(${fmt(pv.n_rows)} rows)</span></h3>
        <div class="grid g3"><label class="f">Label<select id="m_label">${opt(pv.guess.label)}</select></label><label class="f">Recording / segment<select id="m_seg">${opt(pv.guess.segment)}</select></label><label class="f">Time (for sample-rate checks)<select id="m_time">${opt(pv.guess.time)}</select></label></div>
        <div class="small muted" style="margin-top:8px">Channels (expected ${t.input.channels}${(t.input.channel_names || []).length ? ": " + esc(t.input.channel_names.join(", ")) : ""})</div>
        <div class="chips" style="margin-top:4px">${pv.header.map((h, i) => pv.numeric[i] ? `<label class="check small"><input type="checkbox" class="m_ch" value="${esc(h)}" ${pv.guess.channels.includes(h) ? "checked" : ""}>${esc(h)}</label>` : "").join("")}</div>
        <div class="tbl-wrap" style="margin-top:10px"><table class="t"><thead><tr>${pv.header.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${pv.rows.slice(0, 6).map((r) => `<tr>${r.map((v) => `<td class="mono small">${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
        <div class="row" style="margin-top:10px"><button class="btn primary" id="m_go">Ingest with this mapping</button></div></div>`;
      const sync = () => { const used = [$("#m_label").value, $("#m_seg").value, $("#m_time").value]; $$(".m_ch").forEach((c) => { if (used.includes(c.value)) c.checked = false; }); };
      ["#m_label", "#m_seg", "#m_time"].forEach((id) => ($(id).onchange = sync)); sync();
      $("#m_go").onclick = () => send(file, { label: $("#m_label").value || null, segment: $("#m_seg").value || null, time: $("#m_time").value || null, channels: $$(".m_ch:checked").map((x) => x.value) });
    }
    async function doIngest(fn) {
      if (b.dataset.importing === "1") return toast("An import is still running");
      b.dataset.importing = "1";
      const locked = $$("button, input, select", b).filter((x) => !x.disabled);
      locked.forEach((x) => (x.disabled = true));
      const unlock = () => { b.dataset.importing = ""; locked.forEach((x) => { if (x.isConnected) x.disabled = false; }); };
      $("#dsinfo").innerHTML = `<div class="callout info" style="margin-top:12px">Ingesting…</div>`;
      try {
        let r = await fn();
        if (r && r.job) {                       // large file: parsed by a worker in the background
          for (;;) {
            const j = await api(`/api/jobs/${r.job.id}`);
            const last = (j.logs || []).slice(-1)[0];
            $("#dsinfo").innerHTML = `<div class="callout info" style="margin-top:12px">Large file: importing in the background (${esc(j.status)})${last ? ` · ${esc(last.line)}` : ""}</div>`;
            if (j.status === "succeeded") { r = await api(`/api/datasets/${j.result.dataset_id}`); break; }
            if (j.status === "failed" || j.status === "cancelled") throw new Error(j.error || "import failed");
            await new Promise((res) => setTimeout(res, 1500));
          }
        }
        unlock();
        st.ds = r; toast("Dataset ready"); stepData(b);
      } catch (e) { unlock(); if (!(e instanceof Stale)) $("#dsinfo").innerHTML = `<div class="callout bad" style="margin-top:12px">${esc(e.message)}</div>`; }
    }
    if (st.ds) $("#dsinfo").innerHTML = datasetSummary(st.ds);
    wire();
  }

  function stepPrep(b) {
    const d = st.ds, raw1d = !d.frontend && d.shape[1] === 1, P = st.prep;
    P.augment = P.augment || { gain: 0.05, noise: 0, shift: 0 };
    P.split = P.split || { train: 0.7, val: 0.15, test: 0.15, mode: "auto" };
    const names = (d.channel_names && d.channel_names.length === d.shape[0]) ? d.channel_names : Array.from({ length: d.shape[0] }, (_, i) => `ch${i}`);
    const chans = P.channels || names.map((_, i) => i);
    b.innerHTML = `<div class="grid g2" style="align-items:start"><div class="card"><h2>Preprocessing</h2>
      <p class="small muted">Applied to every window in training <b>and on the MCU</b>: the window steps run in the same C, and normalization is mirrored exactly in the generated <code>sai_app_prepare</code>.</p>
      <h3>Window steps ${raw1d ? "" : `<span class="badge">not for ${d.frontend ? "front-end features" : "images"}</span>`}</h3>
      <div class="stack" ${raw1d ? "" : 'style="opacity:.5;pointer-events:none"'}>
        <div class="small muted">Channels</div><div class="chips">${names.map((n, i) => `<label class="check small"><input type="checkbox" class="p_ch" value="${i}" ${chans.includes(i) ? "checked" : ""}>${esc(n)}</label>`).join("")}</div>
        <div class="grid g2"><label class="f">Decimate (block average)<select id="p_dec">${[1, 2, 4, 8].map((v) => `<option ${+(P.decimate || 1) === v ? "selected" : ""} value="${v}">${v === 1 ? "off" : "÷" + v + ` → ${Math.floor(d.shape[2] / v)} samples`}</option>`).join("")}</select></label>
        <label class="check small" style="margin-top:18px"><input type="checkbox" id="p_dm" ${P.demean ? "checked" : ""}>Remove each window's mean (DC, gravity, bias)</label></div></div>
      <h3 style="margin-top:14px">Normalization</h3><div class="grid g2">
        <label class="f">Per channel, fit on the train split<select id="p_norm">${[["zscore", "z-score (mean / std)"], ["robust", "robust (median / IQR)"], ["minmax", "min-max to [-1, 1]"], ["none", "none (raw units)"]].map(([k, v]) => `<option value="${k}" ${(P.normalization || "zscore") === k ? "selected" : ""}>${v}</option>`).join("")}</select></label>
        <label class="f">Clip outliers at ± (normalized)<input id="p_clip" type="number" step="0.5" min="0.5" value="${P.clip ?? ""}" placeholder="off"></label></div>
      <h3 style="margin-top:14px">Training only</h3><div class="grid g3">
        <label class="f">Gain jitter<input id="p_gain" type="number" step="0.01" min="0" max="0.5" value="${P.augment.gain}"></label>
        <label class="f">Noise (σ, normalized)<input id="p_noise" type="number" step="0.01" min="0" max="1" value="${P.augment.noise}"></label>
        <label class="f">Time shift (samples)<input id="p_shift" type="number" min="0" value="${P.augment.shift}" ${raw1d ? "" : "disabled"}></label></div>
      ${st.tpl.task === "classification" ? `<label class="check small" style="margin-top:8px"><input type="checkbox" id="p_cw" ${P.class_weights === false ? "" : "checked"}>Class weights (balance rare classes)</label>` : ""}
      <h3 style="margin-top:14px">Split</h3><div class="grid g4">
        <label class="f">Train<input id="p_tr" type="number" step="0.05" value="${P.split.train}"></label><label class="f">Validation<input id="p_va" type="number" step="0.05" value="${P.split.val}"></label>
        <label class="f">Test<input id="p_te" type="number" step="0.05" value="${P.split.test}"></label>
        <label class="f">Unit<select id="p_mode"><option value="auto" ${P.split.mode !== "window" ? "selected" : ""}>by recording</option><option value="window" ${P.split.mode === "window" ? "selected" : ""}>by window</option></select></label></div>
      <p class="small muted">By recording (when the data has recordings, ${fmt(d.recordings)} here) keeps every window of a recording in one split: honest test accuracy.</p></div>
      <div class="card" id="pv"><h2>Preview</h2><div class="small muted">Computing…</div></div></div>${nav(true, true)}`;
    let timer = null;
    const read = () => {
      const ch = $$(".p_ch:checked").map((x) => +x.value);
      st.prep = { channels: raw1d && ch.length !== names.length ? ch : null, decimate: raw1d ? +$("#p_dec").value : 1, demean: raw1d && $("#p_dm").checked,
        normalization: $("#p_norm").value, clip: $("#p_clip").value === "" ? null : +$("#p_clip").value,
        augment: { gain: +$("#p_gain").value || 0, noise: +$("#p_noise").value || 0, shift: raw1d ? +$("#p_shift").value || 0 : 0 },
        class_weights: $("#p_cw") ? $("#p_cw").checked : true,
        split: { train: +$("#p_tr").value, val: +$("#p_va").value, test: +$("#p_te").value, mode: $("#p_mode").value } };
    };
    const preview = () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        if (!$("#p_norm") || st.step !== 2) return;          // the user already left this step
        read();
        try {
          const r = await api(`/api/datasets/${d.id}/preprocess-preview`, { json: { template_id: st.tpl.id, preprocess: st.prep } });
          if (!$("#pv") || st.step !== 2) return;
          $("#next").disabled = false;
          const nm = (i) => names[(st.prep.channels || names.map((_, k) => k))[i]] || `ch${i}`;
          $("#pv").innerHTML = `<h2>Preview</h2><div class="kpis"><div class="kpi"><div class="l">Raw window</div><div class="v">${r.input_shape.join("×")}</div></div><div class="kpi"><div class="l">Model input</div><div class="v">${r.model_input_shape.join("×")}</div></div><div class="kpi"><div class="l">Input bytes (int8)</div><div class="v">${fmt(r.input_bytes)}</div></div></div>
            ${r.warnings.map((w) => `<div class="callout" style="margin-top:8px">${esc(w)}</div>`).join("")}
            <div class="grid g2" style="margin-top:12px">${r.examples.map((e) => `<div class="card" style="padding:8px"><div class="small"><b>${esc(e.label)}</b></div>${e.before ? `<div class="small muted">before</div>${sparkSeries(e.before)}<div class="small muted">after</div>${sparkSeries(e.after)}` : `<div class="row">${heat(e.before_image, 90)}${heat(e.after_image, 90)}</div>`}</div>`).join("")}</div>
            <div class="tbl-wrap" style="margin-top:10px"><table class="t"><thead><tr><th>Channel (after)</th><th class="n">mean</th><th class="n">std</th><th class="n">p1</th><th class="n">p99</th></tr></thead><tbody>
            ${r.after.map((a, i) => `<tr><td>${esc(nm(i))}</td><td class="n">${fmt(a.mean, 3)}</td><td class="n">${fmt(a.std, 3)}</td><td class="n">${fmt(a.p1, 2)}</td><td class="n">${fmt(a.p99, 2)}</td></tr>`).join("")}</tbody></table></div>`;
        } catch (e) { $("#pv").innerHTML = `<h2>Preview</h2><div class="callout bad">${esc(e.message)}</div>`; $("#next").disabled = true; }
      }, 300);
    };
    $$("input, select", b).forEach((x) => (x.onchange = preview));
    $("#back").onclick = () => { clearTimeout(timer); read(); st.step--; draw(); };
    $("#next").onclick = () => { clearTimeout(timer); read(); st.step++; draw(); };
    preview();
  }

  function stepTarget(b) {
    const mcus = sys.targets.filter((x) => x.status === "available"), road = sys.targets.filter((x) => x.status === "roadmap");
    b.innerHTML = `<div class="grid g2"><div class="card"><h2>Boards</h2><div class="stack">
      ${mcus.map((m) => `<label class="card check" style="padding:12px"><input type="checkbox" name="tgt" value="${m.id}" ${st.targets.includes(m.id) ? "checked" : ""}>
        <div style="flex:1"><div class="row"><b>${esc(m.name)}</b><span class="spacer"></span><span class="badge ${m.toolchain === "available" ? "b-ok" : "b-warn"}">toolchain ${esc(m.toolchain)}</span></div>
        <div class="small muted">${esc(m.core)} @ ${fmt(m.clock_hz / 1e6, 1)} MHz · AI arena ${bytes(m.budget.workspace)} · AI flash ${bytes(m.budget.flash)}</div><div class="small muted">${esc(m.notes)}</div></div></label>`).join("")}
      <div class="small muted">Coming: ${road.map((r) => esc(r.name)).join(" · ")}</div></div></div>
      <div class="card"><h2>Goal</h2><div class="chips" id="goals">${Object.entries(GOAL_LABEL).map(([k, v]) => `<button class="chip ${st.goal === k ? "on" : ""}" data-g="${k}">${v}</button>`).join("")}</div>
      <p class="small muted" style="margin-top:8px" id="goalhelp"></p>
      <h3 style="margin-top:14px">Limits (optional)</h3><div class="grid g3">
        <label class="f">Max RAM (bytes)<input id="mr" type="number" value="${st.budgets.max_ram || ""}" placeholder="board limit"></label>
        <label class="f">Max flash (bytes)<input id="mf" type="number" value="${st.budgets.max_flash || ""}" placeholder="board limit"></label>
        <label class="f">Max latency (ms)<input id="ml" type="number" step="0.01" value="${st.budgets.max_latency_ms || ""}" placeholder="none"></label></div>
      <h3 style="margin-top:14px">Quality</h3><div class="grid g2">
        <label class="f">Minimum ${st.tpl.task === "classification" ? "accuracy" : st.tpl.task === "regression" ? "R²" : "AUC"} (0–1)<input id="mt" type="number" step="0.01" min="0" max="1" value="${st.metric}" placeholder="e.g. 0.95"></label>
        <label class="f">Training epochs<input id="ep" type="number" value="${st.epochs}" min="1" max="200"></label></div>
      <h3 style="margin-top:14px">Training</h3><div class="stack">
        <label class="check small"><input type="checkbox" id="ifp" ${st.initPool ? "checked" : ""} ${(SYS.pool_weights || {})[st.tpl.id] ? "" : "disabled"}>Fine-tune from the pool's base weights ${(SYS.pool_weights || {})[st.tpl.id] ? `<span class="badge b-ok">available</span>` : `<span class="muted">(none trained yet for this model type)</span>`}</label>
        <label class="check small"><input type="checkbox" id="lrs" ${st.lrSearch ? "checked" : ""}>Learning-rate search (2 runs per size, keeps the best on validation)</label>
        ${ME && ME.perms.publish ? `<label class="check small"><input type="checkbox" id="pbm" ${st.poolBench ? "checked" : ""}>Pool benchmark: publish the result on the pool leaderboard (SAI research)</label>` : ""}</div>
      <details id="tadv" style="margin-top:10px" ${Object.keys(st.training).length ? "open" : ""}><summary class="small"><b>Advanced training settings</b> <span class="muted">(leave blank for automatic)</span></summary>
        ${trainingForm(st.training, st.tpl.task)}</details>
      <h3 style="margin-top:14px">Model sizes to try</h3><div class="row">${["tiny", "small", "base"].filter((s) => (st.tpl.variants || {})[s]).map((s) => `<label class="check"><input type="checkbox" name="sz" value="${s}" ${st.sizes.includes(s) ? "checked" : ""}>${s} <span class="muted small">${fmt(st.tpl.sizes[s]?.macs / 1000)}k MACs</span></label><button class="btn ghost sm" data-cust="size:${s}" title="Use ${s} as a baseline and edit it">✎</button>`).join("")}
        <label class="check" id="autolbl" style="${st.auto ? "" : "display:none"}"><input type="checkbox" name="sz" value="auto" ${st.sizes.includes("auto") ? "checked" : ""}>auto <span class="badge b-ok">recommended</span></label><button class="btn ghost sm" data-cust="auto" id="autocust" style="${st.auto ? "" : "display:none"}" title="Use the recommended model as a baseline and edit it">✎</button></div>
        <p class="small muted" style="margin-top:4px">✎ customizes a size as your own baseline: add layers, change channels, kernels, strides…</p>
      </div></div>
      <div id="custed"></div>
      <div class="card" style="margin-top:16px" id="custcard"></div>
      <div class="card" style="margin-top:16px"><div class="row"><h2 style="margin:0">Architectures from the library</h2><span class="spacer"></span><a class="small" href="#/architectures">Browse the library</a></div>
        <p class="small muted" style="margin-top:6px">Architectures that suit this data (parameters, MACs and receptive field are for your input). ✓ = SAI compiled it and it fits that board at the library's reference input; the project then costs it exactly on your data. Each one you tick is trained next to the sizes above and competes for the best plan.</p><div id="libpick"><span class="small muted">Loading…</span></div></div>
      <div class="card" style="margin-top:16px"><div class="row"><h2 style="margin:0">Recommended architecture</h2><span class="spacer"></span><button class="btn sm" id="recbtn">${st.rec ? "Recommend again" : "Recommend an architecture"}</button></div>
        <p class="small muted" style="margin-top:6px">Profiles your data, generates candidate architectures for it, and has SAI cost every candidate exactly on the selected boards before any training: RAM, flash and plan depend on the architecture, not on the weights. Takes a few minutes; the result is added as size <b>auto</b>.</p>
        <div id="recout">${st.rec ? recTable(st.rec) : ""}</div></div>${nav(true, true)}`;
    const help = { balanced: "Best mix of latency, RAM and flash among plans that fit.", fastest: "Lowest latency plan that fits.", "min-ram": "Smallest RAM (workspace + stack): leaves the most memory for your application.", "min-flash": "Smallest code + weights.", accuracy: "Most accurate size that fits, then balanced." };
    const g = () => { $("#goalhelp").textContent = help[st.goal]; };
    g();
    $("#goals").onclick = (e) => { if (e.target.dataset.g) { st.goal = e.target.dataset.g; $$("#goals .chip").forEach((c) => c.classList.toggle("on", c.dataset.g === st.goal)); g(); } };
    const save = () => {
      st.targets = $$("input[name=tgt]:checked").map((x) => x.value); st.sizes = $$("input[name=sz]:checked").map((x) => x.value);
      st.budgets = { max_ram: +$("#mr").value || null, max_flash: +$("#mf").value || null, max_latency_ms: +$("#ml").value || null };
      st.metric = $("#mt").value; st.epochs = +$("#ep").value || 30; st.initPool = $("#ifp").checked; st.lrSearch = $("#lrs").checked; st.poolBench = !!($("#pbm") && $("#pbm").checked);
      st.training = readTrainingForm();
    };
    $("#back").onclick = () => { save(); st.step--; draw(); };
    $("#next").onclick = () => { save(); if (!st.targets.length) return toast("Pick at least one board"); if (!st.sizes.length && !st.archs.length && !st.customs.some((m) => m.include)) return toast("Pick at least one size, architecture or customized model"); st.step++; draw(); };
    let libById = {};
    const kindOf = () => st.ds.height > 1 ? "2d" : st.ds.width >= 8 ? "timeseries" : "features";
    const drawCustoms = () => {
      const c = $("#custcard"); if (!c) return;
      c.style.display = st.customs.length ? "" : "none";
      c.innerHTML = `<h2 style="margin:0 0 6px">Your customized models</h2><p class="small muted">Trained next to the sizes and library architectures you ticked; each competes for the best plan.</p>
        <div class="tbl-wrap"><table class="t"><thead><tr><th></th><th>Name</th><th>Baseline</th><th>Layers</th><th></th></tr></thead><tbody>${st.customs.map((m, i) => `<tr><td><input type="checkbox" data-inc="${i}" ${m.include ? "checked" : ""}></td><td><b>${esc(m.label)}</b><div class="mono small muted">${esc(m.key)}</div></td><td class="small">${esc(m.base)}</td><td class="arch-chain" style="margin:0">${esc(archChain(m.layers))}</td><td style="white-space:nowrap"><button class="btn sm" data-edit="${i}">Edit</button> <button class="btn ghost sm" data-rmc="${i}">Remove</button></td></tr>`).join("")}</tbody></table></div>`;
      c.onclick = (e) => {
        const t = e.target;
        if (t.dataset.inc !== undefined) { st.customs[+t.dataset.inc].include = t.checked; return; }
        if (t.dataset.rmc !== undefined) { st.customs.splice(+t.dataset.rmc, 1); return drawCustoms(); }
        if (t.dataset.edit !== undefined) { const m = st.customs[+t.dataset.edit]; openEditor(m.base, m.baseline, m.layers, m.label, m); }
      };
    };
    const openEditor = (baseName, baseline, layers, label, existing) => {
      save();
      modelEditor($("#custed"), { baseName, baseline, layers, label, ops: SYS.ops, datasetId: st.ds.id, prep: () => st.prep, targets: () => st.targets.length ? st.targets : ["md6605", "cortex-m7"], goal: () => st.goal,
        canSaveToLibrary: !!ME.perms.research, inputKind: kindOf(), task: st.tpl.task,
        onSave: (ls, name) => {
          if (existing) { existing.layers = ls; existing.label = name; }
          else { const slug = baseName.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 14) || "model"; let n = 1; while (st.customs.some((m) => m.key === `c_${slug}_${n}`)) n++; st.customs.push({ key: `c_${slug}_${n}`, label: name, base: baseName, baseline, layers: ls, include: true }); }
          drawCustoms(); toast(`“${name}” will be trained`); $("#custcard").scrollIntoView({ behavior: "smooth", block: "nearest" });
        } });
      $("#custed").scrollIntoView({ behavior: "smooth", block: "start" });
    };
    if (b._custH) b.removeEventListener("click", b._custH, true);     // one handler per draw of this step
    b._custH = (e) => {
      const t = e.target.closest("[data-cust]"); if (!t) return;
      e.preventDefault(); e.stopPropagation();
      const v = t.dataset.cust;
      if (v === "auto") return openEditor("Recommended (auto)", st.auto, st.auto, "Recommended, customized");
      if (v.startsWith("size:")) { const s = v.slice(5); return openEditor(`${st.tpl.name} · ${s}`, st.tpl.variants[s], st.tpl.variants[s], `My ${s}`); }
      if (v.startsWith("lib:")) { const a = libById[v.slice(4)]; return openEditor(a.name, a.layers, a.layers, `My ${a.name}`); }
    };
    b.addEventListener("click", b._custH, true);
    drawCustoms();
    (async () => {
      try {
        const lib = (await api(`/api/architectures?dataset_id=${encodeURIComponent(st.ds.id)}`)).architectures;
        const ok = lib.filter((a) => a.fit && a.fit.ok), no = lib.filter((a) => a.fit && !a.fit.ok);
        const box = $("#libpick"); if (!box) return;
        box.innerHTML = (ok.length ? `<div class="archpick">${ok.map((a) => `<label class="${st.archs.includes(a.id) ? "on" : ""}"><input type="checkbox" name="arch" value="${esc(a.id)}" ${st.archs.includes(a.id) ? "checked" : ""}>
          <div><b>${esc(a.name)}</b> <span class="muted small">${esc(a.family)}</span>${a.source === "custom" ? ` <span class="badge">${a.org_id ? "your organisation" : "SAI"}</span>` : ""}
          <div class="small muted">${fmt(a.fit.params)} params · ${fmt(a.fit.macs / 1000, 1)}k MACs${a.fit.receptive_field ? ` · RF ${a.fit.receptive_field}` : ""}</div>
          <div class="small" style="margin-top:4px">${archChecks(a, st.targets.length ? st.targets : null)} <button class="btn ghost sm" data-cust="lib:${esc(a.id)}" title="Use as a baseline and edit it">✎ Customize</button></div></div></label>`).join("")}</div>` : `<p class="small muted">No library architecture suits this data shape.</p>`)
          + (no.length ? `<details style="margin-top:8px"><summary class="small muted">${no.length} not suitable for this data</summary><ul class="small muted">${no.map((a) => `<li>${esc(a.name)}: ${esc(a.fit.reason)}</li>`).join("")}</ul></details>` : "");
        $$("input[name=arch]").forEach((x) => { x.onchange = () => { x.closest("label").classList.toggle("on", x.checked); st.archs = $$("input[name=arch]:checked").map((y) => y.value); }; });
        libById = Object.fromEntries(lib.map((a) => [a.id, a]));
      } catch (e) { if (!(e instanceof Stale)) $("#libpick").innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; }
    })();
    $("#recbtn").onclick = async () => {
      save();
      if (!st.targets.length) return toast("Pick at least one board");
      $("#recbtn").disabled = true;
      try {
        const job = await api("/api/recommend", { json: { template_id: st.tpl.id, dataset_id: st.ds.id, targets: st.targets, goal: st.goal, budgets: st.budgets, preprocess: st.prep } });
        for (;;) {
          const j = await api(`/api/jobs/${job.id}`);
          const last = (j.logs || []).slice(-1)[0];
          $("#recout").innerHTML = `<div class="callout info">Recommending (${esc(j.status)})${last ? ` · ${esc(last.line)}` : ""}</div>`;
          if (j.status === "succeeded") { st.rec = j.result; break; }
          if (j.status === "failed" || j.status === "cancelled") throw new Error(j.error || "recommendation failed");
          await new Promise((res) => setTimeout(res, 2000));
        }
        if (st.rec.recommended) {
          st.auto = st.rec.recommended.layers;
          if (!st.sizes.includes("auto")) st.sizes.push("auto");
          $("#autolbl").style.display = ""; $("#autolbl input").checked = true; $("#autocust").style.display = "";
        }
        $("#recout").innerHTML = recTable(st.rec);
      } catch (e) { if (!(e instanceof Stale)) $("#recout").innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; }
      finally { if ($("#recbtn")) { $("#recbtn").disabled = false; $("#recbtn").textContent = "Recommend again"; } }
    };
  }

  function trainingForm(t, task) {
    const sel = (id, opts, v, auto) => `<select id="${id}">${auto ? `<option value="">${auto}</option>` : ""}${opts.map((o) => `<option ${v === o ? "selected" : ""}>${o}</option>`).join("")}</select>`;
    const num = (id, v, ph, step = "any") => `<input id="${id}" type="number" step="${step}" value="${v ?? ""}" placeholder="${ph}">`;
    return `<div class="grid g3" style="margin-top:8px">
      <label class="f">Optimizer${sel("t_opt", ["adam", "adamw", "sgd"], t.optimizer, "adam (default)")}</label>
      <label class="f">Learning rate${num("t_lr", t.lr, "auto (5e-3)")}</label>
      <label class="f">LR schedule${sel("t_sch", ["cosine", "step", "constant"], t.schedule, "cosine (default)")}</label>
      <label class="f">Batch size${num("t_bs", t.batch_size, "auto", 1)}</label>
      <label class="f">Weight decay${num("t_wd", t.weight_decay, "1e-4")}</label>
      <label class="f">Early-stop patience (epochs, 0 = off)${num("t_pat", t.patience, "auto", 1)}</label>
      <label class="f">Gradient clip (0 = off)${num("t_clip", t.grad_clip, "2.0")}</label>
      ${task === "classification" ? `<label class="f">Label smoothing (0–0.5)${num("t_ls", t.label_smoothing, "0")}</label>` : ""}
      <label class="f">Random seed${num("t_seed", t.seed, "0", 1)}</label>
      <label class="f">Quantization-aware training${sel("t_qat", ["auto", "always", "off"], t.qat)}</label>
      <label class="f">QAT epochs${num("t_qe", t.qat_epochs, "auto", 1)}</label>
      <label class="check small" style="align-self:end"><input type="checkbox" id="t_ext" ${t.extend_small_data === false ? "" : "checked"}>Extend epochs for small datasets (to 1,500 steps)</label></div>`;
  }
  function readTrainingForm() {
    const v = (id) => { const e = $("#" + id); return e ? e.value.trim() : ""; };
    const t = {}, n = (k, id) => { if (v(id) !== "") t[k] = +v(id); };
    if (v("t_opt")) t.optimizer = v("t_opt");
    if (v("t_sch")) t.schedule = v("t_sch");
    if (v("t_qat") && v("t_qat") !== "auto") t.qat = v("t_qat");
    n("lr", "t_lr"); n("batch_size", "t_bs"); n("weight_decay", "t_wd"); n("patience", "t_pat"); n("grad_clip", "t_clip");
    n("label_smoothing", "t_ls"); n("seed", "t_seed"); n("qat_epochs", "t_qe");
    if ($("#t_ext") && !$("#t_ext").checked) t.extend_small_data = false;
    return t;
  }

  function recTable(r) {
    const arch = (layers) => layers.map((l) => l.op + (l.out ? l.out : "") + (l.k ? "k" + l.k : "") + (l.stride > 1 ? "/s" + l.stride : "")).join(" → ");
    const rec = r.recommended;
    return `${rec ? `<div class="callout ok"><b>Recommended: ${esc(rec.id)}</b> – ${esc(r.why)}. ${fmt(rec.params)} params, ${fmt(rec.macs)} MACs${rec.receptive_field ? `, receptive field ${fmt(rec.receptive_field)}` : ""}; worst board ${bytes(rec.ram)} RAM, ${bytes(rec.flash)} flash, ${fmt(rec.latency_ms, 3)} ms.<div class="mono small" style="margin-top:6px">${esc(arch(rec.layers))} → fc(head)</div></div>` : `<div class="callout warn">${esc(r.why)}</div>`}
      <h3 style="margin-top:12px">Why</h3><ul class="small">${r.rules.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <p class="small muted">Made for goal <b>${esc(GOAL_LABEL[r.goal] || r.goal)}</b> on ${r.targets.map(esc).join(", ")}. If you change them, recommend again.</p>
      <div class="tbl-wrap"><table class="t"><thead><tr><th></th><th>Candidate</th><th class="n">Params</th><th class="n">MACs</th><th class="n">Receptive field</th>${r.targets.map((t) => `<th class="n">${esc(t)} RAM</th><th class="n">flash</th><th class="n">latency</th>`).join("")}<th>Fits?</th></tr></thead><tbody>
      ${r.candidates.map((c) => `<tr class="${c.recommended ? "rec" : c.fits ? "" : "nofit"}"><td>${c.recommended ? "★" : ""}</td><td><span class="mono small" title="${esc(arch(c.layers))}">${esc(c.id)}</span>${c.origin === "pool" ? ` <span class="badge">pool</span>` : c.origin === "library" ? ` <span class="badge b-info">library</span>` : ""}${c.over_cap ? ` <span class="badge b-warn">over cap</span>` : ""}</td><td class="n">${fmt(c.params)}</td><td class="n">${fmt(c.macs)}</td><td class="n">${c.receptive_field ? fmt(c.receptive_field) : "–"}</td>
        ${r.targets.map((t) => { const b = (c.boards || {})[t]; return b ? `<td class="n">${bytes(b.ram)}</td><td class="n">${bytes(b.flash)}</td><td class="n">${fmt(b.latency_ms, 3)} ms</td>` : `<td colspan="3" class="muted">no plan</td>`; }).join("")}
        <td>${c.fits ? `<span class="badge b-ok">fits</span>` : `<span class="badge b-bad" title="${esc(Object.values(c.boards || {}).flatMap((b) => (b && b.violations) || []).join("; "))}">no</span>`}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function stepReview(b) {
    st.name = st.name || `${st.tpl.name} – ${new Date().toLocaleDateString()}`;
    b.innerHTML = `<div class="card"><label class="f">Project name<input id="nm" value="${esc(st.name)}"></label>
      <div class="grid g4" style="margin-top:14px">
        <div class="kpi"><div class="l">Model type</div><div class="v" style="font-size:15px">${esc(st.tpl.icon)} ${esc(st.tpl.name)}</div></div>
        <div class="kpi"><div class="l">Data</div><div class="v" style="font-size:15px">${fmt(st.ds.n)} windows${st.ds.synthetic ? ` <span class="badge b-warn">synthetic</span>` : ""}</div></div>
        <div class="kpi"><div class="l">Boards</div><div class="v" style="font-size:15px">${st.targets.join(", ")}</div></div>
        <div class="kpi"><div class="l">Goal</div><div class="v" style="font-size:15px">${GOAL_LABEL[st.goal]}</div></div></div>
      ${Object.keys(st.training).length ? `<p class="small" style="margin-top:14px">Training settings: ${Object.entries(st.training).map(([k, v]) => `<code>${esc(k)}=${esc(String(v))}</code>`).join(" ")} (the rest automatic)</p>` : ""}
      <p class="small muted" style="margin-top:14px">The pipeline trains ${st.sizes.length} size(s)${st.archs.length ? `, ${st.archs.length} library architecture(s) (${st.archs.map(esc).join(", ")})` : ""}${st.customs.some((m) => m.include) ? ` and ${st.customs.filter((m) => m.include).length} customized model(s) (${st.customs.filter((m) => m.include).map((m) => esc(m.label)).join(", ")})` : ""}, quantizes each to INT8 (with quantization-aware fine-tuning when INT8 loses more than 1%), asks SAI for every plan on every board, picks the best that fits, then compiles, verifies and measures the deployed program's accuracy bit-exactly.</p>
      ${nav(true, true, true, "Build my model →")}</div>`;
    $("#back").onclick = () => { st.name = $("#nm").value; st.step--; draw(); };
    $("#next").onclick = async () => {
      $("#next").disabled = true;
      try {
        const r = await api("/api/projects", { json: { name: $("#nm").value, template_id: st.tpl.id, dataset_id: st.ds.id, targets: st.targets, goal: st.goal, sizes: st.sizes, architectures: st.archs, training: st.training, custom_variants: { ...(st.sizes.includes("auto") && st.auto ? { auto: st.auto } : {}), ...Object.fromEntries(st.customs.filter((m) => m.include).map((m) => [m.key, m.layers])) }, custom_labels: Object.fromEntries(st.customs.filter((m) => m.include).map((m) => [m.key, `${m.label} (from ${m.base})`])), budgets: st.budgets, metric_target: st.metric === "" ? null : +st.metric, epochs: st.epochs, init_from_pool: !!st.initPool, lr_search: !!st.lrSearch, pool_benchmark: !!st.poolBench, preprocess: st.prep } });
        location.hash = `#/job/${r.job.id}`;
      } catch (e) { toast(e.message); $("#next").disabled = false; }
    };
  }
  draw();
});

const LIC = () => (SYS && SYS.licences) || { proprietary: { label: "Own / proprietary data" } };
function datasetSummary(d) {
  const prev = (d.preview || []).map((p) => `<div class="card" style="padding:8px"><div class="small"><b>${esc(p.label)}</b></div>${p.kind === "series" ? sparkSeries(p.series) : heat(p.image)}</div>`).join("");
  return `<div style="margin-top:12px" class="stack">
    <div class="row"><b>${esc(d.name)}</b>${d.synthetic ? `<span class="badge b-warn">synthetic</span>` : ""}<span class="badge">${fmt(d.n)} windows</span><span class="badge">input ${d.shape.join("×")}</span></div>
    ${d.class_counts ? bars(d.class_counts) : d.target_stats ? `<div class="small">target ${fmt(d.target_stats.min, 2)} … ${fmt(d.target_stats.max, 2)} (mean ${fmt(d.target_stats.mean, 2)})</div>` : ""}
    <div class="row small"><span class="badge ${d.commercial_use ? "b-ok" : "b-warn"}">licence: ${esc((LIC()[d.licence] || {}).label || d.licence)}</span>
      ${d.frontend ? `<span class="badge b-info">front end: ${esc(d.frontend.type)} (C, same as on the MCU)</span>` : ""}
      ${d.recordings !== null && d.recordings !== undefined ? `<span class="badge">${fmt(d.recordings)} recordings</span>` : ""}
      ${d.quality ? `<span class="badge">${d.quality.duplicate_windows} duplicates</span>` : ""}</div>
    ${!d.commercial_use ? `<div class="callout">This licence does not allow commercial deployment: packages trained on it are marked <b>evaluation only</b>.</div>` : ""}
    ${(d.warnings || []).map((w) => `<div class="callout">${esc(w)}</div>`).join("")}
    <div class="grid g3">${prev}</div></div>`;
}

/* ------------------------------------------------------------------------------------------------ job page */
route(/^\/architectures$/, async (el) => {
  let kind = "", editing = null;
  async function load() {
    const r = await api("/api/architectures");
    const list = r.architectures.filter((a) => !kind || a.inputs.includes(kind));
    const fams = [...new Set(list.map((a) => a.family))];
    el.innerHTML = `<div class="page-h"><div><h1>Architecture library</h1><p class="muted">Model architectures built only from operators SAI compiles, checked with SAI on every board at a reference input. Pick them in a new project (Board &amp; goal step); the recommender costs them too.</p></div>
      <div class="row">${r.can_publish ? `<button class="btn" id="verall">Verify all with SAI</button>` : ""}${r.can_add ? `<button class="btn primary" id="addarch">+ Add architecture</button>` : ""}</div></div>
      <div class="chips" id="kf" style="margin-bottom:14px">${[["", "All inputs"], ["timeseries", "Time series"], ["2d", "Spectrogram / image"], ["features", "Feature vector"]].map(([k, v]) => `<button class="chip ${k === kind ? "on" : ""}" data-k="${k}">${v}</button>`).join("")}</div>
      <div id="archform"></div>
      ${fams.map((f) => `<h2 style="margin:18px 0 10px">${esc(f)}</h2><div class="arch-grid">${list.filter((a) => a.family === f).map((a) => { const ref = r.reference[a.inputs[0]];
        return `<div class="card"><div class="row"><b>${esc(a.name)}</b><span class="spacer"></span>${a.source === "custom" ? `<span class="badge">${a.org_id ? "your organisation" : "SAI"}</span>` : `<span class="badge b-acc">built-in</span>`}</div>
          <div class="small" style="margin-top:4px">${a.inputs.map((x) => `<span class="badge">${esc(x)}</span>`).join(" ")} ${a.tasks.map((x) => `<span class="badge b-info">${esc(x)}</span>`).join(" ")}</div>
          <p class="small" style="margin:8px 0 0">${esc(a.summary)}</p>${a.notes ? `<p class="small muted" style="margin:4px 0 0">${esc(a.notes)}</p>` : ""}${a.ref ? `<p class="small muted" style="margin:4px 0 0"><b>Based on:</b> ${esc(a.ref)}</p>` : ""}
          <div class="arch-chain">${esc(archChain(a.layers))}</div>
          ${Object.keys(a.checks).length ? `<table class="t small"><thead><tr><th>Board</th><th class="n">RAM</th><th class="n">Flash</th><th class="n">Latency</th><th></th></tr></thead><tbody>${Object.entries(a.checks).map(([t, c]) => `<tr><td>${esc(t)}</td><td class="n">${bytes(c.ram)}</td><td class="n">${bytes(c.flash)}</td><td class="n">${c.latency_ms === null ? "–" : fmt(c.latency_ms, 3) + " ms"}</td><td>${c.fits ? `<span class="badge b-ok">fits</span>` : `<span class="badge b-bad" title="${esc(((c.detail || {}).violations || []).join("; ") || (c.detail || {}).error || "")}">${c.ok ? "no fit" : "no plan"}</span>`}</td></tr>`).join("")}</tbody></table>
            <div class="small muted">SAI ${esc(Object.values(a.checks)[0].sai_version)} · input ${ref.channels}×${ref.height}×${ref.width} · ${ago(Object.values(a.checks)[0].checked)}</div>` : `<p class="small muted">Not checked with SAI yet.</p>`}
          <div class="row" style="margin-top:10px;gap:6px">${r.can_add ? `<button class="btn sm" data-ver="${esc(a.id)}">Verify with SAI</button>` : ""}${a.editable ? `<button class="btn sm ghost" data-edit="${esc(a.id)}">Edit</button><button class="btn sm ghost" data-del="${esc(a.id)}">Delete</button>` : ""}</div></div>`; }).join("")}</div>`).join("") || `<div class="empty small">No architectures for this input.</div>`}`;
    const form = (a) => {
      const v = a || { name: "", family: "Custom", size: "", inputs: ["timeseries"], tasks: ["classification"], layers: [{ op: "conv", out: 8, k: 5, stride: 2 }, { op: "dwconv", k: 3 }, { op: "pwconv", out: 16 }, { op: "gap" }], summary: "", notes: "" };
      $("#archform").innerHTML = `<div class="card" style="margin-bottom:16px"><h2>${a ? "Edit" : "Add"} architecture</h2><div class="grid g3">
        <label class="f">Name<input id="an" value="${esc(v.name)}"></label><label class="f">Family<input id="af" value="${esc(v.family)}"></label><label class="f">Size label<input id="as" value="${esc(v.size)}" placeholder="s, m, l"></label></div>
        <div class="row" style="margin-top:8px">${r.input_kinds.map((k) => `<label class="check small"><input type="checkbox" name="ain" value="${k}" ${v.inputs.includes(k) ? "checked" : ""}>${k}</label>`).join("")}<span class="spacer"></span>${r.tasks.map((k) => `<label class="check small"><input type="checkbox" name="atk" value="${k}" ${v.tasks.includes(k) ? "checked" : ""}>${k}</label>`).join("")}</div>
        <label class="f" style="margin-top:8px">Layers (JSON; SAI operators: conv, dwconv, pwconv, maxpool, avgpool, gap, fc, residual, dropout; the output head is added for you)<textarea id="al" rows="6" class="mono">${esc(JSON.stringify(v.layers, null, 0))}</textarea></label>
        <label class="f">Summary<input id="asu" value="${esc(v.summary)}"></label>
        ${!a && r.can_publish ? `<label class="check small" style="margin-top:8px"><input type="checkbox" id="aglob">Global (every customer sees it)</label>` : ""}
        <div class="row" style="margin-top:10px"><button class="btn primary" id="asave">Save</button><button class="btn ghost" id="acancel">Cancel</button><span class="small muted" id="amsg"></span></div></div>`;
      $("#acancel").onclick = () => { $("#archform").innerHTML = ""; };
      $("#asave").onclick = async () => {
        let layers; try { layers = JSON.parse($("#al").value); } catch { return toast("Layers are not valid JSON"); }
        const body = { name: $("#an").value, family: $("#af").value, size: $("#as").value, inputs: $$("input[name=ain]:checked").map((x) => x.value), tasks: $$("input[name=atk]:checked").map((x) => x.value), layers, summary: $("#asu").value, scope: $("#aglob") && $("#aglob").checked ? "global" : "org" };
        try { const x = await api(a ? `/api/architectures/${a.id}` : "/api/architectures", { method: a ? "PUT" : "POST", json: body }); toast(`Saved ${x.name}`); $("#archform").innerHTML = ""; load(); } catch (e) { $("#amsg").textContent = e.message; }
      };
    };
    if ($("#addarch")) $("#addarch").onclick = () => form(null);
    if ($("#verall")) $("#verall").onclick = () => busy($("#verall"), "Starting…", async () => { const j = await api("/api/architectures/verify", { json: {} }); toast("Verifying every architecture with SAI…"); location.hash = `#/job/${j.id}`; });
    el.onclick = async (e) => {
      const t = e.target;
      if (t.dataset.k !== undefined && t.classList.contains("chip")) { kind = t.dataset.k; return load(); }
      if (t.dataset.ver) { t.disabled = true; t.textContent = "Checking with SAI…"; const j = await api("/api/architectures/verify", { json: { ids: [t.dataset.ver] } });
        for (;;) { const s = await api(`/api/jobs/${j.id}`); if (["succeeded", "failed", "cancelled"].includes(s.status)) { if (s.status !== "succeeded") toast(s.error || s.status); break; } await new Promise((res) => setTimeout(res, 1500)); }
        return load(); }
      if (t.dataset.edit) return form(r.architectures.find((a) => a.id === t.dataset.edit));
      if (t.dataset.del && confirm("Delete this architecture from the library?")) { await api(`/api/architectures/${t.dataset.del}`, { method: "DELETE" }); toast("Deleted"); load(); }
    };
  }
  load();
});

route(/^\/multimodel$/, async (el) => {
  const [av, jobs, sys] = await Promise.all([api("/api/multimodel/available"), api("/api/jobs?kind=train"), system()]);
  const done = jobs.filter((j) => j.status === "succeeded");
  let picked = [];
  el.innerHTML = `<div class="page-h"><div><h1>Several models on one board</h1><p class="muted">Put finished projects on the same MCU, run one after another with <b>one shared workspace</b>. SAI's multi-model plugin plans the models jointly, proves the combined RAM and verifies every build.</p></div></div>
    ${av.multimodel ? "" : `<div class="callout bad">${esc(av.reason)}</div>`}
    <div class="grid g2"><div class="card"><h2>1. Projects</h2><p class="small muted">Finished SAI Core projects (2 to 8). Sequence models (XModels) are not supported by the plugin yet.</p><div id="mmlist"><span class="small muted">Loading…</span></div></div>
    <div class="card"><h2>2. Board and limits</h2><div class="stack"><label class="f">Name<input id="mmname" value="Multi-model build"></label>
      <label class="f">Board<select id="mmtgt">${sys.targets.filter((t) => ["md6605", "cortex-m7"].includes(t.id)).map((t) => `<option value="${t.id}">${esc(t.name)}</option>`).join("")}</select></label>
      <div class="grid g2"><label class="f">Max RAM (bytes)<input id="mmram" type="number" placeholder="board limit"></label><label class="f">Max flash (bytes)<input id="mmflash" type="number" placeholder="board limit"></label></div>
      <button class="btn primary" id="mmgo" ${av.multimodel ? "" : "disabled"}>Build together</button><p class="small muted">The result is evaluation-only (the plugin's own status: test signatures, no board run yet) and its download needs a manager's approval.</p></div></div></div>
    <div class="card" style="margin-top:16px"><h2>Previous multi-model builds</h2><div id="mmprev"></div></div>`;
  const results = await Promise.all(done.slice(0, 40).map((j) => api(`/api/jobs/${j.id}`).catch(() => null)));
  const ok = results.filter((j) => j && j.result && j.result.backend !== "xmodels");
  $("#mmlist").innerHTML = ok.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th></th><th>Project</th><th>Built for</th><th class="n">RAM</th><th class="n">Flash</th></tr></thead><tbody>${ok.map((j) => { const s = j.result.selected; return `<tr><td><input type="checkbox" data-pick="${esc(j.id)}"></td><td><a href="#/job/${esc(j.id)}">${esc(j.params.name || j.id)}</a><div class="small muted">${esc(j.params.template_id)}</div></td><td class="small">${esc(s.target)} · ${esc(s.size)}</td><td class="n">${bytes(s.ram)}</td><td class="n">${bytes(s.flash)}</td></tr>`; }).join("")}</tbody></table></div>` : `<div class="empty small">No finished SAI Core projects yet.</div>`;
  $("#mmlist").onchange = () => { picked = $$("[data-pick]:checked").map((x) => x.dataset.pick); };
  $("#mmgo").onclick = () => {
    if (picked.length < 2) return toast("Pick at least two projects");
    return busy($("#mmgo"), "Starting…", async () => { const j = await api("/api/multimodel", { json: { name: $("#mmname").value, job_ids: picked, target: $("#mmtgt").value, budgets: { max_ram: +$("#mmram").value || null, max_flash: +$("#mmflash").value || null } } }); location.hash = `#/job/${j.id}`; });
  };
  const prev = (await api("/api/jobs?kind=multimodel"));
  $("#mmprev").innerHTML = prev.length ? `<table class="t"><tbody>${prev.map((j) => `<tr><td><a href="#/job/${esc(j.id)}">${esc(j.params.name || j.id)}</a></td><td>${statusBadge(j.status, j.progress)}</td><td class="small">${esc(j.params.target)} · ${(j.params.job_ids || []).length} models</td><td class="small muted">${ago(j.created)}</td></tr>`).join("")}</tbody></table>` : `<p class="small muted">None yet.</p>`;
});

async function refreshPending() {
  const el = $("#pendingCount"); if (!el || !ME) return;
  try {
    const r = await api("/api/packages?status=pending");
    const n = r.can_approve ? r.packages.filter((p) => p.approval.requested_by !== ME.id).length : 0;
    el.hidden = !n; el.textContent = n;
  } catch { /* not signed in yet */ }
}

route(/^\/packages$/, async (el) => {
  let status = "";
  const ST = { not_requested: ["not requested", ""], pending: ["pending approval", "b-pend"], approved: ["approved", "b-ok"], rejected: ["rejected", "b-bad"], revoked: ["revoked", "b-bad"] };
  el.innerHTML = `<div class="page-h"><div><h1>Compiled packages</h1><p class="muted">Every compiled package the backend has produced (SAI-generated C sources, runtime, headers, manifest, memory certificate, signature) with its model and report. These files leave only with a manager's approval; every request, decision and download is audited.</p></div></div>
    <div class="kpis" id="pk"></div><div class="chips" id="pf" style="margin:14px 0">${[["", "All"], ["pending", "Pending"], ["approved", "Approved"], ["not_requested", "Not requested"], ["rejected", "Rejected"], ["revoked", "Revoked"]].map(([k, v]) => `<button class="chip ${k === status ? "on" : ""}" data-s="${k}">${v}</button>`).join("")}</div>
    <div class="card"><div class="tbl-wrap" id="pt"></div></div><div id="pd"></div>`;
  async function load() {
    const all = await api("/api/packages");
    const can = all.can_approve, list = all.packages.filter((p) => !status || p.approval.status === status);
    const cnt = (s) => all.packages.filter((p) => p.approval.status === s).length;
    $("#pk").innerHTML = [["Packages", all.packages.length], ["Pending approval", cnt("pending")], ["Approved", cnt("approved")], ["Downloads", all.packages.reduce((a, p) => a + (p.approval.downloads || 0), 0)]].map(([l, v]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${fmt(v)}</div></div>`).join("");
    $("#pt").innerHTML = list.length ? `<table class="t"><thead><tr><th>Project</th><th>Board · plan</th><th class="n">RAM</th><th class="n">Flash</th><th class="n">Accuracy</th><th>SAI</th><th>Built</th><th>Approval</th><th class="n">Downloads</th><th></th></tr></thead><tbody>
      ${list.map((p) => { const a = p.approval, [lab, cls] = ST[a.status];
        const acts = [];
        if (a.status === "approved") acts.push(`<a class="btn primary sm" href="/api/jobs/${esc(p.job_id)}/download/package.zip" title="compiled package">⬇ package</a>${p.kind === "multimodel" ? "" : `<a class="btn sm" href="/api/jobs/${esc(p.job_id)}/download/model.onnx">model</a>`}<a class="btn sm" href="/api/jobs/${esc(p.job_id)}/download/report.json">report</a>`);
        if (["not_requested", "rejected", "revoked"].includes(a.status)) acts.push(`<button class="btn sm" data-act="request" data-j="${esc(p.job_id)}">Request</button>`);
        if (a.status === "pending" && can && a.requested_by !== ME.id) acts.push(`<button class="btn primary sm" data-act="approve" data-j="${esc(p.job_id)}">Approve</button><button class="btn sm" data-act="reject" data-j="${esc(p.job_id)}">Reject</button>`);
        if (a.status === "pending" && can && a.requested_by === ME.id) acts.push(`<span class="small muted">your request: another manager decides</span>`);
        if (a.status === "approved" && can) acts.push(`<button class="btn sm ghost" data-act="revoke" data-j="${esc(p.job_id)}">Revoke</button>`);
        return `<tr><td><a href="#/job/${esc(p.job_id)}"><b>${esc(p.project || p.job_id)}</b></a><div class="small muted">${esc(p.template_id || "")}${ME.perms.platform_admin ? ` · ${esc(p.org_name || "")}` : ""}${p.deployable === false ? ` · <span class="badge b-warn">evaluation only</span>` : ""}</div></td>
          <td class="small">${esc(p.target || "")} · <code>${esc(p.plan || "")}</code><div class="muted">size ${esc(p.size || "")}${p.signed_key ? ` · signed ${esc(p.signed_key)}` : ""}</div></td>
          <td class="n">${bytes(p.ram)}</td><td class="n">${bytes(p.flash)}</td><td class="n">${p.metric === "accuracy" ? pct(p.accuracy) : fmt(p.accuracy, 3)}</td><td class="mono small">${esc(p.sai || "")}</td><td class="small muted">${ago(p.built)}</td>
          <td><span class="badge ${cls}">${lab}</span>${a.requested_by_name ? `<div class="small muted">req. ${esc(a.requested_by_name)} ${ago(a.requested_at)}${a.request_note ? `: “${esc(a.request_note)}”` : ""}</div>` : ""}${a.decided_by_name ? `<div class="small muted">${esc(a.status)} by ${esc(a.decided_by_name)} ${ago(a.decided_at)}${a.decision_note ? `: “${esc(a.decision_note)}”` : ""}</div>` : ""}</td>
          <td class="n">${fmt(a.downloads || 0)}</td><td><div class="row" style="gap:6px;flex-wrap:nowrap">${acts.join("")}</div><a class="small" href="#" data-hist="${esc(p.job_id)}">history</a></td></tr>`; }).join("")}</tbody></table>`
      : `<div class="empty small">No ${status ? (ST[status] || [status])[0] + " " : ""}packages. A package appears here when a project finishes.</div>`;
    refreshPending();
  }
  el.onclick = async (e) => {
    const t = e.target;
    if (t.dataset.s !== undefined && t.classList.contains("chip")) { status = t.dataset.s; $$("#pf .chip").forEach((c) => c.classList.toggle("on", c.dataset.s === status)); return load(); }
    if (t.dataset.hist) {
      e.preventDefault();
      const d = await api(`/api/packages/${t.dataset.hist}`);
      $("#pd").innerHTML = `<div class="card" style="margin-top:16px"><h2>History · ${esc(t.dataset.hist)}</h2>${d.history.length ? `<table class="t"><tbody>${d.history.map((h) => `<tr><td class="small muted">${new Date(h.ts * 1000).toLocaleString()}</td><td>${esc(h.action.replace("package_", ""))}</td><td class="small">${esc(h.email || "")}</td><td class="small">${esc((h.detail && (h.detail.note || h.detail.file)) || "")}</td></tr>`).join("")}</tbody></table>` : `<p class="small muted">No activity yet.</p>`}</div>`;
      $("#pd").scrollIntoView({ behavior: "smooth" });
      return;
    }
    const act = t.dataset.act, j = t.dataset.j;
    if (!act) return;
    const ask = { request: "Why do you need the compiled package? (shown to the approving manager)", approve: "Approval note (optional)", reject: "Reason for rejecting (required)", revoke: "Reason for revoking (required)" }[act];
    const note = prompt(ask, "");
    if (note === null) return;
    await busy(t, "…", async () => { await api(`/api/packages/${j}/${act}`, { json: { note } }); toast({ request: "Requested", approve: "Approved", reject: "Rejected", revoke: "Revoked" }[act]); });
    load();
  };
  load();
});

route(/^\/jobs$/, async (el) => {
  let timer = null, kind = "", status = "";
  el.innerHTML = `<div class="page-h"><div><h1>Jobs &amp; logs</h1><p class="muted">Every training run, recommendation, import, rebuild and regression, live. Open a job for its training monitor and full log.</p></div></div>
    <div class="row" style="margin-bottom:12px"><select id="jk"><option value="">All kinds</option>${["train", "multimodel", "recommend", "archverify", "ingest", "fitcheck", "analyze", "rebuild", "basetrain", "regression"].map((k) => `<option>${k}</option>`).join("")}</select>
    <select id="js"><option value="">Any status</option>${["running", "queued", "succeeded", "failed", "cancelled"].map((k) => `<option>${k}</option>`).join("")}</select><span class="spacer"></span><span class="small muted" id="jupd"></span></div>
    <div class="card"><div class="tbl-wrap" id="jt"></div></div>`;
  async function load() {
    try {
      const jobs = (await api(`/api/jobs${kind ? `?kind=${kind}` : ""}`)).filter((j) => !status || j.status === status);
      const active = jobs.filter((j) => j.status === "running" || j.status === "queued").length;
      $("#jt").innerHTML = jobs.length ? `<table class="t"><thead><tr><th>Job</th><th>Kind</th><th>Status</th><th>Stage</th><th>Last log line</th><th class="n">Warnings</th><th class="n">Duration</th><th>Started</th></tr></thead><tbody>
        ${jobs.map((j) => `<tr><td><a href="#/job/${esc(j.id)}">${esc(jobTitle(j))}</a><div class="mono small muted">${esc(j.id)}</div></td><td>${esc(j.kind)}</td><td>${statusBadge(j.status, j.progress)}${j.status === "running" ? `<div class="progress" style="height:5px;margin-top:4px;width:90px"><i style="width:${Math.round((j.progress || 0) * 100)}%"></i></div>` : ""}</td>
          <td class="small">${esc(j.stage || "–")}</td><td class="mono small" style="max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(j.last_log || "")}">${esc(j.error || j.last_log || "")}</td>
          <td class="n">${j.warnings ? `<span class="badge b-warn">${j.warnings}</span>` : "–"}</td><td class="n">${dur((["running", "queued"].includes(j.status) ? Date.now() / 1000 : j.updated) - j.created)}</td><td class="small muted">${ago(j.created)}</td></tr>`).join("")}</tbody></table>` : `<div class="empty small">No jobs yet.</div>`;
      $("#jupd").textContent = `${active} active · updated ${new Date().toLocaleTimeString()}`;
    } catch (e) { if (e instanceof Stale) return; toast(e.message); }
    timer = setTimeout(load, 3000);
  }
  $("#jk").onchange = () => { kind = $("#jk").value; clearTimeout(timer); load(); };
  $("#js").onchange = () => { status = $("#js").value; clearTimeout(timer); load(); };
  onLeave(() => clearTimeout(timer));
  load();
});

route(/^\/job\/([\w]+)$/, async (el, jid) => {
  let logs = [], last = 0, timer = null, job = null, rendered = null, metrics = [], mlast = 0;
  const sys = await system();
  el.innerHTML = `<div id="head"></div><div id="live"></div><div id="logbox"></div><div id="res"></div>`;
  async function poll() {
    try {
      job = await api(`/api/jobs/${jid}?log_after=${last}&metrics_after=${mlast}`);
      logs = logs.concat(job.logs); if (job.logs.length) last = job.logs[job.logs.length - 1].seq;
      metrics = metrics.concat(job.metrics || []); if ((job.metrics || []).length) mlast = job.metrics[job.metrics.length - 1].seq;
      drawHead(); drawLive(); drawLog();
      if (["succeeded", "failed", "cancelled"].includes(job.status)) {
        if (job.kind === "recommend" && job.status === "succeeded" && rendered !== job.status) { rendered = job.status; return drawRecommend(job.result); }
        if (job.kind === "archverify" && job.status === "succeeded" && rendered !== job.status) { rendered = job.status; const r = job.result; $("#res").innerHTML = `<div class="card" style="margin-top:16px"><h2>SAI check</h2><p class="small">${r.checked} architecture(s) checked on ${r.targets.map(esc).join(", ")}${r.failed.length ? `; <b>${r.failed.length} failed</b>: ${r.failed.map((f) => esc(f.id + ": " + f.error)).join("; ")}` : ""}.</p><table class="t small"><tbody>${Object.entries(r.fits).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td>${r.targets.map((t) => v.includes(t) ? `<span class="badge b-ok">✓ ${esc(t)}</span>` : `<span class="badge b-bad">✗ ${esc(t)}</span>`).join(" ")}</td></tr>`).join("")}</tbody></table><p class="small"><a href="#/architectures">Open the architecture library</a></p></div>`; return; }
        if (job.kind === "ingest" && job.status === "succeeded" && rendered !== job.status) { rendered = job.status; $("#res").innerHTML = `<div class="callout ok" style="margin-top:16px">Dataset imported: <a href="#/data/${esc(job.result.dataset_id)}">open it</a>.</div>`; return; }
        if (["rebuild", "regression", "basetrain"].includes(job.kind) && job.status === "succeeded" && rendered !== job.status) { rendered = job.status; return drawPlansOnly(job.result); }
        if (rendered !== job.status) { rendered = job.status; drawResult(); if (job.kind === "train" && job.status === "succeeded") drawPackageButton().catch(() => {}); }
        return;
      }
    } catch (e) { if (e instanceof Stale) return; toast(e.message); }
    timer = setTimeout(poll, 1000);
  }
  onLeave(() => clearTimeout(timer));
  let logOpen = null, logFollow = true, logTop = 0, logFilter = "", logWarn = false, logShown = -1, monOpen = null;
  function drawHead() {
    if (logOpen === null) logOpen = job.status !== "succeeded";
    const title = jobTitle(job);
    $("#head").innerHTML = `<div class="page-h"><div><h1>${esc(title)}</h1><div class="row small muted">${statusBadge(job.status, job.progress)} <span class="mono">${esc(job.id)}</span>${job.kind === "train" ? ` · ${esc(job.params.template_id)} · goal ${esc(GOAL_LABEL[job.params.goal] || job.params.goal)}` : ""}</div></div>
      ${job.status === "running" || job.status === "queued" ? `<button class="btn" id="cancel">Cancel</button>` : ""}</div>
      <div class="card"><div class="progress" style="margin-bottom:12px"><i style="width:${Math.round((job.progress || 0) * 100)}%"></i></div>
      <div class="timeline">${(job.stages || []).map((s) => `<div class="tl ${s.status}"><div>${esc(s.title)}</div><div class="s">${esc(s.status)}${s.note ? " · " + esc(s.note) : ""}</div></div>`).join("")}</div>
      ${job.error ? `<div class="callout bad" style="margin-top:12px">${esc(job.error)}</div>` : ""}
      <div class="small muted" style="margin-top:10px">${job.worker ? `worker ${esc(job.worker)} · ` : ""}started ${ago(job.created)} · ${dur(((["succeeded", "failed", "cancelled"].includes(job.status) ? job.updated : Date.now() / 1000) - job.created))} elapsed</div></div>`;
    const c = $("#cancel"); if (c) c.onclick = () => api(`/api/jobs/${jid}/cancel`, { method: "POST" }).then(() => toast("Cancelling…"));
  }
  // ---- live training monitor: one card per training run (size), updated every poll --------------------------
  function drawLive() {
    if (!metrics.length) return;
    const done = ["succeeded", "failed", "cancelled"].includes(job.status);
    if (monOpen === null) monOpen = !done;
    const groups = {};
    metrics.forEach((m) => (groups[m.series] = groups[m.series] || []).push(m));
    const names = Object.keys(groups), latest = metrics[metrics.length - 1].series;
    const card = (name) => {
      const pts = groups[name], p = pts[pts.length - 1], [kind, size] = name.split(":");
      const avg = pts.reduce((a, x) => a + x.seconds, 0) / pts.length;
      const best = pts.reduce((b, x) => (x.val_loss < b.val_loss ? x : b), pts[0]);
      const active = !done && name === latest;
      const state = active ? `<span class="badge b-acc">epoch ${p.epoch}/${p.epochs}</span>` : p.epoch < p.epochs ? `<span class="badge">stopped early at ${p.epoch}/${p.epochs}</span>` : `<span class="badge b-ok">done · ${p.epochs} epochs</span>`;
      const acc = pts.some((x) => x.val_acc !== null && x.val_acc !== undefined);
      return `<div class="card"><div class="row"><b>${kind === "qat" ? "QAT fine-tune" : "Training"} · ${esc(size)}</b><span class="spacer"></span>${state}</div>
        ${active ? `<div class="progress" style="margin:8px 0"><i style="width:${Math.round(p.epoch / p.epochs * 100)}%"></i></div>` : ""}
        <div class="kv"><div>train loss<b>${fmt(p.train_loss, 4)}</b></div><div>val loss<b>${fmt(p.val_loss, 4)}</b></div><div>best val<b>${fmt(best.val_loss, 4)}</b><span class="muted">epoch ${best.epoch}</span></div>${acc ? `<div>val accuracy<b>${pct(p.val_acc)}</b></div>` : `<div></div>`}
          <div>learning rate<b>${Number(p.lr).toExponential(1)}</b></div><div>per epoch<b>${fmt(avg, 2)} s</b></div><div>${active ? "remaining" : "total"}<b>${active ? "≈ " + dur((p.epochs - p.epoch) * avg) : dur(avg * pts.length)}</b></div><div></div></div>
        ${pts.length < 2 ? `<p class="small muted" style="margin:16px 0">Curves appear after the second epoch.</p>` : `${lineChart([{ values: pts.map((x) => x.train_loss), color: "var(--c1)" }, { values: pts.map((x) => x.val_loss), color: "var(--c2)" }], { h: 150, xLabel: "epoch", yLabel: "loss" })}
        ${acc ? lineChart([{ values: pts.map((x) => x.val_acc ?? 0), color: "var(--c3)" }], { h: 110, xLabel: "epoch", yLabel: "validation accuracy" }) : ""}`}
        <div class="legend"><span><i style="background:var(--c1)"></i>train loss</span><span><i style="background:var(--c2)"></i>validation loss</span>${acc ? `<span><i style="background:var(--c3)"></i>validation accuracy</span>` : ""}</div></div>`;
    };
    const tc = (job.params || {}).training || {};
    $("#live").innerHTML = `<details class="card" id="mond" style="margin-top:16px" ${monOpen ? "open" : ""}><summary><b>Training monitor</b> <span class="small muted">${names.length} run(s), ${metrics.length} epochs logged${Object.keys(tc).length ? ` · custom settings: ${Object.entries(tc).map(([k, v]) => `${esc(k)}=${esc(String(v))}`).join(", ")}` : " · automatic settings"}</span></summary>
      <div class="mon" style="margin-top:12px">${names.map(card).join("")}</div></details>`;
    $("#mond").ontoggle = () => { monOpen = $("#mond").open; };
  }
  // ---- log viewer: built once, lines appended; filter, warnings-only, follow, download -----------------------
  function drawLog() {
    if (!$("#logd")) {
      $("#logbox").innerHTML = `<details class="card" id="logd" style="margin-top:16px" ${logOpen ? "open" : ""}><summary><b>Pipeline log</b> <span class="small muted" id="logcount"></span></summary>
        <div class="logbar"><input type="search" id="logq" placeholder="Filter lines…"><label class="check small"><input type="checkbox" id="logw">warnings &amp; errors only</label>
        <label class="check small"><input type="checkbox" id="logf" checked>follow</label><a class="btn sm" href="/api/jobs/${jid}/log.txt">⬇ Download log</a></div>
        <div class="log" id="log" style="max-height:420px"></div></details>`;
      $("#logd").ontoggle = () => { logOpen = $("#logd").open; };
      $("#logq").oninput = () => { logFilter = $("#logq").value.toLowerCase(); logShown = -1; drawLog(); };
      $("#logw").onchange = () => { logWarn = $("#logw").checked; logShown = -1; drawLog(); };
      $("#logf").onchange = () => { logFollow = $("#logf").checked; if (logFollow) $("#log").scrollTop = $("#log").scrollHeight; };
      const lg = $("#log");
      lg.onscroll = () => { const atEnd = lg.scrollTop + lg.clientHeight >= lg.scrollHeight - 8; if (atEnd !== logFollow) { logFollow = atEnd; $("#logf").checked = atEnd; } logTop = lg.scrollTop; };
    }
    const cls = (t) => /error|traceback|failed|exception/i.test(t) ? "le" : /warning|evaluation only|does not fit|diverged/i.test(t) ? "lw" : t.startsWith("== ") ? "ls" : "";
    const warns = logs.filter((l) => cls(l.line) === "lw" || cls(l.line) === "le").length;
    $("#logcount").textContent = `(${logs.length} lines${warns ? `, ${warns} warning/error` : ""})`;
    if (logShown === logs.length) return;
    const show = logs.filter((l) => (!logWarn || ["lw", "le"].includes(cls(l.line))) && (!logFilter || l.line.toLowerCase().includes(logFilter)));
    const hl = (t) => { const e = esc(t); if (!logFilter) return e; const q = esc(logFilter).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); return e.replace(new RegExp(q, "gi"), (m) => `<mark>${m}</mark>`); };
    const lg = $("#log");
    lg.innerHTML = show.map((l) => `<div class="${cls(l.line)}"><span class="muted">${new Date(l.ts * 1000).toLocaleTimeString()}</span>  ${hl(l.line)}</div>`).join("") || `<span class="muted">no matching lines</span>`;
    lg.scrollTop = logFollow ? lg.scrollHeight : logTop;
    logShown = logs.length;
  }
  async function drawPackageButton() {
    const b = $("#pkgbtn"); if (!b) return;
    const a = await api(`/api/packages/${jid}`);
    const who = (x) => esc(x || "");
    if (a.status === "approved") {
      b.innerHTML = GATED.filter((d) => ((job.result || {}).downloads || GATED).includes(d)).map((d, i) => `<a class="btn ${i ? "" : "primary"} sm" href="/api/jobs/${jid}/download/${d}">${i ? "" : "⬇ "}${d}</a>`).join("");
      $("#pkgnote").innerHTML = `<span class="badge b-ok">approved</span> by ${who(a.decided_by_name)} ${ago(a.decided_at)}${a.decision_note ? ` – “${esc(a.decision_note)}”` : ""} · <a href="#/packages">Compiled packages</a>`;
    } else if (a.status === "pending") {
      b.innerHTML = `<button class="btn sm" disabled>⏳ package, model &amp; report awaiting approval</button>`;
      $("#pkgnote").innerHTML = `Requested by ${who(a.requested_by_name)} ${ago(a.requested_at)}. A manager approves it on <a href="#/packages">Compiled packages</a>.`;
    } else {
      const files = GATED.filter((d) => ((job.result || {}).downloads || GATED).includes(d)).map((d) => d.split(".")[0]).join(", ");
      b.innerHTML = `<button class="btn primary sm" id="pkgreq">Request download (${esc(files)})</button>`;
      $("#pkgnote").innerHTML = a.status === "not_requested" ? `The compiled package, the model and the report are released after a manager's approval.` : `<span class="badge b-bad">${esc(a.status)}</span> ${esc(a.decision_note || "")} – you can request again.`;
      $("#pkgreq").onclick = async () => {
        const note = prompt("Why do you need the compiled package? (shown to the approving manager)", "");
        if (note === null) return;
        await busy($("#pkgreq"), "Requesting…", async () => { await api(`/api/packages/${jid}/request`, { json: { note } }); toast("Requested: a manager will review it"); drawPackageButton(); refreshPending(); });
      };
    }
  }
  function drawMulti(r) {
    const pct0 = r.separate_ram ? Math.round(r.saved_ram / r.separate_ram * 100) : 0;
    $("#res").innerHTML = `<div class="callout bad" style="margin-top:16px"><b>Evaluation only:</b> ${esc(r.evaluation_reason)}</div>
      <div class="hero" style="margin-top:16px"><div class="card"><h2>${esc(r.name || "Multi-model build")} · ${esc(r.target)}</h2>
        <div class="kpis"><div class="kpi"><div class="l">Device RAM, one shared build</div><div class="v">${bytes(r.device_ram)}</div><div class="small">${ev(r.evidence === "proven_upper_bound" ? "proven bound" : r.evidence)}</div></div>
        <div class="kpi"><div class="l">Same models side by side</div><div class="v">${bytes(r.separate_ram)}</div><div class="small muted">each with its own workspace</div></div>
        <div class="kpi"><div class="l">Saved</div><div class="v" style="color:var(--ok)">${bytes(r.saved_ram)}</div><div class="small">${pct0}% less RAM</div></div>
        <div class="kpi"><div class="l">Flash (all models)</div><div class="v">${bytes(r.flash)}</div><div class="small">${r.fits ? `<span class="badge b-ok">fits ${esc(r.target)}</span>` : `<span class="badge b-bad">${esc(r.violations.join("; "))}</span>`}</div></div></div>
        <p class="small muted" style="margin-top:10px">Shared workspace ${bytes(r.shared.shared_workspace)} · stack ${bytes(r.shared.stack)} · static ${bytes(r.shared.static)} · caller buffers ${bytes((r.shared.caller_inputs || 0) + (r.shared.caller_outputs || 0))} · SAI verify ${r.verify.passed}/${r.verify.total} builds passed</p>
        <div class="row" style="margin-top:10px"><span id="pkgbtn" class="row" style="gap:8px"></span></div><div id="pkgnote" class="small" style="margin-top:6px"></div></div>
      <div class="card"><h2>Models</h2><div class="tbl-wrap"><table class="t"><thead><tr><th>Model</th><th>ABI</th><th class="n">Workspace</th><th class="n">Stack</th><th class="n">Flash</th><th class="n">Alone (RAM bound)</th></tr></thead><tbody>
        ${r.models.map((m) => `<tr><td><a href="#/job/${esc(m.job_id)}">${esc(m.project)}</a><div class="mono small muted">${esc(m.id)}</div></td><td>${esc(m.abi)}</td><td class="n">${bytes(m.workspace)}</td><td class="n">${bytes(m.stack)}</td><td class="n">${bytes(m.linked_flash)}</td><td class="n">${bytes(m.ram_bound)}</td></tr>`).join("")}</tbody></table></div>
        <p class="small muted" style="margin-top:8px">Run one after another (no concurrent calls); ABI 1.1 builds keep the model input inside the shared workspace.</p></div></div>
      <div class="card" style="margin-top:16px"><h2>What this build does not claim</h2><ul class="small">${r.not_claimed.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
    drawPackageButton().catch(() => {});
  }
  function drawRecommend(r) {
    const rec = r.recommended;
    $("#res").innerHTML = `<div class="card" style="margin-top:16px"><h2>Recommendation</h2>${rec ? `<div class="callout ok"><b>${esc(rec.id)}</b>: ${esc(r.why)}. ${fmt(rec.params)} params, ${fmt(rec.macs)} MACs; worst board ${bytes(rec.ram)} RAM, ${bytes(rec.flash)} flash, ${ms(rec.latency_ms)}.</div>` : `<div class="callout warn">${esc(r.why)}</div>`}
      <ul class="small" style="margin-top:10px">${r.rules.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul><p class="small muted">${r.candidates.length} candidates costed by SAI on ${r.targets.map(esc).join(", ")}. Use it from the new-project wizard (Board &amp; goal step).</p></div>`;
  }
  function drawResult() {
    if (job.status !== "succeeded") return;
    const r = job.result;
    if (job.kind === "fitcheck" || job.kind === "analyze") return drawPlansOnly(r);
    if (job.kind === "multimodel") return drawMulti(r);
    const sel = r.selected, v = r.variants.find((x) => x.size === sel.size), task = r.template.task;
    const mt = (m) => task === "classification" ? pct(m.value) : fmt(m.value, 3);
    $("#res").innerHTML = `
    ${r.deployable === false ? `<div class="callout bad" style="margin-top:16px"><b>Evaluation only:</b> ${esc(r.evaluation_reason || "dataset licence")}. The package carries an EVALUATION_ONLY marker and a compile-time warning.</div>` : ""}
    ${r.backend === "xmodels" ? `<div class="callout info" style="margin-top:16px"><b>Sequence model</b> built by SAI's front end through the XModels backend (${esc(sel.schedule || "")}). The accuracy below is the generated C run on every test window; board runs and rebuilds are for SAI Core packages and are not available for it yet.</div>` : ""}
    ${r.signature ? `<p class="small muted" style="margin-top:10px">Package signed (Ed25519, key <code>${esc(r.signature.key_id)}</code>); verify with <code>tools/verify_package.py</code> and <code>/api/signing/public-key</code>.</p>` : ""}
    ${(r.preprocessing || {}).preprocess ? (() => { const pp = r.preprocessing.preprocess; return `<p class="small muted" style="margin-top:10px">Preprocessing: ${pp.window_ops ? `channels ${pp.channels ? esc(pp.channels.join(",")) : "all"}, decimate ÷${pp.decimate}${pp.demean ? ", mean removed" : ""} (C, on the MCU too) · ` : ""}${esc(pp.normalization)} normalization${pp.clip ? `, clip ±${pp.clip}` : ""} · input ${esc((r.preprocessing.raw_shape || []).join("×"))} → ${esc((r.preprocessing.model_input_shape || []).join("×"))} · <code>sai_app_prepare()</code> in the package.</p>`; })() : ""}
    ${r.split_mode ? `<p class="small muted" style="margin-top:10px">Test split: ${r.split_mode === "recording" ? "whole recordings held out (no leakage between windows of one recording)" : r.split_mode === "mixed" ? "recordings where available, windows elsewhere" : "random windows (no recording information in the data)"}.</p>` : ""}
    ${r.dataset.synthetic ? `<div class="callout" style="margin-top:16px">Trained on <b>synthetic demo data</b>. The pipeline, plans and resource numbers are real; the accuracy is not a field accuracy.</div>` : ""}
    <div class="hero" style="margin-top:16px">
      <div class="card"><div class="row"><h2 style="margin:0">Selected model</h2><span class="spacer"></span><span class="badge b-acc">${esc(sel.size)}</span><span class="badge" style="color:${TARGET_COLOR[sel.target]}">${esc(sel.target)}</span></div>
        <p class="small muted" style="margin:6px 0 12px">Plan <code>${esc(sel.schedule)}</code> chosen by SAI objective <b>${esc(sel.objective)}</b> for goal <b>${esc(GOAL_LABEL[r.goal])}</b>${sel.note ? ` · ${esc(sel.note)}` : ""}</p>
        <div class="kpis">
          <div class="kpi"><div class="l">RAM total</div><div class="v">${bytes(sel.ram)}</div><div class="small">ws ${bytes(sel.workspace)} ${ev(sel.workspace_evidence)} · stack ${bytes(sel.stack)} ${ev(sel.stack_evidence)}</div></div>
          <div class="kpi"><div class="l">Flash (code + weights)</div><div class="v">${bytes(sel.flash)}</div><div class="small">${ev(sel.flash_evidence)}</div></div>
          <div class="kpi"><div class="l">Latency</div><div class="v">${ms(sel.latency.ms)}</div><div class="small">${fmt(sel.latency.cycles)} cycles ${ev(sel.latency.evidence)}</div></div>
          <div class="kpi"><div class="l">MACs / params</div><div class="v">${fmt(v.summary.macs / 1000, 1)}k</div><div class="small">${fmt(v.summary.params)} params</div></div></div></div>
      <div class="card"><h2>${r.exact.headline === "accuracy" ? "Accuracy" : r.exact.headline === "r2" ? "Accuracy (R²)" : "Detection (AUC)"}</h2>
        <div class="grid g3"><div class="stat"><span class="l">Float</span><span class="big">${mt(v.float)}</span></div>
        <div class="stat"><span class="l">INT8 (${esc(v.quant_method || "PTQ")})</span><span class="big">${mt(v.int8)}</span></div>
        <div class="stat"><span class="l">Deployed C, bit-exact</span><span class="big" style="color:var(--ok)">${mt(r.exact)}</span></div></div>
        <p class="small muted" style="margin-top:8px">SAI verify: <b>${r.verify.passed}/${r.verify.total}</b> checks passed (manifest, hashes, memory certificate replay, SBOM, reproducible library…)${r.certificate_check ? ` · certificate re-derived by <code>sai certify --check</code>: <b>${r.certificate_check.passed}/${r.certificate_check.total}</b>` : ""}</p>
        <div class="row" style="margin-top:10px"><span id="pkgbtn" class="row" style="gap:8px"></span>${r.downloads.filter((d) => !GATED.includes(d)).map((d) => `<a class="btn sm" href="/api/jobs/${jid}/download/${d}">${esc(d)}</a>`).join("")}</div><div id="pkgnote" class="small" style="margin-top:6px"></div></div></div>
    <div class="card" style="margin-top:16px"><h2>Model sizes</h2><div class="tbl-wrap"><table class="t"><thead><tr><th>Size</th><th class="n">Params</th><th class="n">MACs</th><th class="n">Float</th><th class="n">INT8</th><th>Quant</th><th class="n">Plans that fit</th><th>Best plan for goal</th></tr></thead><tbody>
      ${r.variants.map((x) => { const rec = x.plans.recommended; return `<tr class="${x.size === sel.size ? "rec" : ""}"><td><b>${x.size}</b></td><td class="n">${fmt(x.summary.params)}</td><td class="n">${fmt(x.summary.macs)}</td><td class="n">${mt(x.float)}</td><td class="n">${mt(x.int8)}</td><td>${esc(x.quant_method || "PTQ")}</td><td class="n">${x.plans.feasible}/${x.plans.candidates.length}</td><td class="small">${rec ? `${esc(rec.target)} · ${bytes(rec.ram)} RAM · ${ms(rec.latency.ms)}` : `<span class="muted">${esc(x.plans.advice || "none")}</span>`}</td></tr>`; }).join("")}
    </tbody></table></div></div>
    <div class="grid g2" style="margin-top:16px">
      <div class="card"><h2>Plan space: RAM vs latency</h2>${planScatter(r.variants, sel)}<div class="legend"><span><i style="background:var(--c1)"></i>cortex-m7</span><span><i style="background:var(--c2)"></i>md6605</span><span>hollow = does not fit</span><span>ring = selected</span></div></div>
      <div class="card">${task === "classification" ? `<h2>Confusion matrix (bit-exact program)</h2>${confusion(r.exact.confusion, r.dataset.labels)}` : `<h2>Training curve (${esc(sel.size)})</h2>`}
        ${lineChart([{ values: v.history.map((h) => h.train_loss), color: "var(--c1)" }, { values: v.history.map((h) => h.val_loss), color: "var(--c2)" }], { xLabel: "epoch", yLabel: "loss" })}<div class="legend"><span><i style="background:var(--c1)"></i>train loss</span><span><i style="background:var(--c2)"></i>validation loss</span></div></div></div>
    <div class="card" style="margin-top:16px"><h2>Every plan SAI produced for the ${esc(sel.size)} model</h2>${planTable(v.plans.candidates)}${spaceNote(v.plans)}</div>
    ${tierCard(r)}
    ${provCard(r, jid)}
    <div class="card" style="margin-top:16px" id="insight"></div>
    <div class="card" style="margin-top:16px" id="hil"></div>`;
    const rb = $("#rebuild");
    if (rb) rb.onclick = () => busy(rb, "Queuing…", async () => { const j = await api(`/api/jobs/${jid}/rebuild`, { method: "POST" }); toast("Rebuild queued"); location.hash = `#/job/${j.id}`; });
    drawHil();
    drawInsight();
  }
  /* insight: what-if sliders, battery life & power, drift */
  async function drawInsight() {
    const box = $("#insight"); if (!box) return;
    let tab = "whatif";
    const draw = () => {
      box.innerHTML = `<div class="row"><h2 style="margin:0">Explore this model</h2><span class="spacer"></span></div><div class="tabs" style="margin-top:8px">${[["whatif", "What-if: window & channels"], ["battery", "Battery life & power"], ["drift", "Drift"], ["compare", "vs TFLite Micro"], ["devboard", "Virtual dev board"]].map(([k, t]) => `<button class="tab ${k === tab ? "on" : ""}" data-it="${k}">${t}</button>`).join("")}</div><div id="ibody"><span class="small muted">Loading…</span></div>`;
      $$("[data-it]", box).forEach((b) => (b.onclick = () => { tab = b.dataset.it; draw(); }));
      ({ whatif: drawWhatIf, battery: drawBattery, drift: drawDrift, compare: drawCompare, devboard: drawDevboardTab })[tab]().catch((e) => { if (!(e instanceof Stale) && $("#ibody")) $("#ibody").innerHTML = `<div class="callout">${esc(e.message)}</div>`; });
    };
    draw();
  }
  async function drawWhatIf() {
    const w = await api(`/api/jobs/${jid}/whatif`);
    const b = w.base;
    let wi = w.windows.indexOf(b.window); if (wi < 0) wi = 0;
    let ch = b.channels;
    const names = w.channel_names || [];
    $("#ibody").innerHTML = `<p class="small muted">Move the sliders to see what a longer or shorter window, or fewer or more sensor channels, would cost on <b>${esc(b.target)}</b>. ${esc(w.note)}</p>
      <div class="grid g2"><label class="f">Window length <b id="wiv"></b><input type="range" id="wi" min="0" max="${w.windows.length - 1}" step="1" value="${wi}"></label>
        <label class="f">Channels <b id="chv"></b><input type="range" id="chs2" min="1" max="${w.max_channels}" step="1" value="${ch}"></label></div>
      <div class="kpis" id="wik" style="margin-top:10px"></div><div id="wiwarn"></div>
      <div class="grid g2" style="margin-top:12px"><div><h3 class="small">RAM by window length (${"<span id='chl'></span>"} channels)</h3><div id="wichart"></div></div><div><h3 class="small">Latency by window length</h3><div id="wichart2"></div></div></div>
      <div class="row" style="margin-top:10px"><button class="btn" id="wicheck">Check exactly with SAI</button><span class="small muted" id="wicres"></span><span class="spacer"></span><span class="small muted">Accuracy changes too: retrain to measure it.</span></div>`;
    const get = (win, c) => w.grid.find((g) => g.window === win && g.channels === c);
    const delta = (a, b0, f = bytes) => a == null || b0 == null ? "" : a === b0 ? `<span class="muted">same</span>` : `<span style="color:${a > b0 ? "var(--bad)" : "var(--ok)"}">${a > b0 ? "+" : "−"}${f(Math.abs(a - b0))}</span>`;
    const upd = () => {
      wi = +$("#wi").value; ch = +$("#chs2").value;
      const win = w.windows[wi], g = get(win, ch);
      $("#wiv").textContent = `${fmt(win)} samples${w.sample_rate_hz ? ` (${fmt(win / w.sample_rate_hz * 1000, 1)} ms of signal)` : ""}${win === b.window ? " · current" : ""}`;
      $("#chv").textContent = `${ch}${ch <= w.raw_channels ? (names.length ? ` (${names.slice(0, ch).join(", ")})` : "") : ` (${ch - w.raw_channels} more than the data has)`}${ch === b.channels ? " · current" : ""}`;
      $("#chl").textContent = ch;
      if (!g || !g.ok) { $("#wik").innerHTML = ""; $("#wiwarn").innerHTML = `<div class="callout bad">${esc((g && g.reason) || "not buildable")}</div>`; return; }
      $("#wik").innerHTML = `<div class="kpi"><div class="l">RAM total</div><div class="v">${bytes(g.ram)}</div><div class="small">${delta(g.ram, b.ram)} ${ev("estimate")}</div></div>
        <div class="kpi"><div class="l">Flash</div><div class="v">${bytes(g.flash)}</div><div class="small">${delta(g.flash, b.flash)}</div></div>
        <div class="kpi"><div class="l">Latency</div><div class="v">${ms(g.latency_ms)}</div><div class="small">${delta(g.latency_ms, b.latency_ms, (x) => ms(x))}</div></div>
        <div class="kpi"><div class="l">MACs</div><div class="v">${fmt(g.macs / 1000, 1)}k</div><div class="small">model input ${esc(g.model_input.join("×"))}</div></div>`;
      $("#wiwarn").innerHTML = g.fits ? `<div class="callout ok" style="margin-top:8px">Fits the ${esc(b.target)} budget.</div>` : `<div class="callout bad" style="margin-top:8px">Does not fit: ${esc(g.violations.join("; "))}</div>`;
      const row = w.windows.map((x) => get(x, ch)).filter((x) => x && x.ok);
      $("#wichart").innerHTML = scatter(row.map((x) => ({ x: x.window, y: x.ram, color: x.fits ? "var(--c1)" : "var(--bad)", hollow: !x.fits, star: x.window === win, title: `${x.window}: ${bytes(x.ram)}` })), { w: 520, h: 200, xLabel: "window (samples)", yLabel: "RAM (B)" });
      $("#wichart2").innerHTML = scatter(row.map((x) => ({ x: x.window, y: x.latency_ms, color: "var(--c2)", star: x.window === win, title: `${x.window}: ${ms(x.latency_ms)}` })), { w: 520, h: 200, xLabel: "window (samples)", yLabel: "latency (ms)" });
    };
    $("#wi").oninput = upd; $("#chs2").oninput = upd; upd();
    $("#wicheck").onclick = () => busy($("#wicheck"), "Starting…", async () => {
      const j = await api(`/api/jobs/${jid}/whatif/check`, { json: { window: w.windows[+$("#wi").value], channels: +$("#chs2").value } });
      $("#wicres").innerHTML = `SAI fit check started: <a href="#/job/${esc(j.id)}">${esc(j.id)}</a>`;
    });
  }
  async function drawBattery() {
    let p = { battery: "2xaa" };
    const run = async () => {
      const r = await api(`/api/jobs/${jid}/battery`, { json: p });
      const bk = r.breakdown_mw, tot = Object.values(bk).reduce((a, x) => a + x, 0) || 1;
      $("#ibody").innerHTML = `<div class="grid g2"><div class="stack">
          <div class="grid g2"><label class="f">Battery<select id="bb">${Object.entries(r.batteries).map(([k, v]) => `<option value="${k}" ${k === (p.battery || "2xaa") ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select></label>
            <label class="f">Capacity (mAh) / voltage<div class="row"><input id="bmah" type="number" value="${r.battery.mah || ""}" style="width:50%"><input id="bv" type="number" step="0.1" value="${r.battery.v || ""}" style="width:50%"></div></label></div>
          <div class="grid g2"><label class="f">Inferences per second<input id="brate" type="number" step="any" value="${r.inferences_per_s}"></label><label class="f">Sensor power (mW)<input id="bsen" type="number" step="any" value="${p.sensor_mw || 0}"></label></div>
          <div class="grid g2"><label class="f">Radio energy per report (µJ)<input id="brad" type="number" step="any" value="${p.radio_uj || 0}"></label><label class="f">Reports per hour<input id="bev" type="number" step="any" value="${p.events_per_hour || 0}"></label></div>
          <div class="grid g2"><label class="f">MCU active power (mW)<input id="bact" type="number" step="any" placeholder="${r.active_mw}" value="${p.active_mw || ""}"></label><label class="f">MCU sleep power (µW)<input id="bsl" type="number" step="any" placeholder="${r.sleep_uw}" value="${p.sleep_uw || ""}"></label></div>
          <p class="small muted">Default MCU power: ${esc(r.profile_basis)} ${ev(r.energy_evidence === "measured" ? "measured" : "assumed")}. Enter your board's numbers to override.</p></div>
        <div><div class="stat"><span class="l">Battery life</span><span class="big">${r.life_days == null ? "–" : r.life_days > 730 ? `${fmt(r.life_days / 365, 1)} years` : `${fmt(r.life_days, 1)} days`}</span></div>
          <p class="small">Average ${fmt(r.avg_mw * 1000, 1)} µW · ${fmt(r.energy_uj, 2)} µJ per inference ${ev(r.energy_evidence === "measured" ? "measured" : r.energy_evidence === "custom" ? "assumed" : "assumed")} · latency ${ms(r.latency_ms)} ${ev(r.latency_evidence)} · duty cycle ${(r.duty_cycle * 100).toFixed(2)}% · up to ${fmt(r.max_inferences_per_s)} inferences/s</p>
          ${r.too_fast ? `<div class="callout bad">The model cannot run that often: one inference takes ${ms(r.latency_ms)}.</div>` : ""}
          <h3 class="small" style="margin-top:10px">Where the energy goes</h3>${bars(Object.fromEntries(Object.entries(bk).map(([k, v]) => [k, +(v / tot * 100).toFixed(1)])))}<p class="small muted">% of average power</p>
          ${r.measured ? `<div class="callout ok" style="margin-top:8px">Power measured on a physical board: ${r.measured.energy_uj ? `${fmt(r.measured.energy_uj, 2)} µJ per inference` : ""}${r.measured.active_mw ? ` · ${fmt(r.measured.active_mw, 1)} mW active` : ""} (${r.measured.runs} run(s)).</div>` : `<div class="callout info" style="margin-top:8px">No power measurement yet. A board-farm board with a power monitor (INA226, Joulescope, PPK2) reports <code>energy_uj</code> per inference with each run, and its number replaces the assumed one here.</div>`}</div></div>`;
      $("#bb").onchange = () => { p = { ...p, battery: $("#bb").value, mah: null, volts: null }; run(); };
      const num = (id) => ($(id).value === "" ? null : +$(id).value);
      $$("#ibody input").forEach((i) => (i.onchange = () => { p = { ...p, mah: num("#bmah"), volts: num("#bv"), inferences_per_s: num("#brate"), sensor_mw: num("#bsen"), radio_uj: num("#brad"), events_per_hour: num("#bev"), active_mw: num("#bact"), sleep_uw: num("#bsl") }; run(); }));
    };
    await run();
  }
  async function drawCompare() {
    let r = await api(`/api/jobs/${jid}/compare`);
    const evb = (e) => ev(e === "replayed" ? "estimate" : e === "assumed" ? "assumed" : e);
    const cell = (sai, other, f, evd) => `<td class="n">${f(sai)}</td><td class="n">${f(other)} ${evd || ""}</td><td class="n">${sai && other ? `<b>${(other / sai).toFixed(1)}×</b>` : "–"}</td>`;
    const draw = () => {
      $("#ibody").innerHTML = `<p class="small muted">The same INT8 model under SAI and under TensorFlow Lite for Microcontrollers. ${esc(r.method)}</p>
        ${r.rows.map((x) => { const t = x.measured || x.tflm; const evs = x.measured ? { arena: "measured", flash: "measured", latency: "measured" } : x.tflm.evidence; return `<div class="card" style="margin-top:10px;padding:12px"><div class="row"><b>${esc(x.board)}</b><span class="badge" style="color:${TARGET_COLOR[x.target]}">${esc(x.target)}</span><span class="spacer"></span><span class="small muted">${esc(x.measured ? x.measured.runtime + " (your measured build)" : x.tflm.runtime)}</span></div>
          ${x.tflm.note ? `<p class="small muted">${esc(x.tflm.note)}</p>` : ""}
          ${x.sai ? `<div class="tbl-wrap"><table class="t"><thead><tr><th></th><th class="n">SAI</th><th class="n">${esc(x.measured ? "TFLM (measured)" : x.tflm.runtime)}</th><th class="n">TFLM / SAI</th></tr></thead><tbody>
            <tr><td>RAM (arena / workspace + stack)</td>${cell(x.sai.ram, t.ram, bytes, x.measured ? ev("measured") : evb(evs.arena))}</tr>
            <tr><td>Flash (code + weights)</td>${cell(x.sai.flash, t.flash, bytes, x.measured ? ev("measured") : evb(evs.flash))}</tr>
            <tr><td>Latency</td>${cell(x.sai.latency_ms, t.latency_ms, (v) => ms(v), x.measured ? ev("measured") : evb(evs.latency))}</tr></tbody></table></div>
            <p class="small muted">SAI: ${ev(x.sai.evidence.ram)} RAM, ${ev(x.sai.evidence.flash)} flash, ${ev(x.sai.evidence.latency)} latency · plan <code>${esc(x.sai.schedule || "")}</code>${x.measured ? "" : ` · TFLM arena = activations ${bytes(x.tflm.arena_parts.activations)} + scratch ${bytes(x.tflm.arena_parts.scratch)} + persistent ${bytes(x.tflm.arena_parts.persistent)}; flash = weights ${bytes(x.tflm.flash_parts.weights_and_biases)} + metadata ${bytes(x.tflm.flash_parts.flatbuffer_metadata)} + code ${bytes(x.tflm.flash_parts.code)}`}</p>
            ${!x.tflm_fits ? `<div class="callout bad">TFLM would not fit this board's ${bytes(x.ram_total)} RAM / ${bytes(x.flash_total)} flash.</div>` : ""}`
          : `<p class="small muted">This project was not planned for ${esc(x.target)}. TFLM model: RAM ${bytes(x.tflm.ram)}, flash ${bytes(x.tflm.flash)}, ${ms(x.tflm.latency_ms)}.</p>`}
          <details style="margin-top:6px"><summary class="small">Enter your own TFLM build's numbers</summary><div class="grid g3" style="margin-top:6px"><label class="f">RAM (B)<input data-m="ram" data-t="${x.target}" type="number"></label><label class="f">Flash (B)<input data-m="flash" data-t="${x.target}" type="number"></label><label class="f">Latency (ms)<input data-m="latency_ms" data-t="${x.target}" type="number" step="any"></label></div><div class="row" style="margin-top:6px"><input data-m="runtime" data-t="${x.target}" placeholder="TFLM version / flags"><button class="btn sm" data-save="${x.target}">Save measured</button></div></details></div>`; }).join("")}`;
      $$("[data-save]").forEach((b) => (b.onclick = () => busy(b, "Saving…", async () => {
        const t = b.dataset.save, body = { target: t };
        $$(`[data-t="${t}"]`).forEach((i) => { if (i.value !== "") body[i.dataset.m] = i.dataset.m === "runtime" ? i.value : +i.value; });
        r = await api(`/api/jobs/${jid}/compare/measured`, { method: "PUT", json: body }); draw();
      })));
    };
    draw();
  }
  async function drawDevboardTab() {
    const av = await api("/api/devboard/available");
    $("#ibody").innerHTML = `<p class="small">Run this project's firmware on an emulated board with your own inputs (dataset windows, a stretch of a recording, or pasted values) and see what the board prints, its outputs and predictions, checked bit-exactly against the host program.</p>
      <div class="row">${Object.entries(av.targets).map(([t, ok]) => `<span class="badge ${ok ? "b-ok" : "b-warn"}">${esc(t)}: ${ok ? "available" : "emulator not installed"}</span>`).join(" ")}<span class="spacer"></span><a class="btn primary" href="#/devboard/${esc(jid)}">Open the virtual dev board</a></div>`;
  }
  async function drawDrift() {
    const [hist, recs, dsets] = await Promise.all([api(`/api/jobs/${jid}/drift`), api("/api/recordings"), api(`/api/datasets?template_id=${encodeURIComponent(job.params.template_id)}`)]);
    const okRecs = recs.filter((r) => r.status === "ready");
    const badge = (s) => `<span class="badge ${s === "stable" ? "b-ok" : s === "watch" ? "b-warn" : "b-bad"}">${esc(s)}</span>`;
    const show = (d) => `<div class="callout ${d.status === "stable" ? "ok" : d.status === "watch" ? "" : "bad"}" style="margin-top:10px">${badge(d.status)} <b>${esc(d.summary)}</b> ${d.windows} windows from ${esc(d.source)} · worst PSI ${d.psi}${(d.top || []).length ? `<div class="small">${d.top.map(esc).join(" · ")}</div>` : ""}</div>
      <div class="grid g2" style="margin-top:10px"><div class="tbl-wrap"><table class="t"><thead><tr><th>Channel</th><th>Feature</th><th class="n">PSI</th><th class="n">Shift (sd)</th><th></th></tr></thead><tbody>${d.inputs.map((r) => `<tr><td>${esc(r.channel)}</td><td>${esc({ mean: "window mean", rms: "window RMS", p2p: "peak-to-peak" }[r.feature])}</td><td class="n">${r.psi}</td><td class="n">${r.shift_sd > 0 ? "+" : ""}${r.shift_sd}</td><td>${badge(r.status)}</td></tr>`).join("")}</tbody></table></div>
      <div>${d.predictions ? `<h3 class="small">Predicted classes (training vs production)</h3><div class="tbl-wrap"><table class="t"><thead><tr><th>Class</th><th class="n">Training</th><th class="n">Production</th></tr></thead><tbody>${d.predictions.labels.map((l, i) => `<tr><td>${esc(l)}</td><td class="n">${pct(d.predictions.reference[i])}</td><td class="n">${pct(d.predictions.current[i])}</td></tr>`).join("")}</tbody></table></div><p class="small">Prediction-mix PSI ${d.predictions.psi} · mean confidence ${pct(d.predictions.confidence)} (training ${pct(d.predictions.confidence_ref)})${d.confidence_drop ? ` <span class="badge b-warn">confidence dropped</span>` : ""}</p>` : `<p class="small muted">Prediction drift needs a Core (CNN) project.</p>`}</div></div>`;
    $("#ibody").innerHTML = `<p class="small muted">Compare data from the field with what this model was trained on: input distributions (population stability index per channel) and the model's own predictions. PSI below 0.1 is stable, 0.1–0.25 worth watching, above 0.25 drift.</p>
      <div class="row" style="flex-wrap:wrap"><select id="dsrc"><optgroup label="Recordings">${okRecs.map((r) => `<option value="rec:${esc(r.id)}">${esc(r.name)} (${fmt(r.samples / r.sample_rate, 1)} s)</option>`).join("")}</optgroup><optgroup label="Datasets">${dsets.filter((d) => d.id !== job.params.dataset_id).map((d) => `<option value="ds:${esc(d.id)}">${esc(d.name)}</option>`).join("")}</optgroup></select><button class="btn primary" id="dgo2">Check drift</button><a class="small" href="#/recordings">Capture or import field data</a></div>
      <div id="dres2"></div>
      ${hist.length ? `<h3 class="small" style="margin-top:14px">History</h3><div class="tbl-wrap"><table class="t"><tbody>${hist.slice().reverse().map((h, i) => `<tr><td>${badge(h.status)}</td><td>${esc(h.source)}</td><td class="n">PSI ${h.psi}</td><td class="muted">${ago(h.created)}</td><td><button class="btn sm" data-h="${i}">Show</button></td></tr>`).join("")}</tbody></table></div>` : ""}`;
    const rev = hist.slice().reverse();
    $$("[data-h]").forEach((b) => (b.onclick = () => { $("#dres2").innerHTML = show(rev[+b.dataset.h]); }));
    $("#dgo2").onclick = () => busy($("#dgo2"), "Comparing…", async () => {
      const v = $("#dsrc").value; if (!v) return toast("Capture or import field data first");
      const [k, id] = v.split(":");
      const d = await api(`/api/jobs/${jid}/drift`, { json: k === "rec" ? { recording_id: id } : { dataset_id: id } });
      $("#dres2").innerHTML = show(d);
    });
  }
  let hilTimer = null;
  onLeave(() => clearTimeout(hilTimer));
  function pollHil() {
    clearTimeout(hilTimer);
    hilTimer = setTimeout(async () => {
      try { const pending = await drawHil(); if (pending) pollHil(); } catch (e) { if (!(e instanceof Stale)) toast(e.message); }
    }, 1500);
  }
  async function drawHil() {
    const keep = $("#hb") ? $("#hb").value : "";
    const [b, runs] = await Promise.all([api("/api/boards"), api(`/api/boards/runs?job_id=${jid}`)]);
    const sel = job.result.selected;
    const avail = b.boards.filter((x) => x.online);
    $("#hil").innerHTML = `<div class="row"><h2 style="margin:0">Run on a board</h2><span class="spacer"></span>
      <select id="hb"><option value="">any ${esc(sel.target)} board</option>${avail.map((x) => `<option value="${esc(x.id)}" data-t="${esc(x.target)}">${esc(x.name)} – ${esc(x.status)}</option>`).join("")}</select>
      <button class="btn primary" id="hrun">Flash & measure</button></div>
      <p class="small muted" style="margin:8px 0">The board builds the HIL harness firmware around the SAI package for its own target, runs the test vectors, and reports outputs, cycles and painted stack. Outputs are compared bit-exactly with the host build; real boards turn the plan's latency into <b>measured</b>. Emulated boards (QEMU) prove the program on the target ISA; their cycles are not latency.</p>
      ${runs.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Run</th><th>Board</th><th>Status</th><th class="n">Estimated</th><th class="n">Measured</th><th class="n">Error</th><th>Outputs</th><th>Stack (painted)</th></tr></thead><tbody>
      ${runs.map((x) => { const res = x.result || {}, est = x.request.estimated_ms, got = res.latency_ms; return `<tr><td class="mono small">${esc(x.id)}</td><td>${esc(x.board_id || "queued")} ${res.simulated ? `<span class="badge b-warn">simulated</span>` : res.emulated ? `<span class="badge b-info" title="real firmware on QEMU: outputs are real, cycles are QEMU's instruction clock">emulated</span>` : ""}</td><td>${statusBadge(x.status)}</td><td class="n">${ms(est)}</td><td class="n">${res.emulated ? `<span class="small muted">${fmt(res.cycles)} emu. cycles</span>` : ms(got)}</td><td class="n">${est && got && !res.emulated ? `${((got - est) / est * 100).toFixed(1)}%` : "–"}</td><td>${res.outputs_match === undefined ? "–" : res.outputs_match ? `<span class="badge b-ok">✓ ${fmt(res.n_vectors)} match</span>` : `<span class="badge b-bad">mismatch</span>`}</td><td>${bytes(res.stack_painted)}</td></tr>`; }).join("")}</tbody></table></div>` : ""}`;
    const openRun = runs.find((x) => x.status === "queued" || x.status === "running");
    $("#hrun").disabled = !!openRun;
    $("#hrun").title = openRun ? `Run ${openRun.id} is ${openRun.status}; the next one can start when it finishes` : "";
    $("#hrun").onclick = () => busy($("#hrun"), "Queuing…", async () => {
      const o = $("#hb").selectedOptions[0];
      await api("/api/boards/runs", { json: { job_id: jid, board_id: o.value || null, target: o.dataset.t || null } });
      toast("Queued on the board farm");
      pollHil();
    });
    if (keep && [...$("#hb").options].some((o) => o.value === keep)) $("#hb").value = keep;
    return runs.some((x) => x.status === "queued" || x.status === "running");
  }
  function drawPlansOnly(r) {
    if (job.kind === "regression" || job.kind === "basetrain") {
      $("#res").innerHTML = job.kind === "regression" ? `<div class="card" style="margin-top:16px"><h2>Regression ${esc(r.run_id)}</h2>
        ${r.alerts.length ? r.alerts.map((a) => `<div class="callout bad">${esc(a)}</div>`).join("") : `<div class="callout ok">No regressions against the previous run.</div>`}
        <div class="tbl-wrap" style="margin-top:10px"><table class="t"><thead><tr><th>Model type</th><th>Board</th><th class="n">RAM</th><th class="n">prev</th><th class="n">Flash</th><th class="n">prev</th><th>HIL</th></tr></thead><tbody>
        ${r.rows.map((x) => `<tr><td>${esc(x.template)}</td><td>${esc(x.target)}</td><td class="n">${bytes(x.ram)}</td><td class="n muted">${bytes(x.prev && x.prev.ram)}</td><td class="n">${bytes(x.flash)}</td><td class="n muted">${bytes(x.prev && x.prev.flash)}</td><td>${x.hil_match === null ? "–" : x.hil_match ? "bit-exact" : "MISMATCH"}</td></tr>`).join("")}</tbody></table></div></div>`
        : `<div class="card" style="margin-top:16px"><h2>Base weights for ${esc(r.template)}</h2><p class="small">Sizes: ${Object.entries(r.sizes).map(([k, v]) => `${esc(k)} (val loss ${v.best_val_loss})`).join(", ")} · licence ${esc(r.licence)}. Customer projects can now fine-tune from them.</p></div>`;
      return;
    }
    if (job.kind === "rebuild") {
      $("#res").innerHTML = `<div class="card" style="margin-top:16px"><h2>Rebuild of <a href="#/job/${esc(job.params.job_id)}">${esc(job.params.job_id)}</a></h2>
        <div class="callout ${r.identical ? "ok" : "bad"}">${r.identical ? "Byte-identical: the package rebuilds exactly from the recorded inputs." : "The rebuild differs from the recorded build."}</div>
        <p class="small" style="margin-top:10px">Build digest <code>${esc(r.build_digest)}</code> · recorded <code>${esc(r.recorded)}</code> · SAI ${esc(r.sai_version)} (recorded ${esc(r.recorded_sai)})</p>
        <div class="tbl-wrap"><table class="t"><tbody>${Object.entries(r.files).map(([f, ok]) => `<tr><td class="mono small">${esc(f)}</td><td>${ok ? `<span class="badge b-ok">same</span>` : `<span class="badge b-bad">differs</span>`}</td></tr>`).join("")}</tbody></table></div></div>`;
      return;
    }
    const s = r.summary;
    $("#res").innerHTML = `${job.kind === "analyze" ? `<div class="card" style="margin-top:16px"><h2>Model</h2><div class="kpis">
        <div class="kpi"><div class="l">Format</div><div class="v">${esc(r.inspect.format)}</div></div><div class="kpi"><div class="l">Nodes</div><div class="v">${fmt(r.inspect.nodes)}</div></div>
        <div class="kpi"><div class="l">Parameters</div><div class="v">${fmt(r.inspect.parameters)}</div></div><div class="kpi"><div class="l">MACs</div><div class="v">${fmt(r.macs)}</div></div></div>
        <p class="small" style="margin-top:8px">Operators: ${Object.entries(r.inspect.operators || {}).map(([k, n]) => `<span class="badge">${esc(k)} ×${n}</span>`).join(" ")}</p></div>` :
      `<div class="card" style="margin-top:16px"><h2>Architecture</h2><div class="kpis"><div class="kpi"><div class="l">Params</div><div class="v">${fmt(s.params)}</div></div><div class="kpi"><div class="l">MACs</div><div class="v">${fmt(s.macs)}</div></div><div class="kpi"><div class="l">Peak activation</div><div class="v">${bytes(s.peak_activation_bytes)}</div></div><div class="kpi"><div class="l">Spatial axes</div><div class="v">${s.spatial_axes}</div></div></div></div>`}
      <div class="card" style="margin-top:16px"><h2>Plans</h2>${planTable(r.plans.candidates)}${r.plans.advice ? `<div class="callout" style="margin-top:10px">${esc(r.plans.advice)}</div>` : ""}${spaceNote(r.plans)}</div>
      <div class="card" style="margin-top:16px"><h2>Explored plan space</h2>${spaceChart(r.plans)}</div>`;
  }
  poll();
});

function tierCard(r) {
  const tc = r.tier_compare || {};
  const rows = Object.entries(tc).filter(([, v]) => v.standard && v.professional && !v.standard.error);
  if (!rows.length) return "";
  const gain = (a, b) => a && b ? `${(((a - b) / a) * 100).toFixed(0)}%` : "–";
  return `<div class="card" style="margin-top:16px"><h2>SAI Standard vs Professional</h2>
    <p class="small muted">The same model, FAST objective, in SAI's two optimization levels. Professional adds SAI's ISA kernels and schedules (SMLAD/SXTB16, RV32 two-channel pack, static requant specialization, packed FC…).</p>
    <div class="tbl-wrap"><table class="t"><thead><tr><th>Board</th><th>Level</th><th>Plan</th><th class="n">Workspace</th><th class="n">Flash</th></tr></thead><tbody>
    ${rows.map(([t, v]) => `<tr><td rowspan="2">${esc(t)}</td><td>Standard</td><td class="mono small">${esc(v.standard.schedule)}</td><td class="n">${bytes(v.standard.workspace)}</td><td class="n">${bytes(v.standard.flash)}</td></tr>
      <tr class="rec"><td><b>Professional</b></td><td class="mono small">${esc(v.professional.schedule)}</td><td class="n">${bytes(v.professional.workspace)}</td><td class="n">${bytes(v.professional.flash)}</td></tr>`).join("")}
    </tbody></table></div><p class="small muted">Professional typically trades some flash for speed (specialized kernels); its latency gain is measured per model on the board farm. Packages are built at Professional, the level SAI supports for MCU builds today.</p></div>`;
}
function provCard(r, jid) {
  const p = r.provenance; if (!p) return "";
  const rows = [["Platform", p.platform_version], ["SAI", p.sai_version], ["Toolchain", p.toolchain], ["Toolchain digest", p.toolchain_digest], ["Build digest", p.build_digest], ["Plan digest", p.plan_digest], ["Dataset SHA-256", p.dataset_sha256], ["Python / torch", `${p.python} / ${p.torch}`], ["Seeds", JSON.stringify(p.seeds)]];
  return `<div class="card" style="margin-top:16px"><div class="row"><h2 style="margin:0">Provenance</h2><span class="spacer"></span>${r.backend === "xmodels" ? `<span class="small muted">built by ${esc(p.compiler || "SAI front end")}</span>` : `<button class="btn sm" id="rebuild">Rebuild & compare</button>`}</div>
    <div class="tbl-wrap" style="margin-top:8px"><table class="t"><tbody>${rows.map(([k, v]) => `<tr><td class="muted" style="width:170px">${k}</td><td class="mono small" style="word-break:break-all">${esc(v ?? "–")}</td></tr>`).join("")}</tbody></table></div></div>`;
}

function planTable(c) {
  return `<div class="tbl-wrap"><table class="t"><thead><tr><th></th><th>Board</th><th>SAI objective</th><th>Plan</th><th class="n">Workspace</th><th class="n">Stack</th><th class="n">RAM</th><th class="n">Flash</th><th class="n">Latency</th><th>Fits?</th></tr></thead><tbody>
  ${c.map((x) => `<tr class="${x.recommended ? "rec" : ""} ${x.fits ? "" : "nofit"}"><td>${x.recommended ? "★" : ""}</td><td class="nw"><span style="color:${TARGET_COLOR[x.target]}">●</span> ${esc(x.target)}</td><td>${x.chosen_by.map((o) => o === "explored" ? `<span class="badge b-info" title="${esc(x.plan_name || "")}: an explored plan SAI built on request (sai compile --plan)">explored</span>` : `<span class="badge">${esc(o)}</span>`).join(" ")}</td><td class="mono small">${esc(x.schedule)}</td>
    <td class="n">${bytes(x.workspace)} ${ev(x.workspace_evidence)}</td><td class="n">${bytes(x.stack)} ${ev(x.stack_evidence)}</td><td class="n"><b>${bytes(x.ram)}</b></td><td class="n">${bytes(x.flash)} ${ev(x.flash_evidence)}</td>
    <td class="n" title="${esc(x.latency.basis)}">${ms(x.latency.ms)} ${ev(x.latency.evidence)}</td><td>${x.fits ? `<span class="badge b-ok">fits</span>` : `<span class="badge b-bad" title="${esc(x.violations.join("; "))}">no</span> <span class="small muted">${esc(x.violations[0] || "")}</span>`}</td></tr>`).join("")}
  </tbody></table></div>`;
}
function spaceNote(p) {
  return `<p class="small muted" style="margin-top:10px">${Object.entries(p.spaces || {}).map(([t, s]) => s.unavailable ? `${esc(t)}: explore unavailable (${esc(s.unavailable)})` : `${esc(t)}: SAI evaluated <b>${fmt(s.evaluated)}</b> candidate plans${s.complete ? " (complete search)" : ""}`).join(" · ")}. RAM = workspace + stack + static RAM.</p>`;
}
function spaceChart(p) {
  const pts = [];
  for (const [t, s] of Object.entries(p.spaces || {})) for (const x of s.plans || []) if (x.workspace && x.rank_score) pts.push({ x: x.workspace, y: x.rank_score, color: TARGET_COLOR[t], r: 4, title: `${t} ${x.name}\nworkspace ${x.workspace} B\nrank ${fmt(x.rank_score)} (${x.basis})` });
  return pts.length ? `${scatter(pts, { xLabel: "workspace (B)", yLabel: "SAI rank score (lower = faster)", logX: true })}<p class="small muted">Top-ranked candidates of SAI's search. Rank scores are SAI's uncalibrated ranking, not latencies.</p>` : `<div class="empty small">No explored candidates</div>`;
}
function planScatter(variants, sel) {
  const pts = [];
  const sizeR = { tiny: 4, small: 6, base: 8 };
  for (const v of variants) for (const c of v.plans.candidates) if (c.latency.ms) pts.push({ x: c.ram, y: c.latency.ms, color: TARGET_COLOR[c.target], hollow: !c.fits, r: sizeR[v.size] || 5, star: v.size === sel.size && c.id === sel.id, title: `${v.size} · ${c.target} · ${c.schedule}\nRAM ${c.ram} B · flash ${c.flash} B\n${c.latency.ms} ms (${c.latency.evidence})${c.fits ? "" : "\n" + c.violations.join("; ")}` });
  return scatter(pts, { xLabel: "RAM (bytes, log)", yLabel: "latency (ms, log)", logX: true, logY: true }) + `<p class="small muted">Dot size = model size (tiny/small/base).</p>`;
}
function confusion(cm, labels) {
  if (!cm) return "";
  const max = Math.max(...cm.flat(), 1);
  return `<div class="tbl-wrap"><table class="cm"><tr><td></td>${labels.map((l) => `<td class="small muted" title="predicted">${esc(l)}</td>`).join("")}</tr>${cm.map((row, i) => `<tr><td class="small muted" style="text-align:right">${esc(labels[i])}</td>${row.map((v, j) => `<td style="background:color-mix(in srgb, ${i === j ? "var(--ok)" : "var(--bad)"} ${Math.round(v / max * 70)}%, transparent)">${v}</td>`).join("")}</tr>`).join("")}</table></div><p class="small muted">Rows: true label · columns: predicted.</p>`;
}

/* ------------------------------------------------------------------------------------------------ studio */
const BLANK = () => ({ name: "", domain: "", icon: "🧪", task: "classification", summary: "", labels: "class_a, class_b",
  input: { kind: "timeseries", channels: 3, height: 1, width: 128, sample_rate_hz: 100 }, datasets_hint: "", synth: "",
  variants: { tiny: [{ op: "conv", out: 8, k: 5, stride: 2, relu: true }, { op: "dwconv", k: 3, relu: true }, { op: "pwconv", out: 16, relu: true }, { op: "gap" }], small: [], base: [] } });

function noAccess(el, what) {
  el.innerHTML = `<div class="page-h"><div><h1>${esc(what)}</h1></div></div><div class="callout bad">Your role (<b>${esc(ME.role)}</b>) cannot open ${esc(what)}. Research pages are for researchers, managers and admins; ask your admin if you need access.</div>`;
}
route(/^\/studio$/, async (el) => {
  if (!ME.perms.research) return noAccess(el, "Model Studio");
  const [sys, mine] = await Promise.all([system(), api("/api/catalog?drafts=true")]);
  const builtins = mine.filter((t) => t.source === "builtin"), research = mine.filter((t) => t.source === "studio");
  let m = BLANK(), size = "tiny", vtimer = null;
  const OPS = Object.keys(sys.ops);
  const PARAMS = LAYER_PARAMS, DEF = LAYER_DEF;
  el.innerHTML = `<div class="page-h"><div><h1>Model Studio</h1><p class="muted">Design a model type for the pool. Every layer maps to a SAI operator; a fit check compiles it for every board in seconds, before any training.</p></div>
    <div class="row"><select id="load"><option value="">Load…</option><optgroup label="Research templates">${research.map((t) => `<option value="${esc(t.id)}">${esc(t.name)} (${esc(t.status)})</option>`).join("")}</optgroup><optgroup label="Copy a built-in">${builtins.map((t) => `<option value="${esc(t.id)}">${esc(t.name)}</option>`).join("")}</optgroup></select><button class="btn" id="newb">New</button></div></div>
    <div class="grid studio-grid"><div class="card" id="meta"></div><div class="stack"><div class="card" id="arch"></div><div class="card" id="fit"><h2>SAI fit check</h2><p class="small muted">Builds the selected size with random weights, exports INT8 and asks SAI for every plan on every board. Resource numbers do not depend on trained weights.</p><button class="btn primary" id="runfit">Run fit check</button><div id="fitres"></div></div></div></div>`;
  const drawMeta = () => {
    const i = m.input;
    $("#meta").innerHTML = `<h2>Model type</h2><div class="stack">
      <label class="f">Name<input id="m_name" value="${esc(m.name)}" placeholder="e.g. Compressor valve fault"></label>
      <div class="grid g2"><label class="f">Domain<input id="m_domain" value="${esc(m.domain)}" placeholder="Predictive maintenance"></label><label class="f">Icon<input id="m_icon" value="${esc(m.icon)}"></label></div>
      <label class="f">Task<select id="m_task">${["classification", "regression", "anomaly"].map((t) => `<option ${m.task === t ? "selected" : ""}>${t}</option>`).join("")}</select></label>
      <label class="f">Summary<textarea id="m_summary" rows="2">${esc(m.summary)}</textarea></label>
      <label class="f">${m.task === "regression" ? "Output name" : "Labels (comma separated)"}<input id="m_labels" value="${esc(Array.isArray(m.labels) ? m.labels.join(", ") : m.labels)}"></label>
      <h3 style="margin:6px 0 0">Input contract</h3>
      <div class="grid g2"><label class="f">Kind<select id="i_kind">${["timeseries", "image", "spectrogram", "features"].map((k) => `<option ${i.kind === k ? "selected" : ""}>${k}</option>`).join("")}</select></label>
      <label class="f">Sample rate (Hz)<input id="i_sr" type="number" value="${i.sample_rate_hz ?? ""}"></label>
      <label class="f">Channels<input id="i_c" type="number" value="${i.channels}"></label><label class="f">Height (1 = 1-D)<input id="i_h" type="number" value="${i.height}"></label>
      <label class="f">Width / window<input id="i_w" type="number" value="${i.width}"></label>
      <label class="f">Demo data<select id="m_synth"><option value="">none</option>${[...new Set(builtins.map((t) => t.synth))].map((s) => `<option ${m.synth === s ? "selected" : ""}>${s}</option>`).join("")}</select></label></div>
      <label class="f">Data guidance for customers<textarea id="m_hint" rows="2">${esc(m.datasets_hint)}</textarea></label>
      <div class="row"><button class="btn" id="save">Save draft</button><button class="btn primary" id="pub">Publish to pool</button>${m.id && m.status === "published" ? `<button class="btn ghost" id="dep">Deprecate</button>` : ""}${m.id ? `<button class="btn ghost" id="del">Delete</button>` : ""}</div>
      ${m.id ? `<div class="small muted">id <code>${esc(m.id)}</code> · ${esc(m.status || "draft")} · version ${esc(m.version || 1)}</div>
        ${(m.history || []).length ? `<div class="small muted">History: ${m.history.slice(-5).map((h) => `v${h.version} ${esc(h.status)}`).join(" → ")}</div>` : ""}` : ""}</div>`;
    $$("#meta input, #meta select, #meta textarea").forEach((x) => (x.onchange = () => { readMeta(); if (x.id === "m_task") drawMeta(); validate(); }));
    $("#save").onclick = () => save("draft"); $("#pub").onclick = () => save("published");
    if ($("#dep")) $("#dep").onclick = () => save("deprecated");
    if ($("#del")) $("#del").onclick = async () => { if (!confirm(`Delete the model type "${m.name}"?`)) return; await api(`/api/catalog/${m.id}`, { method: "DELETE" }); toast("Deleted"); m = BLANK(); drawMeta(); drawArch(); };
  };
  const readMeta = () => {
    m.name = $("#m_name").value; m.domain = $("#m_domain").value; m.icon = $("#m_icon").value; m.task = $("#m_task").value; m.summary = $("#m_summary").value;
    m.labels = $("#m_labels").value; m.datasets_hint = $("#m_hint").value; m.synth = $("#m_synth").value;
    m.input = { kind: $("#i_kind").value, channels: +$("#i_c").value, height: +$("#i_h").value, width: +$("#i_w").value, sample_rate_hz: +$("#i_sr").value || null };
  };
  const labelList = () => (Array.isArray(m.labels) ? m.labels : m.labels.split(",")).map((s) => s.trim()).filter(Boolean);
  const nOut = () => m.task === "classification" ? Math.max(labelList().length, 2) : m.task === "regression" ? 1 : m.input.channels * m.input.height * m.input.width;
  const drawArch = () => {
    const layers = m.variants[size] || (m.variants[size] = []);
    $("#arch").innerHTML = `<div class="row"><h2 style="margin:0">Architecture</h2><span class="spacer"></span><div class="tabs" style="margin:0;border:0">${["tiny", "small", "base"].map((s) => `<button class="tab ${s === size ? "on" : ""}" data-s="${s}">${s} <span class="small muted">${(m.variants[s] || []).length}</span></button>`).join("")}</div></div>
      <div class="tbl-wrap"><table class="t layers"><thead><tr><th>#</th><th>Op</th><th>Parameters</th><th>Output</th><th class="n">MACs</th><th></th></tr></thead><tbody id="lrows">
      ${layers.map((l, i) => `<tr data-i="${i}"><td class="muted">${i}</td><td><select data-f="op">${OPS.map((o) => `<option ${o === l.op ? "selected" : ""}>${o}</option>`).join("")}</select><div class="small muted">${esc(sys.ops[l.op] || "")}</div></td>
        <td><div class="row" style="gap:6px">${PARAMS[l.op].map((p) => paramInput(l, p, 70)).join("")}</div></td>
        <td class="mono small" data-shape="${i}"></td><td class="n mono small" data-macs="${i}"></td>
        <td><button class="btn ghost sm" data-a="up">↑</button><button class="btn ghost sm" data-a="dn">↓</button><button class="btn ghost sm" data-a="rm">✕</button></td></tr>`).join("")}
      <tr class="muted"><td></td><td>head</td><td class="small">fc(${nOut()}), automatic</td><td class="mono small" data-shape="head"></td><td class="n mono small" data-macs="head"></td><td></td></tr></tbody></table></div>
      <div class="row" style="margin-top:10px"><select id="addop">${OPS.map((o) => `<option>${o}</option>`).join("")}</select><button class="btn sm" id="add">+ Add layer</button><span class="spacer"></span><button class="btn sm ghost" id="copy">Copy ${size} to other empty sizes</button></div>
      <div id="vsum" style="margin-top:10px"></div>`;
    $$(".tabs .tab", $("#arch")).forEach((t) => (t.onclick = () => { size = t.dataset.s; drawArch(); }));
    $("#add").onclick = () => { const op = $("#addop").value; layers.push({ op, ...DEF[op] }); drawArch(); };
    $("#copy").onclick = () => { for (const s of ["tiny", "small", "base"]) if (!(m.variants[s] || []).length) m.variants[s] = JSON.parse(JSON.stringify(layers)); drawArch(); };
    $("#lrows").onchange = (e) => {
      const tr = e.target.closest("tr[data-i]"); if (!tr) return; const l = layers[+tr.dataset.i], f = e.target.dataset.f;
      if (f === "op") { layers[+tr.dataset.i] = { op: e.target.value, ...DEF[e.target.value] }; drawArch(); return; }
      l[f] = BOOL_PARAMS[f] ? e.target.checked : e.target.value === "" ? undefined : +e.target.value; validate();
    };
    $("#lrows").onclick = (e) => {
      const a = e.target.dataset.a; if (!a) return; const i = +e.target.closest("tr").dataset.i;
      if (a === "rm") layers.splice(i, 1); if (a === "up" && i > 0) [layers[i - 1], layers[i]] = [layers[i], layers[i - 1]]; if (a === "dn" && i < layers.length - 1) [layers[i + 1], layers[i]] = [layers[i], layers[i + 1]];
      drawArch();
    };
    validate();
  };
  const validate = () => {
    clearTimeout(vtimer);
    vtimer = setTimeout(async () => {
      readMeta();
      const r = await api("/api/studio/validate", { json: { input: m.input, layers: m.variants[size], outputs: nOut() } });
      if (!r.ok) { $("#vsum").innerHTML = `<div class="callout bad">${esc(r.error)}</div>`; return; }
      r.summary.layers.forEach((row, i) => { const key = row.head ? "head" : i; const s = $(`[data-shape="${key}"]`), c = $(`[data-macs="${key}"]`); if (s) s.textContent = row.shape.join("×"); if (c) c.textContent = fmt(row.macs); });
      const s = r.summary;
      $("#vsum").innerHTML = `<div class="kpis"><div class="kpi"><div class="l">Parameters</div><div class="v">${fmt(s.params)}</div></div><div class="kpi"><div class="l">MACs</div><div class="v">${fmt(s.macs)}</div></div><div class="kpi"><div class="l">Weights INT8</div><div class="v">${bytes(s.weight_bytes_int8)}</div></div><div class="kpi"><div class="l">Largest activation</div><div class="v">${bytes(s.peak_activation_bytes)}</div></div></div>
        <p class="small muted" style="margin-top:6px">${s.spatial_axes === 1 ? "1-D graph: SAI's streaming schedules apply." : "2-D graph: SAI explores row-stream and prefix plans."} Before planning, the naive arena ≈ the two largest live activations; SAI's plans usually go far below.</p>`;
    }, 250);
  };
  const body = (status) => ({ ...m, labels: labelList(), status, variants: Object.fromEntries(Object.entries(m.variants).filter(([, v]) => v && v.length)) });
  async function save(status) {
    readMeta();
    try { const t = await api("/api/catalog", { json: body(status) }); m = { ...t, labels: t.labels }; toast(status === "published" ? "Published to the model pool" : "Draft saved"); drawMeta(); drawArch(); }
    catch (e) { toast(e.message); }
  }
  $("#runfit").onclick = async () => {
    if ($("#runfit").disabled) return;
    $("#runfit").disabled = true;
    const doneFit = () => { if ($("#runfit")) $("#runfit").disabled = false; };
    readMeta();
    $("#fitres").innerHTML = `<div class="callout info" style="margin-top:10px">Compiling with SAI…</div>`;
    try {
      const tgts = sys.targets.filter((x) => x.status === "available" && x.id !== "host-cpu").map((x) => x.id);
      const job = await api("/api/studio/fitcheck", { json: { spec: { input: m.input, layers: m.variants[size], outputs: nOut() }, targets: tgts } });
      let ft = null; onLeave(() => clearTimeout(ft));
      const wait = async () => {
        let j;
        try { j = await api(`/api/jobs/${job.id}`); } catch (e) { if (!(e instanceof Stale)) $("#fitres").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(e.message)}</div>`; return; }
        if (["succeeded", "failed", "cancelled"].includes(j.status)) doneFit();
        if (j.status === "cancelled") { $("#fitres").innerHTML = `<div class="callout" style="margin-top:10px">Fit check cancelled.</div>`; return; }
        if (j.status === "succeeded") { $("#fitres").innerHTML = `<div style="margin-top:12px">${planTable(j.result.plans.candidates)}${j.result.plans.advice ? `<div class="callout" style="margin-top:8px">${esc(j.result.plans.advice)}</div>` : ""}${spaceNote(j.result.plans)}<a class="small" href="#/job/${j.id}">Open full report →</a></div>`; return; }
        if (j.status === "failed") { $("#fitres").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(j.error)}</div>`; return; }
        ft = setTimeout(wait, 800);
      };
      wait();
    } catch (e) { doneFit(); $("#fitres").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(e.message)}</div>`; }
  };
  $("#newb").onclick = () => { m = BLANK(); drawMeta(); drawArch(); };
  $("#load").onchange = async (e) => {
    if (!e.target.value) return; const t = await api(`/api/catalog/${e.target.value}`);
    m = JSON.parse(JSON.stringify(t)); if (t.source === "builtin") { delete m.id; m.name = t.name + " (copy)"; m.status = "draft"; }
    m.variants = { tiny: [], small: [], base: [], ...m.variants }; size = "tiny"; drawMeta(); drawArch(); e.target.value = "";
  };
  drawMeta(); drawArch();
});

/* ------------------------------------------------------------------------------------------------ explorer */
route(/^\/explorer$/, async (el) => {
  if (!ME.perms.research) return noAccess(el, "Plan Explorer");
  const jobs = (await api("/api/jobs?kind=analyze")).slice(0, 8);
  el.innerHTML = `<div class="page-h"><div><h1>Plan Explorer</h1><p class="muted">Bring your own INT8 model (.onnx QDQ or .tflite). SAI inspects it and searches every execution plan on every board.</p></div></div>
  <div class="grid g2"><div class="card"><div class="drop" id="drop" role="button" tabindex="0" aria-label="Choose a model file">Drop an INT8 <b>.onnx</b> or <b>.tflite</b>, or click<input type="file" id="file" accept=".onnx,.tflite" hidden></div>
    <div class="row" style="margin-top:12px"><label class="check"><input type="checkbox" class="tg" value="md6605" checked>md6605</label><label class="check"><input type="checkbox" class="tg" value="cortex-m7" checked>cortex-m7</label>
    <label class="f">Goal<select id="goal">${Object.entries(GOAL_LABEL).filter(([k]) => k !== "accuracy").map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></label><label class="f">Max RAM (B)<input id="mr" type="number" style="width:110px"></label></div>
    <div id="st"></div></div>
    <div class="card"><h2>Recent analyses</h2>${jobs.length ? `<table class="t"><tbody>${jobs.map((j) => `<tr><td><a href="#/job/${j.id}">${esc(j.params.filename || j.id)}</a></td><td>${statusBadge(j.status)}</td><td class="muted">${ago(j.created)}</td></tr>`).join("")}</tbody></table>` : `<div class="empty small">None yet</div>`}</div></div>`;
  const d = $("#drop"), f = $("#file");
  let sending = false;
  const go = async (file) => {
    if (sending) return toast("The previous model is still uploading");
    sending = true; d.classList.add("busy");
    const fd = new FormData(); fd.append("file", file); fd.append("targets", $$(".tg:checked").map((x) => x.value).join(",")); fd.append("goal", $("#goal").value); if ($("#mr").value) fd.append("max_ram", $("#mr").value);
    $("#st").innerHTML = `<div class="callout info" style="margin-top:10px">Uploading…</div>`;
    try { const j = await api("/api/explorer/analyze", { method: "POST", body: fd }); location.hash = `#/job/${j.id}`; } catch (e) { $("#st").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(e.message)}</div>`; }
    finally { sending = false; d.classList.remove("busy"); }
  };
  d.onclick = () => f.click(); f.onchange = () => f.files[0] && go(f.files[0]);
  d.ondragover = (e) => { e.preventDefault(); d.classList.add("hover"); }; d.ondragleave = () => d.classList.remove("hover");
  d.ondrop = (e) => { e.preventDefault(); d.classList.remove("hover"); e.dataTransfer.files[0] && go(e.dataTransfer.files[0]); };
});

/* ------------------------------------------------------------------------------------------------ board farm */
route(/^\/boards$/, async (el) => {
  const draw = async () => {
    const [b, reg, pb] = await Promise.all([api("/api/boards"), api("/api/regressions"), api("/api/partner-boards")]);
    const kind = (x) => x.kind === "physical" ? "" : `<span class="badge ${x.kind === "emulated" ? "b-info" : "b-warn"}">${esc(x.kind)}</span>`;
    const dot = (x) => x.status === "quarantined" ? "off" : x.status === "busy" ? "busy" : x.online ? "on" : "off";
    el.innerHTML = `<div class="page-h"><div><h1>Board Farm</h1><p class="muted">MCU boards on lab agents. Every run builds the HIL harness around the SAI package, checks outputs bit-exactly, and — on physical boards — measures cycles and stack.</p></div>
      ${ME.perms.farm_admin ? `<button class="btn" id="regrun">Run regression now</button>` : ""}</div>
    ${b.boards.length && b.boards.every((x) => x.kind !== "physical") ? `<div class="callout info" style="margin-bottom:16px">No physical boards are connected yet. The <b>emulated</b> boards run the real cross-compiled firmware on QEMU (Cortex-M7, RV32IMAFC): outputs are real and checked bit-exactly; QEMU cycles are not latency and never calibrate. Connect a lab agent (Admin → Lab agents) to add NUCLEO-H753ZI or MD6605 EVB boards.</div>` : ""}
    <div class="grid g2"><div class="card"><h2>Boards</h2><div class="tbl-wrap"><table class="t"><thead><tr><th>Board</th><th>Target</th><th>Agent</th><th>Health</th><th>Status</th><th></th></tr></thead><tbody>
      ${b.boards.map((x) => `<tr><td>${esc(x.name)} ${kind(x)}</td><td><span style="color:${TARGET_COLOR[x.target]}">●</span> ${esc(x.target)}</td><td class="mono small">${esc(x.agent_id)}${x.partner ? `<div><span class="badge b-info">hosted by ${esc(x.partner)}</span></div>` : ""}</td>
        <td class="small">${esc(x.health.probe || x.info.probe || "")}${x.fail_count ? ` · ${x.fail_count} fail(s)` : ""}</td><td><span class="dot ${dot(x)}"></span> ${esc(x.status)}</td>
        <td>${x.status === "quarantined" && ME.perms.farm_admin ? `<button class="btn sm" data-reset="${esc(x.id)}">Reset</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="6" class="empty">No boards registered</td></tr>`}
    </tbody></table></div><p class="small muted" style="margin-top:8px">A board that fails 3 runs in a row is quarantined until the lab resets it; agents power-cycle a failing board once (uhubctl) before reporting.</p></div>
    <div class="card"><h2>Latency calibration</h2>${Object.keys(b.calibration).length ? `<table class="t"><thead><tr><th>Target : objective</th><th class="n">cycles/MAC</th><th class="n">overhead</th><th class="n">runs</th></tr></thead><tbody>${Object.entries(b.calibration).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="n">${v.cycles_per_mac.toFixed(3)}</td><td class="n">${fmt(v.overhead)}</td><td class="n">${v.samples}</td></tr>`).join("")}</tbody></table>` : `<p class="small muted">No physical-board measurements yet. Until then latency comes from SAI's kernel cost model where it has one, else a MACs × cycles/MAC heuristic, and is labelled that way everywhere.</p>`}
      <h3 style="margin-top:14px">Connect a lab agent</h3><p class="small">Create a token in Admin → Lab agents, list the boards in <code>boards.json</code> (driver <code>pyocd</code>, <code>jlink</code> or <code>qemu</code>), then on the lab PC:</p>
      <div class="log">pip install numpy pyserial pyocd
python agent/board_agent.py --server ${esc(location.origin)} \\
  --token $SAI_AGENT_TOKEN --boards boards.json</div></div></div>
    <div class="card" style="margin-top:16px"><h2>Partner boards</h2><p class="small muted">Boards beyond the reference ones, mapped to the SAI build they run. A partner can host them in their own lab with a partner lab agent (Admin → Lab agents, with a partner name); they then appear above and take runs like any board.</p>
      ${pb.boards.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Board</th><th>Partner</th><th>MCU</th><th>SAI build</th><th>Status</th><th>In the farm</th>${ME.perms.farm_admin ? "<th></th>" : ""}</tr></thead><tbody>${pb.boards.map((x) => `<tr><td><b>${esc(x.board)}</b>${x.url ? ` <a class="small" href="${esc(x.url)}" target="_blank" rel="noopener">info</a>` : ""}<div class="small muted">${esc(x.vendor)}</div></td><td>${esc(x.partner)}</td><td class="small">${esc(x.mcu)} ${esc(x.core)}</td><td><span style="color:${TARGET_COLOR[x.target]}">●</span> ${esc(x.target)}</td><td><span class="badge ${x.status === "certified" ? "b-ok" : x.status === "community" ? "b-info" : ""}">${esc(x.status)}</span></td><td class="small">${x.online_boards.length ? esc(x.online_boards.join(", ")) : `<span class="muted">not connected</span>`}</td>${ME.perms.farm_admin ? `<td><button class="btn sm ghost" data-pbdel="${esc(x.id)}">Remove</button></td>` : ""}</tr>`).join("")}</tbody></table></div>` : `<div class="empty small">No partner boards yet.</div>`}
      ${ME.perms.farm_admin ? `<details style="margin-top:8px"><summary class="small">Add a partner board</summary><div class="grid g3" style="margin-top:8px"><label class="f">Partner<input id="pbp"></label><label class="f">Board<input id="pbb"></label><label class="f">Vendor<input id="pbv"></label><label class="f">MCU / core<input id="pbm"></label><label class="f">SAI build<select id="pbt">${pb.targets.map((t) => `<option>${t}</option>`).join("")}</select></label><label class="f">Status<select id="pbs"><option>planned</option><option>community</option><option>certified</option></select></label></div><button class="btn sm primary" id="pbadd" style="margin-top:8px">Add</button></details>` : ""}</div>
    <div class="card" style="margin-top:16px"><h2>Nightly regression</h2>${reg.latest.length ? `<p class="small muted">Run ${esc(reg.runs[0].run_id)} · SAI ${esc(reg.runs[0].sai_version)} · ${ago(reg.runs[0].created)}</p>
      <div class="tbl-wrap"><table class="t"><thead><tr><th>Model type (tiny)</th><th>Board</th><th class="n">RAM</th><th class="n">Flash</th><th class="n">Plans that fit</th><th>Emulated HIL</th></tr></thead><tbody>
      ${reg.latest.map((r) => `<tr><td>${esc(r.template)}</td><td>${esc(r.target)}</td><td class="n">${bytes(r.ram)}</td><td class="n">${bytes(r.flash)}</td><td class="n">${r.feasible}</td><td>${r.hil_match === null ? "–" : r.hil_match ? `<span class="badge b-ok">bit-exact</span>` : `<span class="badge b-bad">mismatch</span>`}</td></tr>`).join("")}</tbody></table></div>` : `<p class="small muted">No regression run yet. It checks every pool model type on every board against the current SAI release and alerts on RAM/flash growth, lost fits or HIL mismatches (also run by tools/full_check.sh).</p>`}</div>
    <div class="card" style="margin-top:16px"><h2>Recent runs</h2>${b.runs.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Run</th><th>Job</th><th>Board</th><th>Status</th><th class="n">Estimated</th><th class="n">Result</th><th>Outputs</th><th class="n">Stack</th><th>When</th></tr></thead><tbody>
      ${b.runs.map((x) => { const r = x.result || {}; return `<tr><td class="mono small">${esc(x.id)}</td><td><a href="#/job/${esc(x.job_id)}">${esc(x.job_id)}</a></td><td>${esc(x.board_id || "–")}</td><td>${statusBadge(x.status)}</td><td class="n">${ms(x.request.estimated_ms)}</td>
        <td class="n">${r.emulated ? `<span class="small muted">${fmt(r.cycles)} emu. cycles</span>` : r.simulated ? `${ms(r.latency_ms)} (sim)` : ms(r.latency_ms)}</td>
        <td>${r.outputs_match === undefined ? (r.error ? `<span class="badge b-bad" title="${esc(r.error)}">error</span>` : "–") : r.outputs_match ? `<span class="badge b-ok">match</span>` : `<span class="badge b-bad">mismatch</span>`}</td><td class="n">${bytes(r.stack_painted)}</td><td class="muted">${ago(x.created)}</td></tr>`; }).join("")}</tbody></table></div>` : `<div class="empty small">No runs yet. Open a finished project and press “Flash &amp; measure”.</div>`}</div>`;
    $$("[data-reset]").forEach((bt) => (bt.onclick = () => busy(bt, "Resetting…", async () => {
      try { await api(`/api/boards/${bt.dataset.reset}/reset`, { method: "POST" }); }
      catch (e) { if (!confirm(`${e.message}\n\nReset anyway and fail that run?`)) return; await api(`/api/boards/${bt.dataset.reset}/reset`, { json: { force: true } }); }
      draw();
    })));
    const rr = $("#regrun");
    if (rr) rr.onclick = () => busy(rr, "Starting…", async () => { const j = await api("/api/admin/regression", { json: {} }); location.hash = `#/job/${j.id}`; });
    $$("[data-pbdel]").forEach((x) => (x.onclick = () => busy(x, "…", async () => { await api(`/api/partner-boards/${x.dataset.pbdel}`, { method: "DELETE" }); draw(); })));
    if ($("#pbadd")) $("#pbadd").onclick = () => busy($("#pbadd"), "Adding…", async () => { await api("/api/partner-boards", { json: { partner: $("#pbp").value, board: $("#pbb").value, vendor: $("#pbv").value, mcu: $("#pbm").value, target: $("#pbt").value, status: $("#pbs").value } }); draw(); });
  };
  await draw();
  const t = setInterval(draw, 4000); onLeave(() => clearInterval(t));
});

/* ------------------------------------------------------------------------------------------------ virtual dev board */
route(/^\/devboard\/(job_[0-9a-f]+)$/, async (el, jid) => {
  const [job, av, recs] = await Promise.all([api(`/api/jobs/${jid}`), api("/api/devboard/available"), api("/api/recordings")]);
  const r = job.result, sel = r.selected;
  const targets = Object.entries(av.targets).filter(([, ok]) => ok).map(([t]) => t);
  const okRecs = recs.filter((x) => x.status === "ready" && x.channels.length === (r.preprocessing.raw_shape || [0])[0]);
  let timer = null;
  el.innerHTML = `<div class="page-h"><div><h1>Virtual dev board</h1><p class="muted"><a href="#/job/${esc(jid)}">${esc(job.params.name)}</a> · ${esc(r.template.name || job.params.template_id)} · the real firmware (SAI library + HIL harness, cross-compiled with SAI's flags) booted in QEMU.</p></div></div>
    ${targets.length ? "" : `<div class="callout bad">No emulator or cross compiler is installed on this server.</div>`}
    <div class="grid g2"><div class="card"><h2>1. Board</h2><div class="stack"><label class="f">Virtual board<select id="vt">${targets.map((t) => `<option value="${t}" ${t === sel.target ? "selected" : ""}>${t === "cortex-m7" ? "Cortex-M7 (QEMU MPS2-AN500)" : "RV32IMAFC, MD6605 class (QEMU virt)"}</option>`).join("")}</select></label>
        <p class="small muted">A board other than the project's (${esc(sel.target)}) gets the same model compiled by SAI for it first. Cycle counts are QEMU's instruction clock: they show the program runs on that ISA, not real-board speed.</p></div></div>
      <div class="card"><h2>2. Inputs</h2><div class="tabs" id="vsrc"><button class="tab on" data-s="dataset">Dataset windows</button><button class="tab" data-s="recording">Recording</button><button class="tab" data-s="values">Paste values</button></div><div id="vin"></div>
        <button class="btn primary" id="vgo" style="margin-top:10px" ${targets.length ? "" : "disabled"}>Run on the virtual board</button></div></div>
    <div id="vres" style="margin-top:16px"></div>`;
  let src = "dataset";
  const C = (r.preprocessing.raw_shape || [1])[0], T = (r.preprocessing.frontend || {}).raw_length || (r.preprocessing.raw_shape || [1, 1, 1])[2];
  const drawIn = () => {
    $("#vin").innerHTML = src === "dataset" ? `<label class="f">Window numbers (comma separated, up to 16; empty = 6 random)<input id="vidx" placeholder="e.g. 0, 12, 40"></label>`
      : src === "recording" ? (okRecs.length ? `<div class="grid g3"><label class="f">Recording<select id="vrec">${okRecs.map((x) => `<option value="${esc(x.id)}">${esc(x.name)}</option>`).join("")}</select></label><label class="f">Start (s)<input id="vst" type="number" step="any" value="0"></label><label class="f">Windows<input id="vn" type="number" value="4"></label></div>` : `<p class="small muted">No recording with ${C} channel(s). <a href="#/recordings">Capture or import one</a>.</p>`)
      : `<label class="f">One window per line: ${C} channel(s) × ${T} values, channels after each other, comma separated<textarea id="vval" rows="5" class="mono small" placeholder="${C * T} numbers per line"></textarea></label>`;
  };
  $$("#vsrc .tab").forEach((b) => (b.onclick = () => { src = b.dataset.s; $$("#vsrc .tab").forEach((x) => x.classList.toggle("on", x === b)); drawIn(); }));
  drawIn();
  const show = (s) => {
    const res = s.result;
    if (s.status !== "succeeded") { $("#vres").innerHTML = `<div class="card">${statusBadge(s.status)} ${s.status === "failed" ? `<div class="callout bad" style="margin-top:8px">${esc(s.error)}</div>` : `<span class="small muted">${esc((s.logs || []).slice(-1).map((l) => l.line).join("") || "starting…")}</span>`}</div>`; return; }
    $("#vres").innerHTML = `<div class="grid g2"><div class="card"><div class="row"><h2 style="margin:0">Board output</h2><span class="spacer"></span>${res.all_bit_exact ? `<span class="badge b-ok">bit-exact with the host program</span>` : `<span class="badge b-bad">mismatch</span>`}</div>
        <p class="small muted">${esc(res.board)} · firmware text ${bytes(res.firmware.text)}, bss ${bytes(res.firmware.bss)} · stack used ${bytes(res.stack_painted)} · ${fmt(res.cycles.avg)} instructions per inference ${ev("estimate")}</p>
        <div class="tbl-wrap"><table class="t"><thead><tr><th>Input</th><th></th><th>Prediction</th><th class="n">Confidence</th><th>Expected</th><th>int8 out</th></tr></thead><tbody>
        ${res.windows.map((w) => `<tr><td class="small">${esc(w.name)}</td><td style="width:140px">${sparkSeries(w.input_preview, 140, 34)}</td><td>${w.prediction ? `<b>${esc(w.prediction)}</b>${w.truth ? (w.truth === w.prediction ? ` <span class="badge b-ok">= label</span>` : ` <span class="badge b-warn">label ${esc(w.truth)}</span>`) : ""}` : esc(JSON.stringify(w.value))}</td><td class="n">${w.confidence != null ? pct(w.confidence) : ""}</td><td>${w.bit_exact ? `<span class="badge b-ok">match</span>` : `<span class="badge b-bad">differs</span>`}</td><td class="mono small">[${w.output_int8.join(", ")}]</td></tr>`).join("")}</tbody></table></div></div>
      <div class="card"><h2>UART</h2><div class="log" style="max-height:420px;overflow:auto">${esc(res.uart)}</div></div></div>`;
  };
  $("#vgo").onclick = () => busy($("#vgo"), "Building firmware…", async () => {
    const inputs = { source: src };
    if (src === "dataset") inputs.indices = ($("#vidx").value || "").split(",").map((x) => x.trim()).filter(Boolean).map(Number);
    if (src === "recording") { if (!$("#vrec")) return toast("No recording"); const rec = okRecs.find((x) => x.id === $("#vrec").value); Object.assign(inputs, { recording_id: rec.id, start: Math.round(+$("#vst").value * rec.sample_rate), count: +$("#vn").value }); }
    if (src === "values") inputs.windows = $("#vval").value.trim().split("\n").filter(Boolean).map((l) => { const v = l.split(/[ ,;]+/).filter(Boolean).map(Number); return Array.from({ length: C }, (_, c) => v.slice(c * T, (c + 1) * T)); });
    const s = await api(`/api/jobs/${jid}/devboard`, { json: { target: $("#vt").value, inputs } });
    clearInterval(timer);
    const poll = async () => { const j = await api(`/api/jobs/${s.id}`); show(j); if (["succeeded", "failed", "cancelled"].includes(j.status)) clearInterval(timer); };
    timer = setInterval(() => poll().catch(() => {}), 1500); poll();
  });
  onLeave(() => clearInterval(timer));
});

/* ------------------------------------------------------------------------------------------------ benchmarks */
route(/^\/benchmarks$/, async (el) => {
  const draw = async () => {
    const b = await api("/api/benchmarks");
    const L = b.latest;
    const by = {};
    (L ? L.rows : []).forEach((x) => { (by[x.id] = by[x.id] || {})[x.target] = x; });
    const targets = L ? L.targets : ["md6605", "cortex-m7"];
    const running = b.runs.find((x) => x.status !== "succeeded");
    el.innerHTML = `<div class="page-h"><div><h1>Benchmarks</h1><p class="muted">The MLPerf Tiny reference tasks and every built-in model type, compiled by SAI for every board, next to TFLite Micro on the same graph. A resource benchmark (random weights, no accuracy claim); each run is compared with the previous one, so it doubles as a regression suite across SAI releases.</p></div>
      ${b.can_run ? `<button class="btn primary" id="brun" ${running ? "disabled" : ""}>${running ? `Running (${pct(running.progress)})` : "Run the corpus"}</button>` : ""}</div>
      ${L ? `<p class="small muted">Latest run <a href="#/job/${esc(L.job_id)}">${esc(L.job_id)}</a> · SAI ${esc((L.sai_version || {}).sai || (L.sai_version || {}).version || "")}${L.previous_job ? ` · compared with <a href="#/job/${esc(L.previous_job)}">${esc(L.previous_job)}</a>` : ""}</p>
        ${L.alerts.length ? L.alerts.map((a) => `<div class="callout bad">${esc(a)}</div>`).join("") : `<div class="callout ok">No RAM / flash regression against the previous run.</div>`}` : `<div class="callout info">No benchmark run yet.</div>`}
      <div class="card" style="margin-top:12px"><div class="tbl-wrap"><table class="t"><thead><tr><th>Model</th><th class="n">MACs</th>${targets.map((t) => `<th>${esc(t)}: SAI RAM / flash</th><th>TFLM RAM / flash</th>`).join("")}</tr></thead><tbody>
        ${b.corpus.map((c) => `<tr><td><b>${esc(c.task)}</b><div class="small muted">${esc(c.suite)} · ${esc(c.reference)}</div>${c.deviation ? `<div class="small" style="color:var(--warn)">Differs from the reference: ${esc(c.deviation)}</div>` : ""}</td><td class="n">${fmt(c.summary.macs / 1e6, 2)}M</td>
          ${targets.map((t) => { const x = (by[c.id] || {})[t]; if (!x) return `<td class="muted small">–</td><td class="muted small">–</td>`; const s = x.sai; return `<td class="small">${s ? `${bytes(s.ram)} / ${bytes(s.flash)}${s.fits ? "" : ` <span class="badge b-bad" title="${esc((s.violations || []).join("; "))}">does not fit</span>`}<div class="muted">${ms(s.latency_ms)} ${ev(s.latency_evidence)}${x.prev && x.prev.ram && x.prev.ram !== s.ram ? ` · was ${bytes(x.prev.ram)}` : ""}</div>` : `<span class="badge b-warn" title="${esc(x.error || "")}">no plan</span>`}</td>
            <td class="small">${bytes(x.tflm.ram)} / ${bytes(x.tflm.flash)}${s && s.ram ? ` <b>${(x.tflm.ram / s.ram).toFixed(1)}×</b>` : ""}<div class="muted">${ms(x.tflm.latency_ms)} ${ev("assumed")}</div></td>`; }).join("")}</tr>`).join("")}</tbody></table></div>
        <p class="small muted" style="margin-top:8px">TFLM columns: arena replayed with TFLM's greedy planner on the same graph, flash and latency from assumed code sizes and cycles/MAC (CMSIS-NN on Cortex-M7, reference kernels on RV32). Not an MLPerf submission.</p></div>`;
    if ($("#brun")) $("#brun").onclick = () => busy($("#brun"), "Starting…", async () => { await api("/api/benchmarks/run", { json: {} }); draw(); });
  };
  await draw();
  const t = setInterval(() => draw().catch(() => {}), 6000); onLeave(() => clearInterval(t));
});

/* ------------------------------------------------------------------------------------------------ system */
route(/^\/system$/, async (el) => {
  SYS = null; const s = await system();
  el.innerHTML = `<div class="page-h"><div><h1>Compiler &amp; targets</h1><p class="muted">The platform drives the SAI compiler only through its supported CLI.</p></div></div>
  <div class="grid g2"><div class="card"><h2>SAI</h2><div class="tbl-wrap"><table class="t"><tbody>${Object.entries(s.sai).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="mono">${esc(v)}</td></tr>`).join("")}</tbody></table></div>
    <h3 style="margin-top:14px">Operators available to model authors</h3><div class="tbl-wrap"><table class="t"><tbody>${Object.entries(s.ops).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</tbody></table></div></div>
  <div class="card"><h2>Targets</h2><div class="tbl-wrap"><table class="t"><thead><tr><th>Target</th><th>Core</th><th class="n">Clock</th><th class="n">AI arena</th><th class="n">AI flash</th><th>Toolchain</th></tr></thead><tbody>
  ${s.targets.map((t) => `<tr class="${t.status === "roadmap" ? "nofit" : ""}"><td><b>${esc(t.name)}</b><div class="small muted mono">${esc(t.id)}</div></td><td class="small">${esc(t.core || "")}</td><td class="n">${t.clock_hz ? fmt(t.clock_hz / 1e6, 1) + " MHz" : "–"}</td><td class="n">${bytes(t.budget?.workspace)}</td><td class="n">${bytes(t.budget?.flash)}</td><td>${t.status === "roadmap" ? `<span class="badge">roadmap</span>` : `<span class="badge ${t.toolchain === "available" ? "b-ok" : "b-warn"}">${esc(t.toolchain)}</span>`}</td></tr>`).join("")}
  </tbody></table></div>
  ${s.sai_compat ? `<h3 style="margin-top:14px">SAI contract</h3><p class="small">Report schemas the platform is tested against: ${Object.entries(s.sai_compat.supported).map(([k, v]) => `<span class="badge">${esc(k)} ${esc(v.join(", "))}</span>`).join(" ")}</p>${s.sai_compat.warnings.map((w) => `<div class="callout">${esc(w)}</div>`).join("")}<p class="small muted">Programs measured on the board farm: ${s.measurements}</p>` : ""}
  <h3 style="margin-top:14px">Evidence labels</h3><div class="tbl-wrap"><table class="t"><tbody>${Object.entries(EVIDENCE).map(([k]) => `<tr><td>${ev(k)}</td><td class="small">${esc(EVIDENCE[k][2])}</td></tr>`).join("")}</tbody></table></div></div></div>`;
});

/* ------------------------------------------------------------------------------------------------ datasets */
route(/^\/data$/, async (el) => {
  const [list, cat] = await Promise.all([api("/api/datasets"), api("/api/catalog")]);
  const tn = Object.fromEntries(cat.map((t) => [t.id, t]));
  el.innerHTML = `<div class="page-h"><div><h1>Data</h1><p class="muted">Everything your organisation uploaded, captured, imported or generated, ready to inspect and reuse.</p></div><a class="btn primary" href="#/new">+ New project</a></div>${dataTabs("#/data")}
  <div class="card">${list.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Dataset</th><th>Model type</th><th>Source</th><th>Licence</th><th class="n">Windows</th><th>Shape</th><th class="n">Recordings</th><th>Quality</th><th>Created</th></tr></thead><tbody>
    ${list.map((d) => `<tr><td><a href="#/data/${esc(d.id)}"><b>${esc(d.name)}</b></a></td><td>${esc((tn[d.template] || {}).icon || "")} ${esc((tn[d.template] || {}).name || d.template)}</td><td>${esc(d.source)}${d.frontend ? ` <span class="badge b-info">${esc(d.frontend.type)}</span>` : ""}</td>
      <td><span class="badge ${d.commercial_use ? "b-ok" : "b-warn"}">${esc(d.licence)}</span></td><td class="n">${fmt(d.n)}</td><td class="mono small">${esc((d.shape || []).join("×"))}</td><td class="n">${fmt(d.recordings)}</td>
      <td>${(d.warnings || []).length ? `<span class="badge b-warn" title="${esc(d.warnings.join("\n"))}">${d.warnings.length} warning(s)</span>` : `<span class="badge b-ok">ok</span>`}</td><td class="muted">${ago(d.created)}</td></tr>`).join("")}
  </tbody></table></div>` : `<div class="empty">No datasets yet. Start a <a href="#/new">new project</a> to upload or generate one.</div>`}</div>`;
});

route(/^\/data\/(ds_[0-9a-f]+)$/, async (el, did) => {
  const [d, st, cat] = await Promise.all([api(`/api/datasets/${did}`), api(`/api/datasets/${did}/stats`), api("/api/catalog")]);
  const t = cat.find((x) => x.id === d.template) || { name: d.template, icon: "" };
  const q = d.quality || {};
  let label = null, offset = 0;
  const classes = d.class_counts ? Object.keys(d.class_counts) : [];
  el.innerHTML = `<div class="page-h"><div><h1>${esc(d.name)}</h1><div class="row small muted">${esc(t.icon)} ${esc(t.name)} · <span class="mono">${esc(d.id)}</span> · ${esc(d.source)} · ${ago(d.created)}</div></div>
    <div class="row"><a class="btn primary" href="#/new?template=${encodeURIComponent(d.template)}&dataset=${encodeURIComponent(d.id)}">Use in a new project</a><button class="btn" id="del">Delete</button></div></div>
  <div class="kpis"><div class="kpi"><div class="l">Windows</div><div class="v">${fmt(d.n)}</div></div><div class="kpi"><div class="l">Window shape</div><div class="v">${esc(d.shape.join("×"))}</div></div>
    <div class="kpi"><div class="l">Recordings</div><div class="v">${fmt(d.recordings)}</div></div><div class="kpi"><div class="l">Licence</div><div class="v" style="font-size:15px">${esc(d.licence)} ${d.commercial_use ? `<span class="badge b-ok">commercial ok</span>` : `<span class="badge b-warn">evaluation only</span>`}</div></div>
    ${d.frontend ? `<div class="kpi"><div class="l">Front end (C)</div><div class="v" style="font-size:15px">${esc(d.frontend.type)}</div></div>` : ""}</div>
  ${(d.warnings || []).map((w) => `<div class="callout" style="margin-top:10px">${esc(w)}</div>`).join("")}
  <div class="grid g2" style="margin-top:16px"><div class="card"><h2>${st.task === "regression" ? "Target distribution" : "Classes"}</h2>
      ${st.task === "regression" ? `<div class="small">min ${fmt(d.target_stats.min, 3)} · mean ${fmt(d.target_stats.mean, 3)} · max ${fmt(d.target_stats.max, 3)}</div>${bars(Object.fromEntries(st.per_class[0].histogram.map((h, i) => [fmt(st.per_class[0].edges[i], 2), h])))}`
      : `<div class="tbl-wrap"><table class="t"><thead><tr><th>Class</th><th class="n">Windows</th><th class="n">Recordings</th><th style="width:40%"></th></tr></thead><tbody>
        ${st.per_class.map((c) => { const mx = Math.max(...st.per_class.map((x) => x.n)); return `<tr><td>${esc(c.label)}</td><td class="n">${fmt(c.n)}</td><td class="n">${fmt(c.recordings)}</td><td><div class="progress" style="height:10px"><i style="width:${c.n / mx * 100}%"></i></div></td></tr>`; }).join("")}</tbody></table></div>`}</div>
    <div class="card"><h2>Quality</h2><table class="t"><tbody>
      <tr><td class="muted">Duplicate windows</td><td class="n">${fmt(q.duplicate_windows)}</td></tr><tr><td class="muted">Duplicates across recordings</td><td class="n">${fmt(q.duplicates_across_recordings)}</td></tr>
      <tr><td class="muted">Clipping per channel</td><td class="n">${(q.clipping_fraction_per_channel || []).map((v) => (v * 100).toFixed(1) + "%").join(" · ")}</td></tr>
      ${q.sample_period ? `<tr><td class="muted">Sample period / jitter</td><td class="n">${fmt(q.sample_period * 1e3, 3)} ms / ${(q.sample_period_jitter * 100).toFixed(1)}%</td></tr>` : ""}
    </tbody></table></div></div>
  <div class="card" style="margin-top:16px"><h2>Channels</h2><div class="tbl-wrap"><table class="t"><thead><tr><th>Channel</th><th class="n">mean</th><th class="n">std</th><th class="n">min</th><th class="n">p1</th><th class="n">p99</th><th class="n">max</th>${st.task !== "regression" ? st.per_class.map((c) => `<th class="n">mean · ${esc(c.label)}</th>`).join("") : ""}</tr></thead><tbody>
    ${st.channels.map((c, i) => `<tr><td><b>${esc(c.name)}</b></td><td class="n">${fmt(c.mean, 3)}</td><td class="n">${fmt(c.std, 3)}</td><td class="n">${fmt(c.min, 3)}</td><td class="n">${fmt(c.p1, 3)}</td><td class="n">${fmt(c.p99, 3)}</td><td class="n">${fmt(c.max, 3)}</td>${st.task !== "regression" ? st.per_class.map((k) => `<td class="n">${fmt(k.channel_mean[i], 3)}</td>`).join("") : ""}</tr>`).join("")}
  </tbody></table></div></div>
  <div class="card" style="margin-top:16px"><div class="row"><h2 style="margin:0">Samples</h2><span class="spacer"></span>
    ${classes.length ? `<select id="sl"><option value="">all classes</option>${classes.map((c, i) => `<option value="${i}">${esc(c)}</option>`).join("")}</select>` : ""}
    <button class="btn sm" id="prev">←</button><span class="small muted" id="pos"></span><button class="btn sm" id="nextp">→</button></div><div class="grid g3" id="samples" style="margin-top:12px"></div></div>
  ${(d.used_by || []).length ? `<div class="card" style="margin-top:16px"><h2>Used by</h2><table class="t"><tbody>${d.used_by.map((u) => `<tr><td><a href="#/job/${esc(u.job_id)}">${esc(u.name)}</a></td><td>${statusBadge(u.status)}</td></tr>`).join("")}</tbody></table></div>` : ""}`;
  const names = st.channels.map((c) => c.name);
  const loadSamples = async () => {
    const r = await api(`/api/datasets/${did}/samples?offset=${offset}&n=6${label !== null ? `&label=${label}` : ""}`);
    $("#pos").textContent = r.total ? `${offset + 1}–${Math.min(offset + 6, r.total)} of ${fmt(r.total)}` : "none";
    $("#samples").innerHTML = r.items.map((it) => `<div class="card" style="padding:10px"><div class="row small"><b>${esc(it.label)}</b><span class="spacer"></span><span class="muted mono">#${it.index}${it.group ? " · " + esc(it.group) : ""}</span></div>
      ${it.series ? `${sparkSeries(it.series.slice(0, 3), 300, 90)}<div class="legend">${it.series.slice(0, 3).map((_, k) => `<span><i style="background:var(--c${k + 1})"></i>${esc(names[k] || "ch" + k)}</span>`).join("")}${it.series.length > 3 ? `<span>+${it.series.length - 3} more</span>` : ""}</div>` : heat(it.image, 260)}</div>`).join("") || `<div class="empty small">No windows</div>`;
    $("#prev").disabled = offset === 0; $("#nextp").disabled = offset + 6 >= r.total;
  };
  if ($("#sl")) $("#sl").onchange = (e) => { label = e.target.value === "" ? null : +e.target.value; offset = 0; loadSamples(); };
  $("#prev").onclick = () => { offset = Math.max(0, offset - 6); loadSamples(); };
  $("#nextp").onclick = () => { offset += 6; loadSamples(); };
  $("#del").onclick = () => {
    if (!confirm(`Delete dataset "${d.name}"? Finished projects keep their reports and packages.`)) return;
    return busy($("#del"), "Deleting…", async () => { await api(`/api/datasets/${did}`, { method: "DELETE" }); toast("Dataset deleted"); location.hash = "#/data"; });
  };
  loadSamples();
});

/* ------------------------------------------------------------------------------------------------ data sources */
function dataTabs(on) {
  const t = [["#/data", "Datasets"], ["#/recordings", "Recordings & labelling"], ["#/connectors", "Cloud & database"], ["#/physics", "Physics augmentation"]];
  return `<div class="tabs">${t.map(([h, l]) => `<a class="tab ${h === on ? "on" : ""}" href="${h}">${l}</a>`).join("")}</div>`;
}
const LABEL_COLORS = ["#4f7cff", "#e0a100", "#e5484d", "#2bb673", "#9b59d0", "#00a3bf", "#f76b15", "#7a8599"];
const labelColor = (labels, l) => LABEL_COLORS[Math.max(0, labels.indexOf(l)) % LABEL_COLORS.length];

/* min/max envelope plot of a recording range, label spans behind it; returns svg + the x->sample mapping */
function signalPlot(v, { w = 1000, h = 260, sel = null, hidden = [] } = {}) {
  const pad = { l: 8, r: 8, t: 8, b: 22 };
  const n0 = v.start, n1 = Math.max(v.end, v.start + 1);
  const X = (i) => pad.l + (i - n0) / (n1 - n0) * (w - pad.l - pad.r);
  const chans = v.min.map((_, k) => k).filter((k) => !hidden.includes(k));
  const all = chans.flatMap((k) => [...v.min[k], ...v.max[k]]);
  let lo = all.length ? Math.min(...all) : 0, hi = all.length ? Math.max(...all) : 1; if (lo === hi) { lo -= 1; hi += 1; }
  const Y = (y) => h - pad.b - (y - lo) / (hi - lo) * (h - pad.t - pad.b);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" preserveAspectRatio="none" class="sigplot" role="img" aria-label="signal">`;
  for (const sp of v.spans) {
    const a = Math.max(sp.start, n0), b = Math.min(sp.end, n1);
    s += `<rect x="${X(a)}" y="${pad.t}" width="${Math.max(1, X(b) - X(a))}" height="${h - pad.t - pad.b}" fill="${labelColor(v.labels, sp.label)}" opacity=".16"><title>${esc(sp.label)} ${sp.start}–${sp.end}</title></rect>
      ${X(b) - X(a) > 8 * sp.label.length ? `<text x="${X(a) + 4}" y="${pad.t + 12}" style="fill:${labelColor(v.labels, sp.label)};font-weight:600">${esc(sp.label)}</text>` : ""}`;
  }
  if (v.live && v.live.label) s += `<rect x="${X(Math.max(v.live.start, n0))}" y="${pad.t}" width="${Math.max(1, X(n1) - X(Math.max(v.live.start, n0)))}" height="${h - pad.t - pad.b}" fill="${labelColor(v.labels, v.live.label)}" opacity=".22"/>`;
  chans.forEach((k) => {
    const c = `var(--c${(k % 3) + 1})`;
    const top = v.x.map((x, i) => `${X(x)},${Y(v.max[k][i])}`), bot = v.x.map((x, i) => `${X(x)},${Y(v.min[k][i])}`).reverse();
    s += v.x.length && v.min[k].some((m, i) => m !== v.max[k][i]) ? `<polygon points="${[...top, ...bot].join(" ")}" fill="${c}" fill-opacity=".55" stroke="${c}" stroke-width=".6"/>` : `<polyline points="${top.join(" ")}" fill="none" stroke="${c}" stroke-width="1.2"/>`;
  });
  if (sel) s += `<rect x="${X(Math.min(sel[0], sel[1]))}" y="${pad.t}" width="${Math.abs(X(sel[1]) - X(sel[0]))}" height="${h - pad.t - pad.b}" fill="var(--accent)" opacity=".18" stroke="var(--accent)"/>`;
  const sr = v.sample_rate || 1;
  for (let i = 0; i <= 5; i++) { const n = n0 + (n1 - n0) * i / 5; s += `<text x="${Math.min(w - 30, Math.max(14, X(n)))}" y="${h - 6}" text-anchor="middle">${(n / sr).toFixed((n1 - n0) / sr < 5 ? 2 : 1)} s</text>`; }
  return s + "</svg>";
}

route(/^\/recordings$/, async (el) => {
  const [list, cat] = await Promise.all([api("/api/recordings"), api("/api/catalog")]);
  const ts = cat.filter((t) => t.input && t.input.kind === "timeseries");
  el.innerHTML = `<div class="page-h"><div><h1>Data</h1><p class="muted">Continuous signals to label by time span, captured live or imported, then cut into a training dataset.</p></div></div>${dataTabs("#/recordings")}
  <div class="grid g2"><div class="card"><h2>Import a recording</h2><p class="small muted">CSV: numeric columns become channels; a <span class="mono">time</span> column gives the sample rate; an optional <span class="mono">label</span> column becomes spans you can edit.</p>
      <div class="stack"><label class="f">File<input type="file" id="rf" accept=".csv,.txt"></label>
      <div class="grid g2"><label class="f">Sample rate (Hz)<input id="rsr" type="number" placeholder="from the time column"></label><label class="f">Licence<select id="rlic">${["proprietary", "cc-by-4.0", "cc0", "research-only", "unknown"].map((l) => `<option>${l}</option>`).join("")}</select></label></div>
      <button class="btn primary" id="rup">Import</button></div></div>
    <div class="card"><h2>Live capture</h2><p class="small muted">Stream from a sensor while you watch and mark what is happening. A device, gateway or <span class="mono">tools/capture_client.py</span> posts samples with the capture token.</p>
      <div class="stack"><label class="f">Name<input id="cn" value="Bench capture"></label>
      <label class="f">Model type (sets channels and rate)<select id="ct"><option value="">custom</option>${ts.map((t) => `<option value="${esc(t.id)}">${esc(t.icon)} ${esc(t.name)}</option>`).join("")}</select></label>
      <div class="grid g2"><label class="f">Channels (comma separated)<input id="cc" value="acc_x,acc_y,acc_z"></label><label class="f">Sample rate (Hz)<input id="csr" type="number" value="3200"></label></div>
      <button class="btn primary" id="cgo">Start capture</button></div></div></div>
  <div class="card" style="margin-top:16px"><h2>Recordings</h2>${list.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Name</th><th>Status</th><th>Source</th><th>Channels</th><th class="n">Duration</th><th>Labels</th><th>Created</th></tr></thead><tbody>
    ${list.map((r) => `<tr><td><a href="#/recording/${esc(r.id)}"><b>${esc(r.name)}</b></a></td><td>${r.status === "capturing" ? `<span class="badge b-warn">● capturing</span>` : `<span class="badge b-ok">ready</span>`}</td><td>${esc(r.source)}</td><td class="small">${esc(r.channels.join(", "))}</td>
      <td class="n">${fmt(r.samples / r.sample_rate, 1)} s</td><td>${(r.labels || []).map((l) => `<span class="badge" style="background:${labelColor(r.labels, l)}22;color:${labelColor(r.labels, l)}">${esc(l)}</span>`).join(" ") || `<span class="muted small">none yet</span>`}</td><td class="muted">${ago(r.created)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="empty">No recordings yet.</div>`}</div>`;
  $("#ct").onchange = () => { const t = ts.find((x) => x.id === $("#ct").value); if (!t) return; $("#cc").value = (t.input.channel_names || []).join(","); $("#csr").value = t.input.sample_rate_hz || ""; };
  $("#rup").onclick = () => busy($("#rup"), "Importing…", async () => {
    const f = $("#rf").files[0]; if (!f) return toast("Choose a CSV file");
    const fd = new FormData(); fd.append("file", f); fd.append("licence", $("#rlic").value); if ($("#rsr").value) fd.append("sample_rate", $("#rsr").value);
    const r = await api("/api/recordings/upload", { method: "POST", body: fd }); location.hash = `#/recording/${r.id}`;
  });
  $("#cgo").onclick = () => busy($("#cgo"), "Starting…", async () => {
    const r = await api("/api/recordings/capture", { json: { name: $("#cn").value, channels: $("#cc").value.split(",").map((s) => s.trim()).filter(Boolean), sample_rate: +$("#csr").value, template_id: $("#ct").value || null } });
    sessionStorage.setItem(`captok_${r.id}`, r.token);
    location.hash = `#/recording/${r.id}`;
  });
});

route(/^\/recording\/(rec_[0-9a-f]+)$/, async (el, rid) => {
  let rec = await api(`/api/recordings/${rid}`);
  const [cat, jobs] = await Promise.all([api("/api/catalog"), api("/api/jobs?kind=train")]);
  const ts = cat.filter((t) => t.input && t.input.kind === "timeseries");
  let view = null, win = null, sel = null, hidden = [], cur = (rec.labels || [])[0] || "", suggestion = null, timer = null;
  const live = () => rec.status === "capturing";
  const tok = sessionStorage.getItem(`captok_${rid}`);
  el.innerHTML = `<div class="page-h"><div><h1>${esc(rec.name)}</h1><div class="row small muted"><span class="mono">${esc(rid)}</span> · ${esc(rec.channels.join(", "))} · ${fmt(rec.sample_rate)} Hz · ${esc(rec.source)} · ${esc(rec.licence)}</div></div>
    <div class="row"><a class="btn" href="#/recordings">All recordings</a><button class="btn" id="rdel">Delete</button></div></div>
  <div id="capbox"></div>
  <div class="card"><div class="row" style="flex-wrap:wrap;gap:8px"><b>Label</b><div id="labs" class="row" style="gap:6px;flex-wrap:wrap"></div><input id="newlab" placeholder="new label" style="width:120px"><button class="btn sm" id="addlab">Add</button>
      <span class="spacer"></span><span class="small muted" id="selinfo">Drag on the plot to select a range</span></div>
    <div id="plot" style="margin-top:10px;user-select:none;cursor:crosshair"></div>
    <div class="row" style="margin-top:8px;flex-wrap:wrap;gap:6px"><button class="btn sm" id="zin">Zoom to selection</button><button class="btn sm" id="zout">Zoom out</button><button class="btn sm" id="pl">◀ Pan</button><button class="btn sm" id="pr">Pan ▶</button><button class="btn sm" id="zall">Whole recording</button>
      <span class="legend" id="chs"></span><span class="spacer"></span><button class="btn sm primary" id="apply">Label selection</button><button class="btn sm" id="erase">Erase labels in selection</button></div></div>
  <div class="grid g2" style="margin-top:16px"><div class="card"><h2>Suggest labels with a model</h2><p class="small muted">Runs a finished project's INT8 model over sliding windows of the visible range. Nothing changes until you accept.</p>
      <div class="stack"><label class="f">Project<select id="sj">${jobs.filter((j) => j.status === "succeeded").map((j) => `<option value="${esc(j.id)}">${esc((j.params && j.params.name) || j.name || j.id)}${j.params && j.params.template_id ? ` · ${esc(j.params.template_id)}` : ""}</option>`).join("") || `<option value="">no finished project</option>`}</select></label>
      <label class="f">Minimum confidence <input id="sc" type="range" min="0.3" max="0.99" step="0.01" value="0.6"> <span id="scv" class="mono small">0.60</span></label>
      <div class="row"><button class="btn" id="sgo">Suggest</button><button class="btn primary" id="sacc" disabled>Accept suggestions</button><button class="btn" id="sdis" disabled>Discard</button></div><div id="sres" class="small"></div></div></div>
    <div class="card"><h2>Make a dataset</h2><p class="small muted">Windows of the model type's length; a window takes the label covering at least the given share of it, unlabelled ones are skipped. Splits keep each labelled span together.</p>
      <div class="stack"><label class="f">Model type<select id="dt">${ts.map((t) => `<option value="${esc(t.id)}" ${t.id === rec.template_id ? "selected" : ""}>${esc(t.icon)} ${esc(t.name)} — ${esc((t.labels || []).join(", "))}</option>`).join("")}</select></label>
      <div class="grid g3"><label class="f">Window<input id="dw" type="number" placeholder="model type"></label><label class="f">Hop<input id="dh" type="number" placeholder="model type"></label><label class="f">Min. labelled<input id="dc" type="number" step="0.05" value="0.5"></label></div>
      <label class="f">Name<input id="dn" value="${esc(rec.name)} (labelled)"></label>
      <button class="btn primary" id="dgo">Create dataset</button><div id="dres" class="small"></div></div></div></div>
  <div class="card" style="margin-top:16px"><h2>Spans</h2><div id="spans"></div></div>`;

  const drawLabels = () => {
    const labs = rec.labels || [];
    $("#labs").innerHTML = labs.map((l) => `<button class="btn sm ${l === cur ? "primary" : ""}" data-l="${esc(l)}" style="${l === cur ? "" : `border-color:${labelColor(labs, l)};color:${labelColor(labs, l)}`}">${esc(l)}</button>`).join("") || `<span class="small muted">add your classes →</span>`;
    $$("#labs [data-l]").forEach((b) => (b.onclick = async () => { cur = b.dataset.l; drawLabels(); if (live()) await mark(cur); }));
  };
  const drawSpans = () => {
    const sp = rec.spans || [];
    $("#spans").innerHTML = sp.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Label</th><th class="n">Start (s)</th><th class="n">End (s)</th><th class="n">Length (s)</th><th>By</th><th></th></tr></thead><tbody>${sp.map((s, i) => `<tr><td><span class="badge" style="background:${labelColor(rec.labels, s.label)}22;color:${labelColor(rec.labels, s.label)}">${esc(s.label)}</span></td><td class="n">${fmt(s.start / rec.sample_rate, 3)}</td><td class="n">${fmt(s.end / rec.sample_rate, 3)}</td><td class="n">${fmt((s.end - s.start) / rec.sample_rate, 3)}</td><td class="small muted">${s.by ? "model " + esc(s.by) : "you"}</td><td><button class="btn sm" data-go="${i}">Show</button></td></tr>`).join("")}</tbody></table></div>` : `<div class="empty small">No labels yet.</div>`;
    $$("[data-go]").forEach((b) => (b.onclick = () => { const s = sp[+b.dataset.go]; const pad = Math.round((s.end - s.start) * 0.2); win = [Math.max(0, s.start - pad), s.end + pad]; sel = [s.start, s.end]; load(); }));
  };
  const drawPlot = () => {
    if (!view) return;
    const v = suggestion ? { ...view, labels: rec.labels, spans: suggestion.spans.filter((s) => s.end > view.start && s.start < view.end) } : view;
    $("#plot").innerHTML = signalPlot(v, { sel, hidden }) + (suggestion ? `<div class="small muted">Showing the model's suggestions: accept to keep them, discard to go back to your labels</div>` : "");
    $("#chs").innerHTML = rec.channels.map((c, k) => `<span style="cursor:pointer;opacity:${hidden.includes(k) ? .35 : 1}" data-ch="${k}"><i style="background:var(--c${(k % 3) + 1})"></i>${esc(c)}</span>`).join("");
    $$("[data-ch]").forEach((s) => (s.onclick = () => { const k = +s.dataset.ch; hidden = hidden.includes(k) ? hidden.filter((x) => x !== k) : [...hidden, k]; drawPlot(); }));
    $("#selinfo").textContent = sel ? `selection ${fmt(Math.min(...sel) / rec.sample_rate, 3)}–${fmt(Math.max(...sel) / rec.sample_rate, 3)} s (${fmt(Math.abs(sel[1] - sel[0]))} samples)` : `${fmt(view.start / rec.sample_rate, 2)}–${fmt(view.end / rec.sample_rate, 2)} s of ${fmt(view.samples / rec.sample_rate, 2)} s`;
  };
  const load = async () => {
    const q = live() && !win ? `tail=${Math.round(rec.sample_rate * 10)}` : `start=${win ? win[0] : 0}${win ? `&end=${win[1]}` : ""}`;
    view = await api(`/api/recordings/${rid}/view?points=1000&${q}`);
    drawPlot();
  };
  const refresh = async () => { rec = await api(`/api/recordings/${rid}`); drawLabels(); drawSpans(); await load(); };
  const mark = async (label) => { rec = await api(`/api/recordings/${rid}/mark`, { json: { label } }); drawCap(); drawSpans(); load(); };
  const drawCap = () => {
    if (!live()) { $("#capbox").innerHTML = ""; return; }
    const host = location.origin;
    $("#capbox").innerHTML = `<div class="card" style="margin-bottom:16px;border-color:var(--warn)"><div class="row"><h2 style="margin:0">● Capturing</h2><span class="spacer"></span><span class="mono" id="capn">${fmt(view ? view.samples : 0)} samples</span><button class="btn" id="unmark">Unlabelled</button><button class="btn primary" id="stop">Stop capture</button></div>
      <p class="small muted" style="margin-top:6px">Click a label above to mark what happens from now on. ${rec.live && rec.live.label ? `Marking <b>${esc(rec.live.label)}</b>.` : "Nothing marked."}</p>
      ${tok ? `<details><summary class="small">Device connection</summary><pre class="code small">python tools/capture_client.py --server ${esc(host)} --id ${esc(rid)} --token ${esc(tok)} --serial /dev/ttyACM0
# or: curl -X POST ${esc(host)}/api/capture/${esc(rid)}/push -H "X-Capture-Token: ${esc(tok)}" -H "Content-Type: text/plain" --data-binary @samples.csv</pre><p class="small muted">The token is shown only in this browser tab; it stops working when the capture stops.</p></details>` : ""}</div>`;
    $("#stop").onclick = () => busy($("#stop"), "Stopping…", async () => { rec = await api(`/api/recordings/${rid}/stop`, { method: "POST" }); sessionStorage.removeItem(`captok_${rid}`); clearInterval(timer); toast("Capture stopped: label and build a dataset"); win = null; drawCap(); refresh(); });
    $("#unmark").onclick = () => mark(null);
  };
  // selection by drag
  let dragging = false;
  const toSample = (e) => { const b = $("#plot svg").getBoundingClientRect(); const f = Math.min(1, Math.max(0, (e.clientX - b.left - b.width * 8 / 1000) / (b.width * 984 / 1000))); return Math.round(view.start + f * (view.end - view.start)); };
  $("#plot").onmousedown = (e) => { if (!view) return; dragging = true; sel = [toSample(e), toSample(e)]; drawPlot(); };
  $("#plot").onmousemove = (e) => { if (!dragging) return; sel[1] = toSample(e); drawPlot(); };
  window.addEventListener("mouseup", () => { dragging = false; if (sel && Math.abs(sel[1] - sel[0]) < 2) { sel = null; drawPlot(); } });
  const span = () => (sel ? [Math.min(...sel), Math.max(...sel)] : null);
  $("#zin").onclick = () => { if (span()) { win = span(); sel = null; load(); } };
  $("#zout").onclick = () => { if (!view) return; const c = (view.start + view.end) / 2, w = (view.end - view.start) * 2; win = [Math.max(0, Math.round(c - w / 2)), Math.round(c + w / 2)]; load(); };
  $("#zall").onclick = () => { win = [0, view ? view.samples : 0]; load(); };
  const pan = (d) => { if (!view) return; const w = view.end - view.start, a = Math.max(0, Math.min(view.samples - w, view.start + d * w / 2)); win = [Math.round(a), Math.round(a + w)]; load(); };
  $("#pl").onclick = () => pan(-1); $("#pr").onclick = () => pan(1);
  $("#addlab").onclick = async () => { const l = $("#newlab").value.trim(); if (!l) return; try { rec = await api(`/api/recordings/${rid}/spans`, { method: "PUT", json: { spans: [], labels: [...(rec.labels || []), l] } }); cur = l; $("#newlab").value = ""; drawLabels(); } catch (e) { toast(e.message); } };
  const paint = (label) => busy(label ? $("#apply") : $("#erase"), "Saving…", async () => {
    if (!span()) return toast("Drag on the plot to select a range first");
    if (label === undefined) { if (!cur) return toast("Add or pick a label first"); label = cur; }
    rec = await api(`/api/recordings/${rid}/spans`, { method: "PUT", json: { spans: [{ start: span()[0], end: span()[1], label }] } });
    sel = null; drawLabels(); drawSpans(); load();
  });
  $("#apply").onclick = () => paint(undefined);
  $("#erase").onclick = () => paint("");
  $("#sc").oninput = () => { $("#scv").textContent = (+$("#sc").value).toFixed(2); };
  $("#sgo").onclick = () => busy($("#sgo"), "Running the model…", async () => {
    if (!$("#sj").value) return toast("No finished project to suggest with");
    suggestion = await api(`/api/recordings/${rid}/suggest`, { json: { job_id: $("#sj").value, min_conf: +$("#sc").value, start: view ? view.start : 0, end: view ? view.end : null } });
    const labs = [...new Set([...(rec.labels || []), ...suggestion.labels])]; rec.labels = labs;
    $("#sres").innerHTML = `${suggestion.spans.length} span(s) from ${fmt(suggestion.windows)} windows of ${suggestion.window} samples. <span class="muted">${esc(suggestion.note)}</span>`;
    $("#sacc").disabled = $("#sdis").disabled = !suggestion.spans.length; drawPlot();
  });
  $("#sacc").onclick = () => busy($("#sacc"), "Saving…", async () => { rec = await api(`/api/recordings/${rid}/spans`, { method: "PUT", json: { spans: suggestion.spans } }); suggestion = null; $("#sacc").disabled = $("#sdis").disabled = true; $("#sres").textContent = "Accepted."; drawLabels(); drawSpans(); load(); });
  $("#sdis").onclick = () => { suggestion = null; $("#sacc").disabled = $("#sdis").disabled = true; $("#sres").textContent = ""; drawPlot(); };
  $("#dgo").onclick = () => busy($("#dgo"), "Cutting windows…", async () => {
    const d = await api("/api/recordings/dataset", { json: { recording_ids: [rid], template_id: $("#dt").value, name: $("#dn").value, window: +$("#dw").value || null, hop: +$("#dh").value || null, min_cover: +$("#dc").value || 0.5 } });
    $("#dres").innerHTML = `<div class="callout ok" style="margin-top:8px">Dataset <a href="#/data/${esc(d.id)}">${esc(d.name)}</a>: ${fmt(d.n)} windows (${fmt(d.unlabelled_windows_skipped)} unlabelled skipped).</div>`;
  });
  $("#rdel").onclick = () => { if (!confirm(`Delete recording "${rec.name}"? Datasets made from it are kept.`)) return; return busy($("#rdel"), "Deleting…", async () => { await api(`/api/recordings/${rid}`, { method: "DELETE" }); location.hash = "#/recordings"; }); };
  drawLabels(); drawSpans(); await load(); drawCap();
  if (live()) {
    timer = setInterval(async () => { if (!win) { try { await load(); if ($("#capn")) $("#capn").textContent = `${fmt(view.samples)} samples`; } catch (e) { /* page left */ } } }, 1000);
    onLeave(() => clearInterval(timer));
  }
});

route(/^\/connectors$/, async (el) => {
  const [r, cat] = await Promise.all([api("/api/connectors"), api("/api/catalog")]);
  let kind = "s3", openId = null;
  el.innerHTML = `<div class="page-h"><div><h1>Data</h1><p class="muted">Connect the bucket or database where your sensor data already lives, browse it and import files or query results.</p></div></div>${dataTabs("#/connectors")}
  ${r.private_endpoints ? "" : `<div class="callout info" style="margin-bottom:12px">This hosted platform connects to public endpoints only. Data inside your network (MinIO, an internal database) needs the platform installed in that network (single-customer mode) or a customer-VPC worker.</div>`}
  <div class="grid g2"><div class="card"><h2>Connections</h2><div id="clist"></div></div>
    <div class="card"><h2>New connection</h2><div class="tabs" id="ktabs">${Object.entries(r.kinds).map(([k, v]) => `<button class="tab ${k === kind ? "on" : ""}" data-k="${k}">${esc({ s3: "S3 / GCS", azure: "Azure Blob", influxdb: "InfluxDB", postgres: "PostgreSQL" }[k] || k)}</button>`).join("")}</div><div id="kform"></div></div></div>
  <div class="card" style="margin-top:16px" id="cdetail" hidden></div>`;
  const drawList = () => { $("#clist").innerHTML = r.connections.length ? `<table class="t"><tbody>${r.connections.map((c) => `<tr><td><b>${esc(c.name)}</b>${c.config.via_worker ? ` <span class="badge b-info">VPC worker</span>` : ""}<div class="small muted">${esc(r.kinds[c.kind].label)} · ${esc(c.config.bucket || c.config.container || c.config.database || c.config.org || "")}</div></td><td class="small muted">secrets: ${c.secrets_set.length ? esc(c.secrets_set.join(", ")) + " set" : "none"}</td><td><button class="btn sm" data-open="${esc(c.id)}">Open</button></td></tr>`).join("")}</tbody></table>` : `<div class="empty small">No connections yet.</div>`; $$("[data-open]").forEach((b) => (b.onclick = () => openConn(b.dataset.open))); };
  const drawForm = () => {
    const k = r.kinds[kind];
    $("#kform").innerHTML = `<div class="stack"><label class="f">Name<input id="kn" value="${esc(k.label.split(" (")[0])}"></label>${k.fields.map(([f, l, d]) => typeof d === "boolean" ? `<label class="row small"><input type="checkbox" data-f="${f}" ${d ? "checked" : ""}> ${esc(l)}</label>` : `<label class="f">${esc(l)}<input data-f="${f}" value="${esc(d)}"></label>`).join("")}
      <label class="row small"><input type="checkbox" data-f="via_worker" id="kvw"> Through a customer-VPC worker (private endpoint; the credentials stay on the worker)</label>
      <div id="ksec">${k.secrets.map(([f, l]) => `<label class="f">${esc(l)}<input type="password" data-s="${f}" autocomplete="off"></label>`).join("")}<p class="small muted">Secrets are encrypted at rest and never shown again.</p></div><button class="btn primary" id="ksave">Save connection</button></div>`;
    $("#kvw").onchange = () => { $("#ksec").hidden = $("#kvw").checked; };
    $("#ksave").onclick = () => busy($("#ksave"), "Saving…", async () => {
      const config = {}, secrets = {};
      $$("#kform [data-f]").forEach((i) => (config[i.dataset.f] = i.type === "checkbox" ? i.checked : i.value));
      $$("#kform [data-s]").forEach((i) => (secrets[i.dataset.s] = i.value));
      const c = await api("/api/connectors", { json: { kind, name: $("#kn").value, config, secrets } });
      r.connections.unshift(c); drawList(); openConn(c.id);
    });
  };
  $$("#ktabs [data-k]").forEach((b) => (b.onclick = () => { kind = b.dataset.k; $$("#ktabs .tab").forEach((x) => x.classList.toggle("on", x === b)); drawForm(); }));
  const ts = cat.filter((t) => t.input && t.input.kind !== "image");
  const settle = async (out, label) => {
    if (!out || !out.pending) return out;
    toast(`${label}: waiting for your VPC worker${out.workers_online ? "" : " (none online)"}`);
    for (let i = 0; i < 120; i++) {
      await new Promise((res) => setTimeout(res, 2000));
      const t = await api(`/api/vpc/tasks/${out.task}`);
      if (t.status === "succeeded") return t.result;
      if (t.status === "failed") throw new Error(`VPC worker: ${t.error}`);
    }
    throw new Error("the VPC worker did not answer in 4 minutes");
  };
  const openConn = (cid) => {
    openId = cid;
    const c = r.connections.find((x) => x.id === cid);
    const store = c.kind === "s3" || c.kind === "azure";
    const box = $("#cdetail"); box.hidden = false;
    box.innerHTML = `<div class="row"><h2 style="margin:0">${esc(c.name)}</h2><span class="small muted">${esc(r.kinds[c.kind].label)}</span><span class="spacer"></span><button class="btn sm" id="ctest">Test</button><button class="btn sm" id="cdel">Delete</button></div><div id="ctres" class="small" style="margin-top:6px"></div>
      <div class="grid g2" style="margin-top:12px"><div>${store ? `<div class="row"><input id="cpre" placeholder="prefix (folder)" style="flex:1"><button class="btn sm" id="cbr">Browse</button></div><div id="citems" style="margin-top:8px;max-height:320px;overflow:auto"></div>`
        : `<label class="f">${c.kind === "influxdb" ? "Flux query (pivot fields into columns)" : "SELECT query (read-only)"}<textarea id="cq" rows="7" class="mono small">${esc(c.kind === "influxdb" ? `from(bucket: "${c.config.bucket || "sensors"}")\n  |> range(start: -1h)\n  |> filter(fn: (r) => r._measurement == "vibration")\n  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")\n  |> keep(columns: ["_time", "acc_x", "acc_y", "acc_z"])` : "SELECT ts AS time, acc_x, acc_y, acc_z, label\nFROM vibration\nWHERE machine = 'pump-3'\nORDER BY ts\nLIMIT 200000")}</textarea></label>`}</div>
        <div class="stack"><div class="small"><b>Import as</b> <label><input type="radio" name="cas" value="dataset" checked> dataset (needs a label column)</label> <label><input type="radio" name="cas" value="recording"> recording to label</label></div>
          <label class="f">Model type<select id="cimt">${ts.map((t) => `<option value="${esc(t.id)}">${esc(t.icon)} ${esc(t.name)}</option>`).join("")}</select></label>
          <div class="grid g2"><label class="f">Licence<select id="cil">${["proprietary", "cc-by-4.0", "cc0", "research-only", "unknown"].map((l) => `<option>${l}</option>`).join("")}</select></label><label class="f">Name<input id="cin" placeholder="from the object"></label></div>
          <div><b class="small">Selected:</b> <span id="csel" class="mono small">${store ? "click an object" : "the query"}</span></div>
          <button class="btn primary" id="cimp">Import</button><div id="cires" class="small"></div></div></div>`;
    let key = null;
    $("#ctest").onclick = () => busy($("#ctest"), "Testing…", async () => { const t = await api(`/api/connectors/${cid}/test`, { method: "POST" }); $("#ctres").innerHTML = `<span class="badge ${t.ok ? "b-ok" : "b-bad"}">${t.ok ? "ok" : "failed"}</span> ${esc(t.detail)}`; });
    $("#cdel").onclick = () => { if (!confirm(`Delete connection "${c.name}"?`)) return; busy($("#cdel"), "…", async () => { await api(`/api/connectors/${cid}`, { method: "DELETE" }); r.connections = r.connections.filter((x) => x.id !== cid); box.hidden = true; drawList(); }); };
    if (store) $("#cbr").onclick = () => busy($("#cbr"), "Listing…", async () => {
      const { items } = await settle(await api(`/api/connectors/${cid}/browse?prefix=${encodeURIComponent($("#cpre").value)}`), "Listing");
      $("#citems").innerHTML = items.length ? `<table class="t"><tbody>${items.map((it, i) => `<tr data-i="${i}" style="cursor:pointer"><td class="mono small">${esc(it.key)}</td><td class="n small">${it.size != null ? bytes(it.size) : ""}</td></tr>`).join("")}</tbody></table>` : `<div class="empty small">Nothing under this prefix.</div>`;
      $$("#citems [data-i]").forEach((tr) => (tr.onclick = () => { key = items[+tr.dataset.i].key; $("#csel").textContent = key; $$("#citems tr").forEach((x) => x.classList.toggle("sel", x === tr)); }));
    });
    $("#cimp").onclick = () => busy($("#cimp"), "Importing…", async () => {
      const as = $$('input[name="cas"]').find((x) => x.checked).value;
      if (store && !key) return toast("Pick an object first");
      const body = { as, template_id: $("#cimt").value, licence: $("#cil").value, name: $("#cin").value || null, ...(store ? { key } : { query: $("#cq").value }) };
      const out = await settle(await api(`/api/connectors/${cid}/import`, { json: body }), "Import");
      $("#cires").innerHTML = out.dataset ? `<div class="callout ok">Dataset <a href="#/data/${esc(out.dataset.id)}">${esc(out.dataset.name)}</a>: ${fmt(out.dataset.n)} windows.</div>` : `<div class="callout ok">Recording <a href="#/recording/${esc(out.recording.id)}">${esc(out.recording.name)}</a> ready to label.</div>`;
    });
  };
  drawList(); drawForm();
});

route(/^\/physics$/, async (el) => {
  const [models, cat, dsets] = await Promise.all([api("/api/physics/models"), api("/api/catalog"), api("/api/datasets")]);
  const ts = cat.filter((t) => models[t.synth] && t.input.kind === "timeseries" && !t.frontend);
  let tpl = ts[0];
  el.innerHTML = `<div class="page-h"><div><h1>Data</h1><p class="muted">Simulate fault signatures from <b>your machine's</b> parameters, to fill classes you have little data for, or to augment real recordings with physically plausible variations.</p></div></div>${dataTabs("#/physics")}
  <div class="grid g2"><div class="card"><h2>Machine</h2><div class="stack"><label class="f">Model type<select id="pt">${ts.map((t) => `<option value="${esc(t.id)}">${esc(t.icon)} ${esc(t.name)}</option>`).join("")}</select></label><div id="pp" class="grid g2"></div></div></div>
    <div class="card"><h2>What the physics gives</h2><div id="pf"></div><div id="pex" style="margin-top:8px"></div></div></div>
  <div class="card" style="margin-top:16px"><h2>Generate</h2><div class="grid g3"><label class="f">Windows per class<input id="pn" type="number" value="200"></label>
      <label class="f">Mix with my dataset<select id="pm"><option value="">no: simulated only (licence: synthetic)</option></select></label><label class="f">Name<input id="pname" placeholder="automatic"></label></div>
    <div id="paug" class="grid g3" style="margin-top:8px" hidden><label class="f">Sensor gain error ±%<input id="ag" type="number" value="5"></label><label class="f">Speed change ±%<input id="as" type="number" value="3"></label><label class="f">Added noise SNR (dB)<input id="an" type="number" value="30"></label></div>
    <div class="row" style="margin-top:10px"><label class="small" id="paugl" hidden><input type="checkbox" id="pa" checked> also add one physical variant of every real window</label><span class="spacer"></span><button class="btn primary" id="pgo">Create dataset</button></div><div id="pres" style="margin-top:8px"></div></div>`;
  const params = () => Object.fromEntries($$("#pp [data-p]").map((i) => [i.dataset.p, +i.value]));
  let deb = null;
  const preview = async () => {
    try {
      const r = await api("/api/datasets/physics", { json: { template_id: tpl.id, params: params(), preview: true } });
      const f = r.frequencies;
      $("#pf").innerHTML = `<table class="t"><tbody>${Object.entries(f).map(([k, v]) => `<tr><td class="muted">${esc({ shaft: "Shaft rotation", ftf: "Cage (FTF)", bpfo: "Outer-race defect (BPFO)", bpfi: "Inner-race defect (BPFI)", bsf: "Ball spin (BSF)", supply: "Supply", broken_bar: "Broken-bar sidebands (1±2s)f", eccentricity: "Eccentricity sidebands", switching: "Switching" }[k] || k)}</td><td class="n mono">${Array.isArray(v) ? v.map((x) => fmt(x, 2)).join(" / ") : fmt(v, 2)} Hz</td></tr>`).join("")}</tbody></table>`;
      $("#pex").innerHTML = `<div class="grid g2">${r.examples.map((e) => `<div class="card" style="padding:8px"><b class="small">${esc(e.label)}</b>${sparkSeries(e.series.slice(0, 3), 300, 70)}</div>`).join("")}</div>`;
    } catch (e) { if (!(e instanceof Stale)) $("#pf").innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; }
  };
  const drawParams = () => {
    tpl = ts.find((t) => t.id === $("#pt").value) || ts[0];
    const m = models[tpl.synth];
    $("#pp").innerHTML = m.params.map((p) => `<label class="f">${esc(p.label)}${p.unit ? ` (${esc(p.unit)})` : ""}<input type="number" data-p="${p.key}" value="${p.default}" min="${p.min}" max="${p.max}" step="${p.step || "any"}"></label>`).join("");
    $$("#pp [data-p]").forEach((i) => (i.oninput = () => { clearTimeout(deb); deb = setTimeout(preview, 300); }));
    $("#pm").innerHTML = `<option value="">no: simulated only (licence: synthetic)</option>` + dsets.filter((d) => d.template === tpl.id).map((d) => `<option value="${esc(d.id)}">${esc(d.name)} (${fmt(d.n)} windows)</option>`).join("");
    preview();
  };
  $("#pm").onchange = () => { const on = !!$("#pm").value; $("#paug").hidden = $("#paugl").hidden = !on; };
  $("#pt").onchange = drawParams;
  $("#pgo").onclick = () => busy($("#pgo"), "Simulating…", async () => {
    const mix = $("#pm").value || null;
    const d = await api("/api/datasets/physics", { json: { template_id: tpl.id, params: params(), per_class: +$("#pn").value, mix_with_dataset_id: mix, name: $("#pname").value || null, augment: mix && $("#pa").checked ? { gain_pct: +$("#ag").value, speed_pct: +$("#as").value, snr_db: +$("#an").value } : null } });
    $("#pres").innerHTML = `<div class="callout ok">Dataset <a href="#/data/${esc(d.id)}">${esc(d.name)}</a>: ${fmt(d.n)} windows${d.physics_windows ? `, ${fmt(d.physics_windows)} simulated` : ""} · licence ${esc(d.licence)}.</div>`;
  });
  if (!ts.length) { $("#pp").innerHTML = `<div class="empty small">No model type with a physics model.</div>`; return; }
  drawParams();
});

/* ------------------------------------------------------------------------------------------------ plan & usage */
route(/^\/plan$/, async (el) => {
  const d = await api("/api/plan");
  const bar = (used, max) => max === null || max === undefined ? `<span class="small muted">${fmt(used)} used · unlimited</span>` : `<div class="row small"><div class="progress" style="flex:1;height:10px"><i style="width:${Math.min(100, used / Math.max(max, 1) * 100)}%"></i></div><span class="mono">${fmt(used)} / ${fmt(max)}</span></div>`;
  const yes = (v) => v ? `<span class="badge b-ok">✓</span>` : `<span class="muted">—</span>`;
  el.innerHTML = `<div class="page-h"><div><h1>Plan &amp; usage</h1><p class="muted">${esc(ME.org.name)} is on the <b>${esc(d.plan.label)}</b> plan.</p></div></div>
  <div class="grid g2"><div class="card"><h2>This month</h2><div class="stack">
      <div><div class="small muted">Projects</div>${bar(d.usage.projects, d.plan.projects_per_month)}</div>
      <div><div class="small muted">Board-farm runs</div>${bar(d.usage.farm_runs, d.plan.farm_runs_per_month)}</div></div>
      <h3 style="margin-top:14px">Metered (all time)</h3><p class="small">${Object.entries(d.metered).map(([k, v]) => `${esc(k)}: ${fmt(v.qty, 1)}`).join(" · ") || "–"}</p></div>
    <div class="card"><h2>Deployment rights</h2>
      ${d.deployment.model === "none" ? `<div class="callout">No deployment rights yet: packages are <b>evaluation only</b> (marker file, compile-time warning). Production deployment is licensed per unit (royalty) or as a buyout — contact your SAI representative.</div>` : `<div class="callout ok">Deployment licensed: <b>${esc(d.deployment.model)}</b>${d.deployment.units ? `, ${fmt(d.deployment.units)} units` : ""}. Packages trained on commercially licensed data are production packages.</div>`}
      ${ME.perms.org_admin ? `<h3 style="margin-top:14px">Branding</h3><div class="grid g3"><label class="f">Product name<input id="bn" value="${esc((ME.brand || {}).product_name || "")}"></label><label class="f">Accent colour<input id="ba" type="color" value="${esc((ME.brand || {}).accent || "#2f5bea")}"></label><label class="f">Logo letters<input id="bl" maxlength="2" value="${esc((ME.brand || {}).logo_text || "")}"></label></div><div class="row" style="margin-top:8px"><button class="btn" id="bs">Save branding</button></div>` : ""}</div></div>
  <div class="card" style="margin-top:16px"><h2>Plans</h2><div class="tbl-wrap"><table class="t"><thead><tr><th></th>${Object.values(d.plans).map((p) => `<th>${esc(p.label)}</th>`).join("")}</tr></thead><tbody>
    <tr><td class="muted">For</td>${Object.values(d.plans).map((p) => `<td class="small">${esc(p.summary)}</td>`).join("")}</tr>
    <tr><td class="muted">Projects / month</td>${Object.values(d.plans).map((p) => `<td>${p.projects_per_month === null ? "unlimited" : p.projects_per_month}</td>`).join("")}</tr>
    <tr><td class="muted">Package downloads</td>${Object.values(d.plans).map((p) => `<td>${yes(p.packages)}</td>`).join("")}</tr>
    <tr><td class="muted">Board farm</td>${Object.values(d.plans).map((p) => `<td>${p.farm === "none" ? "—" : p.farm === "emulated" ? "emulated boards" : "emulated + physical (measured)"}</td>`).join("")}</tr>
    <tr><td class="muted">Farm runs / month</td>${Object.values(d.plans).map((p) => `<td>${p.farm_runs_per_month || "—"}</td>`).join("")}</tr>
    <tr><td class="muted">Queue priority</td>${Object.values(d.plans).map((p) => `<td>${p.priority ? "priority" : "standard"}</td>`).join("")}</tr>
    <tr><td class="muted">Support</td>${Object.values(d.plans).map((p) => `<td>${esc(p.support)}</td>`).join("")}</tr>
  </tbody></table></div><p class="small muted" style="margin-top:8px">Prices and deployment royalties are quoted by SAI. Upgrades are applied by your SAI representative.</p></div>`;
  const bs = $("#bs");
  if (bs) bs.onclick = async () => { await api("/api/org/brand", { method: "PUT", json: { product_name: $("#bn").value, accent: $("#ba").value, logo_text: $("#bl").value } }); toast("Branding saved"); boot(); };
});

/* ------------------------------------------------------------------------------------------------ admin */
route(/^\/admin$/, async (el) => {
  const m = ME;
  if (!m.perms.org_admin) { el.innerHTML = `<div class="callout bad">Only admins can open the admin console (your role: <b>${esc(m.role)}</b>).</div>`; return; }
  const tabs = [m.perms.org_admin && ["orgs", "Organisations"], m.perms.org_admin && ["users", "Users"], m.perms.farm_admin && ["agents", "Lab agents"], m.perms.org_admin && ["audit", "Audit log"], m.perms.org_admin && ["opgaps", "Operator gaps"], m.perms.org_admin && ["vpc", "VPC workers"], m.perms.platform_admin && ["queue", "Job queue"]].filter(Boolean);
  let tab = tabs[0][0];
  const draw = async () => {
    el.innerHTML = `<div class="page-h"><div><h1>Admin</h1><p class="muted">${m.mode === "single" ? "Single-organisation install" : "Multi-tenant"} · signed in as ${esc(m.email)} (${esc(m.role)})</p></div></div>
      <div class="tabs">${tabs.map(([k, t]) => `<button class="tab ${k === tab ? "on" : ""}" data-t="${k}">${t}</button>`).join("")}</div><div id="adm"></div>`;
    $$(".tabs .tab", el).forEach((b) => (b.onclick = () => { tab = b.dataset.t; draw(); }));
    const box = $("#adm");
    if (tab === "vpc") {
      const ws = await api("/api/admin/vpc-workers");
      box.innerHTML = `<div class="card"><h2>Customer-VPC workers</h2><p class="small muted">A worker runs inside your network, next to data the platform cannot reach (MinIO, an internal InfluxDB or PostgreSQL). It only makes outbound HTTPS calls to the platform; the credentials of your data sources stay in its local secrets file. Connections marked "through a VPC worker" are browsed and imported by it.</p>
        <div class="tbl-wrap"><table class="t"><thead><tr><th>Worker</th><th>Host</th><th>Last seen</th><th>Status</th><th></th></tr></thead><tbody>${ws.map((w) => `<tr><td><b>${esc(w.name)}</b><div class="mono small muted">${esc(w.id)}</div></td><td class="small">${esc(w.info.host || "")}${(w.info.connections_with_secrets || []).length ? `<div class="muted">secrets for: ${esc(w.info.connections_with_secrets.join(", "))}</div>` : ""}</td><td class="muted">${w.last_seen ? ago(w.last_seen) : "never"}</td><td>${w.disabled ? `<span class="badge b-bad">disabled</span>` : w.online ? `<span class="badge b-ok">online</span>` : `<span class="badge">offline</span>`}</td><td>${w.disabled ? "" : `<button class="btn sm ghost" data-wd="${esc(w.id)}">Disable</button>`}</td></tr>`).join("") || `<tr><td colspan="5" class="empty">No workers yet</td></tr>`}</tbody></table></div>
        <div class="row" style="margin-top:12px"><input id="wn" placeholder="e.g. Plant 3 data gateway"><button class="btn primary" id="wc">Create worker token</button></div><div id="wtok"></div></div>`;
      $$("[data-wd]", box).forEach((b) => (b.onclick = async () => { if (!confirm("Disable this worker? Its token stops working.")) return; await api(`/api/admin/vpc-workers/${b.dataset.wd}/disable`, { method: "POST" }); draw(); }));
      $("#wc").onclick = () => busy($("#wc"), "Creating…", async () => { const w = await api("/api/admin/vpc-workers", { json: { name: $("#wn").value } }); $("#wtok").innerHTML = `<div class="callout ok" style="margin-top:10px">Token for <b>${esc(w.name)}</b> (copy now, it will not be shown again):<div class="log" style="margin-top:6px">export SAI_VPC_TOKEN=${esc(w.token)}
python tools/vpc_worker.py --server ${esc(location.origin)} --secrets vpc-secrets.json</div><p class="small muted">vpc-secrets.json maps each connection name to its secrets, e.g. {"plant MinIO": {"secret_key": "…"}}.</p></div>`; });
    }
    if (tab === "opgaps") {
      const g = await api("/api/admin/operator-gaps");
      box.innerHTML = `<div class="card"><h2>Models refused for a missing operator</h2><p class="small muted">Every time SAI, SAI's front end or the model editor refuses an operator, it is logged here${m.perms.studio || m.perms.platform_admin ? " for all organisations" : " for your organisation"}: what customers need that the compiler cannot lower yet. ${fmt(g.total)} refusal(s) in the last year.</p>
        ${g.operators.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Operator</th><th class="n">Refusals</th><th class="n">Organisations</th><th>Targets</th><th>Where</th><th>Seen with</th><th>Last</th></tr></thead><tbody>${g.operators.map((o) => `<tr><td><b class="mono">${esc(o.op)}</b><div class="small muted">${esc(o.capability || "")}</div></td><td class="n">${fmt(o.count)}</td><td class="n">${fmt(o.orgs)}</td><td class="small">${Object.entries(o.targets).map(([k, v]) => `${esc(k)} ×${v}`).join(", ")}</td><td class="small">${Object.entries(o.sources).map(([k, v]) => `${esc(k)} ×${v}`).join(", ")}</td><td class="small mono">${esc(o.with.join(", "))}</td><td class="muted">${ago(o.last)}</td></tr>`).join("")}</tbody></table></div>
        <h3 class="small" style="margin-top:14px">Latest</h3><div class="tbl-wrap"><table class="t"><tbody>${g.recent.slice(0, 15).map((r) => `<tr><td class="mono">${esc(r.op)}</td><td>${esc(r.target || "")}</td><td>${esc(r.source)}${r.kind ? ` · ${esc(r.kind)}` : ""}</td><td>${r.job_id ? `<a href="#/job/${esc(r.job_id)}">${esc(r.job_id)}</a>` : ""}</td><td class="small">${esc(r.reason)}</td><td class="muted">${ago(r.created)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="empty">No refusals logged.</div>`}</div>`;
    }
    if (tab === "orgs") {
      const orgs = await api("/api/admin/orgs");
      box.innerHTML = `<div class="card"><div class="tbl-wrap"><table class="t"><thead><tr><th>Organisation</th><th>Tier</th><th>Deployment rights</th><th class="n">Users</th><th>Usage (all time)</th><th>Created</th></tr></thead><tbody>
        ${orgs.map((o) => `<tr><td><b>${esc(o.name)}</b><div class="small muted mono">${esc(o.id)}</div></td><td>${m.perms.platform_admin ? `<select data-org="${esc(o.id)}">${["free", "standard", "professional"].map((t) => `<option ${t === o.tier ? "selected" : ""}>${t}</option>`).join("")}</select>` : esc(o.tier)}</td>
          <td>${m.perms.platform_admin ? `<select data-dep="${esc(o.id)}">${["none", "royalty", "buyout"].map((d) => `<option ${d === o.deployment.model ? "selected" : ""}>${d}</option>`).join("")}</select>` : esc(o.deployment.model)}</td><td class="n">${o.users}</td>
          <td class="small">${Object.entries(o.usage).map(([k, v]) => `${esc(k)}: ${fmt(v.qty, 1)}`).join(" · ") || "–"}</td><td class="muted">${ago(o.created)}</td></tr>`).join("")}</tbody></table></div>
        ${m.perms.platform_admin && m.mode !== "single" ? `<div class="row" style="margin-top:12px"><input id="on" placeholder="New customer organisation"><select id="ot"><option>free</option><option>standard</option><option selected>professional</option></select><button class="btn primary" id="oc">Create</button></div>` : ""}</div>`;
      $$("select[data-org]", box).forEach((x) => (x.onchange = async () => { await api(`/api/admin/orgs/${x.dataset.org}/tier`, { json: { tier: x.value } }); toast("Tier updated"); }));
      $$("select[data-dep]", box).forEach((x) => (x.onchange = async () => { await api(`/api/admin/orgs/${x.dataset.dep}/deployment`, { json: { model: x.value } }); toast("Deployment rights updated"); }));
      if ($("#oc")) $("#oc").onclick = async () => { try { await api("/api/admin/orgs", { json: { name: $("#on").value, tier: $("#ot").value } }); toast("Organisation created"); draw(); } catch (e) { toast(e.message); } };
    }
    if (tab === "users") {
      const [us, orgs] = await Promise.all([api("/api/admin/users"), api("/api/admin/orgs")]);
      const on = Object.fromEntries(orgs.map((o) => [o.id, o.name]));
      const roles = ["engineer", "researcher", "manager", "admin"];
      box.innerHTML = `<div class="card"><div class="tbl-wrap"><table class="t"><thead><tr><th>User</th><th>Organisation</th><th>Role</th><th>Status</th><th></th></tr></thead><tbody>
        ${us.map((x) => `<tr><td><b>${esc(x.name)}</b><div class="small muted">${esc(x.email)}</div></td><td>${esc(on[x.org_id] || x.org_id)}</td><td><span class="badge">${esc(x.role)}</span></td><td>${x.disabled ? `<span class="badge b-bad">disabled</span>` : `<span class="badge b-ok">active</span>`}</td><td>${!x.disabled && x.id !== m.id ? `<button class="btn sm ghost" data-dis="${x.id}">Disable</button>` : ""}</td></tr>`).join("")}</tbody></table></div>
        <h3 style="margin-top:14px">Invite a user</h3><div class="grid g4"><label class="f">Name<input id="un"></label><label class="f">Email<input id="ue" type="email"></label><label class="f">Initial password<input id="up" type="text" placeholder="≥ 8 characters"></label>
        <label class="f">Role<select id="ur">${roles.map((r) => `<option>${r}</option>`).join("")}</select></label></div>
        ${m.perms.platform_admin ? `<label class="f" style="margin-top:8px;max-width:320px">Organisation<select id="uo">${orgs.map((o) => `<option value="${esc(o.id)}">${esc(o.name)}</option>`).join("")}</select></label>` : ""}
        <div class="row" style="margin-top:10px"><button class="btn primary" id="uc">Create user</button></div></div>`;
      $$("[data-dis]", box).forEach((b) => (b.onclick = async () => { if (!confirm("Disable this user and sign them out?")) return; await api(`/api/admin/users/${b.dataset.dis}/disable`, { method: "POST" }); draw(); }));
      $("#uc").onclick = async () => { try { await api("/api/admin/users", { json: { name: $("#un").value, email: $("#ue").value, password: $("#up").value, role: $("#ur").value, org_id: $("#uo")?.value } }); toast("User created"); draw(); } catch (e) { toast(e.message); } };
    }
    if (tab === "agents") {
      const ag = await api("/api/admin/agents");
      box.innerHTML = `<div class="card"><p class="small muted">Each lab PC gets its own token. Tokens are shown once and stored hashed.</p><div class="tbl-wrap"><table class="t"><thead><tr><th>Agent</th><th>Created</th><th>Last seen</th><th>Status</th><th></th></tr></thead><tbody>
        ${ag.map((a) => `<tr><td><b>${esc(a.name)}</b><div class="mono small muted">${esc(a.id)}</div></td><td class="muted">${ago(a.created)}</td><td class="muted">${a.last_seen ? ago(a.last_seen) : "never"}</td><td>${a.disabled ? `<span class="badge b-bad">disabled</span>` : `<span class="badge b-ok">active</span>`}</td><td>${a.disabled ? "" : `<button class="btn sm ghost" data-ad="${a.id}">Revoke</button>`}</td></tr>`).join("") || `<tr><td colspan="5" class="empty">No agents yet</td></tr>`}</tbody></table></div>
        <div class="row" style="margin-top:12px"><input id="an" placeholder="e.g. Lab 1 - bench PC"><input id="apn" placeholder="partner (optional, for a partner-hosted lab)"><button class="btn primary" id="ac">Create agent token</button></div><div id="atok"></div></div>`;
      $$("[data-ad]", box).forEach((b) => (b.onclick = async () => { if (!confirm("Revoke this agent token? Its boards go offline.")) return; await api(`/api/admin/agents/${b.dataset.ad}/disable`, { method: "POST" }); draw(); }));
      $("#ac").onclick = async () => { const a = await api("/api/admin/agents", { json: { name: $("#an").value, partner: $("#apn").value } }); $("#atok").innerHTML = `<div class="callout ok" style="margin-top:10px">Token for <b>${esc(a.name)}</b> (copy now, it will not be shown again):<div class="log" style="margin-top:6px">${esc(a.token)}</div></div>`; };
    }
    if (tab === "audit") {
      const log = await api("/api/admin/audit");
      box.innerHTML = `<div class="card"><div class="tbl-wrap"><table class="t"><thead><tr><th>When</th><th>Who</th><th>Action</th><th>Target</th><th>Detail</th></tr></thead><tbody>
        ${log.map((x) => `<tr><td class="muted small">${new Date(x.ts * 1000).toLocaleString()}</td><td class="small">${esc(x.email || "–")}</td><td><span class="badge">${esc(x.action)}</span></td><td class="mono small">${esc(x.target)}</td><td class="mono small">${esc(x.detail ? JSON.stringify(x.detail) : "")}</td></tr>`).join("")}</tbody></table></div></div>`;
    }
    if (tab === "queue") {
      const q = await api("/api/admin/queue");
      const rt = q.retention;
      box.innerHTML = `<div class="card"><div class="kpis">${Object.entries(q.by_status).map(([k, v]) => `<div class="kpi"><div class="l">${esc(k)}</div><div class="v">${fmt(v)}</div></div>`).join("")}</div><p class="small muted" style="margin-top:10px">Busy workers: ${q.busy_workers.map(esc).join(", ") || "none"}${q.paused ? " · <b>paused</b>" : ""}</p></div>
        <div class="card" style="margin-top:16px"><h2>Storage retention</h2><p class="small">Finished projects keep their package, model, report and build. Intermediates are removed when a job ends; the hourly sweep removes uploads older than ${fmt(q.upload_ttl_days)} days, and board builds and failed/cancelled job folders older than ${fmt(q.retention_days)} days.</p>
        <p class="small muted">Last sweep: ${rt ? `${ago(rt.at)}, freed ${bytes(rt.freed_bytes)}` : "never"}</p><button class="btn" id="sweep">Clean up now</button><span id="sweepres" class="small" style="margin-left:10px"></span></div>`;
      $("#sweep").onclick = () => busy($("#sweep"), "Cleaning up…", async () => { const r = await api("/api/admin/retention", { method: "POST" }); $("#sweepres").textContent = `freed ${bytes(r.freed_bytes)} (${Object.entries(r.removed).map(([k, v]) => `${k}: ${v}`).join(", ") || "nothing to remove"})`; });
    }
  };
  draw();
});

/* ------------------------------------------------------------------------------------------------ login */
let ME = null;
function showLogin(msg) {
  leave();                                         // stop every poll and timer of the page we were on
  if ($("#lf") && !msg) return;                    // already showing the form: keep what the user typed
  document.body.classList.add("signed-out");
  $("#main").innerHTML = `<div class="login card"><div class="brand" style="padding:0 0 12px"><span class="logo">◆</span><span><b>SAI</b> Platform<small>Sign in to continue</small></span></div>
    <form id="lf" class="stack"><label class="f">Email<input id="le" type="email" autocomplete="username" required></label>
    <label class="f">Password<input id="lp" type="password" autocomplete="current-password" required></label>
    <div id="lerr">${msg ? `<div class="callout bad">${esc(msg)}</div>` : ""}</div><button class="btn primary" type="submit">Sign in</button>
    <p class="small muted">Accounts are created by your administrator.</p></form></div>`;
  $("#lf").onsubmit = async (e) => {
    e.preventDefault();
    try { await api("/api/auth/login", { json: { email: $("#le").value, password: $("#lp").value }, anyPage: true }); document.body.classList.remove("signed-out"); boot(); }
    catch (err) { if (!(err instanceof Stale)) $("#lerr").innerHTML = `<div class="callout bad">${esc(err.message)}</div>`; }
  };
}

async function boot() {
  try { ME = await api("/api/me"); } catch { return; }
  document.body.classList.remove("signed-out");
  $$("[data-perm]").forEach((a) => (a.hidden = !a.dataset.perm.split("|").some((p) => ME.perms[p])));
  const br = ME.brand || {};
  if (br.accent) { document.documentElement.style.setProperty("--accent", br.accent); }
  if (br.product_name) { document.title = br.product_name; const t = $(".brand span:last-child"); if (t) t.innerHTML = `<b>${esc(br.product_name)}</b><small>Edge AI for MCUs</small>`; }
  if (br.logo_text) { const l = $(".brand .logo"); if (l) l.textContent = br.logo_text; }
  $("#sideFoot").innerHTML = `<div><b>${esc(ME.name)}</b></div><div>${esc(ME.org.name)} · <span class="badge b-acc">${esc(ME.org.tier)}</span></div><div>${esc(ME.role)}</div><a href="#" id="signout">Sign out</a><div id="sysline" style="margin-top:6px"></div>`;
  $("#signout").onclick = async (e) => { e.preventDefault(); await api("/api/auth/logout", { method: "POST" }); SYS = null; history.replaceState(null, "", "#/"); showLogin(); };
  try { const s = await system(); $("#sysline").textContent = `SAI ${s.sai.sai || "?"} · platform ${s.platform_version}`; } catch { /* shown per page */ }
  render();
  refreshPending();
  if (!window.__pendTimer) window.__pendTimer = setInterval(refreshPending, 60000);
}

/* ------------------------------------------------------------------------------------------------ boot */
boot();
