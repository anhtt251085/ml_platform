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

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) { init.method = init.method || "POST"; init.headers = { "Content-Type": "application/json" }; init.body = JSON.stringify(opts.json); }
  const r = await fetch(path, init);
  if (r.status === 401 && !path.startsWith("/api/auth/")) { showLogin(); throw new Error("Please sign in"); }
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body && body.detail ? body.detail : `HTTP ${r.status}`);
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
function onLeave(fn) { cleanup.push(fn); }
async function render() {
  cleanup.forEach((f) => f()); cleanup = [];
  const hash = location.hash.slice(1) || "/";
  const base = hash.split("?")[0];
  const main = $("#main");
  for (const [re, fn] of routes) {
    const m = hash.match(re);
    if (m) {
      $$(".side nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + (base.split("/").slice(0, 2).join("/") || "/")));
      main.innerHTML = `<div class="empty">Loading…</div>`;
      try { await fn(main, ...m.slice(1)); } catch (e) { main.innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; }
      window.scrollTo(0, 0);
      return;
    }
  }
  main.innerHTML = `<div class="empty">Page not found</div>`;
}
window.addEventListener("hashchange", render);

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
  ${rows.map((p) => `<tr><td><a href="#/job/${p.job_id}">${esc(p.name)}</a></td><td>${esc(p.template_id)}</td><td>${esc(GOAL_LABEL[p.config.goal] || p.config.goal)}</td>
    <td>${statusBadge(p.job_status, p.job_progress)}</td><td class="muted">${ago(p.created)}</td></tr>`).join("")}</tbody></table></div>`;
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
  const sizes = Object.entries(t.sizes || {}).map(([k, v]) => v.error ? "" : `<span class="badge" title="${fmt(v.params)} params">${k} · ${fmt(v.macs / 1000, 0)}k MACs</span>`).join(" ");
  return `<div class="card tile ${sel ? "sel" : ""}" data-tid="${esc(t.id)}">
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
      ${Object.entries(t.variants).map(([k, layers]) => `<tr><td>${k}</td><td class="n">${fmt(t.sizes[k]?.params)}</td><td class="n">${fmt(t.sizes[k]?.macs)}</td><td class="mono small">${layers.map((l) => l.op + (l.out ? l.out : "") + (l.k ? "k" + l.k : "") + (l.stride > 1 ? "/s" + l.stride : "")).join(" → ")} → fc(head)</td></tr>`).join("")}
      </tbody></table></div></div>`;
    const bt = $("#basetrain");
    if (bt) bt.onclick = async () => { const j = await api("/api/studio/basetrain", { json: { template_id: bt.dataset.t } }); SYS = null; location.hash = `#/job/${j.id}`; };
    $("#detail").scrollIntoView({ behavior: "smooth" });
  }
});

/* ------------------------------------------------------------------------------------------------ wizard */
route(/^\/new(?:\?template=(.+))?$/, async (el, pre) => {
  SYS = null;
  const [cat, sys] = await Promise.all([api("/api/catalog"), system()]);
  const st = { step: pre ? 1 : 0, tpl: pre ? cat.find((t) => t.id === decodeURIComponent(pre)) : null, ds: null,
    targets: [], goal: "balanced", budgets: {}, metric: "", sizes: ["tiny", "small", "base"], epochs: 30, name: "" };
  if (st.tpl) st.targets = [...st.tpl.targets];
  const STEPS = ["Model type", "Data", "Board & goal", "Review"];
  const draw = () => {
    el.innerHTML = `<div class="page-h"><div><h1>New project</h1><p class="muted">Four steps. The platform does the rest.</p></div></div>
    <div class="steps">${STEPS.map((s, i) => `<div class="step ${i === st.step ? "on" : i < st.step ? "done" : ""}"><b>${i < st.step ? "✓" : i + 1}</b>${s}</div>`).join("")}</div>
    <div id="body"></div>`;
    [stepModel, stepData, stepTarget, stepReview][st.step]($("#body"));
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
      <div class="tabs"><button class="tab on" data-tab="up">Upload file</button><button class="tab" data-tab="url">From URL</button><button class="tab" data-tab="demo">Demo data</button><button class="tab" data-tab="soon">Live sources</button></div>
      <div id="tab"></div></div>
      <div class="card"><h2>${esc(t.icon)} ${esc(t.name)} – data contract</h2>
        ${ts ? `<p class="small"><b>CSV</b>: one row per sample with columns <code>${esc(csvCols)}</code>. Windows of <b>${t.window}</b> samples every <b>${t.hop}</b> are cut per segment; the window label is its majority label.</p>` : ""}
        <p class="small"><b>NPZ</b>: arrays <code>X</code> ${ts ? `[N, ${t.input.channels}, T]` : `[N, ${t.input.channels}, ${t.input.height}, ${t.input.width}]`} and <code>y</code> [N] ${t.task === "regression" ? "(numeric target)" : "(class index or name)"}; optional <code>labels</code>.</p>
        <p class="small muted">${esc(t.datasets_hint || "")}</p>
        <div id="dsinfo"></div></div></div>
      ${nav(true, true, !!st.ds)}`;
    const tabs = {
      up: () => `${licSelect()}<div class="drop" id="drop" style="margin-top:10px">Drop a <b>.csv</b>, <b>.npz</b>${t.frontend ? " or a <b>.zip</b> of label folders with WAV files" : ""} here, or click to choose<input type="file" id="file" accept=".csv,.npz,.txt,.zip" hidden></div><div id="mapbox"></div>
        ${ts ? `<div class="row" style="margin-top:10px"><label class="f">Window<input id="win" type="number" value="${t.window}" style="width:100px"></label><label class="f">Hop<input id="hop" type="number" value="${t.hop}" style="width:100px"></label></div>` : ""}`,
      url: () => `${licSelect()}<label class="f" style="margin-top:10px">HTTPS link to a .csv, .npz or .zip<input id="url" placeholder="https://…/recordings.csv"></label><div class="row" style="margin-top:10px"><button class="btn" id="fetch">Fetch</button></div>`,
      demo: () => `<p class="small">Generate a <b>synthetic</b> dataset with the physics of this task, to try the whole pipeline now. Results on synthetic data say nothing about field accuracy.</p>
        <label class="f">Windows per class<input id="per" type="number" value="300" min="30" max="2000" style="width:120px"></label><div class="row" style="margin-top:10px"><button class="btn" id="gen" ${t.synth ? "" : "disabled"}>Generate demo data</button></div>`,
      soon: () => `<p class="small">Planned connectors (see the implementation plan):</p><ul class="small" style="padding-left:18px"><li>Board-farm data logger: capture straight from your sensor board</li><li>MQTT / OPC-UA live streams with labelling UI</li><li>S3 / GCS / Azure Blob buckets</li><li>Import from Edge Impulse projects or Kaggle datasets</li></ul>`,
    };
    const setTab = (k) => {
      $$(".tab", b).forEach((x) => x.classList.toggle("on", x.dataset.tab === k)); $("#tab").innerHTML = tabs[k]();
      if (k === "up") {
        const d = $("#drop"), f = $("#file");
        d.onclick = () => f.click(); f.onchange = () => f.files[0] && upload(f.files[0]);
        d.ondragover = (e) => { e.preventDefault(); d.classList.add("hover"); }; d.ondragleave = () => d.classList.remove("hover");
        d.ondrop = (e) => { e.preventDefault(); d.classList.remove("hover"); e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]); };
      }
      if (k === "url") $("#fetch").onclick = () => doIngest(() => api("/api/datasets/url", { json: { template_id: t.id, url: $("#url").value, licence: $("#lic").value, window: +($("#win")?.value || 0) || null } }));
      if (k === "demo") $("#gen").onclick = () => doIngest(() => api("/api/datasets/demo", { json: { template_id: t.id, per_class: +$("#per").value } }));
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
      const opt = (sel) => `<option value="">—</option>${pv.header.map((h) => `<option ${h === sel ? "selected" : ""}>${esc(h)}</option>`).join("")}`;
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
      $("#dsinfo").innerHTML = `<div class="callout info" style="margin-top:12px">Ingesting…</div>`;
      try { st.ds = await fn(); toast("Dataset ready"); stepData(b); }
      catch (e) { $("#dsinfo").innerHTML = `<div class="callout bad" style="margin-top:12px">${esc(e.message)}</div>`; }
    }
    if (st.ds) $("#dsinfo").innerHTML = datasetSummary(st.ds);
    wire();
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
      <h3 style="margin-top:14px">Model sizes to try</h3><div class="row">${["tiny", "small", "base"].map((s) => `<label class="check"><input type="checkbox" name="sz" value="${s}" ${st.sizes.includes(s) ? "checked" : ""}>${s} <span class="muted small">${fmt(st.tpl.sizes[s]?.macs / 1000)}k MACs</span></label>`).join("")}</div>
      </div></div>${nav(true, true)}`;
    const help = { balanced: "Best mix of latency, RAM and flash among plans that fit.", fastest: "Lowest latency plan that fits.", "min-ram": "Smallest RAM (workspace + stack): leaves the most memory for your application.", "min-flash": "Smallest code + weights.", accuracy: "Most accurate size that fits, then balanced." };
    const g = () => { $("#goalhelp").textContent = help[st.goal]; };
    g();
    $("#goals").onclick = (e) => { if (e.target.dataset.g) { st.goal = e.target.dataset.g; $$("#goals .chip").forEach((c) => c.classList.toggle("on", c.dataset.g === st.goal)); g(); } };
    const save = () => {
      st.targets = $$("input[name=tgt]:checked").map((x) => x.value); st.sizes = $$("input[name=sz]:checked").map((x) => x.value);
      st.budgets = { max_ram: +$("#mr").value || null, max_flash: +$("#mf").value || null, max_latency_ms: +$("#ml").value || null };
      st.metric = $("#mt").value; st.epochs = +$("#ep").value || 30; st.initPool = $("#ifp").checked; st.lrSearch = $("#lrs").checked; st.poolBench = !!($("#pbm") && $("#pbm").checked);
    };
    $("#back").onclick = () => { save(); st.step--; draw(); };
    $("#next").onclick = () => { save(); if (!st.targets.length) return toast("Pick at least one board"); if (!st.sizes.length) return toast("Pick at least one size"); st.step++; draw(); };
  }

  function stepReview(b) {
    st.name = st.name || `${st.tpl.name} – ${new Date().toLocaleDateString()}`;
    b.innerHTML = `<div class="card"><label class="f">Project name<input id="nm" value="${esc(st.name)}"></label>
      <div class="grid g4" style="margin-top:14px">
        <div class="kpi"><div class="l">Model type</div><div class="v" style="font-size:15px">${esc(st.tpl.icon)} ${esc(st.tpl.name)}</div></div>
        <div class="kpi"><div class="l">Data</div><div class="v" style="font-size:15px">${fmt(st.ds.n)} windows${st.ds.synthetic ? ` <span class="badge b-warn">synthetic</span>` : ""}</div></div>
        <div class="kpi"><div class="l">Boards</div><div class="v" style="font-size:15px">${st.targets.join(", ")}</div></div>
        <div class="kpi"><div class="l">Goal</div><div class="v" style="font-size:15px">${GOAL_LABEL[st.goal]}</div></div></div>
      <p class="small muted" style="margin-top:14px">The pipeline trains ${st.sizes.length} size(s), quantizes each to INT8 (with quantization-aware fine-tuning when INT8 loses more than 1%), asks SAI for every plan on every board, picks the best that fits, then compiles, verifies and measures the deployed program's accuracy bit-exactly.</p>
      ${nav(true, true, true, "Build my model →")}</div>`;
    $("#back").onclick = () => { st.name = $("#nm").value; st.step--; draw(); };
    $("#next").onclick = async () => {
      $("#next").disabled = true;
      try {
        const r = await api("/api/projects", { json: { name: $("#nm").value, template_id: st.tpl.id, dataset_id: st.ds.id, targets: st.targets, goal: st.goal, sizes: st.sizes, budgets: st.budgets, metric_target: st.metric === "" ? null : +st.metric, epochs: st.epochs, init_from_pool: !!st.initPool, lr_search: !!st.lrSearch, pool_benchmark: !!st.poolBench } });
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
route(/^\/job\/([\w]+)$/, async (el, jid) => {
  let logs = [], last = 0, timer = null, job = null, rendered = null;
  const sys = await system();
  el.innerHTML = `<div id="head"></div><div id="res"></div>`;
  async function poll() {
    try {
      job = await api(`/api/jobs/${jid}?log_after=${last}`);
      logs = logs.concat(job.logs); if (job.logs.length) last = job.logs[job.logs.length - 1].seq;
      drawHead();
      if (["succeeded", "failed", "cancelled"].includes(job.status)) {
        if (["rebuild", "regression", "basetrain"].includes(job.kind) && job.status === "succeeded" && rendered !== job.status) { rendered = job.status; return drawPlansOnly(job.result); }
        if (rendered !== job.status) { rendered = job.status; drawResult(); }
        return;
      }
    } catch (e) { toast(e.message); }
    timer = setTimeout(poll, 1000);
  }
  onLeave(() => clearTimeout(timer));
  function drawHead() {
    const title = job.kind === "train" ? (job.params.name || "Project") : job.kind === "fitcheck" ? "Studio fit check" : job.kind === "rebuild" ? "Reproducibility rebuild" : job.kind === "regression" ? "Nightly regression" : job.kind === "basetrain" ? "Pool base weights" : `Analyze ${job.params.filename || "model"}`;
    $("#head").innerHTML = `<div class="page-h"><div><h1>${esc(title)}</h1><div class="row small muted">${statusBadge(job.status, job.progress)} <span class="mono">${esc(job.id)}</span>${job.kind === "train" ? ` · ${esc(job.params.template_id)} · goal ${esc(GOAL_LABEL[job.params.goal] || job.params.goal)}` : ""}</div></div>
      ${job.status === "running" || job.status === "queued" ? `<button class="btn" id="cancel">Cancel</button>` : ""}</div>
      <div class="card"><div class="progress" style="margin-bottom:12px"><i style="width:${Math.round((job.progress || 0) * 100)}%"></i></div>
      <div class="timeline">${(job.stages || []).map((s) => `<div class="tl ${s.status}"><div>${esc(s.title)}</div><div class="s">${esc(s.status)}${s.note ? " · " + esc(s.note) : ""}</div></div>`).join("")}</div>
      ${job.error ? `<div class="callout bad" style="margin-top:12px">${esc(job.error)}</div>` : ""}
      <details style="margin-top:12px" ${job.status !== "succeeded" ? "open" : ""}><summary class="small muted">Pipeline log (${logs.length} lines)</summary><div class="log" id="log">${esc(logs.map((l) => l.line).join("\n"))}</div></details></div>`;
    const lg = $("#log"); if (lg) lg.scrollTop = lg.scrollHeight;
    const c = $("#cancel"); if (c) c.onclick = () => api(`/api/jobs/${jid}/cancel`, { method: "POST" }).then(() => toast("Cancelling…"));
  }
  function drawResult() {
    if (job.status !== "succeeded") return;
    const r = job.result;
    if (job.kind === "fitcheck" || job.kind === "analyze") return drawPlansOnly(r);
    const sel = r.selected, v = r.variants.find((x) => x.size === sel.size), task = r.template.task;
    const mt = (m) => task === "classification" ? pct(m.value) : fmt(m.value, 3);
    $("#res").innerHTML = `
    ${r.deployable === false ? `<div class="callout bad" style="margin-top:16px"><b>Evaluation only:</b> ${esc(r.evaluation_reason || "dataset licence")}. The package carries an EVALUATION_ONLY marker and a compile-time warning.</div>` : ""}
    ${r.signature ? `<p class="small muted" style="margin-top:10px">Package signed (Ed25519, key <code>${esc(r.signature.key_id)}</code>); verify with <code>tools/verify_package.py</code> and <code>/api/signing/public-key</code>.</p>` : ""}
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
        <div class="row" style="margin-top:10px">${r.downloads.map((d) => `<a class="btn ${d === "package.zip" ? "primary" : ""} sm" href="/api/jobs/${jid}/download/${d}">${d === "package.zip" ? "⬇ " : ""}${esc(d)}</a>`).join("")}</div></div></div>
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
    <div class="card" style="margin-top:16px" id="hil"></div>`;
    const rb = $("#rebuild");
    if (rb) rb.onclick = async () => { const j = await api(`/api/jobs/${jid}/rebuild`, { method: "POST" }); toast("Rebuild queued"); location.hash = `#/job/${j.id}`; };
    drawHil();
  }
  async function drawHil() {
    const [b, runs] = await Promise.all([api("/api/boards"), api(`/api/boards/runs?job_id=${jid}`)]);
    const sel = job.result.selected;
    const avail = b.boards.filter((x) => x.online);
    $("#hil").innerHTML = `<div class="row"><h2 style="margin:0">Run on a board</h2><span class="spacer"></span>
      <select id="hb"><option value="">any ${esc(sel.target)} board</option>${avail.map((x) => `<option value="${esc(x.id)}" data-t="${x.target}">${esc(x.name)} – ${esc(x.status)}</option>`).join("")}</select>
      <button class="btn primary" id="hrun">Flash & measure</button></div>
      <p class="small muted" style="margin:8px 0">The board builds the HIL harness firmware around the SAI package for its own target, runs the test vectors, and reports outputs, cycles and painted stack. Outputs are compared bit-exactly with the host build; real boards turn the plan's latency into <b>measured</b>. Emulated boards (QEMU) prove the program on the target ISA; their cycles are not latency.</p>
      ${runs.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Run</th><th>Board</th><th>Status</th><th class="n">Estimated</th><th class="n">Measured</th><th class="n">Error</th><th>Outputs</th><th>Stack (painted)</th></tr></thead><tbody>
      ${runs.map((x) => { const res = x.result || {}, est = x.request.estimated_ms, got = res.latency_ms; return `<tr><td class="mono small">${esc(x.id)}</td><td>${esc(x.board_id || "queued")} ${res.simulated ? `<span class="badge b-warn">simulated</span>` : res.emulated ? `<span class="badge b-info" title="real firmware on QEMU: outputs are real, cycles are QEMU's instruction clock">emulated</span>` : ""}</td><td>${statusBadge(x.status)}</td><td class="n">${ms(est)}</td><td class="n">${res.emulated ? `<span class="small muted">${fmt(res.cycles)} emu. cycles</span>` : ms(got)}</td><td class="n">${est && got && !res.emulated ? `${((got - est) / est * 100).toFixed(1)}%` : "–"}</td><td>${res.outputs_match === undefined ? "–" : res.outputs_match ? `<span class="badge b-ok">✓ ${res.n_vectors} match</span>` : `<span class="badge b-bad">mismatch</span>`}</td><td>${bytes(res.stack_painted)}</td></tr>`; }).join("")}</tbody></table></div>` : ""}`;
    $("#hrun").onclick = async () => {
      const o = $("#hb").selectedOptions[0];
      try { await api("/api/boards/runs", { json: { job_id: jid, board_id: o.value || null, target: o.dataset.t || null } }); toast("Queued on the board farm"); }
      catch (e) { return toast(e.message); }
      const t = setInterval(async () => { await drawHil(); }, 1500); onLeave(() => clearInterval(t)); setTimeout(() => clearInterval(t), 30000);
    };
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
  return `<div class="card" style="margin-top:16px"><div class="row"><h2 style="margin:0">Provenance</h2><span class="spacer"></span><button class="btn sm" id="rebuild">Rebuild & compare</button></div>
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

route(/^\/studio$/, async (el) => {
  const [sys, mine] = await Promise.all([system(), api("/api/catalog?drafts=true")]);
  const builtins = mine.filter((t) => t.source === "builtin"), research = mine.filter((t) => t.source === "studio");
  let m = BLANK(), size = "tiny", vtimer = null;
  const OPS = Object.keys(sys.ops);
  const PARAMS = { conv: ["out", "k", "stride", "dilation", "relu"], dwconv: ["k", "stride", "dilation", "relu"], pwconv: ["out", "relu"], maxpool: ["k"], avgpool: ["k"], gap: [], fc: ["out", "relu"], residual: ["k"], dropout: ["p"] };
  const DEF = { conv: { out: 16, k: 3, stride: 1, relu: true }, dwconv: { k: 3, stride: 1, relu: true }, pwconv: { out: 16, relu: true }, maxpool: { k: 2 }, avgpool: { k: 2 }, gap: {}, fc: { out: 16, relu: true }, residual: { k: 3 }, dropout: { p: 0.1 } };
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
    if ($("#del")) $("#del").onclick = async () => { await api(`/api/catalog/${m.id}`, { method: "DELETE" }); toast("Deleted"); m = BLANK(); drawMeta(); drawArch(); };
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
        <td><div class="row" style="gap:6px">${PARAMS[l.op].map((p) => p === "relu" ? `<label class="check small"><input type="checkbox" data-f="relu" ${l.relu ? "checked" : ""}>ReLU</label>` : `<label class="f">${p}<input data-f="${p}" type="number" step="${p === "p" ? 0.05 : 1}" value="${l[p] ?? ""}"></label>`).join("")}</div></td>
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
      l[f] = f === "relu" ? e.target.checked : e.target.value === "" ? undefined : +e.target.value; validate();
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
    readMeta();
    $("#fitres").innerHTML = `<div class="callout info" style="margin-top:10px">Compiling with SAI…</div>`;
    try {
      const job = await api("/api/studio/fitcheck", { json: { spec: { input: m.input, layers: m.variants[size], outputs: nOut() }, targets: ["md6605", "cortex-m7"] } });
      const wait = async () => {
        const j = await api(`/api/jobs/${job.id}`);
        if (j.status === "succeeded") { $("#fitres").innerHTML = `<div style="margin-top:12px">${planTable(j.result.plans.candidates)}${j.result.plans.advice ? `<div class="callout" style="margin-top:8px">${esc(j.result.plans.advice)}</div>` : ""}${spaceNote(j.result.plans)}<a class="small" href="#/job/${j.id}">Open full report →</a></div>`; return; }
        if (j.status === "failed") { $("#fitres").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(j.error)}</div>`; return; }
        setTimeout(wait, 800);
      };
      wait();
    } catch (e) { $("#fitres").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(e.message)}</div>`; }
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
  const jobs = (await api("/api/jobs?kind=analyze")).slice(0, 8);
  el.innerHTML = `<div class="page-h"><div><h1>Plan Explorer</h1><p class="muted">Bring your own INT8 model (.onnx QDQ or .tflite). SAI inspects it and searches every execution plan on every board.</p></div></div>
  <div class="grid g2"><div class="card"><div class="drop" id="drop">Drop an INT8 <b>.onnx</b> or <b>.tflite</b>, or click<input type="file" id="file" accept=".onnx,.tflite" hidden></div>
    <div class="row" style="margin-top:12px"><label class="check"><input type="checkbox" class="tg" value="md6605" checked>md6605</label><label class="check"><input type="checkbox" class="tg" value="cortex-m7" checked>cortex-m7</label>
    <label class="f">Goal<select id="goal">${Object.entries(GOAL_LABEL).filter(([k]) => k !== "accuracy").map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></label><label class="f">Max RAM (B)<input id="mr" type="number" style="width:110px"></label></div>
    <div id="st"></div></div>
    <div class="card"><h2>Recent analyses</h2>${jobs.length ? `<table class="t"><tbody>${jobs.map((j) => `<tr><td><a href="#/job/${j.id}">${esc(j.params.filename || j.id)}</a></td><td>${statusBadge(j.status)}</td><td class="muted">${ago(j.created)}</td></tr>`).join("")}</tbody></table>` : `<div class="empty small">None yet</div>`}</div></div>`;
  const d = $("#drop"), f = $("#file");
  const go = async (file) => {
    const fd = new FormData(); fd.append("file", file); fd.append("targets", $$(".tg:checked").map((x) => x.value).join(",")); fd.append("goal", $("#goal").value); if ($("#mr").value) fd.append("max_ram", $("#mr").value);
    $("#st").innerHTML = `<div class="callout info" style="margin-top:10px">Uploading…</div>`;
    try { const j = await api("/api/explorer/analyze", { method: "POST", body: fd }); location.hash = `#/job/${j.id}`; } catch (e) { $("#st").innerHTML = `<div class="callout bad" style="margin-top:10px">${esc(e.message)}</div>`; }
  };
  d.onclick = () => f.click(); f.onchange = () => f.files[0] && go(f.files[0]);
  d.ondragover = (e) => { e.preventDefault(); d.classList.add("hover"); }; d.ondragleave = () => d.classList.remove("hover");
  d.ondrop = (e) => { e.preventDefault(); d.classList.remove("hover"); e.dataTransfer.files[0] && go(e.dataTransfer.files[0]); };
});

/* ------------------------------------------------------------------------------------------------ board farm */
route(/^\/boards$/, async (el) => {
  const draw = async () => {
    const [b, reg] = await Promise.all([api("/api/boards"), api("/api/regressions")]);
    const kind = (x) => x.kind === "physical" ? "" : `<span class="badge ${x.kind === "emulated" ? "b-info" : "b-warn"}">${esc(x.kind)}</span>`;
    const dot = (x) => x.status === "quarantined" ? "off" : x.status === "busy" ? "busy" : x.online ? "on" : "off";
    el.innerHTML = `<div class="page-h"><div><h1>Board Farm</h1><p class="muted">MCU boards on lab agents. Every run builds the HIL harness around the SAI package, checks outputs bit-exactly, and — on physical boards — measures cycles and stack.</p></div>
      ${ME.perms.farm_admin ? `<button class="btn" id="regrun">Run regression now</button>` : ""}</div>
    ${b.boards.length && b.boards.every((x) => x.kind !== "physical") ? `<div class="callout info" style="margin-bottom:16px">No physical boards are connected yet. The <b>emulated</b> boards run the real cross-compiled firmware on QEMU (Cortex-M7, RV32IMAFC): outputs are real and checked bit-exactly; QEMU cycles are not latency and never calibrate. Connect a lab agent (Admin → Lab agents) to add NUCLEO-H753ZI or MD6605 EVB boards.</div>` : ""}
    <div class="grid g2"><div class="card"><h2>Boards</h2><div class="tbl-wrap"><table class="t"><thead><tr><th>Board</th><th>Target</th><th>Agent</th><th>Health</th><th>Status</th><th></th></tr></thead><tbody>
      ${b.boards.map((x) => `<tr><td>${esc(x.name)} ${kind(x)}</td><td><span style="color:${TARGET_COLOR[x.target]}">●</span> ${esc(x.target)}</td><td class="mono small">${esc(x.agent_id)}</td>
        <td class="small">${esc(x.health.probe || x.info.probe || "")}${x.fail_count ? ` · ${x.fail_count} fail(s)` : ""}</td><td><span class="dot ${dot(x)}"></span> ${esc(x.status)}</td>
        <td>${x.status === "quarantined" && ME.perms.farm_admin ? `<button class="btn sm" data-reset="${esc(x.id)}">Reset</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="6" class="empty">No boards registered</td></tr>`}
    </tbody></table></div><p class="small muted" style="margin-top:8px">A board that fails 3 runs in a row is quarantined until the lab resets it; agents power-cycle a failing board once (uhubctl) before reporting.</p></div>
    <div class="card"><h2>Latency calibration</h2>${Object.keys(b.calibration).length ? `<table class="t"><thead><tr><th>Target : objective</th><th class="n">cycles/MAC</th><th class="n">overhead</th><th class="n">runs</th></tr></thead><tbody>${Object.entries(b.calibration).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="n">${v.cycles_per_mac.toFixed(3)}</td><td class="n">${fmt(v.overhead)}</td><td class="n">${v.samples}</td></tr>`).join("")}</tbody></table>` : `<p class="small muted">No physical-board measurements yet. Until then latency comes from SAI's kernel cost model where it has one, else a MACs × cycles/MAC heuristic, and is labelled that way everywhere.</p>`}
      <h3 style="margin-top:14px">Connect a lab agent</h3><p class="small">Create a token in Admin → Lab agents, list the boards in <code>boards.json</code> (driver <code>pyocd</code>, <code>jlink</code> or <code>qemu</code>), then on the lab PC:</p>
      <div class="log">pip install numpy pyserial pyocd
python agent/board_agent.py --server ${esc(location.origin)} \\
  --token $SAI_AGENT_TOKEN --boards boards.json</div></div></div>
    <div class="card" style="margin-top:16px"><h2>Nightly regression</h2>${reg.latest.length ? `<p class="small muted">Run ${esc(reg.runs[0].run_id)} · SAI ${esc(reg.runs[0].sai_version)} · ${ago(reg.runs[0].created)}</p>
      <div class="tbl-wrap"><table class="t"><thead><tr><th>Model type (tiny)</th><th>Board</th><th class="n">RAM</th><th class="n">Flash</th><th class="n">Plans that fit</th><th>Emulated HIL</th></tr></thead><tbody>
      ${reg.latest.map((r) => `<tr><td>${esc(r.template)}</td><td>${esc(r.target)}</td><td class="n">${bytes(r.ram)}</td><td class="n">${bytes(r.flash)}</td><td class="n">${r.feasible}</td><td>${r.hil_match === null ? "–" : r.hil_match ? `<span class="badge b-ok">bit-exact</span>` : `<span class="badge b-bad">mismatch</span>`}</td></tr>`).join("")}</tbody></table></div>` : `<p class="small muted">No regression run yet. It checks every pool model type on every board against the current SAI release and alerts on RAM/flash growth, lost fits or HIL mismatches (also nightly in CI).</p>`}</div>
    <div class="card" style="margin-top:16px"><h2>Recent runs</h2>${b.runs.length ? `<div class="tbl-wrap"><table class="t"><thead><tr><th>Run</th><th>Job</th><th>Board</th><th>Status</th><th class="n">Estimated</th><th class="n">Result</th><th>Outputs</th><th class="n">Stack</th><th>When</th></tr></thead><tbody>
      ${b.runs.map((x) => { const r = x.result || {}; return `<tr><td class="mono small">${esc(x.id)}</td><td><a href="#/job/${x.job_id}">${esc(x.job_id)}</a></td><td>${esc(x.board_id || "–")}</td><td>${statusBadge(x.status)}</td><td class="n">${ms(x.request.estimated_ms)}</td>
        <td class="n">${r.emulated ? `<span class="small muted">${fmt(r.cycles)} emu. cycles</span>` : r.simulated ? `${ms(r.latency_ms)} (sim)` : ms(r.latency_ms)}</td>
        <td>${r.outputs_match === undefined ? (r.error ? `<span class="badge b-bad" title="${esc(r.error)}">error</span>` : "–") : r.outputs_match ? `<span class="badge b-ok">match</span>` : `<span class="badge b-bad">mismatch</span>`}</td><td class="n">${bytes(r.stack_painted)}</td><td class="muted">${ago(x.created)}</td></tr>`; }).join("")}</tbody></table></div>` : `<div class="empty small">No runs yet. Open a finished project and press “Flash &amp; measure”.</div>`}</div>`;
    $$("[data-reset]").forEach((bt) => (bt.onclick = async () => { await api(`/api/boards/${bt.dataset.reset}/reset`, { method: "POST" }); draw(); }));
    const rr = $("#regrun");
    if (rr) rr.onclick = async () => { const j = await api("/api/admin/regression", { json: {} }); location.hash = `#/job/${j.id}`; };
  };
  await draw();
  const t = setInterval(draw, 4000); onLeave(() => clearInterval(t));
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
  if (!m.perms.org_admin && !m.perms.farm_admin) { el.innerHTML = `<div class="callout bad">Your role cannot open the admin console.</div>`; return; }
  const tabs = [m.perms.org_admin && ["orgs", "Organisations"], m.perms.org_admin && ["users", "Users"], m.perms.farm_admin && ["agents", "Lab agents"], m.perms.org_admin && ["audit", "Audit log"], m.perms.platform_admin && ["queue", "Job queue"]].filter(Boolean);
  let tab = tabs[0][0];
  const draw = async () => {
    el.innerHTML = `<div class="page-h"><div><h1>Admin</h1><p class="muted">${m.mode === "single" ? "Single-organisation install" : "Multi-tenant"} · signed in as ${esc(m.email)} (${esc(m.role)})</p></div></div>
      <div class="tabs">${tabs.map(([k, t]) => `<button class="tab ${k === tab ? "on" : ""}" data-t="${k}">${t}</button>`).join("")}</div><div id="adm"></div>`;
    $$(".tabs .tab", el).forEach((b) => (b.onclick = () => { tab = b.dataset.t; draw(); }));
    const box = $("#adm");
    if (tab === "orgs") {
      const orgs = await api("/api/admin/orgs");
      box.innerHTML = `<div class="card"><div class="tbl-wrap"><table class="t"><thead><tr><th>Organisation</th><th>Tier</th><th>Deployment rights</th><th class="n">Users</th><th>Usage (all time)</th><th>Created</th></tr></thead><tbody>
        ${orgs.map((o) => `<tr><td><b>${esc(o.name)}</b><div class="small muted mono">${esc(o.id)}</div></td><td>${m.perms.platform_admin ? `<select data-org="${esc(o.id)}">${["free", "standard", "professional"].map((t) => `<option ${t === o.tier ? "selected" : ""}>${t}</option>`).join("")}</select>` : esc(o.tier)}</td><td class="n">${o.users}</td>
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
      const roles = m.perms.platform_admin ? ["member", "org_admin", "researcher", "lab", "platform_admin"] : ["member", "org_admin"];
      box.innerHTML = `<div class="card"><div class="tbl-wrap"><table class="t"><thead><tr><th>User</th><th>Organisation</th><th>Role</th><th>Status</th><th></th></tr></thead><tbody>
        ${us.map((x) => `<tr><td><b>${esc(x.name)}</b><div class="small muted">${esc(x.email)}</div></td><td>${esc(on[x.org_id] || x.org_id)}</td><td><span class="badge">${esc(x.role)}</span></td><td>${x.disabled ? `<span class="badge b-bad">disabled</span>` : `<span class="badge b-ok">active</span>`}</td><td>${!x.disabled && x.id !== m.id ? `<button class="btn sm ghost" data-dis="${x.id}">Disable</button>` : ""}</td></tr>`).join("")}</tbody></table></div>
        <h3 style="margin-top:14px">Invite a user</h3><div class="grid g4"><label class="f">Name<input id="un"></label><label class="f">Email<input id="ue" type="email"></label><label class="f">Initial password<input id="up" type="text" placeholder="≥ 8 characters"></label>
        <label class="f">Role<select id="ur">${roles.map((r) => `<option>${r}</option>`).join("")}</select></label></div>
        ${m.perms.platform_admin ? `<label class="f" style="margin-top:8px;max-width:320px">Organisation<select id="uo">${orgs.map((o) => `<option value="${esc(o.id)}">${esc(o.name)}</option>`).join("")}</select></label>` : ""}
        <div class="row" style="margin-top:10px"><button class="btn primary" id="uc">Create user</button></div></div>`;
      $$("[data-dis]", box).forEach((b) => (b.onclick = async () => { await api(`/api/admin/users/${b.dataset.dis}/disable`, { method: "POST" }); draw(); }));
      $("#uc").onclick = async () => { try { await api("/api/admin/users", { json: { name: $("#un").value, email: $("#ue").value, password: $("#up").value, role: $("#ur").value, org_id: $("#uo")?.value } }); toast("User created"); draw(); } catch (e) { toast(e.message); } };
    }
    if (tab === "agents") {
      const ag = await api("/api/admin/agents");
      box.innerHTML = `<div class="card"><p class="small muted">Each lab PC gets its own token. Tokens are shown once and stored hashed.</p><div class="tbl-wrap"><table class="t"><thead><tr><th>Agent</th><th>Created</th><th>Last seen</th><th>Status</th><th></th></tr></thead><tbody>
        ${ag.map((a) => `<tr><td><b>${esc(a.name)}</b><div class="mono small muted">${esc(a.id)}</div></td><td class="muted">${ago(a.created)}</td><td class="muted">${a.last_seen ? ago(a.last_seen) : "never"}</td><td>${a.disabled ? `<span class="badge b-bad">disabled</span>` : `<span class="badge b-ok">active</span>`}</td><td>${a.disabled ? "" : `<button class="btn sm ghost" data-ad="${a.id}">Revoke</button>`}</td></tr>`).join("") || `<tr><td colspan="5" class="empty">No agents yet</td></tr>`}</tbody></table></div>
        <div class="row" style="margin-top:12px"><input id="an" placeholder="e.g. Lab 1 - bench PC"><button class="btn primary" id="ac">Create agent token</button></div><div id="atok"></div></div>`;
      $$("[data-ad]", box).forEach((b) => (b.onclick = async () => { await api(`/api/admin/agents/${b.dataset.ad}/disable`, { method: "POST" }); draw(); }));
      $("#ac").onclick = async () => { const a = await api("/api/admin/agents", { json: { name: $("#an").value } }); $("#atok").innerHTML = `<div class="callout ok" style="margin-top:10px">Token for <b>${esc(a.name)}</b> (copy now, it will not be shown again):<div class="log" style="margin-top:6px">${esc(a.token)}</div></div>`; };
    }
    if (tab === "audit") {
      const log = await api("/api/admin/audit");
      box.innerHTML = `<div class="card"><div class="tbl-wrap"><table class="t"><thead><tr><th>When</th><th>Who</th><th>Action</th><th>Target</th><th>Detail</th></tr></thead><tbody>
        ${log.map((x) => `<tr><td class="muted small">${new Date(x.ts * 1000).toLocaleString()}</td><td class="small">${esc(x.email || "–")}</td><td><span class="badge">${esc(x.action)}</span></td><td class="mono small">${esc(x.target)}</td><td class="mono small">${esc(x.detail ? JSON.stringify(x.detail) : "")}</td></tr>`).join("")}</tbody></table></div></div>`;
    }
    if (tab === "queue") {
      const q = await api("/api/admin/queue");
      box.innerHTML = `<div class="card"><div class="kpis">${Object.entries(q.by_status).map(([k, v]) => `<div class="kpi"><div class="l">${esc(k)}</div><div class="v">${v}</div></div>`).join("")}</div><p class="small muted" style="margin-top:10px">Busy workers: ${q.busy_workers.map(esc).join(", ") || "none"}</p></div>`;
    }
  };
  draw();
});

/* ------------------------------------------------------------------------------------------------ login */
let ME = null;
function showLogin(msg) {
  document.body.classList.add("signed-out");
  $("#main").innerHTML = `<div class="login card"><div class="brand" style="padding:0 0 12px"><span class="logo">◆</span><span><b>SAI</b> Platform<small>Sign in to continue</small></span></div>
    <form id="lf" class="stack"><label class="f">Email<input id="le" type="email" autocomplete="username" required></label>
    <label class="f">Password<input id="lp" type="password" autocomplete="current-password" required></label>
    <div id="lerr">${msg ? `<div class="callout bad">${esc(msg)}</div>` : ""}</div><button class="btn primary" type="submit">Sign in</button>
    <p class="small muted">Accounts are created by your administrator.</p></form></div>`;
  $("#lf").onsubmit = async (e) => {
    e.preventDefault();
    try { await api("/api/auth/login", { json: { email: $("#le").value, password: $("#lp").value } }); document.body.classList.remove("signed-out"); boot(); }
    catch (err) { $("#lerr").innerHTML = `<div class="callout bad">${esc(err.message)}</div>`; }
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
}

/* ------------------------------------------------------------------------------------------------ boot */
boot();
