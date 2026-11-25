"""Nightly regression: every published pool model type x every board, on the current SAI release.

For each model type's `tiny` size (random weights; resources do not depend on trained weights) it records the
best-fitting plan per board (RAM, flash, latency estimate) and, on the emulated HIL boards, whether the
cross-compiled program's outputs are bit-exact with the host build. Each run is compared with the previous one;
alerts: RAM or flash +5 %, a plan that no longer fits, a HIL mismatch.

    python -m backend.regression [--templates bearing-fault,kws] [--no-hil]     # exit 1 on alerts (cron / CI)
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np

from . import archspec, config, db, exact_eval, hil, qexport, sai_engine, store

GROWTH = 0.05


def _spec(t: dict) -> dict:
    n = len(t["labels"]) if t["task"] == "classification" else 1 if t["task"] == "regression" else \
        t["input"]["channels"] * t["input"]["height"] * t["input"]["width"]
    return archspec.normalize({"input": t["input"], "layers": t["variants"]["tiny"], "outputs": n})


def run(ctx, p: dict):
    rid = "reg_" + uuid.uuid4().hex[:8]
    tpls = [t for t in store.templates() if t.get("variants", {}).get("tiny")]
    if p.get("templates"):
        tpls = [t for t in tpls if t["id"] in p["templates"]]
    work = config.DATA_DIR / "regression" / rid
    work.mkdir(parents=True, exist_ok=True)
    sai_v = sai_engine.version().get("sai")
    rows, models = [], {}
    ctx.stage("plans")
    for i, t in enumerate(tpls):
        spec = _spec(t)
        net = archspec.SpecNet(spec).eval()
        net.fold_bn()
        i_ = spec["input"]
        calib = np.random.default_rng(0).normal(size=(32, i_["channels"], i_["height"], i_["width"])).astype(np.float32)
        path = work / f"{t['id']}.onnx"
        q = qexport.export_qdq(net, calib, path)
        models[t["id"]] = (path, q, calib)
        summ = archspec.summary(spec)
        res = sai_engine.plan_search(path, summ["macs"], summ["spatial_axes"], t.get("targets") or list(sai_engine.MCU_TARGETS),
                                     "min-ram", explore_builds=0)
        for target in {c["target"] for c in res["candidates"]}:
            fit = [c for c in res["candidates"] if c["target"] == target and c["fits"]]
            best = min(fit, key=lambda c: c["ram"]) if fit else None
            rows.append({"template": t["id"], "size": "tiny", "target": target, "ram": best and best["ram"],
                         "flash": best and best["flash"], "cycles": best and best["latency"]["cycles"],
                         "feasible": len(fit), "hil_match": None})
        ctx.log(f"{t['id']}: {', '.join(f'{r['target']} {r['ram']} B RAM' for r in rows if r['template'] == t['id'])}")
        ctx.progress_within("plans", (i + 1) / max(len(tpls), 1))
    ctx.stage("plans", "done")

    ctx.stage("hil")
    if p.get("hil", True):
        ports = [port for port in ("qemu-mps2-an500", "qemu-virt-rv32") if hil.available(port)]
        smoke = [t for t in tpls if t["input"]["height"] == 1][:1] + [t for t in tpls if t["input"]["height"] > 1][:1]
        for t in smoke:
            path, q, calib = models[t["id"]]
            xq = qexport.quantize_input(calib[:8], q["input"])
            ref = exact_eval.run_exact(path, work / f"exact_{t['id']}", xq)
            for port in ports:
                target = hil.PORTS[port]["target"]
                build = work / f"build_{t['id']}_{target}"
                sai_engine.compile_build(path, target, "min-ram", build)
                fw = hil.build(build, port, xq, work / f"fw_{t['id']}_{port}")
                out = hil.run_emulated(fw["elf"], port)
                ok = np.array_equal(np.array(out["outputs"]), ref.astype(int))
                ctx.log(f"HIL {t['id']} on {port}: outputs {'bit-exact' if ok else 'MISMATCH'}")
                for r in rows:
                    if r["template"] == t["id"] and r["target"] == target:
                        r["hil_match"] = int(ok)
        if not ports:
            ctx.log("no emulated boards available (QEMU or toolchain missing): HIL smoke skipped")
    ctx.stage("hil", "done")

    ctx.stage("compare")
    prev = _last_rows()
    alerts = []
    for r in rows:
        old = prev.get((r["template"], r["size"], r["target"]))
        r["prev"] = old
        if old:
            for k in ("ram", "flash"):
                if old[k] and r[k] and r[k] > old[k] * (1 + GROWTH):
                    alerts.append(f"{r['template']} on {r['target']}: {k} {old[k]} -> {r[k]} B (+{(r[k] / old[k] - 1):.0%})")
            if old["feasible"] and not r["feasible"]:
                alerts.append(f"{r['template']} on {r['target']}: no plan fits any more")
        if r["hil_match"] == 0:
            alerts.append(f"{r['template']} on {r['target']}: emulated HIL outputs differ from the host build")
    with db.conn() as c:
        for r in rows:
            c.execute("INSERT INTO regressions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (rid, r["template"], r["size"], r["target"], r["ram"], r["flash"], r["cycles"], r["feasible"],
                       r["hil_match"], sai_v, db.now()))
    for a in alerts:
        ctx.log("ALERT " + a)
    ctx.log(f"{len(rows)} rows, {len(alerts)} alerts")
    import shutil
    shutil.rmtree(work, ignore_errors=True)          # builds and firmware are reproducible; keep only the rows
    ctx.stage("compare", "done")
    ctx.result({"run_id": rid, "sai_version": sai_v, "rows": rows, "alerts": alerts})


def _last_rows(before: str | None = None) -> dict:
    with db.conn() as c:
        rid = c.execute("SELECT run_id FROM regressions ORDER BY created DESC LIMIT 1").fetchone()
        if not rid:
            return {}
        rows = [db.row(r) for r in c.execute("SELECT * FROM regressions WHERE run_id=?", (rid[0],))]
    return {(r["template"], r["size"], r["target"]): r for r in rows}


def summary() -> dict:
    with db.conn() as c:
        runs = [db.row(r) for r in c.execute("SELECT run_id, MAX(created) AS created, MAX(sai_version) AS sai_version,"
                                             " COUNT(*) AS rows FROM regressions GROUP BY run_id ORDER BY created DESC LIMIT 10")]
        latest = [db.row(r) for r in c.execute("SELECT * FROM regressions WHERE run_id=? ORDER BY template, target",
                                               (runs[0]["run_id"],))] if runs else []
    return {"runs": runs, "latest": latest}


class _Cli:
    """Minimal job context for running outside the queue."""

    def __init__(self):
        self.result_ = None

    def stage(self, *a, **k):
        pass

    def progress_within(self, *a):
        pass

    def log(self, line):
        print(line, flush=True)

    def result(self, r):
        self.result_ = r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="")
    ap.add_argument("--no-hil", action="store_true")
    a = ap.parse_args()
    db.init()
    ctx = _Cli()
    run(ctx, {"templates": [t for t in a.templates.split(",") if t] or None, "hil": not a.no_hil})
    out = Path(config.DATA_DIR) / "regression" / "latest.json"
    import json
    out.write_text(json.dumps(ctx.result_, indent=1, default=str))
    print(f"report: {out}")
    sys.exit(1 if ctx.result_["alerts"] else 0)


if __name__ == "__main__":
    main()
