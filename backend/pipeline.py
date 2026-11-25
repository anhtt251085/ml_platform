"""Job orchestration: the end-to-end pipeline a customer starts with one click, plus the Studio fit check and the
model analyzer. Jobs run on a thread pool now; the job/stage/log model maps 1:1 onto a queue (Celery, Arq,
Cloud Tasks) when training moves to GPU workers."""

from __future__ import annotations

import json
import shutil
import traceback
import uuid
import zipfile
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import archspec, config, data, db, exact_eval, features, qexport, sai_engine, trainer
from . import boards as boards_mod

STAGES = {
    "train": [("data", "Validate data"), ("train", "Train model sizes"), ("quantize", "INT8 quantization"),
              ("plan", "SAI plan search"), ("select", "Select model + plan"), ("compile", "Compile + verify"),
              ("exact", "Bit-exact accuracy"), ("package", "Package")],
    "fitcheck": [("build", "Build graph"), ("quantize", "INT8 export"), ("plan", "SAI plan search")],
    "analyze": [("inspect", "Inspect model"), ("plan", "SAI plan search")],
    "rebuild": [("compile", "Recompile"), ("compare", "Compare with the original build")],
    "basetrain": [("data", "Prepare data"), ("train", "Train base weights"), ("save", "Store in the pool")],
    "regression": [("plans", "Pool x boards plan check"), ("hil", "Emulated HIL smoke"), ("compare", "Compare with last run")],
}
KINDS = tuple(STAGES)
STALE_RUNNING_S = 6 * 3600
GOALS = ("balanced", "fastest", "min-ram", "min-flash", "accuracy")


# ------------------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, job_id: str, kind: str):
        self.id, self.kind = job_id, kind
        self.dir = config.DATA_DIR / "jobs" / job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stages = [{"key": k, "title": t, "status": "pending"} for k, t in STAGES[kind]]
        self.seq = 0

    def log(self, line: str):
        self.seq += 1
        with db.conn() as c:
            c.execute("INSERT INTO job_logs VALUES (?,?,?,?)", (self.id, self.seq, db.now(), str(line)))

    def _save(self, **fields):
        sets = ", ".join(f"{k}=?" for k in fields) + ", updated=?"
        with db.conn() as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), db.now(), self.id))

    def stage(self, key: str, status: str = "running", note: str | None = None):
        for i, s in enumerate(self.stages):
            if s["key"] == key:
                s["status"] = status
                s["started" if status == "running" else "finished"] = db.now()
                if note:
                    s["note"] = note
                done = sum(1 for x in self.stages if x["status"] in ("done", "skipped"))
                self._save(stage=key, stages=db.j(self.stages), progress=done / len(self.stages))
                if status == "running":
                    self.log(f"== {s['title']}")
                return

    def progress_within(self, key: str, frac: float):
        idx = [s["key"] for s in self.stages].index(key)
        self._save(progress=(idx + max(0.0, min(frac, 1.0))) / len(self.stages))

    def cancelled(self) -> bool:
        with db.conn() as c:
            r = c.execute("SELECT cancel_requested FROM jobs WHERE id=?", (self.id,)).fetchone()
        return bool(r and r[0])

    def result(self, res: dict):
        self._save(result=db.j(res))


def create(kind: str, params: dict, project_id: str | None = None, org_id: str | None = None,
           user_id: str | None = None, priority: int = 0) -> dict:
    """Enqueue a job. A worker (in-process thread or `python -m backend.worker`) claims and runs it."""
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind}")
    jid = "job_" + uuid.uuid4().hex[:10]
    stages = [{"key": k, "title": t, "status": "pending"} for k, t in STAGES[kind]]
    with db.conn() as c:
        c.execute("INSERT INTO jobs (id, kind, project_id, status, stage, progress, stages, result, error, params,"
                  " created, updated, org_id, user_id, priority, attempts, cancel_requested)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)",
                  (jid, kind, project_id, "queued", None, 0.0, db.j(stages), None, None, db.j(params), db.now(),
                   db.now(), org_id, user_id, priority))
    _WAKE.set()
    return get(jid)


def get(jid: str, log_after: int = 0) -> dict | None:
    with db.conn() as c:
        r = db.row(c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())
        if not r:
            return None
        logs = [dict(x) for x in c.execute("SELECT seq, ts, line FROM job_logs WHERE job_id=? AND seq>? ORDER BY seq",
                                           (jid, log_after))]
    for k in ("stages", "result", "params"):
        r[k] = db.uj(r[k])
    r["logs"] = logs
    return r


def list_jobs(kind: str | None = None, limit: int = 50, org_id: str | None = None) -> list[dict]:
    where, args = [], []
    if kind:
        where.append("kind=?"); args.append(kind)
    if org_id:
        where.append("org_id=?"); args.append(org_id)
    q = ("SELECT id, kind, project_id, status, stage, progress, error, created, updated, params, org_id, worker"
         " FROM jobs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created DESC LIMIT ?")
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute(q, (*args, limit))]
    for r in rows:
        r["params"] = db.uj(r["params"], {})
    return rows


def cancel(jid: str):
    with db.conn() as c:
        c.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (jid,))
        c.execute("UPDATE jobs SET status='cancelled', error='cancelled before start' WHERE id=? AND status='queued'",
                  (jid,))


def queue_stats() -> dict:
    with db.conn() as c:
        rows = c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        workers = [r[0] for r in c.execute("SELECT DISTINCT worker FROM jobs WHERE status='running'")]
    return {"by_status": {r[0]: r[1] for r in rows}, "busy_workers": workers, "paused": _PAUSED.is_set()}


# ---- workers -----------------------------------------------------------------------------------------------------
_WAKE = threading.Event()
_PAUSED = threading.Event()        # drain for maintenance: running jobs finish, nothing new starts in this process


def pause_workers(paused: bool):
    (_PAUSED.set if paused else _PAUSED.clear)()


def claim(worker: str, kinds: list[str] | None = None) -> dict | None:
    """Atomically take the next queued job (priority, then age). Safe across processes: the UPDATE only succeeds
    for the worker that still sees the row queued."""
    kinds = kinds or list(KINDS)
    marks = ",".join("?" * len(kinds))
    won = None
    with db.conn() as c:
        for r in c.execute(f"SELECT id FROM jobs WHERE status='queued' AND kind IN ({marks})"
                           " ORDER BY priority DESC, created LIMIT 5", kinds).fetchall():
            cur = c.execute("UPDATE jobs SET status='running', worker=?, attempts=attempts+1, updated=?"
                            " WHERE id=? AND status='queued'", (worker, db.now(), r[0]))
            if cur.rowcount == 1:
                won = r[0]
                break
    return get(won) if won else None          # read after the commit


def requeue_stale():
    """Jobs left 'running' by a dead worker (no update for STALE_RUNNING_S) go back to the queue once."""
    with db.conn() as c:
        c.execute("UPDATE jobs SET status='queued', worker=NULL WHERE status='running' AND updated<? AND attempts<2",
                  (db.now() - STALE_RUNNING_S,))
        c.execute("UPDATE jobs SET status='failed', error='worker lost twice' WHERE status='running' AND updated<?",
                  (db.now() - STALE_RUNNING_S,))


def work_forever(name: str, kinds: list[str] | None = None, stop: threading.Event | None = None):
    stop = stop or threading.Event()
    while not stop.is_set():
        if _PAUSED.is_set():
            stop.wait(0.5)
            continue
        job = claim(name, kinds)
        if job:
            _run(job["id"], job["kind"], job["params"] or {})
            continue
        _WAKE.wait(1.0)
        _WAKE.clear()


def start_inprocess_workers(n: int) -> list[threading.Thread]:
    requeue_stale()
    host = socket.gethostname()
    threads = []
    for i in range(n):
        t = threading.Thread(target=work_forever, args=(f"{host}:{os.getpid()}:t{i}", config.WORKER_KINDS or None),
                             daemon=True, name=f"job-worker-{i}")
        t.start()
        threads.append(t)
    return threads


def _run(jid: str, kind: str, params: dict):
    ctx = Ctx(jid, kind)
    t0 = time.time()
    try:
        {"train": run_train, "fitcheck": run_fitcheck, "analyze": run_analyze, "rebuild": run_rebuild,
         "basetrain": run_basetrain, "regression": _regression}[kind](ctx, params)
        ctx._save(status="succeeded", progress=1.0)
        ctx.log("job finished")
    except InterruptedError:
        ctx._save(status="cancelled", error="cancelled by user")
        ctx.log("job cancelled")
    except Exception as e:                                     # noqa: BLE001 - every failure is reported to the UI
        for s in ctx.stages:
            if s["status"] == "running":
                s["status"] = "failed"
        ctx._save(status="failed", error=f"{type(e).__name__}: {e}", stages=db.j(ctx.stages))
        ctx.log("ERROR " + "".join(traceback.format_exception_only(type(e), e)).strip())
        ctx.log(traceback.format_exc()[-2000:])
    finally:
        from . import auth
        with db.conn() as c:
            r = c.execute("SELECT org_id FROM jobs WHERE id=?", (jid,)).fetchone()
        auth.meter(r[0] if r else None, f"job_seconds:{kind}", time.time() - t0, jid)


# ------------------------------------------------------------------------------------------------------------------
def get_template(tid: str) -> dict:
    from . import store
    t = store.template(tid)
    if not t:
        raise ValueError(f"unknown model type {tid}")
    return t


def run_train(ctx: Ctx, p: dict):
    tpl = get_template(p["template_id"])
    task = tpl["task"]
    targets = [t for t in p.get("targets") or tpl.get("targets") or list(sai_engine.MCU_TARGETS) if t in sai_engine.MCU_TARGETS]
    goal = p.get("goal", "balanced")
    sizes = p.get("sizes") or ["tiny", "small", "base"]
    epochs = int(p.get("epochs") or 30)
    acc_target = p.get("metric_target")

    # 1 data ---------------------------------------------------------------------------------------------------
    ctx.stage("data")
    X, y, labels, meta = data.load(p["dataset_id"])
    ctx.log(f"dataset {meta.get('name')} : {len(X)} windows, input {list(X.shape[1:])}, labels {labels}")
    for w in meta.get("warnings", []):
        ctx.log("warning: " + w)
    groups = data.load_groups(p["dataset_id"])
    idx = trainer.split(y, task, groups=groups)
    mean, std = trainer.norm_stats(X[idx["train"]])
    Xn = ((X - mean) / std).astype(np.float32)
    y_stats = (float(y[idx["train"]].mean()), float(y[idx["train"]].std() + 1e-6)) if task == "regression" else None
    ctx.log(f"split train {len(idx['train'])} / val {len(idx['val'])} / test {len(idx['test'])} "
            f"({idx['split_mode']}-level: {'whole recordings per split' if idx['split_mode'] == 'recording' else 'windows'})")
    ctx.stage("data", "done")

    # 2 train ----------------------------------------------------------------------------------------------------
    ctx.stage("train")
    variants = []
    for si, size in enumerate(sizes):
        layers = tpl["variants"].get(size)
        if not layers:
            continue
        try:
            spec = trainer.build_spec(layers, X, task, len(labels))
        except archspec.SpecError as e:
            ctx.log(f"size {size}: skipped ({e})")
            continue
        summ = archspec.summary(spec)
        ctx.log(f"size {size}: {summ['params']:,} params, {summ['macs']:,} MACs")
        base = pool_weights(tpl["id"], size) if p.get("init_from_pool") else None
        lrs = [3e-3, 1e-2] if p.get("lr_search") else [None]
        best_net = None
        for lr in lrs:
            init = None
            if base is not None:
                init = archspec.SpecNet(spec)
                copied = trainer.init_from(init, base["state"])
                ctx.log(f"size {size}: fine-tuning from pool base weights ({len(copied)} tensors copied)")
            kw = {"lr": lr if lr else (2e-3 if base is not None else 5e-3)}
            net, hist = trainer.train(spec, Xn, y, idx, task, epochs=epochs, log=ctx.log, cancelled=ctx.cancelled,
                                      progress=lambda f, si=si: ctx.progress_within("train", (si + f) / len(sizes)),
                                      y_stats=y_stats, net=init, **kw)
            if len(lrs) > 1:
                ctx.log(f"size {size}: lr {kw['lr']} -> best validation loss {net.best_val:.4f}")
            if best_net is None or net.best_val < best_net[0].best_val:
                best_net = (net, hist)
        net, hist = best_net
        with torch.no_grad():
            out = net(torch.from_numpy(Xn[idx["test"]])).numpy().reshape(len(idx["test"]), -1)
        thr = None
        if task == "anomaly":
            with torch.no_grad():
                vo = net(torch.from_numpy(Xn[idx["val"]])).numpy().reshape(len(idx["val"]), -1)
            thr = float(np.percentile(((vo - Xn[idx["val"]].reshape(len(idx["val"]), -1)) ** 2).mean(1), 99))
        fm = trainer.metrics(task, y[idx["test"]], out, labels, y_stats, Xn[idx["test"]].reshape(len(idx["test"]), -1), thr)
        ctx.log(f"size {size}: float {fm['headline']} = {fm['value']}")
        variants.append({"size": size, "spec": spec, "summary": summ, "net": net, "history": hist,
                         "float": fm, "threshold": thr})
    if not variants:
        raise ValueError("no model size fits this input; check the window length")
    ctx.stage("train", "done")

    # 3 quantize -------------------------------------------------------------------------------------------------
    ctx.stage("quantize")
    rng = np.random.default_rng(0)
    calib = Xn[rng.choice(idx["train"], size=min(512, len(idx["train"])), replace=False)]
    xt, xv = Xn[idx["test"]], Xn[idx["val"]]
    yv = y[idx["val"]]

    def _m(out, xs, ys, v):
        if task == "anomaly" and not (ys == 1).any():     # validation holds normal data only: score fidelity
            return {"value": -round(float(((out - xs.reshape(len(xs), -1)) ** 2).mean()), 6)}
        return trainer.metrics(task, ys, out, labels, y_stats, xs.reshape(len(xs), -1), v["threshold"])

    def int8_metric(v, path, xs, ys):
        xq = qexport.quantize_input(xs, v["q"]["input"])
        out = qexport.dequantize_output(qexport.ort_int8_outputs(path, xq).reshape(len(xs), -1), v["q"]["output"])
        return _m(out, xs, ys, v)

    def float_metric(net, xs, ys):
        with torch.no_grad():
            out = net(torch.from_numpy(xs)).numpy().reshape(len(xs), -1)
        return _m(out, xs, ys, v)

    for v in variants:
        v["net"].fold_bn()
        path = ctx.dir / f"model_{v['size']}.onnx"
        v["q"] = qexport.export_qdq(v["net"], calib, path)
        v["onnx"] = path
        fval = float_metric(v["net"], xv, yv)["value"]
        ptq = int8_metric(v, path, xv, yv)["value"]
        v["quant_method"] = "PTQ"
        drop = (fval or 0) - (ptq or 0)
        if task == "anomaly":                      # relative growth of the reconstruction error
            drop = (ptq - fval) / max(abs(fval), 1e-9) * -0.1 if fval else 0.0
        ctx.log(f"size {v['size']}: post-training INT8 on validation {ptq} (float {fval})")
        if drop > 0.01:
            ctx.log(f"size {v['size']}: drop {drop:.3f} > 0.01, quantization-aware fine-tuning")
            act, _ = qexport.calibrate(v["net"], calib)
            backup = {k: t.clone() for k, t in v["net"].state_dict().items()}
            v["net"], _ = trainer.qat_finetune(v["net"], act, Xn, y, idx, task, epochs=max(5, epochs // 3),
                                               cancelled=ctx.cancelled, y_stats=y_stats)
            qat_path = ctx.dir / f"model_{v['size']}_qat.onnx"
            q_qat = qexport.export_qdq(v["net"], calib, qat_path, q=act)
            cand = {**v, "q": q_qat}
            qat = int8_metric(cand, qat_path, xv, yv)["value"]
            ctx.log(f"size {v['size']}: QAT INT8 on validation {qat}")
            if (qat or 0) > (ptq or 0):
                qat_path.replace(path)
                v["q"], v["quant_method"] = q_qat, "QAT"
            else:
                v["net"].load_state_dict(backup)
                qat_path.unlink(missing_ok=True)
        v["int8"] = int8_metric(v, path, xt, y[idx["test"]])
        ctx.log(f"size {v['size']}: INT8 ({v['quant_method']}, onnxruntime) test {v['int8']['headline']} = "
                f"{v['int8']['value']} (float {v['float']['value']})")
    ctx.stage("quantize", "done")

    # 4 plan -----------------------------------------------------------------------------------------------------
    ctx.stage("plan")
    cal = boards_mod.calibration()
    for i, v in enumerate(variants):
        v["plans"] = sai_engine.plan_search(v["onnx"], v["summary"]["macs"], v["summary"]["spatial_axes"], targets,
                                            goal if goal != "accuracy" else "balanced", p.get("budgets"), cal,
                                            log=ctx.log, measurements=boards_mod.measurements(),
                                            explore_builds=int(p.get("explore_builds", 3)))
        ctx.progress_within("plan", (i + 1) / len(variants))
        ctx.log(f"size {v['size']}: {v['plans']['feasible']} of {len(v['plans']['candidates'])} plans fit")
    ctx.stage("plan", "done")

    # 5 select ---------------------------------------------------------------------------------------------------
    ctx.stage("select")
    rows = []
    for v in variants:
        for c in v["plans"]["candidates"]:
            if c["fits"]:
                rows.append({**c, "size": v["size"], "metric": v["int8"]["value"]})
    if not rows:
        raise ValueError("no trained size has a plan that fits the board and your limits. "
                         + (variants[0]["plans"].get("advice") or ""))
    meets = [r for r in rows if acc_target in (None, "") or (r["metric"] or 0) >= float(acc_target)]
    note = None
    if not meets:
        note = f"no size reached the target {acc_target}; picked the best available"
        meets = [r for r in rows if r["metric"] == max(x["metric"] or 0 for x in rows)]
    if goal == "accuracy":
        top = max(r["metric"] or 0 for r in meets)
        chosen = sai_engine._pick([r for r in meets if (r["metric"] or 0) >= top - 1e-9], "balanced")
    else:
        chosen = sai_engine._pick(meets, goal)
    v = next(x for x in variants if x["size"] == chosen["size"])
    ctx.log(f"selected size {chosen['size']} on {chosen['target']} ({chosen['objective']}, {chosen['schedule']}): "
            f"RAM {chosen['ram']} B, flash {chosen['flash']} B, ~{chosen['latency']['ms']} ms, "
            f"INT8 {v['int8']['headline']} {v['int8']['value']}" + (f" - {note}" if note else ""))
    ctx.stage("select", "done", note)

    # 6 compile --------------------------------------------------------------------------------------------------
    ctx.stage("compile")
    build = ctx.dir / "build"
    if build.exists():
        shutil.rmtree(build)
    final = ctx.dir / "model.onnx"            # the delivered model; rebuilds compile this exact file
    shutil.copy(v["onnx"], final)
    man = sai_engine.compile_build(final, chosen["target"], chosen["objective"], build, chosen.get("plan_name"),
                                   chosen.get("candidate_id"))
    if chosen.get("plan_name"):
        ctx.log(f"built explored plan {chosen['plan_name']} (candidate {chosen['candidate_id'][:12]})")
    ctx.log(f"sai compile: {man.get('compile_status')} -> build/")
    ver = sai_engine.verify(build, final)
    checks = ver.get("checks", [])
    passed = sum(1 for c in checks if c.get("passed"))
    ctx.log(f"sai verify: {passed}/{len(checks)} checks passed")
    cert = _read_json(build / "memory-certificate.json")
    cert_check = sai_engine.certify_check(build, final)
    ctx.log(f"sai certify --check: {cert_check['passed']}/{cert_check['total']} certificate checks re-derived")
    ctx.stage("compile", "done")

    # 7 exact ----------------------------------------------------------------------------------------------------
    ctx.stage("exact")
    xq = qexport.quantize_input(xt, v["q"]["input"])
    oq = exact_eval.run_exact(v["onnx"], ctx.dir / "exact", xq)
    exact = trainer.metrics(task, y[idx["test"]], qexport.dequantize_output(oq, v["q"]["output"]), labels, y_stats,
                            xt.reshape(len(xt), -1), v["threshold"])
    ctx.log(f"bit-exact SAI program: {exact['headline']} = {exact['value']} on {len(xt)} test windows")
    n = min(32, len(xq))
    np.savez(ctx.dir / "hil_vectors.npz", x=xq[:n], expected=oq[:n])
    ctx.stage("exact", "done")

    # 8 package --------------------------------------------------------------------------------------------------
    ctx.stage("package")
    pre = {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist(), "input": v["q"]["input"],
           "output": v["q"]["output"], "task": task, "labels": labels, "y_stats": y_stats,
           "threshold": v["threshold"], "layout": "NCHW, 1-D signals as [1, C, 1, T]",
           "frontend": meta.get("frontend")}
    deployable = bool(meta.get("commercial_use"))
    if not deployable:
        ctx.log(f"dataset licence '{meta.get('licence')}' does not allow commercial use: package is EVALUATION ONLY")
    (ctx.dir / "preprocessing.json").write_text(json.dumps(pre, indent=1))
    (ctx.dir / "sai_app.h").write_text(_app_header(pre, tpl, deployable))
    if meta.get("frontend"):
        (ctx.dir / "sai_frontend_config.h").write_text(features.config_header(meta["frontend"]))
    (ctx.dir / "main_example.c").write_text(_example_main(pre, tpl))
    report = {
        "template": {k: tpl.get(k) for k in ("id", "name", "domain", "task")},
        "dataset": {k: meta.get(k) for k in ("id", "name", "source", "n", "shape", "labels", "class_counts", "synthetic",
                                             "licence", "commercial_use", "recordings", "quality", "frontend")},
        "deployable": deployable, "split_mode": idx["split_mode"],
        "goal": goal, "targets": targets, "budgets": p.get("budgets"), "metric_target": acc_target,
        "variants": [{"size": x["size"], "summary": {k: x["summary"][k] for k in ("macs", "params", "spatial_axes")},
                      "layers": x["summary"]["layers"], "spec": x["spec"], "float": x["float"], "int8": x["int8"],
                      "quant_method": x.get("quant_method"),
                      "history": x["history"], "plans": x["plans"]} for x in variants],
        "selected": {**chosen, "note": note},
        "exact": exact, "verify": {"passed": passed, "total": len(checks), "checks": checks},
        "memory_certificate": cert, "certificate_check": cert_check, "preprocessing": pre, "sai": sai_engine.version(),
        "provenance": provenance(build, p["dataset_id"], chosen),
        "tier_compare": sai_engine.tier_compare(v["onnx"], [chosen["target"]]),
    }
    (ctx.dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (ctx.dir / "MODEL_CARD.md").write_text(_model_card(report, tpl, meta))
    with zipfile.ZipFile(ctx.dir / "package.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(build.rglob("*")):
            if f.is_file():
                z.write(f, Path("sai_package") / f.relative_to(build))
        for name in ("model.onnx", "preprocessing.json", "sai_app.h", "main_example.c", "report.json", "MODEL_CARD.md"):
            z.write(ctx.dir / name, Path("sai_package") / name)
        if meta.get("frontend"):
            z.write(ctx.dir / "sai_frontend_config.h", Path("sai_package") / "frontend" / "sai_frontend_config.h")
            for f in ("sai_features.c", "sai_features.h"):
                z.write(features.SRC / f, Path("sai_package") / "frontend" / f)
        if not deployable:
            z.writestr("sai_package/EVALUATION_ONLY.txt", "Trained on data whose licence ("
                       f"{meta.get('licence')}) does not allow commercial use. Not for deployment.\n")
    ctx.log("package.zip ready")
    ctx.stage("package", "done")
    report["downloads"] = ["package.zip", "model.onnx", "report.json", "MODEL_CARD.md"]
    report["selected_model_path"] = str(v["onnx"])
    ctx.result(report)


def run_fitcheck(ctx: Ctx, p: dict):
    ctx.stage("build")
    spec = archspec.normalize(p["spec"])
    summ = archspec.summary(spec)
    ctx.log(f"{len(spec['layers'])} layers + head, {summ['params']:,} params, {summ['macs']:,} MACs")
    torch.manual_seed(0)
    net = archspec.SpecNet(spec)
    net.eval()
    net.fold_bn()
    ctx.stage("build", "done")
    ctx.stage("quantize")
    i = spec["input"]
    calib = np.random.default_rng(0).normal(0, 1, (64, i["channels"], i["height"], i["width"])).astype(np.float32)
    path = ctx.dir / "fitcheck.onnx"
    qexport.export_qdq(net, calib, path)
    ctx.log("random-weight INT8 graph exported (resources do not depend on trained weights; accuracy is not measured)")
    ctx.stage("quantize", "done")
    ctx.stage("plan")
    targets = [t for t in (p.get("targets") or sai_engine.MCU_TARGETS) if t in sai_engine.MCU_TARGETS]
    plans = sai_engine.plan_search(path, summ["macs"], summ["spatial_axes"], targets, p.get("goal", "balanced"),
                                   p.get("budgets"), boards_mod.calibration(), log=ctx.log,
                                   measurements=boards_mod.measurements())
    ctx.stage("plan", "done")
    ctx.result({"spec": spec, "summary": summ, "plans": plans})


def run_analyze(ctx: Ctx, p: dict):
    ctx.stage("inspect")
    path = Path(p["model_path"])
    info = sai_engine.inspect(path)
    macs = _onnx_macs(path) if path.suffix == ".onnx" else None
    ctx.log(f"{info.get('format')} model, {info.get('nodes')} nodes, {info.get('parameters')} parameters, "
            f"operators {info.get('operators')}" + (f", {macs:,} MACs" if macs else ""))
    ctx.stage("inspect", "done")
    ctx.stage("plan")
    axes = 1 if any(len(t.get("shape", [])) == 4 and t["shape"][2] == 1 for t in info.get("inputs", [])) else 2
    targets = [t for t in (p.get("targets") or sai_engine.MCU_TARGETS) if t in sai_engine.MCU_TARGETS]
    plans = sai_engine.plan_search(path, macs or 0, axes, targets, p.get("goal", "balanced"), p.get("budgets"),
                                   boards_mod.calibration(), log=ctx.log, measurements=boards_mod.measurements())
    if not macs:
        for c in plans["candidates"]:
            if c["latency"]["evidence"] == "heuristic":
                c["latency"] = {"cycles": None, "ms": None, "evidence": "unavailable",
                                "basis": "MAC count unknown for this format"}
    ctx.stage("plan", "done")
    ctx.result({"inspect": info, "macs": macs, "plans": plans})


# ------------------------------------------------------------------------------------------------------------------
def start_board_run(job_id: str, target: str | None, board_id: str | None, org_id: str | None = None) -> dict:
    job = get(job_id)
    if not job or job["status"] != "succeeded" or job["kind"] != "train":
        raise ValueError("board runs need a finished training job")
    res = job["result"]
    sel = res["selected"]
    target = target or sel["target"]
    cands = [c for v in res["variants"] if v["size"] == sel["size"] for c in v["plans"]["candidates"]
             if c["target"] == target]
    cand = next((c for c in cands if c["id"] == sel["id"]), None) or \
        next((c for c in cands if c["objective"] == sel["objective"]), cands[0] if cands else sel)
    vec = np.load(config.DATA_DIR / "jobs" / job_id / "hil_vectors.npz")
    macs = next(v["summary"]["macs"] for v in res["variants"] if v["size"] == sel["size"])
    jobdir = config.DATA_DIR / "jobs" / job_id
    if target == sel["target"]:
        build_dir = jobdir / "build"
    else:   # the board runs the same model compiled for its own target (SAI is bit-exact across targets)
        build_dir = jobdir / f"hil_build_{target}"
        if not (build_dir / "manifest.json").exists():
            sai_engine.compile_build(jobdir / "model.onnx", target, cand.get("objective") if cand.get("objective")
                                     in sai_engine.OBJECTIVES else "fast", build_dir,
                                     cand.get("plan_name"), cand.get("candidate_id"))
    req = {"model_path": res["selected_model_path"], "objective": cand.get("objective"), "macs": macs,
           "schedule": cand.get("schedule"), "estimated_cycles": cand["latency"]["cycles"],
           "estimated_ms": cand["latency"]["ms"], "estimate_evidence": cand["latency"]["evidence"],
           "stack_bound": cand.get("stack"), "expected_outputs": vec["expected"].tolist(),
           "plan_digest": cand.get("plan_digest"), "plan_name": cand.get("plan_name"), "build_dir": str(build_dir),
           "package": f"/api/jobs/{job_id}/download/package.zip"}
    return boards_mod.request_run(job_id, target, board_id, req, org_id)


def provenance(build: Path, dataset_id: str, chosen: dict) -> dict:
    """Everything needed to rebuild this package byte for byte and to trace where it came from."""
    import hashlib
    import platform as _pf
    man = _read_json(build / "manifest.json") or {}
    rep = _read_json(build / "compile-report.json") or {}
    npz = config.DATA_DIR / "datasets" / dataset_id / "data.npz"
    return {
        "platform_version": config.PLATFORM_VERSION, "python": _pf.python_version(), "torch": torch.__version__,
        "seeds": {"split": 0, "torch": 0, "calibration": 0},
        "dataset_sha256": hashlib.sha256(npz.read_bytes()).hexdigest() if npz.exists() else None,
        "sai_version": man.get("compiler_version"), "build_digest": man.get("build_digest"),
        "plan_digest": man.get("plan_digest"), "semantic_digest": man.get("semantic_digest"),
        "toolchain_digest": man.get("toolchain_digest"),
        "toolchain": ((rep.get("toolchain") or {}).get("build_record") or {}).get("toolchain"),
        "target": chosen["target"], "objective": chosen["objective"], "schedule": chosen["schedule"],
        "plan_name": chosen.get("plan_name"), "candidate_id": chosen.get("candidate_id"),
    }


def run_rebuild(ctx: Ctx, p: dict):
    """Recompile a finished project's model with the recorded target/objective; byte-identical or not."""
    src = get(p["job_id"])
    if not src or src["status"] != "succeeded" or src["kind"] != "train":
        raise ValueError("rebuild needs a finished training job")
    prov = (src["result"] or {}).get("provenance") or {}
    model = config.DATA_DIR / "jobs" / p["job_id"] / "model.onnx"
    ctx.stage("compile")
    out = ctx.dir / "build"
    if out.exists():
        shutil.rmtree(out)
    sai_engine.compile_build(model, prov["target"], prov["objective"], out, prov.get("plan_name"),
                             prov.get("candidate_id"))
    ctx.stage("compile", "done")
    ctx.stage("compare")
    man = _read_json(out / "manifest.json") or {}
    orig = config.DATA_DIR / "jobs" / p["job_id"] / "build"
    files = {}
    for f in sorted(x for x in orig.rglob("*") if x.is_file()):
        g = out / f.relative_to(orig)
        files[str(f.relative_to(orig))] = g.exists() and g.read_bytes() == f.read_bytes()
    same = man.get("build_digest") == prov.get("build_digest") and all(files.values())
    ctx.log(f"build digest {man.get('build_digest')} vs recorded {prov.get('build_digest')}: "
            f"{'identical' if same else 'DIFFERENT'}")
    for name, ok in files.items():
        if not ok:
            ctx.log(f"  differs: {name}")
    ctx.stage("compare", "done")
    ctx.result({"identical": same, "build_digest": man.get("build_digest"), "recorded": prov.get("build_digest"),
                "files": files, "sai_version": man.get("compiler_version"), "recorded_sai": prov.get("sai_version")})


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _onnx_macs(path: Path) -> int | None:
    try:
        import onnx
        m = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
        shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
                  for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
        inits = {i.name: list(i.dims) for i in m.graph.initializer}
        total = 0
        for n in m.graph.node:
            if n.op_type in ("Conv", "QLinearConv"):
                w = inits.get(n.input[1 if n.op_type == "Conv" else 3])
                out = shapes.get(n.output[0])
                if w and out:
                    total += int(np.prod(out[1:])) * int(np.prod(w[1:]))
            elif n.op_type in ("Gemm", "MatMul", "QLinearMatMul"):
                w = inits.get(n.input[1 if n.op_type != "QLinearMatMul" else 3])
                if w:
                    total += int(np.prod(w))
        return total or None
    except Exception:
        return None


def _app_header(pre: dict, tpl: dict, deployable: bool = True) -> str:
    mean, std = pre["mean"], pre["std"]
    s, z = pre["input"]["scale"], pre["input"]["zero_point"]
    a = [1.0 / (sd * s) for sd in std]
    b = [-m / (sd * s) + z for m, sd in zip(mean, std)]
    q16 = lambda v: int(round(v * 65536))  # noqa: E731
    labels = ", ".join(f'"{l}"' for l in pre["labels"])
    os_, oz = pre["output"]["scale"], pre["output"]["zero_point"]
    lines = [
        "/* Generated by SAI Platform: input quantization and output decoding for this model. */",
        "#ifndef SAI_APP_H", "#define SAI_APP_H", "#include <stdint.h>", "",
        *([] if deployable else ["#define SAI_PACKAGE_EVALUATION_ONLY 1",
                                 "#warning \"SAI package trained on data not licensed for commercial use: evaluation only\"", ""]),
        f"#define SAI_APP_CHANNELS {len(mean)}",
        f"#define SAI_APP_NUM_LABELS {len(pre['labels'])}",
        f"static const char *const sai_app_labels[] = {{ {labels} }};", "",
        "/* q = clamp(round(x * A[c] + B[c])) with A, B in Q16.16: integer-only, raw sensor units in. */",
        "static const int32_t sai_app_a_q16[] = { " + ", ".join(str(q16(v)) for v in a) + " };",
        "static const int32_t sai_app_b_q16[] = { " + ", ".join(str(q16(v)) for v in b) + " };", "",
        "static inline int8_t sai_app_quantize(int32_t raw, int c) {",
        "    int64_t v = (int64_t)raw * sai_app_a_q16[c] + sai_app_b_q16[c];",
        "    int32_t q = (int32_t)((v + (v >= 0 ? 32768 : -32768)) / 65536);",
        "    return (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);", "}", "",
        "/* Floating-point version for float sensor units (MCUs with an FPU). */",
        "static inline int8_t sai_app_quantize_f(float x, int c) {",
        "    static const float A[] = { " + ", ".join(f"{v:.9g}f" for v in a) + " };",
        "    static const float B[] = { " + ", ".join(f"{v:.9g}f" for v in b) + " };",
        "    float q = x * A[c] + B[c]; q = q < -128.f ? -128.f : q > 127.f ? 127.f : q;",
        "    return (int8_t)(q >= 0 ? q + 0.5f : q - 0.5f);", "}", "",
        f"#define SAI_APP_OUT_SCALE {os_:.9g}f", f"#define SAI_APP_OUT_ZP {oz}",
    ]
    if pre["task"] == "classification":
        lines += ["", "/* Argmax over INT8 logits (one output scale, so the order is the float order). */",
                  "static inline int sai_app_argmax(const int8_t *logits) {",
                  "    int best = 0; for (int i = 1; i < SAI_APP_NUM_LABELS; i++) if (logits[i] > logits[best]) best = i;",
                  "    return best;", "}"]
    elif pre["task"] == "regression":
        m, sd = pre["y_stats"]
        lines += ["", f"/* {tpl['labels'][0]} = ((q - zp) * scale) * {sd:.6g} + {m:.6g} */",
                  "static inline float sai_app_decode(int8_t q) {",
                  f"    return ((float)(q - SAI_APP_OUT_ZP) * SAI_APP_OUT_SCALE) * {sd:.9g}f + {m:.9g}f;", "}"]
    else:
        lines += ["", f"#define SAI_APP_ANOMALY_THRESHOLD {pre['threshold']:.9g}f",
                  "/* score = mean((deq(out) - normalized input)^2); anomaly when score > threshold. */"]
    lines += ["", "#endif", ""]
    return "\n".join(lines)


def _example_main(pre: dict, tpl: dict) -> str:
    fe = pre.get("frontend")
    if fe:
        return _example_main_frontend(pre, tpl, fe)
    return f"""/* Minimal integration of the generated model ({tpl['name']}). Link lib/libsai_model.a (or the sources
   in model/) and call sai_infer() once per window. No heap; the workspace is a static array. */
#include <stdint.h>
#include "model/sai_model.h"
#include "sai_app.h"

static uint8_t workspace[SAI_MODEL_WORKSPACE_BYTES + 1] __attribute__((aligned(SAI_MODEL_WORKSPACE_ALIGNMENT)));
static int8_t input[SAI_MODEL_INPUT0_BYTES];
static int8_t output[SAI_MODEL_OUTPUT0_BYTES];
static sai_model_t model;

int sai_setup(void) {{
    sai_run_context_t ctx = {{ workspace, SAI_MODEL_WORKSPACE_BYTES, 0 }};
    return sai_model_init(&model, &ctx) == SAI_OK ? 0 : -1;
}}

/* raw[c * T + t]: one window, channel-major, in sensor units. */
int sai_infer(const int32_t *raw, int T) {{
    for (int c = 0; c < SAI_APP_CHANNELS; c++)
        for (int t = 0; t < T; t++)
            input[c * T + t] = sai_app_quantize(raw[c * T + t], c);
    size_t ni, no;
    const int64_t *si = sai_model_input_shape(0, &ni), *so = sai_model_output_shape(0, &no);
    sai_tensor_view_t in = {{ input, sizeof input, si, ni, SAI_MODEL_INPUT0_DTYPE }};
    sai_tensor_buffer_t out = {{ output, sizeof output, so, no, SAI_MODEL_OUTPUT0_DTYPE }};
    if (sai_model_invoke(&model, &in, 1, &out, 1) != SAI_OK) return -1;
    {"return sai_app_argmax(output);" if pre["task"] == "classification" else "return 0; /* decode output[] with sai_app.h */"}
}}
"""


def _example_main_frontend(pre: dict, tpl: dict, fe: dict) -> str:
    call = ("sai_mfcc(raw, SAI_FRONTEND_RAW_LENGTH, &sai_frontend_mfcc, work, feat)" if fe["type"] == "mfcc" else
            "sai_band_energy(raw, SAI_FRONTEND_RAW_LENGTH, SAI_FRONTEND_N_FFT, SAI_FRONTEND_BANDS, work, feat)")
    return f"""/* {tpl['name']}: raw signal -> SAI feature front end (same C as training) -> model. No heap. */
#include <stdint.h>
#include "model/sai_model.h"
#include "sai_app.h"
#include "frontend/sai_frontend_config.h"

static uint8_t workspace[SAI_MODEL_WORKSPACE_BYTES + 1] __attribute__((aligned(SAI_MODEL_WORKSPACE_ALIGNMENT)));
static float work[SAI_FEAT_WORK_FLOATS];
static float feat[SAI_MODEL_INPUT0_BYTES];
static int8_t input[SAI_MODEL_INPUT0_BYTES];
static int8_t output[SAI_MODEL_OUTPUT0_BYTES];
static sai_model_t model;

int sai_setup(void) {{
    sai_run_context_t ctx = {{ workspace, SAI_MODEL_WORKSPACE_BYTES, 0 }};
    return sai_model_init(&model, &ctx) == SAI_OK ? 0 : -1;
}}

/* raw: SAI_FRONTEND_RAW_LENGTH samples ({'audio in [-1, 1]' if fe['type'] == 'mfcc' else 'sensor units'}). */
int sai_infer(const float *raw) {{
    if ({call} != SAI_FEAT_OK) return -1;
    for (unsigned i = 0; i < SAI_MODEL_INPUT0_BYTES; i++)
        input[i] = sai_app_quantize_f(feat[i], 0);
    size_t ni, no;
    const int64_t *si = sai_model_input_shape(0, &ni), *so = sai_model_output_shape(0, &no);
    sai_tensor_view_t in = {{ input, sizeof input, si, ni, SAI_MODEL_INPUT0_DTYPE }};
    sai_tensor_buffer_t out = {{ output, sizeof output, so, no, SAI_MODEL_OUTPUT0_DTYPE }};
    if (sai_model_invoke(&model, &in, 1, &out, 1) != SAI_OK) return -1;
    {"return sai_app_argmax(output);" if pre["task"] == "classification" else "return 0; /* decode output[] with sai_app.h */"}
}}
"""


def _regression(ctx, p):
    from . import regression
    regression.run(ctx, p)


def pool_weights(tid: str, size: str) -> dict | None:
    f = config.DATA_DIR / "pool" / tid / f"{size}.pt"
    return torch.load(f, map_location="cpu", weights_only=False) if f.exists() else None


def pool_weights_index() -> dict:
    root = config.DATA_DIR / "pool"
    out = {}
    for d in root.iterdir() if root.exists() else []:
        meta = d / "meta.json"
        if meta.exists():
            out[d.name] = json.loads(meta.read_text())
    return out


def run_basetrain(ctx: Ctx, p: dict):
    """Researcher job: train base weights of a pool model type (on a given dataset or its synthetic demo data)
    that customer projects can fine-tune from."""
    tpl = get_template(p["template_id"])
    task = tpl["task"]
    ctx.stage("data")
    if p.get("dataset_id"):
        did = p["dataset_id"]
    else:
        meta0 = data.from_synthetic(tpl, int(p.get("per_class", 300)))
        from . import store
        store.add_dataset(meta0, None)
        did = meta0["id"]
    X, y, labels, meta = data.load(did)
    idx = trainer.split(y, task, groups=data.load_groups(did))
    mean, std = trainer.norm_stats(X[idx["train"]])
    Xn = ((X - mean) / std).astype(np.float32)
    y_stats = (float(y[idx["train"]].mean()), float(y[idx["train"]].std() + 1e-6)) if task == "regression" else None
    ctx.stage("data", "done")
    ctx.stage("train")
    out_dir = config.DATA_DIR / "pool" / tpl["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for size in p.get("sizes") or list(tpl["variants"]):
        spec = trainer.build_spec(tpl["variants"][size], X, task, len(labels))
        net, _ = trainer.train(spec, Xn, y, idx, task, epochs=int(p.get("epochs", 30)), log=ctx.log,
                               cancelled=ctx.cancelled, y_stats=y_stats)
        torch.save({"state": net.state_dict(), "spec": spec, "labels": labels, "dataset": did,
                    "licence": meta.get("licence")}, out_dir / f"{size}.pt")
        saved[size] = {"best_val_loss": round(float(net.best_val), 5), "params": archspec.summary(spec)["params"]}
        ctx.log(f"size {size}: base weights stored (val loss {net.best_val:.4f})")
    ctx.stage("train", "done")
    ctx.stage("save")
    info = {"template": tpl["id"], "sizes": saved, "dataset": did, "licence": meta.get("licence"),
            "created": db.now(), "job": ctx.id}
    (out_dir / "meta.json").write_text(json.dumps(info, indent=1))
    ctx.stage("save", "done")
    ctx.result(info)


def _model_card(r: dict, tpl: dict, meta: dict) -> str:
    sel = r["selected"]
    v = next(x for x in r["variants"] if x["size"] == sel["size"])
    synthetic = "**Trained on SYNTHETIC demo data - not valid for field use.**\n\n" if meta.get("synthetic") else ""
    if not meta.get("commercial_use"):
        synthetic += f"**EVALUATION ONLY: dataset licence `{meta.get('licence')}` does not allow commercial deployment.**\n\n"
    return f"""# Model card - {tpl['name']}

{synthetic}| | |
|---|---|
| Task | {tpl['task']} ({tpl['domain']}) |
| Labels | {', '.join(meta.get('labels', []))} |
| Data | {meta.get('name')} - {meta.get('n')} windows, input {meta.get('shape')} |
| Size | {sel['size']} - {v['summary']['params']:,} parameters, {v['summary']['macs']:,} MACs |
| Target | {sel['target']} - plan `{sel['schedule']}` ({sel['objective']}) |
| Float {v['float']['headline']} | {v['float']['value']} |
| INT8 (onnxruntime) | {v['int8']['value']} |
| INT8 (bit-exact SAI program) | {r['exact']['value']} |
| RAM | {sel['ram']:,} B (workspace {sel['workspace']} [{sel['workspace_evidence']}], stack {sel['stack']} [{sel['stack_evidence']}]) |
| Flash | {sel['flash']} B [{sel['flash_evidence']}] |
| Latency | {sel['latency']['ms']} ms [{sel['latency']['evidence']}: {sel['latency']['basis']}] |
| SAI verify | {r['verify']['passed']}/{r['verify']['total']} checks |

Evidence labels: `static_exact` / `linked_exact` / `proven_upper_bound` come from SAI; `heuristic` and
`sai_cost_model` latencies are planning numbers until a board-farm run measures this build.

Intended use: {tpl['summary']} Limitations: accuracy holds for data like the training set only; validate on
your hardware and sensor chain. Labels and data provenance are the customer's responsibility.
"""
