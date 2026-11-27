# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Job orchestration: the end-to-end pipeline a customer starts with one click, plus the Studio fit check and the
model analyzer. Jobs run on a thread pool now; the job/stage/log model maps 1:1 onto a queue (Celery, Arq,
Cloud Tasks) when training moves to GPU workers."""

from __future__ import annotations

import json
import math
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

from . import archspec, config, data, db, exact_eval, features, preprocess, qexport, sai_engine, trainer
from . import boards as boards_mod

STAGES = {
    "train": [("data", "Validate data"), ("preprocess", "Preprocess & normalize"), ("train", "Train model sizes"), ("quantize", "INT8 quantization"),
              ("plan", "SAI plan search"), ("select", "Select model + plan"), ("compile", "Compile + verify"),
              ("exact", "Bit-exact accuracy"), ("package", "Package")],
    "fitcheck": [("build", "Build graph"), ("quantize", "INT8 export"), ("plan", "SAI plan search")],
    "analyze": [("inspect", "Inspect model"), ("plan", "SAI plan search")],
    "rebuild": [("compile", "Recompile"), ("compare", "Compare with the original build")],
    "basetrain": [("data", "Prepare data"), ("train", "Train base weights"), ("save", "Store in the pool")],
    "ingest": [("ingest", "Parse and validate"), ("store", "Store dataset")],
    "archverify": [("verify", "SAI check per architecture and board")],
    "multimodel": [("collect", "Collect models"), ("compile", "SAI multi-model build"), ("package", "Package")],
    "recommend": [("profile", "Profile the data"), ("generate", "Generate candidates"), ("cost", "SAI cost per board"),
                  ("pick", "Recommend")],
    "regression": [("plans", "Pool x boards plan check"), ("hil", "Emulated HIL smoke"), ("compare", "Compare with last run")],
}
KINDS = tuple(STAGES)
STALE_RUNNING_S = 30 * 60          # no heartbeat (log/save) for this long = the worker is gone


def _finite(o):
    """NaN/inf -> None, recursively, so results are valid JSON (the API refuses NaN)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.generic):
        return _finite(o.item())
    return o
GOALS = ("balanced", "fastest", "min-ram", "min-flash", "accuracy")


# ------------------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, job_id: str, kind: str):
        self.id, self.kind = job_id, kind
        self.dir = config.DATA_DIR / "jobs" / job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stages = [{"key": k, "title": t, "status": "pending"} for k, t in STAGES[kind]]
        with db.conn() as c:            # a requeued job continues its log numbering
            self.seq = c.execute("SELECT COALESCE(MAX(seq), 0) FROM job_logs WHERE job_id=?", (job_id,)).fetchone()[0]
            self.mseq = c.execute("SELECT COALESCE(MAX(seq), 0) FROM job_metrics WHERE job_id=?", (job_id,)).fetchone()[0]
        self._beat = 0.0

    def metric(self, series: str, data: dict):
        """One point of a live training curve (series e.g. "train:tiny", "qat:tiny")."""
        self.mseq += 1
        with db.conn() as c:
            c.execute("INSERT INTO job_metrics VALUES (?,?,?,?,?)", (self.id, self.mseq, db.now(), series, db.j(data)))

    def log(self, line: str):
        self.seq += 1
        now = db.now()
        with db.conn() as c:
            c.execute("INSERT INTO job_logs VALUES (?,?,?,?)", (self.id, self.seq, now, str(line)))
            if now - self._beat > 30:       # heartbeat: a job that logs is alive (see requeue_stale)
                c.execute("UPDATE jobs SET updated=? WHERE id=?", (now, self.id))
                self._beat = now

    def _save(self, **fields):
        sets = ", ".join(f"{k}=?" for k in fields) + ", updated=?"
        with db.conn() as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), db.now(), self.id))

    def stage(self, key: str, status: str = "running", note: str | None = None):
        if status == "running" and self.cancelled():
            raise InterruptedError("cancelled")
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
        self._save(result=json.dumps(_finite(res), default=str, allow_nan=False))


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


ACTIVE = ("queued", "running")


def active_jobs(kinds, match=lambda params: True, org_id: str | None = None) -> list[dict]:
    """Queued or running jobs of these kinds whose params satisfy `match` (oldest first)."""
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    q = ("SELECT id, kind, status, params, org_id FROM jobs WHERE status IN ('queued', 'running') AND kind IN ("
         + ",".join("?" * len(kinds)) + ")" + (" AND org_id=?" if org_id else "") + " ORDER BY created")
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute(q, (*kinds, *([org_id] if org_id else [])))]
    out = []
    for r in rows:
        r["params"] = db.uj(r["params"], {})
        if match(r["params"]):
            out.append(r)
    return out


def get(jid: str, log_after: int = 0, metrics_after: int | None = None) -> dict | None:
    with db.conn() as c:
        r = db.row(c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())
        if not r:
            return None
        logs = [dict(x) for x in c.execute("SELECT seq, ts, line FROM job_logs WHERE job_id=? AND seq>? ORDER BY seq",
                                           (jid, log_after))]
        if metrics_after is not None:
            r["metrics"] = [{"seq": m["seq"], "ts": m["ts"], "series": m["series"], **db.uj(m["data"], {})}
                            for m in c.execute("SELECT seq, ts, series, data FROM job_metrics WHERE job_id=? AND seq>?"
                                               " ORDER BY seq", (jid, metrics_after))]
    for k in ("stages", "result", "params"):
        r[k] = db.uj(r[k])
    r["logs"] = logs
    return r


def log_text(jid: str) -> str:
    with db.conn() as c:
        rows = c.execute("SELECT ts, line FROM job_logs WHERE job_id=? ORDER BY seq", (jid,)).fetchall()
    import datetime as _dt
    return "".join(f"{_dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z  {line}\n"
                   for t, line in rows)


def list_jobs(kind: str | None = None, limit: int = 50, org_id: str | None = None) -> list[dict]:
    where, args = [], []
    if kind:
        where.append("kind=?"); args.append(kind)
    if org_id:
        where.append("org_id=?"); args.append(org_id)
    q = ("SELECT id, kind, project_id, status, stage, progress, error, created, updated, params, org_id, worker,"
         " (SELECT line FROM job_logs l WHERE l.job_id=jobs.id ORDER BY seq DESC LIMIT 1) AS last_log,"
         " (SELECT COUNT(*) FROM job_logs l WHERE l.job_id=jobs.id AND (line LIKE '%warning%' OR line LIKE '%error%'))"
         " AS warnings FROM jobs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created DESC LIMIT ?")
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
    return {"by_status": {r[0]: r[1] for r in rows}, "busy_workers": workers, "paused": paused()}


# ---- workers -----------------------------------------------------------------------------------------------------
_WAKE = threading.Event()
_PAUSED = threading.Event()        # drain for maintenance: running jobs finish, nothing new starts in this process


def pause_workers(paused: bool):
    """Shared by every worker process (stored in the database)."""
    db.set_setting("queue_paused", bool(paused))
    (_PAUSED.set if paused else _PAUSED.clear)()


def paused() -> bool:
    return bool(db.get_setting("queue_paused", False)) or _PAUSED.is_set()


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
    """Never dies: an error in the loop itself (e.g. a locked database) is logged and retried after a pause."""
    stop = stop or threading.Event()
    last_reap = 0.0
    while not stop.is_set():
        try:
            if time.time() - last_reap > 60:
                requeue_stale()
                last_reap = time.time()
                if time.time() - float((db.get_setting("retention_last") or {}).get("at", 0)) > 3600:
                    from . import retention
                    retention.sweep()
            if paused():
                stop.wait(1.0)
                continue
            job = claim(name, kinds)
            if job:
                _run(job["id"], job["kind"], job["params"] or {})
                continue
            _WAKE.wait(1.0)
            _WAKE.clear()
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
            stop.wait(5.0)


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
    t0 = time.time()
    try:
        ctx = Ctx(jid, kind)
    except Exception as e:                                  # noqa: BLE001 - never leave the job "running"
        with db.conn() as c:
            c.execute("UPDATE jobs SET status='failed', error=?, updated=? WHERE id=?", (f"{type(e).__name__}: {e}", db.now(), jid))
        return
    try:
        {"train": run_train, "fitcheck": run_fitcheck, "analyze": run_analyze, "rebuild": run_rebuild,
         "basetrain": run_basetrain, "regression": _regression, "ingest": run_ingest,
         "recommend": run_recommend, "archverify": run_archverify, "multimodel": run_multimodel}[kind](ctx, params)
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
    tcfg = trainer.normalize_training(p.get("training"), p.get("epochs"))
    epochs = tcfg["epochs"]
    acc_target = p.get("metric_target")

    # 1 data ---------------------------------------------------------------------------------------------------
    ctx.stage("data")
    X, y, labels, meta = data.load(p["dataset_id"])
    ctx.log(f"dataset {meta.get('name')} : {len(X)} windows, input {list(X.shape[1:])}, labels {labels}")
    for w in meta.get("warnings", []):
        ctx.log("warning: " + w)
    pcfg = preprocess.normalize_cfg(p.get("preprocess"), meta, X.shape[1:], task)
    sp = pcfg["split"]
    groups = data.load_groups(p["dataset_id"])
    idx = trainer.split(y, task, groups=groups, frac=(sp["train"], sp["val"], sp["test"]), mode=sp["mode"])
    for name in ("train", "val", "test"):
        if len(idx[name]) == 0:
            raise ValueError(f"the {name} split is empty: add data or change the split fractions")
    y_stats = (float(y[idx["train"]].mean()), float(y[idx["train"]].std() + 1e-6)) if task == "regression" else None
    ctx.log(f"split train {len(idx['train'])} / val {len(idx['val'])} / test {len(idx['test'])} "
            f"({idx['split_mode']}-level: {'whole recordings per split' if idx['split_mode'] == 'recording' else 'windows'})")
    ctx.stage("data", "done")

    # 1b preprocess ------------------------------------------------------------------------------------------------
    ctx.stage("preprocess")
    raw_shape = list(X.shape[1:])
    X = preprocess.window_ops(X, pcfg)
    if pcfg["window_ops"]:
        ctx.log(f"window steps (C): channels {pcfg['channels'] or 'all'}, decimate {pcfg['decimate']}, "
                f"demean {pcfg['demean']}: {raw_shape} -> {list(X.shape[1:])}")
    off, sc = preprocess.fit_norm(X[idx["train"]], pcfg)
    Xn = preprocess.apply_norm(X, off, sc, pcfg["clip"])
    mean, std = off.reshape(1, -1, 1, 1), sc.reshape(1, -1, 1, 1)
    ctx.log(f"normalization {pcfg['normalization']} (fit on the train split)"
            + (f", clip +-{pcfg['clip']}" if pcfg["clip"] else "")
            + f"; augmentation {pcfg['augment']}; class weights {'on' if pcfg['class_weights'] else 'off'}")
    ctx.stage("preprocess", "done")

    # 2 train ----------------------------------------------------------------------------------------------------
    ctx.stage("train")
    variants = []
    custom = p.get("custom_variants") or {}
    from . import seqfront
    Xseq = None                         # [N, T, F] for sequence models (LSTM / GRU / attention), made on first use
    for si, size in enumerate(sizes):
        layers = custom.get(size) or tpl["variants"].get(size)
        if size in custom:
            ctx.log(f"size {size}: architecture from the recommender ({len(layers)} layers)")
        if not layers:
            continue
        try:
            spec = trainer.build_spec(layers, X, task, len(labels))
        except archspec.SpecError as e:
            ctx.log(f"size {size}: skipped ({e})")
            continue
        summ = archspec.summary(spec)
        seq = spec.get("layout") == "sequence"
        if seq and not seqfront.available():
            ctx.log(f"warning: size {size}: skipped, {seqfront.why_unavailable()}")
            continue
        if seq and Xseq is None:
            Xseq = archspec.to_sequence(Xn)
        Xin = Xseq if seq else Xn
        ctx.log(f"size {size}: {summ['params']:,} params, {summ['macs']:,} MACs"
                + (" (sequence model: SAI front end, XModels backend)" if seq else ""))
        base = pool_weights(tpl["id"], size) if p.get("init_from_pool") else None
        lrs = [3e-3, 1e-2] if p.get("lr_search") and not tcfg["lr"] else [None]
        best_net = None
        for lr in lrs:
            init = None
            if base is not None:
                init = archspec.SpecNet(spec)
                copied = trainer.init_from(init, base["state"])
                ctx.log(f"size {size}: fine-tuning from pool base weights ({len(copied)} tensors copied)")
            kw = {"lr": lr if lr else (2e-3 if base is not None else 5e-3)}
            run_cfg = {**tcfg, "lr": lr or tcfg["lr"]}
            series = f"train:{size}" + (f"@lr{kw['lr']:g}" if len(lrs) > 1 else "")
            net, hist = trainer.train(spec, Xin, y, idx, task, epochs=epochs, log=ctx.log, cancelled=ctx.cancelled,
                                      cfg=run_cfg, metric=lambda d, s=series: ctx.metric(s, d),
                                      augment=pcfg["augment"], class_weights=pcfg["class_weights"],
                                      progress=lambda f, si=si: ctx.progress_within("train", (si + f) / len(sizes)),
                                      y_stats=y_stats, net=init, **kw)
            if len(lrs) > 1:
                ctx.log(f"size {size}: lr {kw['lr']} -> best validation loss {net.best_val:.4f}")
            if best_net is None or net.best_val < best_net[0].best_val:
                best_net = (net, hist)
        net, hist = best_net
        with torch.no_grad():
            out = net(torch.from_numpy(Xin[idx["test"]])).numpy().reshape(len(idx["test"]), -1)
        thr = None
        if task == "anomaly":
            with torch.no_grad():
                vo = net(torch.from_numpy(Xin[idx["val"]])).numpy().reshape(len(idx["val"]), -1)
            thr = float(np.percentile(((vo - Xn[idx["val"]].reshape(len(idx["val"]), -1)) ** 2).mean(1), 99))
        fm = trainer.metrics(task, y[idx["test"]], out, labels, y_stats, Xn[idx["test"]].reshape(len(idx["test"]), -1), thr)
        ctx.log(f"size {size}: float {fm['headline']} = {fm['value']}")
        variants.append({"size": size, "spec": spec, "summary": summ, "net": net, "history": hist,
                         "float": fm, "threshold": thr, "seq": seq, "backend": "xmodels" if seq else "core"})
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
        if v["seq"]:                       # the SAI front end quantizes sequence models itself (sai-int/1)
            path = ctx.dir / f"model_{v['size']}.onnx"
            seqfront.export_float(v["net"], v["spec"], path)
            v["onnx"], v["quant_method"], v["int8"] = path, "SAI front end", dict(v["float"])
            ctx.log(f"size {v['size']}: float ONNX exported for the SAI front end (it quantizes with its own contract)")
            continue
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
        if tcfg["qat"] == "always" or (tcfg["qat"] == "auto" and drop > 0.01):
            ctx.log(f"size {v['size']}: " + ("quantization-aware fine-tuning (always on)" if tcfg["qat"] == "always"
                                              else f"drop {drop:.3f} > 0.01, quantization-aware fine-tuning"))
            act, _ = qexport.calibrate(v["net"], calib)
            backup = {k: t.clone() for k, t in v["net"].state_dict().items()}
            v["net"], _ = trainer.qat_finetune(v["net"], act, Xn, y, idx, task,
                                               epochs=tcfg["qat_epochs"] or max(5, epochs // 3), log=ctx.log,
                                               cancelled=ctx.cancelled, y_stats=y_stats, cfg=tcfg,
                                               metric=lambda d, s=f"qat:{v['size']}": ctx.metric(s, d))
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
        if task == "anomaly":      # the shipped threshold must match the INT8 program, not the float network
            xvq = qexport.quantize_input(xv, v["q"]["input"])
            vo = qexport.dequantize_output(qexport.ort_int8_outputs(path, xvq).reshape(len(xv), -1), v["q"]["output"])
            v["threshold"] = float(np.percentile(((vo - xv.reshape(len(xv), -1)) ** 2).mean(1), 99))
        v["int8"] = int8_metric(v, path, xt, y[idx["test"]])
        ctx.log(f"size {v['size']}: INT8 ({v['quant_method']}, onnxruntime) test {v['int8']['headline']} = "
                f"{v['int8']['value']} (float {v['float']['value']})")
    ctx.stage("quantize", "done")

    # 4 plan -----------------------------------------------------------------------------------------------------
    ctx.stage("plan")
    cal = boards_mod.calibration()
    seq_data = None
    for i, v in enumerate(variants):
        if v["seq"]:
            if seq_data is None:
                cal_i = np.random.default_rng(0).choice(idx["train"], size=min(512, len(idx["train"])), replace=False)
                seq_data = (ctx.dir / "seq_calib.npz", ctx.dir / "seq_test.npz")
                seqfront.write_data(seq_data[0], Xseq[cal_i], y[cal_i], task)
                seqfront.write_data(seq_data[1], Xseq[idx["test"]], y[idx["test"]], task)
            v["plans"] = seqfront.plans(v["onnx"], targets, goal, p.get("budgets"), ctx.dir / f"seq_{v['size']}",
                                        seq_data[0], seq_data[1], log=ctx.log)
            acc = next((c["accuracy"] for c in v["plans"]["candidates"] if c.get("accuracy")), None) or {}
            if task == "classification" and acc.get("integer_top1") is not None:
                v["int8"] = {**v["float"], "headline": "accuracy", "value": round(float(acc["integer_top1"]), 4),
                             "basis": "SAI front end integer oracle"}
            ctx.progress_within("plan", (i + 1) / len(variants))
            ctx.log(f"size {v['size']}: {v['plans']['feasible']} of {len(v['plans']['candidates'])} boards fit")
            continue
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
    note = None
    if all(r["metric"] is None for r in rows):        # e.g. anomaly data without anomalies in the test split
        meets = rows
        if acc_target not in (None, "") or goal == "accuracy":
            note = "the quality metric is undefined for this data (no test positives); chose by resources"
        goal = "balanced" if goal == "accuracy" else goal
    else:
        meets = [r for r in rows if acc_target in (None, "") or (r["metric"] if r["metric"] is not None else -1e9)
                 >= float(acc_target)]
        if not meets:
            note = f"no size reached the target {acc_target}; picked the best available"
            top = max(x["metric"] for x in rows if x["metric"] is not None)
            meets = [r for r in rows if r["metric"] == top]
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

    if v["seq"]:
        return _finish_sequence(ctx, p, tpl, meta, v, variants, chosen, note, goal, targets, acc_target, tcfg, idx, y,
                                labels, task, y_stats, pcfg, off, sc, mean, std, raw_shape, X, Xseq, xt)

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
           "preprocess": pcfg, "raw_shape": raw_shape, "model_input_shape": list(X.shape[1:]),
           "output": v["q"]["output"], "task": task, "labels": labels, "y_stats": y_stats,
           "threshold": v["threshold"], "layout": "NCHW, 1-D signals as [1, C, 1, T]",
           "frontend": meta.get("frontend")}
    from . import commercial
    with db.conn() as c:
        org_id = (c.execute("SELECT org_id FROM jobs WHERE id=?", (ctx.id,)).fetchone() or [None])[0]
    rights = commercial.deployment_rights(org_id)
    deployable = bool(meta.get("commercial_use")) and rights
    eval_reason = None if deployable else (
        f"dataset licence '{meta.get('licence')}' does not allow commercial use" if not meta.get("commercial_use")
        else "the organisation has no SAI deployment rights yet (royalty or buyout)")
    if not deployable:
        ctx.log(f"{eval_reason}: package is EVALUATION ONLY")
    (ctx.dir / "preprocessing.json").write_text(json.dumps(pre, indent=1))
    (ctx.dir / "sai_app.h").write_text(_app_header(pre, tpl, deployable))
    (ctx.dir / "sai_app_prepare.h").write_text(preprocess.c_prepare(pcfg, off, sc, v["q"]["input"],
                                                                     raw_shape if pcfg["window_ops"] else list(X.shape[1:])))
    if meta.get("frontend"):
        (ctx.dir / "sai_frontend_config.h").write_text(features.config_header(meta["frontend"]))
    (ctx.dir / "main_example.c").write_text(_example_main(pre, tpl))
    report = {
        "template": {k: tpl.get(k) for k in ("id", "name", "domain", "task", "version", "source")},
        "pool_benchmark": bool(p.get("pool_benchmark")),
        "dataset": {k: meta.get(k) for k in ("id", "name", "source", "n", "shape", "labels", "class_counts", "synthetic",
                                             "licence", "commercial_use", "recordings", "quality", "frontend")},
        "deployable": deployable, "evaluation_reason": eval_reason, "split_mode": idx["split_mode"],
        "goal": goal, "targets": targets, "budgets": p.get("budgets"), "metric_target": acc_target,
        "training": tcfg,
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
        for name in ("model.onnx", "preprocessing.json", "sai_app.h", "sai_app_prepare.h", "main_example.c",
                     "report.json", "MODEL_CARD.md"):
            z.write(ctx.dir / name, Path("sai_package") / name)
        for f in ("sai_prep.c", "sai_prep.h"):
            z.write(preprocess.SRC / f, Path("sai_package") / f)
        if meta.get("frontend"):
            z.write(ctx.dir / "sai_frontend_config.h", Path("sai_package") / "frontend" / "sai_frontend_config.h")
            for f in ("sai_features.c", "sai_features.h"):
                z.write(features.SRC / f, Path("sai_package") / "frontend" / f)
        if not deployable:
            z.writestr("sai_package/EVALUATION_ONLY.txt", f"Evaluation only: {eval_reason}. Not for deployment.\n")
    sig = commercial.sign_package(ctx.dir / "package.zip", {
        "job": ctx.id, "org": org_id, "deployable": deployable, "platform_version": config.PLATFORM_VERSION,
        "sai_version": (man or {}).get("compiler_version"), "plan_digest": (man or {}).get("plan_digest"),
        "target": chosen["target"]})
    ctx.log(f"package signed (Ed25519, key {sig['key_id']})")
    ctx.log("package.zip ready")
    ctx.stage("package", "done")
    report["downloads"] = ["package.zip", "model.onnx", "report.json", "MODEL_CARD.md"]
    report["signature"] = sig
    report["selected_model_path"] = str(ctx.dir / "model.onnx")
    from . import retention
    freed = retention.job_intermediates(ctx.dir)
    if freed:
        ctx.log(f"removed {freed / 1e6:.1f} MB of intermediates (per-size models, evaluation build)")
    ctx.result(report)


def _finish_sequence(ctx, p, tpl, meta, v, variants, chosen, note, goal, targets, acc_target, tcfg, idx, y, labels,
                     task, y_stats, pcfg, off, sc, mean, std, raw_shape, X, Xseq, xt):
    """Compile, bit-exact evaluation and package for a sequence model built by SAI's front end (XModels)."""
    from . import commercial, seqfront
    ctx.stage("compile")
    build = ctx.dir / "build"
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(chosen["build_dir"], build)
    final = ctx.dir / "model.onnx"
    shutil.copy(v["onnx"], final)
    ver = chosen.get("verification") or {}
    checks = [{"name": "integer oracle vs generated C (host)", "passed": ver.get("mismatches") == 0,
               "detail": f"{ver.get('vectors')} vectors, {ver.get('mismatches')} mismatches"},
              {"name": "sanitizers (ASan / UBSan)", "passed": "clean" in str(ver.get("sanitized", "")),
               "detail": ver.get("sanitized")}]
    q7 = ver.get("cortex_m7_qemu") or {}
    if q7:
        checks.append({"name": "Cortex-M7 build on QEMU", "passed": "exact" in str(q7.get("result", "")),
                       "detail": q7.get("result")})
    passed = sum(1 for c in checks if c["passed"])
    ctx.log(f"SAI front end build for {chosen['target']}: plan {chosen['schedule']}, {passed}/{len(checks)} checks "
            "passed (output EVALUATION_ONLY: the front end is not qualified yet)")
    ctx.stage("compile", "done")

    ctx.stage("exact")
    outs, q = seqfront.run_exact(build / "library", Xseq[idx["test"]], ctx.dir / "exact")
    exact = trainer.metrics(task, y[idx["test"]], outs, labels, y_stats, xt.reshape(len(xt), -1), v["threshold"])
    exact["basis"] = "generated C run on the host (sai_xm_invoke), every test window"
    ctx.log(f"bit-exact generated C: {exact['headline']} = {exact['value']} on {len(xt)} test windows")
    ctx.stage("exact", "done")

    ctx.stage("package")
    seq_width = 1 if X.shape[2] == 1 else int(X.shape[3])
    pre = {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist(), "input": q["input"],
           "preprocess": pcfg, "raw_shape": raw_shape, "model_input_shape": list(Xseq.shape[1:]),
           "output": q["output"], "task": task, "labels": labels, "y_stats": y_stats, "threshold": v["threshold"],
           "layout": "[time, features] (sequence model)", "frontend": meta.get("frontend")}
    with db.conn() as c:
        org_id = (c.execute("SELECT org_id FROM jobs WHERE id=?", (ctx.id,)).fetchone() or [None])[0]
    reasons = ["SAI's front end and XModels backend are not qualified yet (their own status: EVALUATION_ONLY)"]
    if not meta.get("commercial_use"):
        reasons.append(f"dataset licence '{meta.get('licence')}' does not allow commercial use")
    if not commercial.deployment_rights(org_id):
        reasons.append("the organisation has no SAI deployment rights yet (royalty or buyout)")
    eval_reason = "; ".join(reasons)
    ctx.log(f"package is EVALUATION ONLY: {eval_reason}")
    (ctx.dir / "preprocessing.json").write_text(json.dumps(pre, indent=1))
    (ctx.dir / "sai_app.h").write_text(_app_header(pre, tpl, False))
    (ctx.dir / "sai_app_prepare.h").write_text(preprocess.c_prepare(
        pcfg, off, sc, q["input"], raw_shape if pcfg["window_ops"] else list(X.shape[1:]), seq_width=seq_width))
    (ctx.dir / "main_example.c").write_text(seqfront.EXAMPLE_MAIN)
    front = chosen.get("accuracy") or {}
    report = {
        "template": {k: tpl.get(k) for k in ("id", "name", "domain", "task", "version", "source")},
        "pool_benchmark": bool(p.get("pool_benchmark")),
        "dataset": {k: meta.get(k) for k in ("id", "name", "source", "n", "shape", "labels", "class_counts", "synthetic",
                                             "licence", "commercial_use", "recordings", "quality", "frontend")},
        "deployable": False, "evaluation_reason": eval_reason, "split_mode": idx["split_mode"], "backend": "xmodels",
        "goal": goal, "targets": targets, "budgets": p.get("budgets"), "metric_target": acc_target, "training": tcfg,
        "variants": [{"size": x["size"], "summary": {k: x["summary"][k] for k in ("macs", "params", "spatial_axes")},
                      "layers": x["summary"]["layers"], "spec": x["spec"], "float": x["float"], "int8": x["int8"],
                      "quant_method": x.get("quant_method"), "backend": x["backend"],
                      "history": x["history"], "plans": x["plans"]} for x in variants],
        "selected": {**chosen, "note": note},
        "exact": exact, "verify": {"passed": passed, "total": len(checks), "checks": checks},
        "front_end": {"accuracy": front, "route": "xmodels", "techniques": None},
        "memory_certificate": None, "certificate_check": None, "preprocessing": pre, "sai": sai_engine.version(),
        "provenance": {"platform_version": config.PLATFORM_VERSION, "torch": torch.__version__,
                       "seeds": {"split": 0, "torch": int(tcfg.get("seed", 0)), "calibration": 0},
                       "compiler": "SAI front end (sai_front) + XModels", "target": chosen["target"],
                       "objective": chosen["objective"], "schedule": chosen["schedule"]},
        "tier_compare": None,
    }
    (ctx.dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (ctx.dir / "MODEL_CARD.md").write_text(_model_card(report, tpl, meta))
    with zipfile.ZipFile(ctx.dir / "package.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(build.rglob("*")):
            if f.is_file():
                z.write(f, Path("sai_package") / f.relative_to(build))
        for name in ("model.onnx", "preprocessing.json", "sai_app.h", "sai_app_prepare.h", "main_example.c",
                     "report.json", "MODEL_CARD.md"):
            z.write(ctx.dir / name, Path("sai_package") / name)
        for f in ("sai_prep.c", "sai_prep.h"):
            z.write(preprocess.SRC / f, Path("sai_package") / f)
        z.writestr("sai_package/EVALUATION_ONLY.txt", f"Evaluation only: {eval_reason}. Not for deployment.\n")
    sig = commercial.sign_package(ctx.dir / "package.zip", {
        "job": ctx.id, "org": org_id, "deployable": False, "platform_version": config.PLATFORM_VERSION,
        "sai_version": "sai_front", "plan_digest": None, "target": chosen["target"]})
    ctx.log(f"package signed (Ed25519, key {sig['key_id']}); package.zip ready")
    ctx.stage("package", "done")
    report["downloads"] = ["package.zip", "model.onnx", "report.json", "MODEL_CARD.md"]
    report["signature"] = sig
    report["selected_model_path"] = str(final)
    from . import retention
    retention.job_intermediates(ctx.dir)
    ctx.result(report)


def run_fitcheck(ctx: Ctx, p: dict):
    ctx.stage("build")
    spec = archspec.normalize(p["spec"])
    summ = archspec.summary(spec)
    ctx.log(f"{len(spec['layers'])} layers + head, {summ['params']:,} params, {summ['macs']:,} MACs")
    if spec.get("layout") == "sequence":          # SAI's front end (XModels) builds sequence models
        from . import archlib
        ctx.stage("build", "done")
        ctx.stage("quantize")
        ctx.log("sequence model: float graph with random weights; SAI's front end quantizes it and plans every board")
        ctx.stage("quantize", "done")
        ctx.stage("plan")
        targets = [t for t in (p.get("targets") or sai_engine.MCU_TARGETS) if t in sai_engine.MCU_TARGETS]
        plans = archlib.random_plans(spec, targets, ctx.dir, p.get("goal", "balanced"), p.get("budgets"),
                                     log=ctx.log, name="fitcheck")
        ctx.stage("plan", "done")
        ctx.result({"spec": spec, "summary": summ, "plans": plans})
        return
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
def start_board_run(job_id: str, target: str | None, board_id: str | None, org_id: str | None = None,
                    allowed_kinds: list[str] | None = None) -> dict:
    job = get(job_id)
    if not job or job["status"] != "succeeded" or job["kind"] != "train":
        raise ValueError("board runs need a finished training job")
    res = job["result"]
    if res.get("backend") == "xmodels":
        raise ValueError("board runs of sequence models (SAI front end, XModels) are not available yet: the lab "
                         "harness runs SAI Core packages; the front end verified this build on QEMU")
    sel = res["selected"]
    target = target or sel["target"]
    if target not in sai_engine.MCU_TARGETS:
        raise ValueError(f"target must be one of {sai_engine.MCU_TARGETS}")
    if board_id:
        b = next((x for x in boards_mod.list_boards() if x["id"] == board_id), None)
        if not b or b["target"] != target:
            raise ValueError("that board does not exist or serves another target")
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
           "allowed_kinds": allowed_kinds,
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
    if (src["result"] or {}).get("backend") == "xmodels":
        raise ValueError("rebuild compares SAI Core builds; this project was built by SAI's front end (XModels)")
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


def _cf(v) -> str:
    """A C float literal ("5000" -> "5000.0f"; inf/nan cannot occur: the header is generated from finite values)."""
    t = f"{float(v):.9g}"
    return (t if any(ch in t for ch in ".en") else t + ".0") + "f"


def _c_str(text: str) -> str:
    """A C string literal for any label (quotes, backslashes, newlines, non-ASCII escaped)."""
    out = []
    for ch in str(text):
        if ch in '"\\':
            out.append("\\" + ch)
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.extend(f"\\x{b:02x}\"\"" for b in ch.encode())
    return '"' + "".join(out) + '"'


def _app_header(pre: dict, tpl: dict, deployable: bool = True) -> str:
    mean, std = pre["mean"], pre["std"]
    s, z = pre["input"]["scale"], pre["input"]["zero_point"]
    a = [1.0 / (sd * s) for sd in std]
    b = [-m / (sd * s) + z for m, sd in zip(mean, std)]
    q16 = lambda v: int(round(v * 65536))  # noqa: E731
    labels = ", ".join(_c_str(lb) for lb in pre["labels"])
    int_ok = all(abs(v) * 65536 < 2 ** 31 - 1 for v in a + b)
    os_, oz = pre["output"]["scale"], pre["output"]["zero_point"]
    lines = [
        "/* Generated by SAI Platform: input quantization and output decoding for this model. */",
        "#ifndef SAI_APP_H", "#define SAI_APP_H", "#include <stdint.h>", "",
        *([] if deployable else ["#define SAI_PACKAGE_EVALUATION_ONLY 1",
                                 "#warning \"SAI package trained on data not licensed for commercial use: evaluation only\"", ""]),
        f"#define SAI_APP_CHANNELS {len(mean)}",
        f"#define SAI_APP_NUM_LABELS {len(pre['labels'])}",
        f"static const char *const sai_app_labels[] = {{ {labels} }};", "",
        *(["/* q = clamp(round(x * A[c] + B[c])) with A, B in Q16.16: integer-only, raw sensor units in",
            "   (approximate: use sai_app_prepare() from sai_app_prepare.h for the exact training preprocessing). */",
            "static const int32_t sai_app_a_q16[] = { " + ", ".join(str(q16(v)) for v in a) + " };",
            "static const int32_t sai_app_b_q16[] = { " + ", ".join(str(q16(v)) for v in b) + " };", "",
            "static inline int8_t sai_app_quantize(int32_t raw, int c) {",
            "    int64_t v = (int64_t)raw * sai_app_a_q16[c] + sai_app_b_q16[c];",
            "    int32_t q = (int32_t)((v + (v >= 0 ? 32768 : -32768)) / 65536);",
            "    return (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);", "}", ""] if int_ok else
          ["/* No Q16 integer path: this input scaling does not fit Q16.16. Use sai_app_prepare() (float). */", ""]),
        "/* Floating-point version for float sensor units (MCUs with an FPU). */",
        "static inline int8_t sai_app_quantize_f(float x, int c) {",
        "    static const float A[] = { " + ", ".join(_cf(v) for v in a) + " };",
        "    static const float B[] = { " + ", ".join(_cf(v) for v in b) + " };",
        "    float q = x * A[c] + B[c]; q = q < -128.f ? -128.f : q > 127.f ? 127.f : q;",
        "    return (int8_t)(q >= 0 ? q + 0.5f : q - 0.5f);", "}", "",
        f"#define SAI_APP_OUT_SCALE {_cf(os_)}", f"#define SAI_APP_OUT_ZP {oz}",
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
                  f"    return ((float)(q - SAI_APP_OUT_ZP) * SAI_APP_OUT_SCALE) * {_cf(sd)} + {_cf(m)};", "}"]
    else:
        lines += ["", f"#define SAI_APP_ANOMALY_THRESHOLD {_cf(pre['threshold'])}",
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


def run_multimodel(ctx: Ctx, p: dict):
    """Several finished projects on one board with one shared workspace (SAI's multi-model plugin)."""
    from . import commercial, multimodel
    if not multimodel.available():
        raise ValueError(multimodel.why_unavailable())
    target, budgets = p["target"], {k: v for k, v in (p.get("budgets") or {}).items() if v}
    ctx.stage("collect")
    wl = ctx.dir / "workload"
    used, models, reasons = set(), [], []
    for jid in p["job_ids"]:
        j = get(jid)
        res = j["result"] or {}
        name = (j["params"] or {}).get("name") or jid
        mid = multimodel.model_id(name, used)
        dst = wl / "models" / f"{mid}.onnx"
        multimodel.copy_model(config.DATA_DIR / "jobs" / jid, dst)
        models.append({"id": mid, "job_id": jid, "project": name, "template_id": (j["params"] or {}).get("template_id"),
                       "sha256": multimodel.sha256(dst)})
        if res.get("deployable") is False and res.get("evaluation_reason"):
            reasons.append(f"{name}: {res['evaluation_reason']}")
        ctx.log(f"model {mid}: project {name} ({jid}), {res.get('selected', {}).get('target')} build, "
                f"sha256 {models[-1]['sha256'][:12]}")
    (wl / "workload.yaml").write_text(multimodel.workload(p.get("name"), models, target, budgets))
    ctx.log(f"workload: {len(models)} models, run one after another on {target}, minimum peak RAM, shared workspace")
    ctx.stage("collect", "done")

    ctx.stage("compile")
    r = multimodel.run(["inspect", "workload.yaml"], wl, timeout=600)
    for line in (r.stdout or "").strip().splitlines()[-4:]:
        ctx.log("inspect: " + line.strip())
    out = ctx.dir / "bundle"
    r = multimodel.run(["eval-bundle", "workload.yaml", "--target", target, "--output", str(out)], wl)
    if r.returncode or not (out / "eval-bundle.json").exists():
        msg = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        raise ValueError(f"sai-multimodel eval-bundle failed (exit {r.returncode}): " + (msg[-1] if msg else ""))
    bundle = json.loads((out / "eval-bundle.json").read_text())
    summ = multimodel.summarize(bundle, models, target, budgets)
    for m in summ["models"]:
        ctx.log(f"{m['id']}: ABI {m['abi']}, workspace {m['workspace']} B, stack {m['stack']} B, "
                f"flash {m['linked_flash']} B")
    ctx.log(f"one shared build: {summ['device_ram']:,} B device RAM ({summ['evidence']}) vs {summ['separate_ram']:,} B "
            f"side by side: saves {summ['saved_ram']:,} B; flash {summ['flash']:,} B; SAI verify "
            f"{summ['verify']['passed']}/{summ['verify']['total']} builds")
    for v in summ["violations"]:
        ctx.log("warning: " + v)
    ctx.stage("compile", "done")

    ctx.stage("package")
    reason = "; ".join(["SAI's multi-model plugin output is EVALUATION_ONLY (test signatures, no board run yet)"]
                       + reasons)
    report = {"kind": "multimodel", "name": p.get("name"), **summ, "deployable": False,
              "evaluation_reason": reason, "sai": bundle.get("sai"), "budgets": budgets,
              "workload": (wl / "workload.yaml").read_text()}
    (ctx.dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    multimodel.package(ctx.dir, out, report, reason)
    with db.conn() as c:
        org_id = (c.execute("SELECT org_id FROM jobs WHERE id=?", (ctx.id,)).fetchone() or [None])[0]
    sig = commercial.sign_package(ctx.dir / "package.zip", {
        "job": ctx.id, "org": org_id, "deployable": False, "platform_version": config.PLATFORM_VERSION,
        "sai_version": (bundle.get("sai") or {}).get("version"), "plan_digest": None, "target": target})
    ctx.log(f"package signed (Ed25519, key {sig['key_id']}); package.zip ready")
    ctx.stage("package", "done")
    report["downloads"] = ["package.zip", "report.json"]
    report["signature"] = sig
    shutil.rmtree(wl / "models", ignore_errors=True)       # copies of the projects' models; the bundle keeps builds
    ctx.result(report)


def run_archverify(ctx: Ctx, p: dict):
    """Check library architectures with SAI at their reference input on every board (see backend/archlib.py)."""
    from . import archlib
    ctx.stage("verify")
    archs = [a for a in archlib.all_visible(None, platform=True) if not p.get("ids") or a["id"] in p["ids"]]
    targets = [t for t in p.get("targets") or sai_engine.MCU_TARGETS if t in sai_engine.MCU_TARGETS]
    ctx.log(f"{len(archs)} architecture(s) x {len(targets)} board(s), SAI {sai_engine.version().get('sai')}")
    res, failed = {}, []
    for i, a in enumerate(archs):
        if ctx.cancelled():
            raise InterruptedError("cancelled")
        try:
            res[a["id"]] = archlib.verify(a, targets, ctx.dir, log=ctx.log)
        except Exception as e:            # one bad entry must not stop the others; it is reported
            failed.append({"id": a["id"], "error": str(e)})
            ctx.log(f"warning: {a['id']}: {e}")
        ctx.progress_within("verify", (i + 1) / len(archs))
    ctx.stage("verify", "done")
    ctx.result({"checked": len(res), "failed": failed, "targets": targets,
                "fits": {aid: [t for t, r in v.items() if r["fits"]] for aid, v in res.items()}})


def run_recommend(ctx: Ctx, p: dict):
    """Recommend an architecture for this dataset, boards and goal, costed exactly by SAI before any training."""
    from . import recommend
    tpl = get_template(p["template_id"])
    task = tpl["task"]
    X, y, labels, meta = data.load(p["dataset_id"])
    pcfg = preprocess.normalize_cfg(p.get("preprocess"), meta, X.shape[1:], task)
    X = preprocess.window_ops(X, pcfg)
    rate = (tpl.get("input") or {}).get("sample_rate_hz")
    if rate and pcfg.get("decimate", 1) > 1:
        rate = rate / pcfg["decimate"]
    n_out = len(labels) if task == "classification" else 1 if task == "regression" else int(np.prod(X.shape[1:]))
    targets = [t for t in p.get("targets") or tpl.get("targets") or list(sai_engine.MCU_TARGETS) if t in sai_engine.MCU_TARGETS]
    ctx.log(f"dataset {meta.get('name')}: {len(X)} windows, input {list(X.shape[1:])}; boards {', '.join(targets)}; "
            f"goal {p.get('goal', 'balanced')}")
    from . import archlib
    lib = [(a["id"], a["layers"]) for a in archlib.all_visible(p.get("_org_id"), platform=False)
           if not archspec.is_sequence({"layers": a["layers"]}) and archlib.compatible(a, X.shape[1:], task, n_out)["ok"]]
    res = recommend.recommend(X, y, task, labels, tpl, n_out, p.get("goal", "balanced"), targets, p.get("budgets"),
                              ctx.dir, library=lib, sample_rate_hz=rate, n_train=int(round(len(X) * pcfg["split"]["train"])),
                              max_candidates=int(p.get("max_candidates") or 8), calibration=boards_mod.calibration(),
                              measurements=boards_mod.measurements(), log=ctx.log, cancelled=ctx.cancelled,
                              stage=ctx.stage)
    rec = res["recommended"]
    ctx.log(f"recommended {rec['id']}: {res['why']}" if rec else res["why"])
    ctx.result(_finite(res))


def run_ingest(ctx: Ctx, p: dict):
    """Large uploads are parsed by a worker, not by the API process: the file was streamed to uploads/."""
    from . import store
    path = Path(p["path"])
    if config.DATA_DIR / "uploads" not in path.resolve().parents:
        raise ValueError("bad staged upload path")
    ctx.stage("ingest")
    tpl = get_template(p["template_id"])
    ctx.log(f"parsing {p['filename']} ({path.stat().st_size / 1e6:.1f} MB) for {tpl['name']}")
    try:
        meta = data.ingest_bytes(path.read_bytes(), p["filename"], tpl, window=p.get("window"), hop=p.get("hop"),
                                 mapping=p.get("mapping"), licence=p.get("licence") or "unknown")
    finally:
        path.unlink(missing_ok=True)
    ctx.log(f"{meta['n']:,} windows, input {meta['shape']}, {len(meta.get('warnings', []))} warning(s)")
    ctx.stage("ingest", "done")
    ctx.stage("store")
    with db.conn() as c:
        org_id = (c.execute("SELECT org_id FROM jobs WHERE id=?", (ctx.id,)).fetchone() or [None])[0]
    store.add_dataset(meta, org_id)
    ctx.stage("store", "done")
    ctx.result({"dataset": {k: v for k, v in meta.items() if k != "preview"}, "dataset_id": meta["id"]})


def _regression(ctx, p):
    from . import regression
    regression.run(ctx, p)


def leaderboard() -> dict:
    """Pool program leaderboard: per model type, every pool-benchmark project (real data, run by SAI research)
    with its bit-exact accuracy and the resources of the selected plan; best first."""
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute("SELECT id, params, result, updated FROM jobs WHERE kind='train' AND"
                                             " status='succeeded' ORDER BY updated DESC LIMIT 500")]
    out: dict[str, list] = {}
    for r in rows:
        params, res = db.uj(r["params"], {}), db.uj(r["result"], {})
        if not params.get("pool_benchmark") or not res:
            continue
        sel, ds = res["selected"], res.get("dataset") or {}
        v = next((x for x in res["variants"] if x["size"] == sel["size"]), {})
        out.setdefault(res["template"]["id"], []).append({
            "job_id": r["id"], "updated": r["updated"], "size": sel["size"], "target": sel["target"],
            "metric": res["exact"]["headline"], "value": res["exact"]["value"], "float": v.get("float", {}).get("value"),
            "ram": sel["ram"], "flash": sel["flash"], "latency": sel["latency"], "schedule": sel["schedule"],
            "dataset": ds.get("name"), "licence": ds.get("licence"), "recordings": ds.get("recordings"),
            "n": ds.get("n"), "split": res.get("split_mode"), "synthetic": bool(ds.get("synthetic")),
            "template_version": res["template"].get("version"), "params": v.get("summary", {}).get("params"),
            "macs": v.get("summary", {}).get("macs")})
    for k in out:
        out[k].sort(key=lambda e: (e["synthetic"], -(e["value"] or 0), e["ram"] or 1e12))
    return out


def pool_weights(tid: str, size: str) -> dict | None:
    f = config.DATA_DIR / "pool" / tid / f"{size}.pt"
    return torch.load(f, map_location="cpu", weights_only=True) if f.exists() else None


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
    if not r.get("deployable", True):
        synthetic += f"**EVALUATION ONLY: {r.get('evaluation_reason')}.**\n\n"
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
