# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""SAI Platform HTTP API and static frontend.

    uvicorn backend.app:app --port 8000

Every /api route except /api/health, /api/auth/login and the lab-agent routes needs a session (cookie or bearer).
Customer data (datasets, projects, jobs, board runs) is scoped to the caller's organisation.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

from . import archspec, auth, boards, commercial, config, data, db, pipeline, sai_engine, store

app = FastAPI(title=config.BRAND_NAME, version=config.PLATFORM_VERSION)
FRONT = config.ROOT / "frontend"
_started = {"sim": None, "workers": []}
SINGLE_ORG = "org_customer"


@app.on_event("startup")
def _startup():
    db.init()
    auth.bootstrap()
    if config.MODE == "single" and auth.org(SINGLE_ORG) is None:
        auth.create_org(config.SINGLE_ORG_NAME, "professional", SINGLE_ORG)
    if config.SIM_BOARDS and _started["sim"] is None:
        # emulated QEMU boards when the toolchains + QEMU exist (real firmware), else the simulated stand-in
        _started["sim"] = boards.EmulatedAgent() if boards.EmulatedAgent.usable() else boards.SimulatedAgent()
        _started["sim"].start()
    if config.INPROC_WORKERS and not _started["workers"]:
        _started["workers"] = pipeline.start_inprocess_workers(config.INPROC_WORKERS)


def _bad(e: Exception, code: int = 400):
    raise HTTPException(code, str(e))


def _own(u: dict, org_id: str | None, what: str = "resource"):
    if not auth.can_see(u, org_id):
        raise HTTPException(404, f"unknown {what}")


# ---- auth --------------------------------------------------------------------------------------------------------
class LoginIn(BaseModel):
    email: str
    password: str


@app.get("/api/health")
def health():
    return {"ok": True, "version": config.PLATFORM_VERSION, "schema": db.version(), "mode": config.MODE}


def _secure(request: Request) -> bool:
    if config.COOKIE_SECURE in ("1", "true", "yes"):
        return True
    if config.COOKIE_SECURE in ("0", "false", "no"):
        return False
    return request.url.scheme == "https"


@app.post("/api/auth/login")
def login(body: LoginIn, response: Response, request: Request):
    try:
        token, u = auth.login(body.email, body.password)
    except PermissionError as e:
        auth.audit(None, "login_failed", body.email.lower())
        raise HTTPException(401, str(e)) from None
    response.set_cookie("sai_session", token, httponly=True, samesite="lax", secure=_secure(request),
                        max_age=auth.SESSION_S)
    auth.audit(u, "login")
    return {"user": u, "token": token}


@app.post("/api/auth/logout")
def logout(response: Response, u: dict = Depends(auth.current)):
    auth.logout(u["token"])
    response.delete_cookie("sai_session")
    return {"ok": True}


@app.get("/api/me")
def me(u: dict = Depends(auth.current)):
    perms = {k: u["role"] in v for k, v in auth.PERMS.items()}
    s = commercial.settings(u["org_id"])
    return {**{k: v for k, v in u.items() if k != "token"}, "perms": perms, "mode": config.MODE,
            "brand": {"product_name": config.BRAND_NAME, **s["brand"]}, "plan": commercial.plan_of(u),
            "deployment": s["deployment"]}


class PasswordIn(BaseModel):
    old: str
    new: str


@app.post("/api/me/password")
def change_password(body: PasswordIn, u: dict = Depends(auth.current)):
    try:
        auth.verify_password(u["email"], body.old)
        if len(body.new) < 8:
            raise ValueError("password must have at least 8 characters")
    except (PermissionError, ValueError) as e:
        _bad(e)
    with db.conn() as c:     # new password; every other session of this user is signed out
        c.execute("UPDATE users SET pw_hash=? WHERE id=?", (auth.hash_pw(body.new), u["id"]))
        c.execute("DELETE FROM sessions WHERE user_id=? AND token_hash<>?", (u["id"], auth._th(u["token"])))
    auth.audit(u, "password_changed")
    return {"ok": True}


# ---- admin -------------------------------------------------------------------------------------------------------
class OrgIn(BaseModel):
    name: str
    tier: str = "free"


class UserIn(BaseModel):
    email: str
    name: str
    role: str = "member"
    password: str
    org_id: str | None = None


@app.get("/api/admin/orgs")
def admin_orgs(u: dict = Depends(auth.require("org_admin"))):
    rows = auth.orgs() if u["role"] == "platform_admin" else [auth.org(u["org_id"])]
    for r in rows:
        r["usage"] = auth.usage(r["id"])
        r["deployment"] = commercial.settings(r["id"])["deployment"]
        r.pop("branding", None)
        r["users"] = len(auth.users(r["id"]))
    return rows


@app.post("/api/admin/orgs")
def admin_org_create(body: OrgIn, u: dict = Depends(auth.require("platform_admin"))):
    if config.MODE == "single":
        raise HTTPException(400, "single-organisation install: organisations are fixed")
    try:
        o = auth.create_org(body.name, body.tier)
    except ValueError as e:
        _bad(e)
    auth.audit(u, "org_created", o["id"], {"tier": body.tier})
    return o


@app.post("/api/admin/orgs/{oid}/tier")
def admin_org_tier(oid: str, body: dict, u: dict = Depends(auth.require("platform_admin"))):
    try:
        auth.set_tier(oid, body.get("tier", ""))
    except ValueError as e:
        _bad(e)
    auth.audit(u, "tier_changed", oid, {"tier": body.get("tier")})
    return auth.org(oid)


@app.post("/api/admin/orgs/{oid}/deployment")
def admin_org_deployment(oid: str, body: dict, u: dict = Depends(auth.require("platform_admin"))):
    try:
        commercial.set_deployment(oid, body.get("model", "none"), body.get("units"), body.get("contract"))
    except ValueError as e:
        _bad(e)
    auth.audit(u, "deployment_rights_set", oid, body)
    return commercial.settings(oid)["deployment"]


@app.get("/api/plan")
def plan_usage(u: dict = Depends(auth.current)):
    return {"plan": commercial.plan_of(u), "usage": commercial.counts(u["org_id"]), "plans": commercial.PLANS,
            "deployment": commercial.settings(u["org_id"])["deployment"], "metered": auth.usage(u["org_id"])}


@app.put("/api/org/brand")
def org_brand(body: dict, u: dict = Depends(auth.require("org_admin"))):
    b = commercial.set_brand(u["org_id"], body)
    auth.audit(u, "branding_changed", u["org_id"], b)
    return b


@app.get("/api/signing/public-key")
def signing_key():
    return {"key_id": commercial.key_id(), "algorithm": "Ed25519", "public_key_pem": commercial.public_key_pem()}


@app.get("/api/admin/users")
def admin_users(u: dict = Depends(auth.require("org_admin"))):
    return auth.users(auth.scope(u))


@app.post("/api/admin/users")
def admin_user_create(body: UserIn, u: dict = Depends(auth.require("org_admin"))):
    org_id = body.org_id if u["role"] == "platform_admin" and body.org_id else u["org_id"]
    if u["role"] != "platform_admin" and body.role not in ("org_admin", "member"):
        raise HTTPException(403, "org admins can create org_admin and member users only")
    if not auth.org(org_id):
        raise HTTPException(400, "unknown organisation")
    try:
        nu = auth.create_user(org_id, body.email, body.name, body.role, body.password)
    except ValueError as e:
        _bad(e)
    auth.audit(u, "user_created", nu["id"], {"email": nu["email"], "role": nu["role"], "org": org_id})
    return nu


@app.post("/api/admin/users/{uid}/disable")
def admin_user_disable(uid: str, u: dict = Depends(auth.require("org_admin"))):
    t = auth.user(uid)
    if not t or not auth.can_see(u, t["org_id"]) or uid == u["id"]:
        raise HTTPException(404, "unknown user")
    with db.conn() as c:
        c.execute("UPDATE users SET disabled=1 WHERE id=?", (uid,))
        c.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
    auth.audit(u, "user_disabled", uid)
    return {"ok": True}


@app.get("/api/admin/audit")
def admin_audit(u: dict = Depends(auth.require("org_admin"))):
    return auth.audit_log(auth.scope(u))


@app.get("/api/admin/agents")
def admin_agents(u: dict = Depends(auth.require("farm_admin"))):
    return auth.agents()


@app.post("/api/admin/agents")
def admin_agent_create(body: dict, u: dict = Depends(auth.require("farm_admin"))):
    a, token = auth.create_agent(body.get("name") or "lab agent")
    auth.audit(u, "agent_created", a["id"], {"name": a["name"]})
    return {**a, "token": token, "note": "shown once; store it on the lab PC"}


@app.post("/api/admin/agents/{aid}/disable")
def admin_agent_disable(aid: str, u: dict = Depends(auth.require("farm_admin"))):
    auth.disable_agent(aid)
    auth.audit(u, "agent_disabled", aid)
    return {"ok": True}


@app.get("/api/admin/queue")
def admin_queue(u: dict = Depends(auth.require("platform_admin"))):
    from . import retention
    return {**pipeline.queue_stats(), "retention": db.get_setting("retention_last"),
            "retention_days": retention.RETENTION_DAYS, "upload_ttl_days": retention.UPLOAD_TTL_DAYS}


@app.post("/api/admin/retention")
def admin_retention(u: dict = Depends(auth.require("platform_admin"))):
    from . import retention
    r = retention.sweep()
    auth.audit(u, "retention_sweep", "", {"freed_bytes": r["freed_bytes"], "removed": r["removed"]})
    return r


@app.post("/api/admin/queue/pause")
def admin_queue_pause(body: dict, u: dict = Depends(auth.require("platform_admin"))):
    pipeline.pause_workers(bool(body.get("paused")))
    auth.audit(u, "queue_paused" if body.get("paused") else "queue_resumed")
    return pipeline.queue_stats()


# ---- system ------------------------------------------------------------------------------------------------------
@app.get("/api/system")
def system(u: dict = Depends(auth.current)):
    return {"sai": sai_engine.version(), "targets": sai_engine.target_status(), "goals": pipeline.GOALS,
            "ops": {k: v["sai"] for k, v in archspec.OPS.items()}, "calibration": boards.calibration(),
            "sim_boards": config.SIM_BOARDS, "platform_version": config.PLATFORM_VERSION, "mode": config.MODE,
            "schema": db.version(), "licences": data.LICENCES, "pool_weights": pipeline.pool_weights_index(),
            "sai_compat": {"supported": {k: sorted(v) for k, v in sai_engine.SUPPORTED_SCHEMAS.items()},
                           "warnings": sorted(sai_engine.COMPAT_WARNINGS)},
            "measurements": len(boards.measurements())}


# ---- model pool --------------------------------------------------------------------------------------------------
@app.get("/api/catalog")
def catalog(drafts: bool = False, u: dict = Depends(auth.current)):
    out = []
    for t in store.templates(include_drafts=drafts and u["role"] in auth.PERMS["studio"]):
        t = dict(t)
        t["sizes"] = {}
        for size, layers in t["variants"].items():
            try:
                n_out = len(t["labels"]) if t["task"] == "classification" else 1 if t["task"] == "regression" else \
                    t["input"]["channels"] * t["input"]["height"] * t["input"]["width"]
                spec = archspec.normalize({"input": t["input"], "layers": layers, "outputs": n_out})
                s = archspec.summary(spec)
                t["sizes"][size] = {"macs": s["macs"], "params": s["params"]}
            except archspec.SpecError as e:
                t["sizes"][size] = {"error": str(e)}
        out.append(t)
    return out


@app.get("/api/catalog/{tid}")
def catalog_item(tid: str, u: dict = Depends(auth.current)):
    t = store.template(tid)
    if not t or (t.get("status") == "draft" and u["role"] not in auth.PERMS["studio"]):
        raise HTTPException(404, "unknown model type")
    return t


@app.post("/api/catalog")
def catalog_save(body: dict, u: dict = Depends(auth.require("studio"))):
    if body.get("status") == "published" and u["role"] not in auth.PERMS["publish"]:
        raise HTTPException(403, "only researchers publish to the pool")
    body["author"] = body.get("author") or u["name"]
    try:
        t = store.save_template(body, u["org_id"])
    except (ValueError, archspec.SpecError) as e:
        _bad(e)
    auth.audit(u, f"template_{t['status']}", t["id"], {"version": t.get("version")})
    return t


@app.delete("/api/catalog/{tid}")
def catalog_delete(tid: str, u: dict = Depends(auth.require("studio"))):
    try:
        store.delete_template(tid)
    except ValueError as e:
        _bad(e)
    auth.audit(u, "template_deleted", tid)
    return {"ok": True}


# ---- datasets ----------------------------------------------------------------------------------------------------
def _tpl(tid, u: dict | None = None):
    """A template the caller may use: customers get published (and deprecated, for old projects) ones only."""
    t = store.template(tid)
    if not t or (u is not None and t.get("status") == "draft" and u["role"] not in auth.PERMS["studio"]):
        raise HTTPException(404, "unknown model type")
    return t


async def _stage_upload(file: UploadFile, limit: int) -> Path:
    """Stream an upload to uploads/ in 1 MB chunks (constant memory), stopping at the limit."""
    dst = config.DATA_DIR / "uploads" / f"ingest_{uuid.uuid4().hex}"
    total = 0
    try:
        with open(dst, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, f"file larger than {limit // (1 << 20)} MB")
                f.write(chunk)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    return dst


MAX_MODEL_BYTES = 256 * 1024 * 1024        # an MCU model is far smaller; this bounds the explorer upload


def _ds(did: str, u: dict) -> dict:
    d = store.dataset(did)
    if not d:
        raise HTTPException(404, "unknown dataset")
    _own(u, d.get("org_id"), "dataset")
    return d


@app.post("/api/datasets/preview")
async def dataset_preview(file: UploadFile = File(...), u: dict = Depends(auth.current)):
    """CSV header, first rows and a guessed column mapping for the mapping step."""
    raw = await file.read(2_000_000)
    try:
        return data.csv_preview(raw)
    except data.DataError as e:
        _bad(e)


@app.post("/api/datasets/upload")
async def dataset_upload(template_id: str = Form(...), file: UploadFile = File(...), window: int | None = Form(None),
                         hop: int | None = Form(None), mapping: str | None = Form(None),
                         licence: str | None = Form(None), u: dict = Depends(auth.current)):
    tpl = _tpl(template_id, u)
    try:
        m = json.loads(mapping) if mapping else None
    except json.JSONDecodeError as e:
        _bad(e)
    staged = await _stage_upload(file, config.MAX_UPLOAD_BYTES)
    size = staged.stat().st_size
    if size > config.INLINE_INGEST_BYTES:
        # big file: a worker parses it; the client polls the job and gets the dataset from its result
        job = pipeline.create("ingest", {"path": str(staged), "filename": file.filename or "upload.csv",
                                         "template_id": template_id, "window": window, "hop": hop, "mapping": m,
                                         "licence": licence or "unknown"}, org_id=u["org_id"], user_id=u["id"], priority=1)
        auth.audit(u, "dataset_upload_queued", job["id"], {"name": file.filename, "bytes": size})
        auth.meter(u["org_id"], "upload_bytes", size, job["id"])
        return JSONResponse({"job": {"id": job["id"], "status": job["status"]}}, status_code=202)
    try:
        raw = staged.read_bytes()
        # parsing / feature extraction is CPU work: keep it off the event loop
        meta = await run_in_threadpool(data.ingest_bytes, raw, file.filename or "upload.csv", tpl, window=window,
                                       hop=hop, mapping=m, licence=licence or "unknown")
    except data.DataError as e:
        _bad(e)
    finally:
        staged.unlink(missing_ok=True)
    store.add_dataset(meta, u["org_id"])
    auth.audit(u, "dataset_uploaded", meta["id"], {"name": meta.get("name"), "n": meta.get("n")})
    auth.meter(u["org_id"], "upload_bytes", size, meta["id"])
    return meta


class UrlIn(BaseModel):
    template_id: str
    url: str
    window: int | None = None
    hop: int | None = None
    licence: str = "unknown"


@app.post("/api/datasets/url")
def dataset_url(body: UrlIn, u: dict = Depends(auth.current)):
    try:
        meta = data.ingest_url(body.url, _tpl(body.template_id, u), window=body.window, hop=body.hop,
                               licence=body.licence)
    except (data.DataError, OSError) as e:
        _bad(e)
    store.add_dataset(meta, u["org_id"])
    auth.audit(u, "dataset_fetched", meta["id"], {"url": body.url})
    return meta


class DemoIn(BaseModel):
    template_id: str
    per_class: int = 300


@app.post("/api/datasets/demo")
def dataset_demo(body: DemoIn, u: dict = Depends(auth.current)):
    t = _tpl(body.template_id, u)
    if not t.get("synth"):
        raise HTTPException(400, "this model type has no synthetic demo data; upload a dataset")
    meta = data.from_synthetic(t, max(30, min(body.per_class, 2000)))
    store.add_dataset(meta, u["org_id"])
    return meta


@app.get("/api/datasets")
def dataset_list(template_id: str | None = None, u: dict = Depends(auth.current)):
    return store.datasets(template_id, auth.scope(u))


@app.get("/api/datasets/{did}")
def dataset_get(did: str, u: dict = Depends(auth.current)):
    d = _ds(did, u)
    d["used_by"] = store.dataset_usage(did)
    return d


@app.delete("/api/datasets/{did}")
def dataset_delete(did: str, u: dict = Depends(auth.current)):
    _ds(did, u)
    running = [x for x in store.dataset_usage(did) if x["status"] in ("queued", "running")]
    if running:
        raise HTTPException(409, f"dataset is used by {len(running)} running project(s); wait or cancel them first")
    store.delete_dataset(did)
    auth.audit(u, "dataset_deleted", did)
    return {"ok": True}


@app.get("/api/datasets/{did}/stats")
def dataset_stats(did: str, u: dict = Depends(auth.current)):
    """Per-channel statistics overall and per class, recordings per class, for the data page."""
    _ds(did, u)
    X, y, labels, meta = data.load(did)
    C = X.shape[1]
    flat = np.moveaxis(X, 1, 0).reshape(C, -1)
    names = meta.get("channel_names") or [f"ch{c}" for c in range(C)]
    chans = [{"name": names[c] if c < len(names) else f"ch{c}", "mean": float(flat[c].mean()), "std": float(flat[c].std()),
              "min": float(flat[c].min()), "max": float(flat[c].max()),
              "p1": float(np.percentile(flat[c], 1)), "p99": float(np.percentile(flat[c], 99))} for c in range(C)]
    per_class = []
    groups = data.load_groups(did)
    if meta.get("task") != "regression":
        for k in sorted(set(int(v) for v in y)):
            m = y == k
            per_class.append({"label": labels[k] if k < len(labels) else str(k), "n": int(m.sum()),
                              "recordings": int(len(set(groups[m].tolist()))) if groups is not None else None,
                              "channel_mean": [float(v) for v in X[m].mean(axis=(0, 2, 3))],
                              "channel_std": [float(v) for v in X[m].std(axis=(0, 2, 3))]})
    else:
        hist, edges = np.histogram(y.astype(float), bins=20)
        per_class = [{"histogram": hist.tolist(), "edges": edges.round(4).tolist()}]
    return {"channels": chans, "per_class": per_class, "task": meta.get("task")}


@app.get("/api/datasets/{did}/samples")
def dataset_samples(did: str, label: int | None = None, offset: int = 0, n: int = 6, u: dict = Depends(auth.current)):
    """Windows for the sample browser (series decimated to <= 256 points per channel; images as-is)."""
    _ds(did, u)
    X, y, labels, meta = data.load(did)
    groups = data.load_groups(did)
    idx = np.arange(len(X)) if label is None or meta.get("task") == "regression" else np.where(y == label)[0]
    n = max(1, min(int(n), 24))
    sel = idx[max(0, offset):max(0, offset) + n]
    out = []
    for i in sel:
        ex = X[i]
        item = {"index": int(i), "group": str(groups[i]) if groups is not None else None,
                "label": (labels[int(y[i])] if meta.get("task") != "regression" and int(y[i]) < len(labels)
                          else f"{float(y[i]):.3f}")}
        if ex.shape[1] == 1:
            step = max(1, ex.shape[2] // 256)
            item["series"] = ex[:, 0, ::step].round(5).tolist()
        else:
            item["image"] = ex[0].round(4).tolist()
        out.append(item)
    return {"total": int(len(idx)), "offset": int(offset), "items": out}


class PrepPreviewIn(BaseModel):
    template_id: str | None = None
    preprocess: dict = {}


@app.post("/api/datasets/{did}/preprocess-preview")
def dataset_prep_preview(did: str, body: PrepPreviewIn, u: dict = Depends(auth.current)):
    from . import preprocess
    _ds(did, u)
    X, y, labels, meta = data.load(did)
    task = meta.get("task", "classification")
    try:
        cfg = preprocess.normalize_cfg(body.preprocess, meta, X.shape[1:], task)
        return {"config": cfg, **preprocess.preview(X, y, labels, meta, cfg, task)}
    except preprocess.PrepError as e:
        _bad(e)


# ---- projects ----------------------------------------------------------------------------------------------------
class ProjectIn(BaseModel):
    name: str
    template_id: str
    dataset_id: str
    targets: list[str] = ["md6605", "cortex-m7"]
    goal: str = "balanced"
    sizes: list[str] = ["tiny", "small", "base"]
    budgets: dict = {}
    metric_target: float | None = None
    epochs: int = 30
    init_from_pool: bool = False      # fine-tune from the pool's base weights when the shapes match
    pool_benchmark: bool = False      # SAI research only: publish the result on the pool leaderboard
    lr_search: bool = False           # try two learning rates per size, keep the better validation loss
    preprocess: dict = {}             # preprocessing stage settings (backend/preprocess.py)


@app.post("/api/projects")
def project_create(body: ProjectIn, u: dict = Depends(auth.current)):
    _tpl(body.template_id, u)
    d = _ds(body.dataset_id, u)
    from . import preprocess
    try:
        preprocess.normalize_cfg(body.preprocess, d, d["shape"], d.get("task", "classification"))
    except preprocess.PrepError as e:
        _bad(e)
    if body.goal not in pipeline.GOALS:
        raise HTTPException(400, f"goal must be one of {pipeline.GOALS}")
    cfg = body.model_dump()
    cfg["epochs"] = max(1, min(cfg["epochs"], 200))
    if cfg.get("pool_benchmark") and u["role"] not in auth.PERMS["publish"]:
        raise HTTPException(403, "only SAI research publishes pool benchmarks")
    plan = commercial.check_project(u)
    job = pipeline.create("train", cfg, org_id=u["org_id"], user_id=u["id"], priority=plan["priority"])
    prj = store.create_project(body.name or "Untitled", body.template_id, body.dataset_id, cfg, job["id"], u["org_id"])
    with db.conn() as c:
        c.execute("UPDATE jobs SET project_id=? WHERE id=?", (prj["id"], job["id"]))
    auth.audit(u, "project_created", prj["id"], {"template": body.template_id, "job": job["id"]})
    return {"project": prj, "job": job}


@app.get("/api/projects")
def project_list(u: dict = Depends(auth.current)):
    return store.projects(auth.scope(u))


@app.get("/api/projects/{pid}")
def project_get(pid: str, u: dict = Depends(auth.current)):
    p = store.project(pid)
    if not p:
        raise HTTPException(404, "unknown project")
    _own(u, p.get("org_id"), "project")
    return p


# ---- jobs --------------------------------------------------------------------------------------------------------
def _job(jid: str, u: dict, log_after: int = 0) -> dict:
    j = pipeline.get(jid, log_after)
    if not j:
        raise HTTPException(404, "unknown job")
    _own(u, j.get("org_id"), "job")
    return j


@app.get("/api/jobs")
def job_list(kind: str | None = None, u: dict = Depends(auth.current)):
    return pipeline.list_jobs(kind, org_id=auth.scope(u))


@app.get("/api/jobs/{jid}")
def job_get(jid: str, log_after: int = 0, u: dict = Depends(auth.current)):
    j = _job(jid, u, log_after)
    # server-side paths are not the customer's business
    (j.get("params") or {}).pop("model_path", None)
    if isinstance(j.get("result"), dict):
        j["result"].pop("selected_model_path", None)
    return j


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str, u: dict = Depends(auth.current)):
    _job(jid, u)
    pipeline.cancel(jid)
    auth.audit(u, "job_cancelled", jid)
    return {"ok": True}


@app.post("/api/jobs/{jid}/rebuild")
def job_rebuild(jid: str, u: dict = Depends(auth.current)):
    _job(jid, u)
    j = pipeline.create("rebuild", {"job_id": jid}, org_id=u["org_id"], user_id=u["id"], priority=1)
    auth.audit(u, "rebuild_requested", jid, {"rebuild_job": j["id"]})
    return j


DOWNLOADS = {"package.zip": "application/zip", "model.onnx": "application/octet-stream",
             "report.json": "application/json", "MODEL_CARD.md": "text/markdown"}


def _file(jid: str, name: str) -> Path:
    if name not in DOWNLOADS or not jid.startswith("job_") or "/" in jid or ".." in jid:
        raise HTTPException(404, "not found")
    p = config.DATA_DIR / "jobs" / jid / name
    if not p.exists():
        raise HTTPException(404, "not built yet")
    return p


@app.get("/api/jobs/{jid}/download/{name}")
def job_download(jid: str, name: str, u: dict = Depends(auth.current)):
    _job(jid, u)
    p = _file(jid, name)
    if name in ("package.zip", "model.onnx"):
        commercial.check_package_download(u)
    auth.audit(u, "download", jid, {"file": name})
    auth.meter(u["org_id"], "download", 1, f"{jid}/{name}")
    return FileResponse(p, media_type=DOWNLOADS[name], filename=f"{jid}_{name}")


# ---- researcher studio & explorer ---------------------------------------------------------------------------------
@app.post("/api/studio/validate")
def studio_validate(body: dict, u: dict = Depends(auth.current)):
    try:
        spec = archspec.normalize(body)
        return {"ok": True, "spec": spec, "summary": archspec.summary(spec)}
    except archspec.SpecError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/studio/fitcheck")
def studio_fitcheck(body: dict, u: dict = Depends(auth.require("studio"))):
    try:
        archspec.normalize(body.get("spec") or {})
    except archspec.SpecError as e:
        _bad(e)
    return pipeline.create("fitcheck", body, org_id=u["org_id"], user_id=u["id"], priority=2)


class BaseTrainIn(BaseModel):
    template_id: str
    sizes: list[str] | None = None
    dataset_id: str | None = None
    per_class: int = 300
    epochs: int = 30


@app.post("/api/studio/basetrain")
def studio_basetrain(body: BaseTrainIn, u: dict = Depends(auth.require("publish"))):
    _tpl(body.template_id)
    if body.dataset_id:
        _ds(body.dataset_id, u)
    j = pipeline.create("basetrain", body.model_dump(), org_id=u["org_id"], user_id=u["id"])
    auth.audit(u, "basetrain_requested", body.template_id, {"job": j["id"]})
    return j


@app.get("/api/pool/leaderboard")
def pool_leaderboard(u: dict = Depends(auth.current)):
    return pipeline.leaderboard()


@app.get("/api/pool/weights")
def pool_weights(u: dict = Depends(auth.current)):
    return pipeline.pool_weights_index()


@app.post("/api/explorer/analyze")
async def explorer_analyze(file: UploadFile = File(...), targets: str = Form("md6605,cortex-m7"),
                           goal: str = Form("balanced"), max_ram: int | None = Form(None),
                           u: dict = Depends(auth.current)):
    name = (file.filename or "model.onnx").lower()
    if not name.endswith((".onnx", ".tflite")):
        raise HTTPException(400, "upload an INT8 .onnx or .tflite model")
    staged = await _stage_upload(file, MAX_MODEL_BYTES)            # streamed to disk, constant memory
    path = staged.with_name(f"{uuid.uuid4().hex}{Path(name).suffix}")
    staged.replace(path)
    auth.audit(u, "model_uploaded", path.name, {"filename": file.filename})
    return pipeline.create("analyze", {"model_path": str(path), "filename": file.filename,
                                       "targets": [t for t in targets.split(",") if t], "goal": goal,
                                       "budgets": {"max_ram": max_ram} if max_ram else {}},
                           org_id=u["org_id"], user_id=u["id"], priority=1)


# ---- board farm --------------------------------------------------------------------------------------------------
@app.get("/api/boards")
def board_list(u: dict = Depends(auth.current)):
    return {"boards": boards.list_boards(), "runs": boards.list_runs(limit=30, org_id=auth.scope(u)),
            "calibration": boards.calibration()}


class RunIn(BaseModel):
    job_id: str
    target: str | None = None
    board_id: str | None = None


@app.post("/api/boards/runs")
def board_run(body: RunIn, u: dict = Depends(auth.current)):
    _job(body.job_id, u)
    bd = {b["id"]: b for b in boards.list_boards()}
    if body.board_id and body.board_id not in bd:
        raise HTTPException(404, "unknown board")
    plan = commercial.check_board_run(u, bd[body.board_id]["kind"] if body.board_id else "emulated")
    allowed = None if plan["farm"] == "all" else ["emulated", "simulated"]      # enforced again when a board claims it
    try:
        r = pipeline.start_board_run(body.job_id, body.target, body.board_id, u["org_id"], allowed)
    except (ValueError, KeyError) as e:
        _bad(e)
    auth.audit(u, "board_run_requested", r["id"], {"job": body.job_id, "target": r["target"]})
    return r


@app.get("/api/boards/runs")
def board_runs(job_id: str | None = None, u: dict = Depends(auth.current)):
    if job_id:
        _job(job_id, u)
    return boards.list_runs(job_id, org_id=auth.scope(u))


@app.post("/api/agent/register")
def agent_register(body: dict, authorization: str | None = Header(None)):
    aid = auth.agent_from_token(authorization)
    return {"agent_id": aid, "boards": boards.register(aid, body.get("boards", []))}


@app.post("/api/agent/poll")
def agent_poll(body: dict, authorization: str | None = Header(None)):
    aid = auth.agent_from_token(authorization)
    boards.heartbeat(aid, body.get("health"))
    if body.get("heartbeat_only"):          # the agent is busy with a run: keep its boards online, hand out nothing
        return {"run": None}
    run = boards.claim(aid)
    if run:
        run["request"].pop("model_path", None)
    return {"run": run}


@app.post("/api/agent/runs/{rid}/result")
def agent_result(rid: str, body: dict, authorization: str | None = Header(None)):
    aid = auth.agent_from_token(authorization)
    run = boards.get_run(rid)
    if not run or boards.board_agent(run["board_id"]) != aid:
        raise HTTPException(404, "unknown run")
    try:
        r = boards.complete(rid, body, agent_id=aid)
    except PermissionError as e:
        raise HTTPException(409, str(e)) from None
    auth.meter(run.get("org_id"), "farm_runs", 1, rid)
    return {"id": r["id"], "status": r["status"], "outputs_match": (r["result"] or {}).get("outputs_match")}


@app.get("/api/agent/runs/{rid}/package")
def agent_package(rid: str, authorization: str | None = Header(None)):
    aid = auth.agent_from_token(authorization)
    run = boards.get_run(rid)
    if not run or boards.board_agent(run["board_id"]) != aid:
        raise HTTPException(404, "unknown run")
    bdir = Path(run["request"].get("build_dir") or "")
    if bdir.is_dir() and config.DATA_DIR in bdir.resolve().parents:
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(bdir.rglob("*")):
                if f.is_file():
                    z.write(f, Path("sai_package") / f.relative_to(bdir))
            z.writestr("sai_package/hil_vectors.json", json.dumps(
                np.load(config.DATA_DIR / "jobs" / run["job_id"] / "hil_vectors.npz")["x"].tolist()))
        return Response(buf.getvalue(), media_type="application/zip")
    return FileResponse(_file(run["job_id"], "package.zip"), media_type="application/zip")


@app.post("/api/boards/{bid}/reset")
def board_reset(bid: str, u: dict = Depends(auth.require("farm_admin"))):
    boards.reset_board(bid)
    auth.audit(u, "board_reset", bid)
    return {"ok": True}


@app.post("/api/admin/regression")
def regression_start(body: dict, u: dict = Depends(auth.require("farm_admin"))):
    j = pipeline.create("regression", {"templates": body.get("templates"), "hil": body.get("hil", True)},
                        org_id=u["org_id"], user_id=u["id"], priority=-1)
    auth.audit(u, "regression_requested", j["id"])
    return j


@app.get("/api/regressions")
def regressions(u: dict = Depends(auth.current)):
    from . import regression
    return regression.summary()


# ---- frontend ----------------------------------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled(request, exc):
    """Log the details server-side; the client gets an id to quote, not a stack trace or file paths."""
    import logging
    eid = uuid.uuid4().hex[:8]
    logging.getLogger("sai-platform").exception("unhandled error %s on %s %s", eid, request.method, request.url.path)
    return JSONResponse({"detail": f"internal error (reference {eid})"}, status_code=500)


@app.middleware("http")
async def _security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' "
                            "'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; "
                            "base-uri 'none'")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.scheme == "https":
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return resp


app.mount("/", StaticFiles(directory=FRONT, html=True), name="frontend")
