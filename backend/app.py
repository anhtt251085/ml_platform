"""SAI Platform HTTP API and static frontend.

    uvicorn backend.app:app --port 8000

Every /api route except /api/health, /api/auth/login and the lab-agent routes needs a session (cookie or bearer).
Customer data (datasets, projects, jobs, board runs) is scoped to the caller's organisation.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import archspec, auth, boards, config, data, db, pipeline, sai_engine, store

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
        _started["sim"] = boards.SimulatedAgent()
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


@app.post("/api/auth/login")
def login(body: LoginIn, response: Response):
    try:
        token, u = auth.login(body.email, body.password)
    except PermissionError as e:
        auth.audit(None, "login_failed", body.email.lower())
        raise HTTPException(401, str(e)) from None
    response.set_cookie("sai_session", token, httponly=True, samesite="lax", secure=config.COOKIE_SECURE,
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
    return {**{k: v for k, v in u.items() if k != "token"}, "perms": perms, "mode": config.MODE,
            "brand": config.BRAND_NAME}


class PasswordIn(BaseModel):
    old: str
    new: str


@app.post("/api/me/password")
def change_password(body: PasswordIn, u: dict = Depends(auth.current)):
    try:
        auth.login(u["email"], body.old)
        if len(body.new) < 8:
            raise ValueError("password must have at least 8 characters")
    except (PermissionError, ValueError) as e:
        _bad(e)
    with db.conn() as c:
        c.execute("UPDATE users SET pw_hash=? WHERE id=?", (auth.hash_pw(body.new), u["id"]))
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
    return pipeline.queue_stats()


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
def _tpl(tid):
    t = store.template(tid)
    if not t:
        raise HTTPException(404, "unknown model type")
    return t


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
    raw = await file.read()
    try:
        m = json.loads(mapping) if mapping else None
        meta = data.ingest_bytes(raw, file.filename or "upload.csv", _tpl(template_id), window=window, hop=hop,
                                 mapping=m, licence=licence or "unknown")
    except (data.DataError, json.JSONDecodeError) as e:
        _bad(e)
    store.add_dataset(meta, u["org_id"])
    auth.audit(u, "dataset_uploaded", meta["id"], {"name": meta.get("name"), "n": meta.get("n")})
    auth.meter(u["org_id"], "upload_bytes", len(raw), meta["id"])
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
        meta = data.ingest_url(body.url, _tpl(body.template_id), window=body.window, hop=body.hop,
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
    t = _tpl(body.template_id)
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
    return _ds(did, u)


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
    lr_search: bool = False           # try two learning rates per size, keep the better validation loss


@app.post("/api/projects")
def project_create(body: ProjectIn, u: dict = Depends(auth.current)):
    _tpl(body.template_id)
    _ds(body.dataset_id, u)
    if body.goal not in pipeline.GOALS:
        raise HTTPException(400, f"goal must be one of {pipeline.GOALS}")
    cfg = body.model_dump()
    cfg["epochs"] = max(1, min(cfg["epochs"], 200))
    job = pipeline.create("train", cfg, org_id=u["org_id"], user_id=u["id"])
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
    return _job(jid, u, log_after)


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
    raw = await file.read()
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, "file too large")
    path = config.DATA_DIR / "uploads" / f"{uuid.uuid4().hex}{Path(name).suffix}"
    path.write_bytes(raw)
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
    try:
        r = pipeline.start_board_run(body.job_id, body.target, body.board_id, u["org_id"])
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
    boards.heartbeat(aid)
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
    r = boards.complete(rid, body)
    auth.meter(run.get("org_id"), "farm_runs", 1, rid)
    return r


@app.get("/api/agent/runs/{rid}/package")
def agent_package(rid: str, authorization: str | None = Header(None)):
    aid = auth.agent_from_token(authorization)
    run = boards.get_run(rid)
    if not run or boards.board_agent(run["board_id"]) != aid:
        raise HTTPException(404, "unknown run")
    return FileResponse(_file(run["job_id"], "package.zip"), media_type="application/zip")


# ---- frontend ----------------------------------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled(request, exc):
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


app.mount("/", StaticFiles(directory=FRONT, html=True), name="frontend")
