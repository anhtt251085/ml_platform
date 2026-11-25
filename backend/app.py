"""SAI Platform HTTP API and static frontend.

    uvicorn backend.app:app --port 8000
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import archspec, boards, config, data, db, pipeline, sai_engine, store

app = FastAPI(title="SAI Platform", version="0.1.0")
FRONT = config.ROOT / "frontend"
_sim = None


@app.on_event("startup")
def _startup():
    global _sim
    db.init()
    if config.SIM_BOARDS and _sim is None:
        _sim = boards.SimulatedAgent()
        _sim.start()


def _bad(e: Exception, code: int = 400):
    raise HTTPException(code, str(e))


# ---- system ------------------------------------------------------------------------------------------------------
@app.get("/api/system")
def system():
    return {"sai": sai_engine.version(), "targets": sai_engine.target_status(), "goals": pipeline.GOALS,
            "ops": {k: v["sai"] for k, v in archspec.OPS.items()}, "calibration": boards.calibration(),
            "sim_boards": config.SIM_BOARDS}


# ---- model pool --------------------------------------------------------------------------------------------------
@app.get("/api/catalog")
def catalog(drafts: bool = False):
    out = []
    for t in store.templates(include_drafts=drafts):
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
def catalog_item(tid: str):
    t = store.template(tid)
    if not t:
        raise HTTPException(404, "unknown model type")
    return t


@app.post("/api/catalog")
def catalog_save(body: dict):
    try:
        return store.save_template(body)
    except (ValueError, archspec.SpecError) as e:
        _bad(e)


@app.delete("/api/catalog/{tid}")
def catalog_delete(tid: str):
    try:
        store.delete_template(tid)
    except ValueError as e:
        _bad(e)
    return {"ok": True}


# ---- datasets ----------------------------------------------------------------------------------------------------
def _tpl(tid):
    t = store.template(tid)
    if not t:
        raise HTTPException(404, "unknown model type")
    return t


@app.post("/api/datasets/upload")
async def dataset_upload(template_id: str = Form(...), file: UploadFile = File(...), window: int | None = Form(None),
                         hop: int | None = Form(None)):
    raw = await file.read()
    try:
        meta = data.ingest_bytes(raw, file.filename or "upload.csv", _tpl(template_id), window=window, hop=hop)
    except data.DataError as e:
        _bad(e)
    store.add_dataset(meta)
    return meta


class UrlIn(BaseModel):
    template_id: str
    url: str
    window: int | None = None
    hop: int | None = None


@app.post("/api/datasets/url")
def dataset_url(body: UrlIn):
    try:
        meta = data.ingest_url(body.url, _tpl(body.template_id), window=body.window, hop=body.hop)
    except (data.DataError, OSError) as e:
        _bad(e)
    store.add_dataset(meta)
    return meta


class DemoIn(BaseModel):
    template_id: str
    per_class: int = 300


@app.post("/api/datasets/demo")
def dataset_demo(body: DemoIn):
    t = _tpl(body.template_id)
    if not t.get("synth"):
        raise HTTPException(400, "this model type has no synthetic demo data; upload a dataset")
    meta = data.from_synthetic(t, max(30, min(body.per_class, 2000)))
    store.add_dataset(meta)
    return meta


@app.get("/api/datasets")
def dataset_list(template_id: str | None = None):
    return store.datasets(template_id)


@app.get("/api/datasets/{did}")
def dataset_get(did: str):
    d = store.dataset(did)
    if not d:
        raise HTTPException(404, "unknown dataset")
    return d


# ---- projects (the customer flow) ---------------------------------------------------------------------------------
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


@app.post("/api/projects")
def project_create(body: ProjectIn):
    _tpl(body.template_id)
    if not store.dataset(body.dataset_id):
        raise HTTPException(400, "unknown dataset")
    if body.goal not in pipeline.GOALS:
        raise HTTPException(400, f"goal must be one of {pipeline.GOALS}")
    cfg = body.model_dump()
    cfg["epochs"] = max(1, min(cfg["epochs"], 200))
    job = pipeline.create("train", cfg)
    prj = store.create_project(body.name or "Untitled", body.template_id, body.dataset_id, cfg, job["id"])
    with db.conn() as c:
        c.execute("UPDATE jobs SET project_id=? WHERE id=?", (prj["id"], job["id"]))
    return {"project": prj, "job": job}


@app.get("/api/projects")
def project_list():
    return store.projects()


@app.get("/api/projects/{pid}")
def project_get(pid: str):
    p = store.project(pid)
    if not p:
        raise HTTPException(404, "unknown project")
    return p


# ---- jobs --------------------------------------------------------------------------------------------------------
@app.get("/api/jobs")
def job_list(kind: str | None = None):
    return pipeline.list_jobs(kind)


@app.get("/api/jobs/{jid}")
def job_get(jid: str, log_after: int = 0):
    j = pipeline.get(jid, log_after)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    pipeline.cancel(jid)
    return {"ok": True}


DOWNLOADS = {"package.zip": "application/zip", "model.onnx": "application/octet-stream",
             "report.json": "application/json", "MODEL_CARD.md": "text/markdown"}


@app.get("/api/jobs/{jid}/download/{name}")
def job_download(jid: str, name: str):
    if name not in DOWNLOADS or not jid.startswith("job_") or "/" in jid:
        raise HTTPException(404, "not found")
    p = config.DATA_DIR / "jobs" / jid / name
    if not p.exists():
        raise HTTPException(404, "not built yet")
    return FileResponse(p, media_type=DOWNLOADS[name], filename=f"{jid}_{name}")


# ---- researcher studio & explorer ---------------------------------------------------------------------------------
@app.post("/api/studio/validate")
def studio_validate(body: dict):
    try:
        spec = archspec.normalize(body)
        return {"ok": True, "spec": spec, "summary": archspec.summary(spec)}
    except archspec.SpecError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/studio/fitcheck")
def studio_fitcheck(body: dict):
    try:
        archspec.normalize(body.get("spec") or {})
    except archspec.SpecError as e:
        _bad(e)
    return pipeline.create("fitcheck", body)


@app.post("/api/explorer/analyze")
async def explorer_analyze(file: UploadFile = File(...), targets: str = Form("md6605,cortex-m7"),
                           goal: str = Form("balanced"), max_ram: int | None = Form(None)):
    name = (file.filename or "model.onnx").lower()
    if not name.endswith((".onnx", ".tflite")):
        raise HTTPException(400, "upload an INT8 .onnx or .tflite model")
    raw = await file.read()
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, "file too large")
    path = config.DATA_DIR / "uploads" / f"{uuid.uuid4().hex}{Path(name).suffix}"
    path.write_bytes(raw)
    return pipeline.create("analyze", {"model_path": str(path), "filename": file.filename,
                                       "targets": [t for t in targets.split(",") if t], "goal": goal,
                                       "budgets": {"max_ram": max_ram} if max_ram else {}})


# ---- board farm --------------------------------------------------------------------------------------------------
@app.get("/api/boards")
def board_list():
    return {"boards": boards.list_boards(), "runs": boards.list_runs(limit=30), "calibration": boards.calibration()}


class RunIn(BaseModel):
    job_id: str
    target: str | None = None
    board_id: str | None = None


@app.post("/api/boards/runs")
def board_run(body: RunIn):
    try:
        return pipeline.start_board_run(body.job_id, body.target, body.board_id)
    except (ValueError, KeyError) as e:
        _bad(e)


@app.get("/api/boards/runs")
def board_runs(job_id: str | None = None):
    return boards.list_runs(job_id)


def _agent(token: str | None):
    if token != f"Bearer {config.AGENT_TOKEN}":
        raise HTTPException(401, "bad agent token")


@app.post("/api/agent/register")
def agent_register(body: dict, authorization: str | None = Header(None)):
    _agent(authorization)
    return {"boards": boards.register(body["agent_id"], body.get("boards", []))}


@app.post("/api/agent/poll")
def agent_poll(body: dict, authorization: str | None = Header(None)):
    _agent(authorization)
    boards.heartbeat(body["agent_id"])
    run = boards.claim(body["agent_id"])
    if run:
        run["request"].pop("model_path", None)
    return {"run": run}


@app.post("/api/agent/runs/{rid}/result")
def agent_result(rid: str, body: dict, authorization: str | None = Header(None)):
    _agent(authorization)
    try:
        return boards.complete(rid, body)
    except KeyError:
        raise HTTPException(404, "unknown run") from None


@app.get("/api/agent/runs/{rid}/package")
def agent_package(rid: str, authorization: str | None = Header(None)):
    _agent(authorization)
    run = boards.get_run(rid)
    if not run:
        raise HTTPException(404, "unknown run")
    return job_download(run["job_id"], "package.zip")


# ---- frontend ----------------------------------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled(request, exc):
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


app.mount("/", StaticFiles(directory=FRONT, html=True), name="frontend")
