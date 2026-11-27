# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""While something trains or runs, the actions that would duplicate or break it are refused (409 naming the job):
duplicate recommendations, SAI checks, base training, regressions, rebuilds and board runs; deleting a dataset or
model type in use; resetting a board in the middle of a run; sweeping an upload a queued job still needs.
The job queue is paused here, so every job created stays queued and the checks are deterministic."""

from __future__ import annotations

import os
import tempfile
import time
import uuid

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import pytest  # noqa: E402

from backend import config, db, pipeline  # noqa: E402


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from backend.app import app
    with TestClient(app) as c:
        r = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                            "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
        c.headers["Authorization"] = f"Bearer {r.json()['token']}"
        c.post("/api/admin/queue/pause", json={"paused": True})
        try:
            yield c
        finally:
            for j in pipeline.active_jobs(list(pipeline.STAGES)):
                pipeline.cancel(j["id"])
            with db.conn() as con:
                con.execute("UPDATE jobs SET status='cancelled' WHERE status IN ('queued', 'running')")
                con.execute("UPDATE hil_runs SET status='cancelled' WHERE status IN ('queued', 'running')")
            c.post("/api/admin/queue/pause", json={"paused": False})


def _finished_job(org_id: str) -> str:
    jid = "job_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO jobs (id, kind, project_id, status, stage, progress, stages, result, error, params,"
                  " created, updated, org_id, user_id, priority, attempts, cancel_requested)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)",
                  (jid, "train", None, "succeeded", "package", 1.0, "[]", db.j({"selected": {"target": "md6605"}}),
                   None, db.j({"name": "p", "template_id": "gas-sense"}), db.now(), db.now(), org_id, None, 0))
    return jid


def test_active_jobs_helper(client):
    j = pipeline.create("recommend", {"dataset_id": "ds_x", "template_id": "t"})
    assert [x["id"] for x in pipeline.active_jobs("recommend", lambda p: p["dataset_id"] == "ds_x")] == [j["id"]]
    assert pipeline.active_jobs("recommend", lambda p: p["dataset_id"] == "other") == []


def test_duplicate_jobs_are_refused_with_the_running_job(client):
    ds = client.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 10}).json()
    body = {"template_id": "gas-sense", "dataset_id": ds["id"], "targets": ["md6605"], "goal": "min-ram"}
    first = client.post("/api/recommend", json=body).json()
    dup = client.post("/api/recommend", json=body)
    assert dup.status_code == 409 and first["id"] in dup.text
    assert client.post("/api/recommend", json={**body, "goal": "fastest"}).status_code == 200   # different question
    # the dataset a queued job reads cannot be deleted
    d = client.delete(f"/api/datasets/{ds['id']}")
    assert d.status_code == 409 and "reads this dataset" in d.text
    # SAI checks of the library: same entries, or the whole library while a part runs
    v1 = client.post("/api/architectures/verify", json={"ids": ["ds-tcn-s"]}).json()
    assert client.post("/api/architectures/verify", json={"ids": ["ds-tcn-s", "mlp-s"]}).status_code == 409
    assert client.post("/api/architectures/verify", json={}).status_code == 409
    assert client.post("/api/architectures/verify", json={"ids": ["mlp-s"]}).status_code == 200
    assert v1["kind"] == "archverify"
    # base weights per model type, regression runs
    assert client.post("/api/studio/basetrain", json={"template_id": "gas-sense"}).status_code == 200
    assert client.post("/api/studio/basetrain", json={"template_id": "gas-sense"}).status_code == 409
    assert client.post("/api/admin/regression", json={}).status_code == 200
    assert client.post("/api/admin/regression", json={}).status_code == 409


def test_model_type_in_use_cannot_be_deleted(client):
    pipeline.create("train", {"template_id": "gas-sense", "dataset_id": "ds_none", "name": "x"})
    r = client.delete("/api/catalog/gas-sense")
    assert r.status_code == 409 and "model type" in r.text


def test_rebuild_and_board_runs_one_at_a_time(client):
    org = client.get("/api/me").json()["org_id"]
    jid = _finished_job(org)
    queued = pipeline.create("train", {"template_id": "gas-sense", "dataset_id": "d", "name": "x"}, org_id=org)
    assert client.post(f"/api/jobs/{queued['id']}/rebuild").status_code == 400      # not finished yet
    assert client.post(f"/api/jobs/{jid}/rebuild").status_code == 200
    assert client.post(f"/api/jobs/{jid}/rebuild").status_code == 409
    with db.conn() as c:
        c.execute("INSERT INTO hil_runs (id, job_id, board_id, target, status, created, updated, request, result, org_id)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?)", ("hil_open1", jid, None, "md6605", "queued", db.now(), db.now(),
                                                   "{}", None, org))
    r = client.post("/api/boards/runs", json={"job_id": jid})
    assert r.status_code == 409 and "hil_open1" in r.text


def test_board_reset_does_not_kill_a_live_run_without_force(client):
    boards = client.get("/api/boards").json()["boards"]
    bid = boards[0]["id"]
    org = client.get("/api/me").json()["org_id"]
    jid = _finished_job(org)
    with db.conn() as c:
        c.execute("UPDATE boards SET status='busy' WHERE id=?", (bid,))
        c.execute("INSERT INTO hil_runs (id, job_id, board_id, target, status, created, updated, request, result, org_id)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?)", ("hil_live1", jid, bid, boards[0]["target"], "running", db.now(),
                                                   db.now(), "{}", None, org))
    r = client.post(f"/api/boards/{bid}/reset")
    assert r.status_code == 409 and "hil_live1" in r.text
    assert client.post(f"/api/boards/{bid}/reset", json={"force": True}).status_code == 200
    run = next(x for x in client.get("/api/boards/runs").json() if x["id"] == "hil_live1")
    assert run["status"] == "failed"


def test_retention_keeps_uploads_queued_jobs_need(client):
    from backend import retention
    up = config.DATA_DIR / "uploads"
    keep, drop = up / f"ingest_{uuid.uuid4().hex}", up / f"ingest_{uuid.uuid4().hex}"
    for f in (keep, drop):
        f.write_bytes(b"x")
        old = time.time() - (retention.UPLOAD_TTL_DAYS + 1) * retention.DAY
        os.utime(f, (old, old))
    pipeline.create("ingest", {"path": str(keep), "filename": "big.csv", "template_id": "gas-sense"})
    retention.sweep()
    assert keep.exists() and not drop.exists()
