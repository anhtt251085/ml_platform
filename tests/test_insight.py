# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Project insight: what-if sliders, battery life and power, drift checks, and the log of operators SAI refused."""

from __future__ import annotations

import io
import os
import tempfile
import time
import uuid

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import archspec, db, insight, opgaps, sai_engine  # noqa: E402

db.init()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from backend.app import app
    with TestClient(app) as c:
        r = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                            "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
        c.headers["Authorization"] = f"Bearer {r.json()['token']}"
        yield c


SPEC = {"input": {"channels": 3, "height": 1, "width": 256},
        "layers": [{"op": "conv", "out": 8, "k": 5, "stride": 2}, {"op": "dwconv", "k": 3}, {"op": "pwconv", "out": 12},
                   {"op": "maxpool", "k": 2}, {"op": "pwconv", "out": 16}, {"op": "gap"}], "outputs": 4}


def _result(dataset_id: str | None = None) -> dict:
    spec = archspec.normalize(SPEC)
    summ = archspec.summary(spec)
    sel = {"size": "tiny", "target": "md6605", "objective": "min-ram", "ram": 1200, "workspace": 944, "flash": 6000,
           "latency": {"ms": 3.0, "cycles": 200000, "evidence": "heuristic"}, "metric": 0.95}
    return {"selected": sel, "variants": [{"size": "tiny", "spec": spec, "summary": {"macs": summ["macs"]}}],
            "preprocessing": {"raw_shape": [3, 1, 256], "preprocess": {"decimate": 1}, "layout": "NCHW"},
            "template": {"id": "bearing-fault", "task": "classification"}, "dataset": {"id": dataset_id}}


def _job(org: str, dataset_id: str | None = None, template: str = "bearing-fault") -> str:
    jid = "job_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO jobs (id, kind, project_id, status, stage, progress, stages, result, error, params,"
                  " created, updated, org_id, user_id, priority, attempts, cancel_requested)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)",
                  (jid, "train", None, "succeeded", "package", 1.0, "[]", db.j(_result(dataset_id)), None,
                   db.j({"name": "p", "template_id": template, "dataset_id": dataset_id, "goal": "balanced"}),
                   db.now(), db.now(), org, None, 0))
    return jid


# ---- what-if ------------------------------------------------------------------------------------------------------
def test_whatif_scales_from_the_projects_own_numbers():
    res = _result()
    same = insight.scale(res, 256, 3)
    assert (same["ram"], same["flash"], same["latency_ms"]) == (1200, 6000, 3.0)
    longer = insight.scale(res, 512, 3)
    assert longer["latency_ms"] == pytest.approx(6.0, rel=0.02) and longer["ram"] > 1200 and longer["flash"] == 6000
    more = insight.scale(res, 256, 6)
    assert more["flash"] == 6000 + 3 * 5 * 8          # first conv: 3 more input channels x k5 x 8 filters
    tiny = insight.scale(res, 16, 3)
    assert tiny["ok"] is False or tiny["model_input"][2] == 16
    big = insight.scale(res, 8192, 3)
    assert big["fits"] is False and any("workspace" in v for v in big["violations"])
    grid = insight.whatif_grid(res, {"input": {"sample_rate_hz": 3200, "channel_names": ["x", "y", "z"]}, "hop": 128})
    assert 256 in grid["windows"] and grid["max_channels"] == 6 and len(grid["grid"]) == len(grid["windows"]) * 6


def test_whatif_api_and_sai_check_job(client):
    org = client.get("/api/me").json()["org_id"]
    jid = _job(org)
    g = client.get(f"/api/jobs/{jid}/whatif").json()
    assert g["base"]["window"] == 256 and g["base"]["ram"] == 1200
    client.post("/api/admin/queue/pause", json={"paused": True})
    try:
        j = client.post(f"/api/jobs/{jid}/whatif/check", json={"window": 512, "channels": 2}).json()
        assert j["kind"] == "fitcheck"
        assert client.post(f"/api/jobs/{jid}/whatif/check", json={"window": 512, "channels": 2}).status_code == 409
    finally:
        from backend import pipeline
        pipeline.cancel(j["id"])
        client.post("/api/admin/queue/pause", json={"paused": False})


# ---- battery and power --------------------------------------------------------------------------------------------
def test_battery_life_arithmetic(client):
    org = client.get("/api/me").json()["org_id"]
    jid = _job(org)
    r = client.post(f"/api/jobs/{jid}/battery", json={"battery": "cr2032", "inferences_per_s": 1,
                                                      "active_mw": 30, "sleep_uw": 10}).json()
    # 30 mW x 3 ms = 90 uJ per inference; 1/s -> 0.09 mW + ~0.01 mW sleep
    assert r["energy_uj"] == pytest.approx(90) and r["avg_mw"] == pytest.approx(0.09 + 0.01 * (1 - 0.003), rel=1e-3)
    assert r["life_hours"] == pytest.approx(225 * 3.0 * 0.85 / r["avg_mw"], rel=1e-3)
    assert r["energy_evidence"] == "custom"
    fast = client.post(f"/api/jobs/{jid}/battery", json={"inferences_per_s": 1000}).json()
    assert fast["too_fast"] is True


def test_measured_board_power_replaces_the_assumed_profile(client):
    org = client.get("/api/me").json()["org_id"]
    jid = _job(org)
    with db.conn() as c:
        for i, (sim, e) in enumerate([(False, 40.0), (False, 44.0), (True, 999.0)]):
            c.execute("INSERT INTO hil_runs (id, job_id, board_id, target, status, created, updated, request, result,"
                      " org_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (f"hil_pw{jid}{i}", jid, None, "md6605", "succeeded", db.now(), db.now(), "{}",
                       db.j({"simulated": sim, "energy_uj": e, "latency_ms": 2.5, "active_power_mw": 16.5}), org))
    r = client.post(f"/api/jobs/{jid}/battery", json={"inferences_per_s": 2}).json()
    assert r["energy_evidence"] == "measured" and r["energy_uj"] == pytest.approx(42.0)       # simulated run ignored
    assert r["latency_evidence"] == "measured" and r["measured"]["runs"] == 2


# ---- drift --------------------------------------------------------------------------------------------------------
def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    e = insight._edges(a)
    assert insight.psi(insight._hist(a, e), insight._hist(rng.normal(0, 1, 3000), e)) < 0.02
    assert insight.psi(insight._hist(a, e), insight._hist(rng.normal(1.5, 1, 3000), e)) > 0.25


def test_drift_check_against_the_training_data(client):
    ds = client.post("/api/datasets/demo", json={"template_id": "bearing-fault", "per_class": 60}).json()
    org = client.get("/api/me").json()["org_id"]
    jid = _job(org, ds["id"])
    from backend import data
    X, y, _, _ = data.load(ds["id"])
    # same machine, a fresh recording of the same mix: stable
    same = client.post("/api/datasets/demo", json={"template_id": "bearing-fault", "per_class": 30}).json()
    r = client.post(f"/api/jobs/{jid}/drift", json={"dataset_id": same["id"]}).json()
    assert r["status"] in ("stable", "watch") and r["windows"] == 120, r
    # the sensor was remounted: 3x gain on every channel
    from backend import recordings
    sig = np.concatenate([x[:, 0, :].T for x in X[::4]]) * 3.0
    rec = recordings.create(org, None, "after remount", ["acc_x", "acc_y", "acc_z"], 3200)
    recordings.store_array(rec, sig)
    d = client.post(f"/api/jobs/{jid}/drift", json={"recording_id": rec["id"]}).json()
    assert d["status"] == "drift" and any("rms" in t for t in d["top"]), d
    assert d["inputs"][0]["channel"] == "acc_x"
    hist = client.get(f"/api/jobs/{jid}/drift").json()
    assert [h["status"] for h in hist][-1] == "drift" and len(hist) == 2
    # wrong number of channels
    rec2 = recordings.create(org, None, "one axis", ["acc_x"], 3200)
    recordings.store_array(rec2, sig[:, :1])
    assert client.post(f"/api/jobs/{jid}/drift", json={"recording_id": rec2["id"]}).status_code == 400


# ---- operator gaps ------------------------------------------------------------------------------------------------
def test_sai_refusals_are_logged_with_job_context(client):
    tok = opgaps.JOB.set(("job_ctx1", "analyze", "org_ctx"))
    try:
        opgaps.from_sai_error({"code": "graph.unsupported", "capability": "operator_or_attribute",
                               "location": {"node": 0, "op": "GridSample"}, "reason": "No exact lowering",
                               "status": "unsupported", "target": "generic-core"},
                              ["estimate", "/nonexistent/m.onnx", "--target", "md6605"])
        opgaps.from_sai_error({"code": "model.parse_failed"}, ["inspect", "x.onnx"])       # not an operator gap
    finally:
        opgaps.JOB.reset(tok)
    with pytest.raises(archspec.SpecError):
        archspec.normalize({"input": {"channels": 1, "height": 1, "width": 64}, "layers": [{"op": "conv3d"}], "outputs": 2})
    s = client.get("/api/admin/operator-gaps").json()
    ops = {o["op"]: o for o in s["operators"]}
    assert ops["GridSample"]["targets"] == {"md6605": 1} and ops["GridSample"]["sources"] == {"sai": 1}
    assert "conv3d" in ops and ops["conv3d"]["sources"].get("platform", 0) >= 1
    gs = next(r for r in s["recent"] if r["op"] == "GridSample")
    assert gs["job_id"] == "job_ctx1" and gs["kind"] == "analyze"
    assert opgaps.from_text("error: unsupported layer 'Bidirectional' in model", "sai-front", "md6605") is True


def test_engineers_cannot_read_the_operator_log(client):
    email = f"eng{uuid.uuid4().hex[:6]}@test.local"
    r = client.post("/api/admin/users", json={"email": email, "name": "Eng", "password": "engineer-pass-1",
                                              "role": "engineer"})
    assert r.status_code == 200, r.text
    from fastapi.testclient import TestClient

    from backend.app import app
    with TestClient(app) as e:
        t = e.post("/api/auth/login", json={"email": email, "password": "engineer-pass-1"}).json()["token"]
        e.headers["Authorization"] = f"Bearer {t}"
        assert e.get("/api/admin/operator-gaps").status_code == 403


@pytest.mark.skipif(not sai_engine.available(), reason="no SAI compiler (set SAI_REPO)")
def test_a_real_sai_refusal_is_logged(client):
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 1, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 1, 16])
    g = helper.make_graph([helper.make_node("Erf", ["x"], ["e"]), helper.make_node("Relu", ["e"], ["y"])], "g", [x], [y])
    buf = io.BytesIO()
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), buf)
    j = client.post("/api/explorer/analyze", files={"file": ("erf.onnx", buf.getvalue())},
                    data={"targets": "md6605"}).json()
    for _ in range(120):
        s = client.get(f"/api/jobs/{j['id']}").json()
        if s["status"] in ("succeeded", "failed"):
            break
        time.sleep(1)
    assert s["status"] == "failed"
    rows = [r for r in client.get("/api/admin/operator-gaps").json()["recent"] if r["job_id"] == j["id"]]
    assert rows and rows[0]["op"] == "Erf" and rows[0]["source"] == "sai"
    erf = next(o for o in client.get("/api/admin/operator-gaps").json()["operators"] if o["op"] == "Erf")
    assert "Relu" in erf["with"]
