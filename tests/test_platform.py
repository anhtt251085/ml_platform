"""Platform tests. Run: SAI_PLATFORM_DATA=$(mktemp -d) pytest -q tests"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")
ADMIN = (os.environ["SAI_PLATFORM_ADMIN_EMAIL"], os.environ["SAI_PLATFORM_ADMIN_PASSWORD"])

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from backend import archspec, catalog, data, qexport, sai_engine, trainer  # noqa: E402

HAS_SAI = shutil.which(os.environ.get("SAI_BIN", "sai")) is not None
need_sai = pytest.mark.skipif(not HAS_SAI, reason="sai CLI not installed")


def test_spec_validation_and_shapes():
    spec = archspec.normalize({"input": {"channels": 3, "width": 128},
                               "layers": [{"op": "conv", "out": 8, "k": 5, "stride": 2}, {"op": "maxpool", "k": 2},
                                          {"op": "residual"}, {"op": "gap"}], "outputs": 4})
    rows = archspec.shapes(spec)
    assert rows[0]["shape"] == [8, 1, 64] and rows[1]["shape"] == [8, 1, 32] and rows[-1]["shape"] == [4]
    with pytest.raises(archspec.SpecError):
        archspec.normalize({"input": {"channels": 1, "width": 4}, "layers": [{"op": "maxpool", "k": 8}], "outputs": 2})
    with pytest.raises(archspec.SpecError):
        archspec.normalize({"input": {"channels": 1, "width": 64}, "layers": [{"op": "fc", "out": 4}, {"op": "conv", "out": 4}]})
    with pytest.raises(archspec.SpecError):
        archspec.normalize({"input": {"channels": 1, "width": 64}, "layers": [{"op": "lstm"}]})


@pytest.mark.parametrize("tid", [t["id"] for t in catalog.BUILTIN])
def test_every_builtin_variant_is_valid(tid):
    t = catalog.builtin()[tid]
    n = len(t["labels"]) if t["task"] == "classification" else 1 if t["task"] == "regression" else \
        t["input"]["channels"] * t["input"]["height"] * t["input"]["width"]
    for layers in t["variants"].values():
        archspec.normalize({"input": t["input"], "layers": layers, "outputs": n})


def test_bn_fold_preserves_function():
    spec = archspec.normalize({"input": {"channels": 2, "width": 32}, "layers": [{"op": "conv", "out": 4}, {"op": "gap"}], "outputs": 3})
    net = archspec.SpecNet(spec)
    net.train()
    x = torch.randn(16, 2, 1, 32)
    for _ in range(3):
        net(x)                                       # move BN running stats
    net.eval()
    with torch.no_grad():
        before = net(x)
        net.fold_bn()
        after = net(x)
    assert torch.allclose(before, after, atol=1e-5)


@need_sai
@pytest.mark.parametrize("shape,layers", [
    ((3, 1, 128), [{"op": "conv", "out": 8, "k": 5, "stride": 2}, {"op": "dwconv"}, {"op": "pwconv", "out": 12},
                   {"op": "residual"}, {"op": "gap"}, {"op": "fc", "out": 8, "relu": True}]),
    ((1, 24, 32), [{"op": "conv", "out": 8, "k": 3, "stride": 2}, {"op": "avgpool", "k": 2}, {"op": "gap"}]),
])
def test_export_compiles_and_host_is_exact(tmp_path, shape, layers):
    from backend import exact_eval
    spec = archspec.normalize({"input": {"channels": shape[0], "height": shape[1], "width": shape[2]}, "layers": layers, "outputs": 4})
    net = archspec.SpecNet(spec).eval()
    net.fold_bn()
    x = np.random.default_rng(0).normal(size=(64, *shape)).astype(np.float32)
    q = qexport.export_qdq(net, x, tmp_path / "m.onnx")
    est = sai_engine.estimate(tmp_path / "m.onnx", "cortex-m7", "min-ram")
    assert est["resources"]["tensor_workspace"]["evidence"] == "static_exact"
    xq = qexport.quantize_input(x[:20], q["input"])
    sai_out = exact_eval.run_exact(tmp_path / "m.onnx", tmp_path / "exact", xq).astype(int)
    ort_out = qexport.ort_int8_outputs(tmp_path / "m.onnx", xq).reshape(20, -1).astype(int)
    # onnxruntime computes QDQ convs in float with one rounding; SAI uses exact integer requantization. 1-LSB
    # differences can grow through later layers, so this is a framework-agreement check, not the contract
    # (SAI's own `verify` replays its frozen vectors).
    d = np.abs(sai_out - ort_out)
    assert d.mean() <= 1.0 and d.max() <= 4


def test_qat_finetune_runs_and_exports(tmp_path):
    X = np.random.default_rng(0).normal(size=(120, 2, 1, 32)).astype(np.float32)
    y = (X[:, 0].mean(axis=(1, 2)) > 0).astype(int)
    idx = trainer.split(y, "classification")
    spec = trainer.build_spec([{"op": "conv", "out": 4}, {"op": "gap"}], X, "classification", 2)
    net, _ = trainer.train(spec, X, y, idx, "classification", epochs=3, min_steps=20)
    net.fold_bn()
    act, _ = qexport.calibrate(net, X[:32])
    net, _ = trainer.qat_finetune(net, act, X, y, idx, "classification", epochs=2)
    assert net.act_q is None
    q = qexport.export_qdq(net, X[:32], tmp_path / "q.onnx", q=act)
    assert q["input"]["scale"] == pytest.approx(act["x"][0])


def test_csv_windowing_respects_segments():
    tpl = catalog.builtin()["dcdc-fault"]
    rows = ["v_out,i_l,label,segment"]
    for seg in range(4):
        for t in range(300):
            rows.append(f"{12 + 0.01 * t},{2.0},{'normal' if seg % 2 == 0 else 'short'},{seg}")
    meta = data.from_csv_bytes("\n".join(rows).encode(), tpl, "t.csv", window=128, hop=64)
    assert meta["shape"] == [2, 1, 128]
    assert meta["n"] == 4 * 3                       # (300-128)//64 + 1 = 3 windows per segment, none crossing
    assert meta["class_counts"] == {"normal": 6, "short": 6}


def test_npz_rejects_pickles():
    buf = io.BytesIO()
    np.savez(buf, X=np.array([object()], dtype=object), y=np.array([0]))
    with pytest.raises(data.DataError):
        data.from_npz_bytes(buf.getvalue(), catalog.builtin()["gas-sense"], "x.npz")


def test_plan_pick_respects_goal():
    mk = lambda i, lat, ram, fl: {"id": i, "latency": {"ms": lat}, "ram": ram, "flash": fl}  # noqa: E731
    rows = [mk("a", 1.0, 900, 5000), mk("b", 2.0, 300, 6000), mk("c", 3.0, 800, 3000)]
    assert sai_engine._pick(rows, "fastest")["id"] == "a"
    assert sai_engine._pick(rows, "min-ram")["id"] == "b"
    assert sai_engine._pick(rows, "min-flash")["id"] == "c"


def login(c, email=None, pw=None):
    r = c.post("/api/auth/login", json={"email": email or ADMIN[0], "password": pw or ADMIN[1]})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def wait_job(c, jid, h=None, timeout=300):
    for _ in range(timeout):
        j = c.get(f"/api/jobs/{jid}", headers=h).json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(1)
    raise AssertionError("job timeout")


@need_sai
def test_api_end_to_end_with_board_run():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        c.headers.update(login(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 60}).json()
        r = c.post("/api/projects", json={"name": "t", "template_id": "gas-sense", "dataset_id": ds["id"],
                                          "sizes": ["tiny"], "epochs": 5, "goal": "min-ram"}).json()
        jid = r["job"]["id"]
        j = wait_job(c, jid)
        assert j["status"] == "succeeded", j["error"]
        res = j["result"]
        assert res["verify"]["passed"] == res["verify"]["total"]
        assert res["selected"]["fits"]
        assert c.get(f"/api/jobs/{jid}/download/package.zip").status_code == 200
        run = c.post("/api/boards/runs", json={"job_id": jid}).json()
        for _ in range(60):
            rr = next(x for x in c.get("/api/boards/runs").json() if x["id"] == run["id"])
            if rr["status"] not in ("queued", "running"):
                break
            time.sleep(1)
        assert rr["status"] == "succeeded" and rr["result"]["outputs_match"] is True
        assert rr["result"]["simulated"] is True
        # simulated runs never calibrate latency
        assert c.get("/api/boards").json()["calibration"] == {}
        # M1: provenance recorded and a rebuild is byte-identical
        prov = res["provenance"]
        assert prov["build_digest"] and prov["dataset_sha256"] and prov["sai_version"]
        rb = wait_job(c, c.post(f"/api/jobs/{jid}/rebuild").json()["id"])
        assert rb["status"] == "succeeded" and rb["result"]["identical"] is True, rb["result"]
        # Standard vs Professional comparison from SAI
        tc = res["tier_compare"][res["selected"]["target"]]
        assert tc["standard"]["schedule"] and tc["professional"]["schedule"]


def test_agent_tokens_are_per_agent():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        assert c.post("/api/agent/poll", json={}).status_code == 401
        # the shared dev token is refused unless explicitly enabled
        assert c.post("/api/agent/poll", json={}, headers={"Authorization": "Bearer dev-agent-token"}).status_code == 401
        h = login(c)
        tok = c.post("/api/admin/agents", json={"name": "lab9"}, headers=h).json()
        ah = {"Authorization": f"Bearer {tok['token']}"}
        ok = c.post("/api/agent/register", json={"boards": [{"id": "L9", "target": "md6605"}]}, headers=ah)
        assert ok.status_code == 200 and ok.json()["agent_id"] == tok["id"]
        assert c.post("/api/agent/poll", json={}, headers=ah).json() == {"run": None}
        c.post(f"/api/admin/agents/{tok['id']}/disable", headers=h)
        assert c.post("/api/agent/poll", json={}, headers=ah).status_code == 401


def test_auth_roles_and_org_isolation():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        assert c.get("/api/projects").status_code == 401
        h = login(c)
        a = c.post("/api/admin/orgs", json={"name": "Acme", "tier": "standard"}, headers=h).json()
        b = c.post("/api/admin/orgs", json={"name": "Globex", "tier": "free"}, headers=h).json()
        for org, email in ((a, "eng@acme.test"), (b, "eng@globex.test")):
            r = c.post("/api/admin/users", headers=h, json={"email": email, "name": "Eng", "role": "member",
                                                            "password": "password-1", "org_id": org["id"]})
            assert r.status_code == 200, r.text
        ha, hb = login(c, "eng@acme.test", "password-1"), login(c, "eng@globex.test", "password-1")
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 30}, headers=ha).json()
        assert c.get(f"/api/datasets/{ds['id']}", headers=ha).status_code == 200
        assert c.get(f"/api/datasets/{ds['id']}", headers=hb).status_code == 404          # other org: invisible
        assert all(d["id"] != ds["id"] for d in c.get("/api/datasets", headers=hb).json())
        assert c.post("/api/projects", headers=hb, json={"name": "x", "template_id": "gas-sense",
                                                          "dataset_id": ds["id"]}).status_code == 404
        # members cannot use the Studio, publish, or administer
        assert c.post("/api/studio/fitcheck", json={"spec": {}}, headers=ha).status_code == 403
        assert c.post("/api/catalog", json={"name": "x"}, headers=ha).status_code == 403
        assert c.get("/api/admin/agents", headers=ha).status_code == 403
        assert c.post("/api/admin/users", headers=ha, json={"email": "x@y.z", "name": "x", "role": "member",
                                                             "password": "password-1"}).status_code == 403
        assert c.post("/api/auth/login", json={"email": "eng@acme.test", "password": "wrong"}).status_code == 401
        me = c.get("/api/me", headers=ha).json()
        assert me["org"]["tier"] == "standard" and not me["perms"]["studio"]
        actions = [x["action"] for x in c.get("/api/admin/audit", headers=h).json()]
        assert "org_created" in actions and "login_failed" in actions


def test_queue_claim_is_exclusive_and_cancellable():
    from backend import db, pipeline
    db.init()
    pipeline.pause_workers(True)
    try:
        time.sleep(1.2)                                    # let in-process workers notice the pause
        a = pipeline.create("analyze", {"model_path": "/nonexistent.onnx"}, priority=5)["id"]
        b = pipeline.create("analyze", {"model_path": "/nonexistent.onnx"}, priority=5)["id"]
        pipeline.cancel(b)
        assert pipeline.get(b)["status"] == "cancelled"
        first = pipeline.claim("w1", ["analyze"])
        assert first["id"] == a and first["status"] == "running" and first["worker"] == "w1"
        assert pipeline.claim("w2", ["analyze"]) is None or pipeline.claim("w2", ["analyze"])["id"] not in (a, b)
    finally:
        pipeline.pause_workers(False)


def test_migration_from_v01_database(tmp_path, monkeypatch):
    import sqlite3
    from backend import config, db
    path = tmp_path / "old.sqlite3"
    c = sqlite3.connect(path)
    c.executescript(db.MIGRATIONS[0])
    c.execute("INSERT INTO jobs (id, kind, status, created, updated) VALUES ('job_old', 'train', 'succeeded', 0, 0)")
    c.commit(); c.close()
    monkeypatch.setattr(config, "DB_PATH", path)
    db.init()
    assert db.version() == len(db.MIGRATIONS)
    with db.conn() as k:
        assert k.execute("SELECT org_id, cancel_requested FROM jobs WHERE id='job_old'").fetchone()[1] == 0
