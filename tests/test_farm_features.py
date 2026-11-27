# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Compiler / farm features: TFLM + CMSIS-NN comparison, the MLPerf Tiny benchmark corpus, customer-VPC workers,
partner boards and the virtual dev board."""

from __future__ import annotations

import http.server
import json
import os
import socket
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import archspec, baselines, db, hil, mlperf, sai_engine, vpc  # noqa: E402

db.init()
ROOT = Path(__file__).resolve().parents[1]
HAS_QEMU = sai_engine.available() and hil.available("qemu-mps2-an500")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from backend.app import app
    with TestClient(app) as c:
        r = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                            "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
        c.headers["Authorization"] = f"Bearer {r.json()['token']}"
        yield c


# ---- TFLM comparison ---------------------------------------------------------------------------------------------
def test_greedy_planner_reuses_memory_like_tflm():
    # a chain a -> b -> c -> d: a and c never live together, so c reuses a's slot
    ts = [{"name": "a", "size": 100, "first": 0, "last": 1}, {"name": "b", "size": 60, "first": 1, "last": 2},
          {"name": "c", "size": 90, "first": 2, "last": 3}, {"name": "d", "size": 10, "first": 3, "last": 4}]
    assert baselines.greedy_plan(ts) == 112 + 64            # 16-byte aligned: 100->112, 60->64
    # everything alive at once: the sum
    assert baselines.greedy_plan([{**t, "first": 0, "last": 9} for t in ts]) == 112 + 64 + 96 + 16


def test_residual_input_stays_alive_until_the_add():
    spec = archspec.normalize({"input": {"channels": 8, "height": 1, "width": 64},
                               "layers": [{"op": "residual", "k": 3}, {"op": "gap"}], "outputs": 2})
    ts, ops = baselines.tensors(spec)
    assert [o["op"] for o in ops][:3] == ["conv", "conv", "add"]
    assert ts[0]["last"] >= 2                                # the block input feeds the Add (op 2)
    assert baselines.greedy_plan(ts) >= 3 * 512              # input, conv1 out, conv2 out coexist


def test_tflm_model_is_labelled_and_arm_only_for_cmsis():
    spec = archspec.normalize(mlperf.CORPUS[0]["spec"])
    m7, rv = baselines.tflm(spec, "cortex-m7"), baselines.tflm(spec, "md6605")
    assert m7["runtime"] == "TFLM + CMSIS-NN" and rv["kernels"] == "reference" and "Arm" in rv["note"]
    assert m7["evidence"] == {"arena": "replayed", "flash": "assumed", "latency": "assumed"}
    assert m7["arena_parts"]["scratch"] > 0 and rv["arena_parts"]["scratch"] == 0
    assert m7["flash"] > sum(r["params"] for r in archspec.shapes(spec))


def test_compare_api_and_measured_override(client):
    org = client.get("/api/me").json()["org_id"]
    spec = archspec.normalize({"input": {"channels": 3, "height": 1, "width": 128},
                               "layers": [{"op": "conv", "out": 8, "k": 5, "stride": 2}, {"op": "gap"}], "outputs": 4})
    sel = {"size": "tiny", "target": "md6605", "ram": 900, "flash": 4000, "workspace": 700,
           "latency": {"ms": 1.0, "evidence": "heuristic"}}
    res = {"selected": sel, "variants": [{"size": "tiny", "spec": spec, "summary": {}, "plans": {"candidates": [
        {**sel, "fits": True}, {**sel, "target": "cortex-m7", "ram": 1200, "fits": True}]}}]}
    jid = "job_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO jobs (id, kind, status, stage, progress, stages, result, params, created, updated, org_id,"
                  " priority, attempts, cancel_requested) VALUES (?,?,?,?,?,?,?,?,?,?,?,0,0,0)",
                  (jid, "train", "succeeded", "package", 1, "[]", db.j(res), db.j({"template_id": "bearing-fault"}),
                   db.now(), db.now(), org))
    r = client.get(f"/api/jobs/{jid}/compare").json()
    md = next(x for x in r["rows"] if x["target"] == "md6605")
    assert md["sai"]["ram"] == 900 and md["ratio"]["ram"] == round(md["tflm"]["ram"] / 900, 2)
    m7 = next(x for x in r["rows"] if x["target"] == "cortex-m7")
    assert m7["sai"]["ram"] == 1200
    r = client.put(f"/api/jobs/{jid}/compare/measured", json={"target": "md6605", "ram": 5400, "flash": 30000,
                                                             "latency_ms": 4.2, "runtime": "TFLM 2.14 -Os"}).json()
    md = next(x for x in r["rows"] if x["target"] == "md6605")
    assert md["measured"]["ram"] == 5400 and md["ratio"]["ram"] == 6.0


# ---- MLPerf corpus -----------------------------------------------------------------------------------------------
def test_mlperf_reference_models_build():
    import torch
    from backend import catalog
    entries = mlperf.corpus(list(catalog.builtin().values()))
    ids = [e["id"] for e in entries]
    assert ids[:4] == ["mlperf-kws-dscnn", "mlperf-vww-mbv1", "mlperf-ic-resnet8", "mlperf-ad-fcae"]
    assert any(i.startswith("tpl-") for i in ids)
    vww = entries[1]
    assert vww["summary"]["macs"] > 7_000_000 and vww["deviation"] is None          # MobileNetV1 0.25 at 96x96
    for e in entries[:4]:
        i = e["spec"]["input"]
        out = archspec.SpecNet(e["spec"])(torch.zeros(1, i["channels"], i["height"], i["width"]))
        assert out.shape[-1] == e["spec"]["outputs"]


def test_benchmark_run_is_for_researchers_and_one_at_a_time(client):
    b = client.get("/api/benchmarks").json()
    assert b["can_run"] is True and len(b["corpus"]) >= 4
    client.post("/api/admin/queue/pause", json={"paused": True})
    try:
        j = client.post("/api/benchmarks/run", json={"ids": ["mlperf-kws-dscnn"]}).json()
        assert j["kind"] == "benchmark"
        assert client.post("/api/benchmarks/run", json={}).status_code == 409
    finally:
        from backend import pipeline
        pipeline.cancel(j["id"])
        client.post("/api/admin/queue/pause", json={"paused": False})


# ---- customer-VPC worker -----------------------------------------------------------------------------------------
CSV = ("time,acc_x,acc_y,acc_z,label\n" + "".join(
    f"{i / 3200:.6f},{np.sin(i / 7):.4f},{np.cos(i / 9):.4f},{0.1 * np.sin(i / 3):.4f},{'normal' if i < 3000 else 'ball'}\n"
    for i in range(6000))).encode()


class _FakeS3(http.server.BaseHTTPRequestHandler):
    """A MinIO-like endpoint inside 'the customer network' (plain http on 127.0.0.1)."""
    seen_auth: list = []

    def do_GET(self):
        self.seen_auth.append(self.headers.get("Authorization", ""))
        if "list-type=2" in self.path:
            body = (b'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    b"<Contents><Key>line3/run.csv</Key><Size>%d</Size><LastModified>2026-11-01T00:00:00Z</LastModified>"
                    b"</Contents></ListBucketResult>" % len(CSV))
        elif self.path.endswith("/line3/run.csv"):
            body = CSV
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def test_vpc_worker_end_to_end_with_private_endpoint(client, monkeypatch, tmp_path):
    import uvicorn

    from backend.app import app
    monkeypatch.setattr(vpc, "WAIT_S", 0.3)
    monkeypatch.delenv("SAI_PLATFORM_ALLOW_PRIVATE_CONNECTORS", raising=False)
    s3 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeS3)
    threading.Thread(target=s3.serve_forever, daemon=True).start()
    w = client.post("/api/admin/vpc-workers", json={"name": "plant-1"}).json()
    con = client.post("/api/connectors", json={"kind": "s3", "name": "plant MinIO", "config": {
        "endpoint": f"http://127.0.0.1:{s3.server_port}", "bucket": "sensors", "path_style": True,
        "access_key": "plantkey", "via_worker": True}, "secrets": {"secret_key": "must-not-be-stored"}}).json()
    assert con["secrets_set"] == [] and con["config"]["via_worker"] is True
    # the platform itself cannot reach the private endpoint, the worker will
    pend = client.get(f"/api/connectors/{con['id']}/browse")
    assert pend.status_code == 202 and pend.json()["workers_online"] == 0
    imp = client.post(f"/api/connectors/{con['id']}/import", json={"key": "line3/run.csv", "template_id": "bearing-fault",
                                                                   "licence": "proprietary"})
    assert imp.status_code == 202
    # serve the platform over real HTTP and run the worker against it
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    try:
        (tmp_path / "secrets.json").write_text(json.dumps({"plant MinIO": {"secret_key": "local-only"}}))
        env = {**os.environ, "SAI_VPC_TOKEN": w["token"], "SAI_PLATFORM_DATA": str(tmp_path / "w")}
        import subprocess
        p = subprocess.run([sys.executable, str(ROOT / "tools" / "vpc_worker.py"), "--server", f"http://127.0.0.1:{port}",
                            "--secrets", str(tmp_path / "secrets.json"), "--once"], env=env, capture_output=True,
                           text=True, timeout=120)
        assert p.returncode == 0, p.stderr
    finally:
        server.should_exit = True
        s3.shutdown()
    assert all("Credential=plantkey/" in a for a in _FakeS3.seen_auth)          # signed with the worker's secret
    browse = client.get(f"/api/vpc/tasks/{pend.json()['task']}").json()
    assert browse["status"] == "succeeded" and browse["result"]["items"][0]["key"] == "line3/run.csv"
    t = client.get(f"/api/vpc/tasks/{imp.json()['task']}").json()
    assert t["status"] == "succeeded", t
    ds = t["result"]["dataset"]
    assert ds["source"] == "connector:s3" and set(ds["class_counts"]) == {"normal", "ball"}
    with db.conn() as c:
        assert "local-only" not in str([tuple(r) for r in c.execute("SELECT * FROM connections")])
        assert "local-only" not in str([tuple(r) for r in c.execute("SELECT * FROM vpc_tasks")])


def test_vpc_tokens_are_org_scoped(client):
    a = client.post("/api/admin/orgs", json={"name": f"VPC-{uuid.uuid4().hex[:4]}", "tier": "professional"}).json()
    w_other, tok_other = vpc.create_worker(a["id"], "other org")
    org = client.get("/api/me").json()["org_id"]
    con = client.post("/api/connectors", json={"kind": "azure", "name": "plant blob", "config": {
        "account": "plantstore", "container": "c", "via_worker": True}}).json()
    t = vpc.submit(org, con["id"], "browse", {"prefix": ""})
    r = client.post("/api/vpc/claim", json={}, headers={"Authorization": f"Bearer {tok_other}"})
    assert r.status_code == 204                                                # not its organisation's task
    assert client.post("/api/vpc/claim", json={}, headers={"Authorization": "Bearer nope"}).status_code == 401
    vpc.finish(t["id"], False, error="test cleanup")


# ---- partner boards ------------------------------------------------------------------------------------------------
def test_partner_boards_and_partner_hosted_agents(client):
    r = client.post("/api/partner-boards", json={"partner": "Acme Labs", "board": "Acme RV-66 EVB", "vendor": "Acme",
                                                 "mcu": "RV32IMAFC 66 MHz", "target": "md6605", "status": "community"})
    assert r.status_code == 200
    assert client.post("/api/partner-boards", json={"partner": "x", "board": "y", "target": "esp32"}).status_code == 400
    ag = client.post("/api/admin/agents", json={"name": "acme lab pc", "partner": "Acme Labs"}).json()
    assert ag["partner"] == "Acme Labs"
    from backend import boards
    boards.register(ag["id"], [{"id": "ACME-RV-01", "name": "Acme RV-66 EVB #1", "target": "md6605"}])
    pb = client.get("/api/partner-boards").json()["boards"]
    row = next(b for b in pb if b["board"] == "Acme RV-66 EVB")
    assert row["online_boards"] == ["ACME-RV-01"]
    lst = client.get("/api/boards").json()["boards"]
    assert next(b for b in lst if b["id"] == "ACME-RV-01")["partner"] == "Acme Labs"


# ---- virtual dev board (needs SAI + QEMU + cross compiler) ---------------------------------------------------------
@pytest.mark.skipif(not HAS_QEMU, reason="sai, Cortex-M toolchain or QEMU missing")
def test_virtual_dev_board_runs_the_package_on_your_inputs(client):
    ds = client.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 40}).json()
    job = client.post("/api/projects", json={"name": "devboard", "template_id": "gas-sense", "dataset_id": ds["id"],
                                             "sizes": ["tiny"], "epochs": 3, "targets": ["cortex-m7"]}).json()["job"]
    for _ in range(600):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("succeeded", "failed"):
            break
        time.sleep(1)
    assert j["status"] == "succeeded", j.get("error")
    s = client.post(f"/api/jobs/{job['id']}/devboard", json={"target": "cortex-m7",
                                                             "inputs": {"source": "dataset", "indices": [0, 5, 77]}}).json()
    for _ in range(300):
        r = client.get(f"/api/jobs/{s['id']}").json()
        if r["status"] in ("succeeded", "failed"):
            break
        time.sleep(1)
    assert r["status"] == "succeeded", r.get("error")
    res = r["result"]
    assert res["all_bit_exact"] is True and len(res["windows"]) == 3
    assert res["windows"][0]["prediction"] in res["labels"] and res["windows"][0]["truth"] in res["labels"]
    assert "SAI-HIL" in res["uart"] and res["cycles"]["evidence"] == "emulated"
    cmp = client.get(f"/api/jobs/{job['id']}/compare").json()
    m7 = next(x for x in cmp["rows"] if x["target"] == "cortex-m7")
    assert m7["sai"]["ram"] and m7["tflm"]["runtime"] == "TFLM + CMSIS-NN"
