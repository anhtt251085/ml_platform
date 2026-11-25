"""M4 board farm: HIL harness firmware on emulated boards, the lab-agent protocol over HTTP, quarantine, regression."""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import threading
import time

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import archspec, exact_eval, hil, qexport, sai_engine  # noqa: E402

from backend import sai_engine as _sai_engine  # noqa: E402

HAS_SAI = _sai_engine.available()
HAS_QEMU = HAS_SAI and hil.available("qemu-mps2-an500") and hil.available("qemu-virt-rv32")
need_qemu = pytest.mark.skipif(not HAS_QEMU, reason="sai, MCU toolchains or QEMU missing")


@need_qemu
@pytest.mark.parametrize("port", ["qemu-mps2-an500", "qemu-virt-rv32"])
def test_harness_is_bit_exact_on_emulated_isa(tmp_path, port):
    spec = archspec.normalize({"input": {"channels": 2, "width": 64}, "layers": [{"op": "conv", "out": 8, "k": 5, "stride": 2},
                               {"op": "dwconv"}, {"op": "pwconv", "out": 8}, {"op": "gap"}, {"op": "fc", "out": 8, "relu": True}],
                               "outputs": 3})
    net = archspec.SpecNet(spec).eval()
    net.fold_bn()
    x = np.random.default_rng(5).normal(size=(32, 2, 1, 64)).astype(np.float32)
    q = qexport.export_qdq(net, x, tmp_path / "m.onnx")
    xq = qexport.quantize_input(x[:6], q["input"])
    ref = exact_eval.run_exact(tmp_path / "m.onnx", tmp_path / "exact", xq)
    target = hil.PORTS[port]["target"]
    sai_engine.compile_build(tmp_path / "m.onnx", target, "fast", tmp_path / "b")
    fw = hil.build(tmp_path / "b", port, xq, tmp_path / "fw")
    res = hil.run_emulated(fw["elf"], port)
    assert np.array_equal(np.array(res["outputs"]), ref.astype(int))
    bound = sai_engine.estimate(tmp_path / "m.onnx", target, "fast")["resources"]["stack"]["value"]
    assert res["stack_painted"] >= bound * 0.8          # painted includes the harness frames; same order as SAI's proof
    assert res["cycles"] > 0


def test_physical_board_firmware_links():
    """The NUCLEO and MD6605 ports must at least link with SAI's production flags (CI catches bring-up breakage)."""
    if not HAS_SAI or not hil.available("stm32h753-nucleo") or not hil.available("md6605-evb"):
        pytest.skip("toolchains missing")
    d = tempfile.mkdtemp()
    from pathlib import Path
    spec = archspec.normalize({"input": {"channels": 1, "width": 32}, "layers": [{"op": "conv", "out": 4}, {"op": "gap"}], "outputs": 2})
    net = archspec.SpecNet(spec).eval()
    net.fold_bn()
    q = qexport.export_qdq(net, np.random.randn(16, 1, 1, 32).astype(np.float32), Path(d) / "m.onnx")
    xq = qexport.quantize_input(np.random.randn(2, 1, 1, 32).astype(np.float32), q["input"])
    for port in ("stm32h753-nucleo", "md6605-evb"):
        t = hil.PORTS[port]["target"]
        sai_engine.compile_build(Path(d) / "m.onnx", t, "min-ram", Path(d) / f"b_{t}")
        fw = hil.build(Path(d) / f"b_{t}", port, xq, Path(d) / f"fw_{port}")
        assert fw["text"] and fw["text"] < 65536
    assert fw["bss"] < 8192                             # MD6605: inside its 8 KB SRAM


def _login(c):
    r = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                        "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _wait(c, path, done, timeout=300):
    for _ in range(timeout):
        j = c.get(path).json()
        if done(j):
            return j
        time.sleep(1)
    raise AssertionError("timeout " + path)


@need_qemu
def test_lab_agent_protocol_over_http_with_qemu_driver(tmp_path):
    import uvicorn
    from fastapi.testclient import TestClient
    from backend.app import app
    sys_path = os.path.join(os.path.dirname(__file__), "..", "agent")
    import sys
    sys.path.insert(0, sys_path)
    import board_agent
    with TestClient(app) as c:
        c.headers.update(_login(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 40}).json()
        job = c.post("/api/projects", json={"name": "hil", "template_id": "gas-sense", "dataset_id": ds["id"], "sizes": ["tiny"],
                                            "epochs": 3, "targets": ["md6605", "cortex-m7"]}).json()["job"]
        j = _wait(c, f"/api/jobs/{job['id']}", lambda j: j["status"] in ("succeeded", "failed"))
        assert j["status"] == "succeeded", j["error"]
        token = c.post("/api/admin/agents", json={"name": "ci-qemu"}).json()["token"]
        # a real HTTP server for the agent
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        try:
            agent = board_agent.Agent(f"http://127.0.0.1:{port}", token,
                                      [{"id": "CI-RV32", "name": "QEMU RV32", "target": "md6605", "port": "qemu-virt-rv32",
                                        "driver": "qemu"}], work=tmp_path)
            agent.register()
            # the other target than the selected one: the server compiles the model for md6605 on demand
            run = c.post("/api/boards/runs", json={"job_id": job["id"], "target": "md6605", "board_id": "CI-RV32"}).json()
            assert agent.poll_once()
            rr = _wait(c, "/api/boards/runs", lambda rs: next(x for x in rs if x["id"] == run["id"])["status"] != "running")
            rr = next(x for x in rr if x["id"] == run["id"])
            assert rr["status"] == "succeeded", rr["result"]
            assert rr["result"]["outputs_match"] is True and rr["result"]["emulated"] is True
            boards = {b["id"]: b for b in c.get("/api/boards").json()["boards"]}
            assert boards["CI-RV32"]["kind"] == "emulated" and boards["CI-RV32"]["health"]["probe"] == "ok"
        finally:
            server.should_exit = True
            th.join(5)


def test_failing_board_is_quarantined_and_reset():
    from backend import boards, db
    db.init()
    boards.register("lab-q", [{"id": "Q-1", "target": "cortex-m7"}])
    for _ in range(boards.QUARANTINE_AFTER):
        boards.request_run("job_x", "cortex-m7", "Q-1", {})
        run = boards.claim("lab-q")
        assert run is not None
        boards.complete(run["id"], {"ok": False, "error": "probe lost"})
    b = next(x for x in boards.list_boards() if x["id"] == "Q-1")
    assert b["status"] == "quarantined"
    boards.request_run("job_x", "cortex-m7", "Q-1", {})
    assert boards.claim("lab-q") is None                 # quarantined boards get no work
    boards.reset_board("Q-1")
    assert boards.claim("lab-q") is not None


@need_qemu
def test_regression_job_records_and_compares():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        c.headers.update(_login(c))
        for _ in range(2):
            j = c.post("/api/admin/regression", json={"templates": ["dcdc-fault"], "hil": True}).json()
            j = _wait(c, f"/api/jobs/{j['id']}", lambda j: j["status"] in ("succeeded", "failed"))
            assert j["status"] == "succeeded", j["error"]
        r = j["result"]
        assert r["alerts"] == [] and {row["target"] for row in r["rows"]} == {"md6605", "cortex-m7"}
        assert all(row["prev"] for row in r["rows"])          # second run compared with the first
        assert c.get("/api/regressions").json()["latest"]
