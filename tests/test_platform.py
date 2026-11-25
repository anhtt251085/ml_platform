"""Platform tests. Run: SAI_PLATFORM_DATA=$(mktemp -d) pytest -q tests"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))

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


@need_sai
def test_api_end_to_end_with_board_run():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 60}).json()
        r = c.post("/api/projects", json={"name": "t", "template_id": "gas-sense", "dataset_id": ds["id"],
                                          "sizes": ["tiny"], "epochs": 5, "goal": "min-ram"}).json()
        jid = r["job"]["id"]
        for _ in range(300):
            j = c.get(f"/api/jobs/{jid}").json()
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(1)
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


def test_agent_endpoints_require_token():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        assert c.post("/api/agent/poll", json={"agent_id": "x"}).status_code == 401
        ok = c.post("/api/agent/register", json={"agent_id": "lab9", "boards": [{"id": "L9", "target": "md6605"}]},
                    headers={"Authorization": "Bearer dev-agent-token"})
        assert ok.status_code == 200 and ok.json()["boards"][0]["id"] == "L9"
