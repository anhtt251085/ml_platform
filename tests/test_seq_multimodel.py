# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Sequence models (LSTM / GRU / RNN / attention through SAI's front end, XModels backend) and several models on
one board (SAI's multi-model plugin). The SAI-dependent tests run when SAI_REPO is a product/sai-unified checkout."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import time
import zipfile

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from backend import archlib, archspec, multimodel, preprocess, sai_engine, seqfront  # noqa: E402
from tests.approval_helper import release  # noqa: E402

TS = {"channels": 6, "height": 1, "width": 64}
need_front = pytest.mark.skipif(not seqfront.available(), reason="needs SAI_REPO on product/sai-unified (sai_front)")
need_mm = pytest.mark.skipif(not (multimodel.available() and sai_engine.available()),
                             reason="needs SAI_REPO with extensions/sai_multimodel")


# ---- builder ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layers", [
    [{"op": "lstm", "units": 16}],
    [{"op": "gru", "units": 12, "bidirectional": True}],
    [{"op": "rnn", "units": 8}],
    [{"op": "lstm", "units": 16, "sequences": True}, {"op": "gru", "units": 8}],
    [{"op": "conv", "out": 12, "k": 5, "stride": 2}, {"op": "gru", "units": 16}],
    [{"op": "fc", "out": 16}, {"op": "attention", "heads": 4}, {"op": "layernorm"}, {"op": "fc", "out": 8}, {"op": "gap"}],
])
def test_sequence_specs_count_like_pytorch(layers):
    spec = archspec.normalize({"input": TS, "layers": layers, "outputs": 3})
    assert spec["layout"] == "sequence" and archspec.summary(spec)["backend"] == "xmodels"
    net = archspec.SpecNet(spec)
    assert archspec.summary(spec)["params"] == sum(p.numel() for p in net.parameters())
    t, f = archspec.seq_dims(spec["input"])
    assert tuple(net(torch.zeros(2, t, f)).shape) == (2, 3)


@pytest.mark.parametrize("layers, msg", [
    ([{"op": "lstm", "sequences": True}], "end the sequence part"),
    ([{"op": "lstm", "sequences": True}, {"op": "gap"}], "right after a recurrent layer"),
    ([{"op": "dwconv", "k": 3}, {"op": "lstm"}], "not dwconv"),
    ([{"op": "fc", "out": 10}, {"op": "attention", "heads": 4}, {"op": "gap"}], "do not split"),
    ([{"op": "lstm"}, {"op": "gru"}], "already reduced"),
    ([{"op": "conv", "out": 4, "k": 31, "stride": 1, "dilation": 4}, {"op": "lstm"}], "shorter than the kernel"),
])
def test_sequence_specs_explain_what_is_not_supported(layers, msg):
    with pytest.raises(archspec.SpecError, match=msg):
        archspec.normalize({"input": TS, "layers": layers, "outputs": 3})


def test_sequence_layout_of_1d_and_2d_data():
    x = np.arange(2 * 3 * 1 * 5, dtype=np.float32).reshape(2, 3, 1, 5)
    s = archspec.to_sequence(x)
    assert s.shape == (2, 5, 3) and s[1, 4, 2] == x[1, 2, 0, 4]
    y = np.arange(2 * 2 * 4 * 3, dtype=np.float32).reshape(2, 2, 4, 3)       # [C=2, H=4, W=3]
    s2 = archspec.to_sequence(y)
    assert s2.shape == (2, 4, 6) and s2[1, 3, 1 * 3 + 2] == y[1, 1, 3, 2]
    assert archspec.seq_dims({"channels": 2, "height": 4, "width": 3}) == (4, 6)


@pytest.mark.parametrize("shape, width", [((3, 1, 16), 1), ((2, 4, 8), 8)])
def test_c_prepare_writes_the_sequence_layout(tmp_path, shape, width):
    """The package's sai_app_prepare produces exactly the quantized to_sequence() input the model was trained on."""
    rng = np.random.default_rng(0)
    c, h, w = shape
    raw = rng.normal(0, 2, (1, c, h, w)).astype(np.float32)
    cfg = preprocess.normalize_cfg({}, {}, shape, "classification")
    off, sc = preprocess.fit_norm(raw, cfg)
    q = {"scale": 0.05, "zero_point": -3}
    (tmp_path / "sai_app_prepare.h").write_text(preprocess.c_prepare(cfg, off, sc, q, list(shape), seq_width=width))
    for f in ("sai_prep.c", "sai_prep.h"):
        (tmp_path / f).write_bytes((preprocess.SRC / f).read_bytes())
    (tmp_path / "m.c").write_text('#include <stdio.h>\n#include "sai_app_prepare.h"\nint main(void){static float r[%d];'
                                  'static int8_t q[%d];for(int i=0;i<%d;i++) if(scanf("%%f",&r[i])!=1) return 2;'
                                  'if(sai_app_prepare(r,q)) return 3;for(int i=0;i<%d;i++) printf("%%d\\n",q[i]);'
                                  'return 0;}' % ((c * h * w,) * 4))
    exe = tmp_path / "m"
    subprocess.run(["cc", "-O1", "-o", str(exe), str(tmp_path / "m.c"), str(tmp_path / "sai_prep.c"), "-lm"], check=True)
    out = subprocess.run([str(exe)], input=" ".join(f"{v:.9g}" for v in raw.reshape(-1)), capture_output=True,
                         text=True, check=True).stdout.split()
    norm = preprocess.apply_norm(raw, off, sc, None)
    want = np.clip(np.rint(archspec.to_sequence(norm).reshape(-1) / q["scale"]) + q["zero_point"], -128, 127)
    assert [int(v) for v in out] == want.astype(int).tolist()


def test_library_has_sequence_architectures():
    seq = [a for a in archlib.BUILTIN if archspec.is_sequence({"layers": a["layers"]})]
    assert {"lstm-s", "gru-s", "bilstm-s", "crnn-s", "tiny-transformer"} <= {a["id"] for a in seq}
    for a in seq:
        for kind in a["inputs"]:
            archspec.normalize({"input": archlib.REFERENCE[kind], "layers": a["layers"], "outputs": 4})
    assert archlib.compatible(next(a for a in seq if a["id"] == "gru-s"), (1, 49, 10), "classification", 12)["ok"]


# ---- front-end and plugin glue (no SAI needed) -------------------------------------------------------------------
def test_front_end_report_becomes_a_plan_row():
    res = {"report": {"objective": "balanced", "backend": {"plan": "lstm_chain:chain", "workspace_bytes": 128,
                                                           "cortex_m7_instructions": 250000},
                      "accuracy": {"top1_agreement": 1.0, "integer_top1": 0.97}},
           "compile_report": {"memory": {"workspace_bytes": 128, "input_bytes": 256, "output_bytes": 5},
                              "device_ram": {"stack_static_bound": 216},
                              "code": {"cortex_m7_object": {"text": 2616, "rodata": 3325}},
                              "plan": {"selected": "lstm_chain:chain", "measured_shortlist": [
                                  {"label": "lstm_chain:chain", "instructions": 435711, "isa": "rv32"}]}},
           "dir": "/x"}
    r = seqfront.row(res, "md6605", {})
    assert r["ram"] == 128 + 261 + 216 and r["flash"] == 2616 + 3325 and r["fits"]
    assert r["latency"]["cycles"] == 435711 and r["latency"]["evidence"] == "qemu_instructions"
    tight = seqfront.row(res, "md6605", {"max_ram": 300})
    assert not tight["fits"] and "your limit" in tight["violations"][0]


def test_multimodel_workload_and_ram_accounting():
    import yaml
    used = set()
    ids = [multimodel.model_id(n, used) for n in ("Gas leak detector", "Gas leak detector", "2nd model!",
                                                   "Bearing fault detection – 11/25/2026", "SAI core test")]
    assert ids[:3] == ["gas_leak_detector", "gas_leak_detector_2", "m_2nd_model"]
    import re
    assert all(re.fullmatch(r"[a-z][a-z0-9_]{0,23}", i) for i in ids) and ids[4].startswith("m_sai")
    models = [{"id": i, "sha256": "0" * 64, "job_id": f"job_{n}", "project": i, "template_id": "t"} for n, i in enumerate(ids[:2])]
    doc = yaml.safe_load(multimodel.workload("Plant #1", models, "md6605", {"max_ram": 4000}))
    assert doc["workload_id"] == "plant_1" and doc["mode"] == "independent_sequential"
    assert doc["models"][0]["sha256"] == "0" * 64 and doc["constraints"]["peak_total_ram_bytes"] == 4000
    bundle = {"abi_search": {"selected": {"abi": {ids[0]: "1.1", ids[1]: "1.0"}, "device_ram": 2000,
                                          "shared_workspace": 944, "stack": 256, "static": 32,
                                          "evidence": "proven_upper_bound"}},
              "builds": [{"id": ids[0], "abi": "1.0", "workspace": 344, "static": 20, "stack": 224, "input_bytes": 256,
                          "output_bytes": 5, "linked_flash": 4726},
                         {"id": ids[0], "abi": "1.1", "workspace": 600, "static": 20, "stack": 224, "input_bytes": 256,
                          "output_bytes": 5, "linked_flash": 4800},
                         {"id": ids[1], "abi": "1.0", "workspace": 944, "static": 12, "stack": 256, "input_bytes": 768,
                          "output_bytes": 4, "linked_flash": 7000}],
              "sai_verify": {"a": {"result": "passed"}, "b": {"result": "passed"}}, "not_claimed": ["no board"]}
    s = multimodel.summarize(bundle, models, "md6605", {})
    assert s["separate_ram"] == 344 + 944 + 20 + 12 + 256 + (256 + 5 + 768 + 4)
    assert s["saved_ram"] == s["separate_ram"] - 2000 and s["flash"] == 4800 + 7000 and s["fits"]
    assert s["verify"] == {"passed": 2, "total": 2, "checks": bundle["sai_verify"]}


def _client():
    from fastapi.testclient import TestClient

    from backend.app import app
    return TestClient(app)


def _login(c):
    r = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                        "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _wait(c, jid, timeout=1500):
    for _ in range(timeout):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(1)
    raise AssertionError("job timeout")


def test_multimodel_api_validates_its_input():
    with _client() as c:
        c.headers.update(_login(c))
        r = c.post("/api/multimodel", json={"job_ids": ["job_x"], "target": "md6605"})
        if not multimodel.available():
            assert r.status_code == 400 and "multi-model plugin" in r.text
            return
        assert r.status_code == 400 and "2 to 8" in r.text
        assert c.post("/api/multimodel", json={"job_ids": ["job_a", "job_b"], "target": "z80"}).status_code == 400


# ---- with SAI -----------------------------------------------------------------------------------------------------
@need_front
def test_sequence_project_end_to_end():
    with _client() as c:
        c.headers.update(_login(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "har-imu", "per_class": 25}).json()
        r = c.post("/api/projects", json={"name": "gru", "template_id": "har-imu", "dataset_id": ds["id"], "sizes": [],
                                          "epochs": 3, "targets": ["md6605", "cortex-m7"],
                                          "architectures": ["gru-s"]}).json()
        j = _wait(c, r["job"]["id"])
        assert j["status"] == "succeeded", j["error"]
        res = j["result"]
        assert res["backend"] == "xmodels" and res["deployable"] is False and "not qualified" in res["evaluation_reason"]
        assert res["selected"]["backend"] == "xmodels" and res["exact"]["value"] is not None
        assert res["verify"]["passed"] == res["verify"]["total"] >= 2
        release(c, j["id"])
        z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{j['id']}/download/package.zip").content))
        names = set(z.namelist())
        assert {"sai_package/library/sai_xm_model.c", "sai_package/library/sai_xm_model.h",
                "sai_package/sai_app_prepare.h", "sai_package/EVALUATION_ONLY.txt"} <= names
        run = c.post("/api/boards/runs", json={"job_id": j["id"]})
        assert run.status_code == 400 and "sequence models" in run.text
        # the multi-model plugin takes SAI Core projects only
        if multimodel.available():
            mm = c.post("/api/multimodel", json={"job_ids": [j["id"], j["id"] + "x"], "target": "md6605"})
            assert mm.status_code in (400, 404)


@need_front
def test_sequence_fitcheck_and_library_check():
    with _client() as c:
        c.headers.update(_login(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 20}).json()
        job = c.post("/api/models/fitcheck", json={"dataset_id": ds["id"], "layers": [{"op": "lstm", "units": 12}],
                                                   "targets": ["md6605"]}).json()
        j = _wait(c, job["id"])
        assert j["status"] == "succeeded", j["error"]
        rows = j["result"]["plans"]["candidates"]
        assert rows and rows[0]["backend"] == "xmodels" and rows[0]["fits"]
        v = _wait(c, c.post("/api/architectures/verify", json={"ids": ["gru-s"], "targets": ["md6605"]}).json()["id"])
        assert v["status"] == "succeeded" and v["result"]["fits"]["gru-s"] == ["md6605"]


@need_mm
def test_two_projects_share_one_workspace_on_md6605():
    with _client() as c:
        c.headers.update(_login(c))
        jobs = []
        for tid in ("gas-sense", "har-imu"):
            ds = c.post("/api/datasets/demo", json={"template_id": tid, "per_class": 20}).json()
            r = c.post("/api/projects", json={"name": f"{tid} monitor", "template_id": tid, "dataset_id": ds["id"],
                                              "sizes": ["tiny"], "epochs": 2, "targets": ["md6605"]}).json()
            j = _wait(c, r["job"]["id"])
            assert j["status"] == "succeeded", j["error"]
            jobs.append(j["id"])
        m = _wait(c, c.post("/api/multimodel", json={"name": "Plant", "job_ids": jobs, "target": "md6605"}).json()["id"])
        assert m["status"] == "succeeded", m["error"]
        r = m["result"]
        assert r["kind"] == "multimodel" and len(r["models"]) == 2 and r["fits"]
        assert 0 < r["device_ram"] < r["separate_ram"] and r["saved_ram"] > 0
        assert r["verify"]["passed"] == r["verify"]["total"] >= 2 and r["not_claimed"]
        pk = next(p for p in c.get("/api/packages").json()["packages"] if p["job_id"] == m["id"])
        assert pk["ram"] == r["device_ram"] and pk["approval"]["status"] == "not_requested"
        assert c.get(f"/api/jobs/{m['id']}/download/package.zip").status_code == 403
        release(c, m["id"])
        z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{m['id']}/download/package.zip").content))
        assert "sai_multimodel_package/eval-bundle.json" in z.namelist()
