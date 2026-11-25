"""Contract tests: the exact SAI CLI JSON fields the platform reads. Run against every SAI release (CI full job);
a failure here means SAI changed a report the platform depends on."""

from __future__ import annotations

import os
import shutil
import tempfile

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import archspec, qexport, sai_engine  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which(os.environ.get("SAI_BIN", "sai")) is None, reason="sai CLI not installed")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("contract")
    out = {}
    for name, inp, layers in [
        ("m1d", {"channels": 2, "width": 64}, [{"op": "conv", "out": 8, "k": 5}, {"op": "dwconv"}, {"op": "pwconv", "out": 8},
                                               {"op": "gap"}]),
        ("m2d", {"channels": 1, "height": 16, "width": 16}, [{"op": "conv", "out": 8, "k": 3, "stride": 2}, {"op": "dwconv"},
                                                            {"op": "pwconv", "out": 12}, {"op": "gap"}]),
    ]:
        spec = archspec.normalize({"input": inp, "layers": layers, "outputs": 3})
        net = archspec.SpecNet(spec).eval()
        net.fold_bn()
        x = np.random.default_rng(0).normal(size=(32, inp["channels"], inp.get("height", 1), inp["width"])).astype(np.float32)
        qexport.export_qdq(net, x, d / f"{name}.onnx")
        out[name] = d / f"{name}.onnx"
    return out


def test_version_and_targets():
    v = sai_engine.version()
    assert {"sai", "abi"} <= set(v)
    ids = {t["id"] for t in sai_engine.run(["targets"])["targets"]}
    assert {"host-cpu", "cortex-m7", "md6605"} <= ids


@pytest.mark.parametrize("target", ["cortex-m7", "md6605"])
def test_estimate_fields(models, target):
    e = sai_engine.estimate(models["m1d"], target, "fast")
    assert e["schema_version"] in {"1.0"}
    for k in ("tensor_workspace", "linked_sai_flash", "stack", "static_ram"):
        assert {"value", "evidence"} <= set(e["resources"][k])
    assert {"schedule", "tier", "program_id"} <= set(e["provenance"]["plan"])
    assert "latency_cycles" in e["metrics"] and "plan_digest" in e
    assert sai_engine.estimate(models["m1d"], target, "fast", tier="core")["provenance"]["plan"]["tier"] == "core"


def test_explore_fields(models):
    ex = sai_engine.explore(models["m2d"], "cortex-m7", 2, max_seconds=20)
    assert "ranking" in ex and ex["ranking"]
    r = ex["ranking"][0]
    assert {"name", "candidate_id"} <= set(r)
    assert "value" in r["workspace"] and {"value", "basis"} <= set(r["rank_score"])
    assert {"evaluated", "complete"} <= set(ex["search"])


def test_compile_plan_verify_certify(models, tmp_path):
    ex = sai_engine.explore(models["m2d"], "cortex-m7", 2, max_seconds=20)
    r = ex["ranking"][-1]
    sai_engine.compile_build(models["m2d"], "cortex-m7", "fast", tmp_path / "b", r["name"], r["candidate_id"])
    for f in ("manifest.json", "estimate-report.json", "memory-certificate.json", "compile-report.json",
              "model/sai_model.c", "model/sai_model.h", "lib/libsai_model.a"):
        assert (tmp_path / "b" / f).exists(), f
    v = sai_engine.verify(tmp_path / "b", models["m2d"])
    assert v["checks"] and all({"check", "passed"} <= set(c) for c in v["checks"])
    c = sai_engine.certify_check(tmp_path / "b", models["m2d"])
    assert c["total"] > 0 and c["passed"] == c["total"]


def test_inspect_fields(models):
    i = sai_engine.inspect(models["m1d"])
    assert {"format", "nodes", "parameters", "operators", "inputs"} <= set(i)


def test_explored_plans_become_candidates(models):
    res = sai_engine.plan_search(models["m2d"], 10000, 2, ["cortex-m7"], "min-ram", explore_builds=2)
    explored = [c for c in res["candidates"] if c["objective"] == "explored"]
    assert explored, "no explored plan was built"
    e = explored[0]
    assert e["plan_name"] and e["candidate_id"] and e["flash_evidence"] == "linked_exact"
    assert e["stack_evidence"] == "proven_upper_bound"


def test_measured_evidence_from_board_runs(models):
    base = sai_engine.plan_search(models["m1d"], 5000, 1, ["cortex-m7"], "fastest", explore_builds=0)
    c = base["candidates"][0]
    meas = {f"cortex-m7:{c['plan_digest']}": {"cycles": 12345, "runs": 3, "board": "LAB-H753-01"}}
    again = sai_engine.plan_search(models["m1d"], 5000, 1, ["cortex-m7"], "fastest", explore_builds=0, measurements=meas)
    hit = next(x for x in again["candidates"] if x["plan_digest"] == c["plan_digest"])
    assert hit["latency"]["evidence"] == "measured" and hit["latency"]["cycles"] == 12345
