# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Architecture recommender: data profile, rules, candidate generation, goal-based pick, API, and (with SAI) the
recommend job feeding a training run."""

from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import archspec, catalog, recommend, sai_engine  # noqa: E402

TPL = {t["id"]: t for t in catalog.BUILTIN}


def _sines(n=200, c=2, w=256, cycles=(4, 16), noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(w) / w
    y = rng.integers(0, 2, n)
    X = np.stack([np.stack([np.sin(2 * np.pi * cycles[k] * t + rng.uniform(0, 6.3)) for _ in range(c)])
                  for k in y]) + noise * rng.normal(size=(n, c, w))
    return X[:, :, None, :].astype(np.float32), y


def test_profile_finds_the_signal_and_rules_follow_it():
    X, y = _sines()
    p = recommend.profile(X, y, "classification", ["a", "b"], sample_rate_hz=1000)
    assert p["kind"] == "timeseries" and p["classes"] == 2 and p["n_train"] == 140
    sp = p["spectrum"]
    assert sp["peak_cycles_per_window"] in (4, 16) and sp["low10_cycles_per_window"] <= 5 and sp["flatness"] < 0.5
    rl = recommend.rules(p, conv=True)
    assert rl["param_cap"] == 3500                     # 25 per training window
    # slowest significant component: 4 cycles per 256 samples -> period 64 -> two cycles = 128 samples
    assert 100 <= rl["rf_target"] <= 256
    # strongest component 64 or 16 samples per cycle -> stride keeps >= 4 samples per cycle
    assert rl["max_stride"] == min(4, (256 // sp["peak_cycles_per_window"]) // 4)
    assert any("receptive field" in r for r in rl["reasons"])


def test_broadband_noise_gets_short_receptive_field():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 1, 1, 256)).astype(np.float32)
    p = recommend.profile(X, np.zeros(120, int), "classification", ["a", "b"], None)
    rl = recommend.rules(p, conv=True)
    assert p["spectrum"]["flatness"] > 0.5 and rl["rf_target"] == 32


def test_receptive_field():
    layers = [{"op": "conv", "k": 5, "stride": 2}, {"op": "dwconv", "k": 3}, {"op": "maxpool", "k": 2},
              {"op": "dwconv", "k": 3}, {"op": "gap"}]
    # 1 + 4 = 5 (jump 2); + 2*2 = 9; pool + 1*2 = 11 (jump 4); + 2*4 = 19
    assert recommend.receptive_field(layers) == 19


@pytest.mark.parametrize("tid", ["bearing-fault", "kws", "thermal-presence", "vibration-anomaly", "gas-sense"])
def test_candidates_are_valid_and_respect_the_rules(tid):
    tpl = TPL[tid]
    i = tpl["input"]
    n = 400
    X = np.random.default_rng(0).normal(size=(n, i["channels"], i.get("height", 1), i["width"])).astype(np.float32)
    y = np.arange(n) % len(tpl["labels"])
    task = tpl["task"]
    n_out = len(tpl["labels"]) if task == "classification" else 1 if task == "regression" else int(X[0].size)
    p = recommend.profile(X, y, task, tpl["labels"], i.get("sample_rate_hz"))
    conv = tid != "vibration-anomaly"
    rl = recommend.rules(p, conv)
    cands = recommend.candidates(p, rl, tpl, n_out, max_candidates=6)
    gen = [c for c in cands if c["origin"] == "generated"]
    assert 1 <= len(gen) <= 6 and any(c["origin"] == "pool" for c in cands)
    for c in gen:
        spec = archspec.normalize({"input": {"channels": i["channels"], "height": i.get("height", 1), "width": i["width"]},
                                   "layers": c["layers"], "outputs": n_out})
        assert archspec.summary(spec)["params"] == c["params"] <= rl["param_cap"]
        if conv:
            assert c["layers"][0]["stride"] <= rl["max_stride"]
    if conv and any(c["receptive_field"] >= rl["rf_target"] for c in gen):
        assert all(c["receptive_field"] >= rl["rf_target"] for c in gen)


def _c(cid, params, ram, flash, ms, fits=True, over=False):
    return {"id": cid, "params": params, "ram": ram, "flash": flash, "latency_ms": ms, "fits": fits, "over_cap": over}


def test_pick_follows_the_goal():
    rl = {"param_cap": 5000, "param_floor": 500}
    cands = [_c("tiny", 300, 500, 2000, 0.1),             # below the capacity floor
             _c("lean", 800, 900, 4000, 0.5), _c("fast", 1500, 1500, 5000, 0.2),
             _c("big", 4500, 2000, 9000, 1.5), _c("huge", 9000, 1200, 3000, 0.1, over=True),
             _c("nofit", 3000, 100, 100, 0.01, fits=False)]
    assert recommend.pick(cands, rl, "min-ram")[0]["id"] == "lean"
    assert recommend.pick(cands, rl, "fastest")[0]["id"] == "fast"
    assert recommend.pick(cands, rl, "min-flash")[0]["id"] == "lean"
    assert recommend.pick(cands, rl, "accuracy")[0]["id"] == "big"
    assert recommend.pick(cands, rl, "balanced")[0]["id"] in ("lean", "fast", "big")
    best, why = recommend.pick([_c("x", 900, 1, 1, 1, fits=False)], rl, "balanced")
    assert best is None and "No candidate fits" in why


def _client():
    from fastapi.testclient import TestClient

    from backend.app import app
    return TestClient(app)


def _login(c):
    r = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                        "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _wait(c, jid, timeout=600):
    for _ in range(timeout):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(1)
    raise AssertionError("job timeout")


def test_api_validates_recommend_and_custom_variants():
    with _client() as c:
        c.headers.update(_login(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 20}).json()
        assert c.post("/api/recommend", json={"template_id": "gas-sense", "dataset_id": ds["id"],
                                              "goal": "nonsense"}).status_code == 400
        bad = c.post("/api/projects", json={"name": "x", "template_id": "gas-sense", "dataset_id": ds["id"],
                                            "sizes": ["auto"], "custom_variants": {"auto": [{"op": "lstm"}]}})
        assert bad.status_code == 400 and "unknown op" in bad.text
        bad = c.post("/api/projects", json={"name": "x", "template_id": "gas-sense", "dataset_id": ds["id"],
                                            "sizes": ["auto"], "custom_variants": {"../x": [{"op": "gap"}]}})
        assert bad.status_code == 400


@pytest.mark.skipif(not sai_engine.available(), reason="no SAI compiler (set SAI_REPO)")
def test_recommend_job_then_train_the_recommended_architecture():
    with _client() as c:
        c.headers.update(_login(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 40}).json()
        job = c.post("/api/recommend", json={"template_id": "gas-sense", "dataset_id": ds["id"], "targets": ["md6605"],
                                             "goal": "min-ram", "max_candidates": 2}).json()
        j = _wait(c, job["id"])
        assert j["status"] == "succeeded", j["error"]
        r = j["result"]
        rec = r["recommended"]
        assert rec and rec["boards"]["md6605"]["fits"] and rec["params"] >= r["rules"]["param_floor"]
        assert sum(x["recommended"] for x in r["candidates"]) == 1
        from backend import config
        assert not list((config.DATA_DIR / "jobs" / job["id"]).glob("rec_*.onnx"))      # no leftovers
        p = c.post("/api/projects", json={"name": "auto", "template_id": "gas-sense", "dataset_id": ds["id"],
                                          "targets": ["md6605"], "goal": "min-ram", "sizes": ["auto"], "epochs": 3,
                                          "custom_variants": {"auto": rec["layers"]}}).json()
        t = _wait(c, p["job"]["id"])
        assert t["status"] == "succeeded", t["error"]
        res = t["result"]
        assert res["selected"]["size"] == "auto"
        v = next(x for x in res["variants"] if x["size"] == "auto")
        assert v["summary"]["params"] == rec["params"]
        # the trained model's plan costs no more than the recommender predicted from random weights (training also
        # builds explored plans, which can only add cheaper options)
        assert res["selected"]["ram"] <= rec["boards"]["md6605"]["ram"]
