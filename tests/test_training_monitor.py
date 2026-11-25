# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Training settings (validation and effect) and monitoring (per-epoch metrics, log download, jobs list)."""

from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import trainer  # noqa: E402


@pytest.mark.parametrize("bad, msg", [({"optimizer": "lion"}, "optimizer"), ({"lr": 5}, "lr"), ({"lrr": 1e-3}, "unknown"),
                                      ({"batch_size": "x"}, "batch_size"), ({"schedule": "poly"}, "schedule"),
                                      ({"qat": "sometimes"}, "qat"), ({"label_smoothing": 0.9}, "label_smoothing")])
def test_training_settings_are_validated(bad, msg):
    with pytest.raises(trainer.TrainCfgError, match=msg):
        trainer.normalize_training(bad)


def test_blank_settings_mean_automatic():
    t = trainer.normalize_training({"lr": "", "batch_size": None}, 12)
    assert t["lr"] is None and t["batch_size"] is None and t["epochs"] == 12 and t["optimizer"] == "adam"


def _toy(n=120, w=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, 1, w)).astype(np.float32)
    y = (X[:, 0, 0, :].mean(1) > 0).astype(int)
    idx = {"train": np.arange(0, 80), "val": np.arange(80, 100), "test": np.arange(100, n)}
    spec = trainer.build_spec([{"op": "conv", "out": 4, "k": 3}, {"op": "gap"}], X, "classification", 2)
    return X, y, idx, spec


@pytest.mark.parametrize("opt", ["adam", "adamw", "sgd"])
def test_settings_take_effect_and_every_epoch_is_reported(opt):
    X, y, idx, spec = _toy()
    cfg = trainer.normalize_training({"optimizer": opt, "lr": 0.01, "schedule": "step", "batch_size": 16,
                                      "patience": 0, "extend_small_data": False, "label_smoothing": 0.1}, 6)
    pts, lines = [], []
    net, hist = trainer.train(spec, X, y, idx, "classification", cfg=cfg, metric=pts.append, log=lines.append)
    assert len(hist) == len(pts) == 6                         # patience 0 and no extension: exactly 6 epochs
    assert pts[0]["lr"] == pytest.approx(0.01) and pts[-1]["lr"] == pytest.approx(0.01 * 0.3 ** 2)   # step x0.3 / 2 ep
    assert all(0 <= p["val_acc"] <= 1 and p["seconds"] >= 0 and p["epochs"] == 6 for p in pts)
    assert any(f"training: {opt} lr 0.01 (step), batch 16, 6 epochs" in x for x in lines)


def test_seed_changes_training_and_is_reproducible():
    X, y, idx, spec = _toy()
    run = lambda seed: trainer.train(spec, X, y, idx, "classification",  # noqa: E731
                                     cfg=trainer.normalize_training({"seed": seed, "extend_small_data": False}, 3))[1]
    assert run(1) == run(1) and run(1) != run(2)


def test_small_data_extension_can_be_turned_off():
    X, y, idx, spec = _toy()
    _, auto = trainer.train(spec, X, y, idx, "classification", cfg=trainer.normalize_training({"patience": 0}, 3))
    _, off = trainer.train(spec, X, y, idx, "classification",
                           cfg=trainer.normalize_training({"patience": 0, "extend_small_data": False}, 3))
    assert len(off) == 3 and len(auto) > 3


def _client():
    from fastapi.testclient import TestClient

    from backend.app import app
    return TestClient(app)


def _login(c):
    r = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                        "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_project_training_settings_metrics_and_logs():
    with _client() as c:
        c.headers.update(_login(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 20}).json()
        base = {"name": "mon", "template_id": "gas-sense", "dataset_id": ds["id"], "targets": ["md6605"], "sizes": ["tiny"]}
        bad = c.post("/api/projects", json={**base, "training": {"optimizer": "lion"}})
        assert bad.status_code == 400 and "optimizer" in bad.text
        r = c.post("/api/projects", json={**base, "epochs": 4, "training": {"optimizer": "sgd", "lr": 0.02,
                                                                            "extend_small_data": False, "patience": 0}})
        assert r.status_code == 200, r.text
        jid = r.json()["job"]["id"]
        seen = []
        for _ in range(300):        # poll like the UI: only new metrics each time
            j = c.get(f"/api/jobs/{jid}?metrics_after={seen[-1]['seq'] if seen else 0}").json()
            seen += j["metrics"]
            if j["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.5)
        train = [m for m in seen if m["series"] == "train:tiny"]
        assert [m["epoch"] for m in train] == [1, 2, 3, 4]       # every epoch exactly once
        assert train[0]["lr"] == pytest.approx(0.02) and "val_acc" in train[0]
        txt = c.get(f"/api/jobs/{jid}/log.txt")
        assert txt.status_code == 200 and "attachment" in txt.headers["content-disposition"]
        assert "training: sgd lr 0.02 (cosine), batch" in txt.text and txt.text.split("\n")[0].endswith("== Validate data")
        row = next(x for x in c.get("/api/jobs").json() if x["id"] == jid)
        assert row["last_log"] and "warnings" in row
        if j["status"] == "succeeded":
            assert j["result"]["training"]["optimizer"] == "sgd"
        # another org's user cannot read the log
        o = c.post("/api/admin/orgs", json={"name": "Other", "tier": "free"}).json()
        c.post("/api/admin/users", json={"email": "o@other.test", "name": "O", "role": "member", "org_id": o["id"],
                                         "password": "other-pass-123"})
        r2 = c.post("/api/auth/login", json={"email": "o@other.test", "password": "other-pass-123"})
        h = {"Authorization": f"Bearer {r2.json()['token']}"}
        assert c.get(f"/api/jobs/{jid}/log.txt", headers=h).status_code == 404
