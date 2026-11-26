# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Model editor: any user takes a size or library architecture as a baseline, edits it (layers and parameters),
sees it checked live on their data, checks it with SAI, and trains it."""

from __future__ import annotations

import os
import tempfile
import time
import uuid

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import pytest  # noqa: E402

from backend import archlib, sai_engine  # noqa: E402

BASE = next(a for a in archlib.BUILTIN if a["id"] == "ds-tcn-xs")["layers"]


def _client():
    from fastapi.testclient import TestClient

    from backend.app import app
    return TestClient(app)


def _h(c, email, pw):
    r = c.post("/api/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _engineer(c):
    ah = _h(c, os.environ["SAI_PLATFORM_ADMIN_EMAIL"], os.environ["SAI_PLATFORM_ADMIN_PASSWORD"])
    org = c.post("/api/admin/orgs", headers=ah, json={"name": f"Ed{uuid.uuid4().hex[:4]}", "tier": "professional"}).json()
    email = f"eng-{uuid.uuid4().hex[:6]}@editor.test"
    c.post("/api/admin/users", headers=ah, json={"email": email, "name": "Eng", "role": "engineer",
                                                 "password": "password-123", "org_id": org["id"]})
    return _h(c, email, "password-123"), ah


def _wait(c, jid, h, timeout=600):
    for _ in range(timeout):
        j = c.get(f"/api/jobs/{jid}", headers=h).json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(1)
    raise AssertionError("job timeout")


def test_preview_checks_an_edited_model_on_the_users_data():
    with _client() as c:
        h, ah = _engineer(c)            # engineers edit models too; only the research pages are closed to them
        ds = c.post("/api/datasets/demo", headers=h, json={"template_id": "gas-sense", "per_class": 20}).json()
        edited = BASE[:1] + [{"op": "dwconv", "k": 5}, {"op": "pwconv", "out": 20}, {"op": "dropout", "p": 0.2}] + BASE[1:]
        r = c.post("/api/models/preview", headers=h, json={"dataset_id": ds["id"], "layers": edited}).json()
        assert r["ok"] and r["spec"]["input"] == {"channels": 8, "height": 1, "width": 32}
        assert r["spec"]["outputs"] == len(ds["labels"]) and r["summary"]["params"] > 0 and r["receptive_field"] > 1
        assert len(r["summary"]["layers"]) == len(edited) + 1 and r["summary"]["layers"][-1]["head"]
        # the preprocessing the user chose changes what the model sees
        r2 = c.post("/api/models/preview", headers=h, json={"dataset_id": ds["id"], "layers": edited,
                                                            "preprocess": {"decimate": 2}}).json()
        assert r2["ok"] and r2["spec"]["input"]["width"] == 16
        # impossible edits come back as a message, not an error page
        bad = c.post("/api/models/preview", headers=h, json={"dataset_id": ds["id"], "layers": [{"op": "conv3d"}]}).json()
        assert not bad["ok"] and "unknown op" in bad["error"]
        bad = c.post("/api/models/preview", headers=h, json={"dataset_id": ds["id"],
                                                             "layers": [{"op": "fc", "out": 8}, {"op": "conv", "out": 4}]}).json()
        assert not bad["ok"] and "convolution after" in bad["error"]
        # another organisation's dataset is not reachable
        h2, _ = _engineer(c)
        assert c.post("/api/models/preview", headers=h2, json={"dataset_id": ds["id"], "layers": BASE}).status_code == 404


def test_projects_take_customized_models_with_their_names():
    with _client() as c:
        h, _ = _engineer(c)
        ds = c.post("/api/datasets/demo", headers=h, json={"template_id": "gas-sense", "per_class": 20}).json()
        base = {"name": "custom", "template_id": "gas-sense", "dataset_id": ds["id"], "targets": ["md6605"], "sizes": []}
        assert c.post("/api/projects", headers=h, json={**base, "custom_variants": {"bad name!": BASE}}).status_code == 400
        bad = c.post("/api/projects", headers=h, json={**base, "custom_variants": {"c_x_1": [{"op": "conv", "out": 0}]},
                                                       "custom_labels": {"c_x_1": "Broken"}})
        assert bad.status_code == 400 and "Broken" in bad.text
        r = c.post("/api/projects", headers=h, json={**base, "epochs": 2, "custom_variants": {"c_DS_TCN_XS_1": BASE},
                                                     "custom_labels": {"c_DS_TCN_XS_1": "My XS (from DS-TCN XS)"}})
        assert r.status_code == 200, r.text
        p = c.get(f"/api/jobs/{r.json()['job']['id']}", headers=h).json()["params"]
        assert p["sizes"] == ["c_DS_TCN_XS_1"] and p["architecture_names"]["c_DS_TCN_XS_1"] == "My XS (from DS-TCN XS)"
        c.post(f"/api/jobs/{r.json()['job']['id']}/cancel", headers=h)


def test_anomaly_custom_models_are_checked_with_the_real_output_size():
    with _client() as c:
        h, _ = _engineer(c)
        ds = c.post("/api/datasets/demo", headers=h, json={"template_id": "vibration-anomaly", "per_class": 30}).json()
        ae = next(a for a in archlib.BUILTIN if a["id"] == "ae-s")["layers"]
        r = c.post("/api/models/preview", headers=h, json={"dataset_id": ds["id"], "layers": ae}).json()
        assert r["ok"] and r["spec"]["outputs"] == 64 and r["task"] == "anomaly"


def test_fitcheck_is_open_to_engineers():
    with _client() as c:
        h, _ = _engineer(c)
        ds = c.post("/api/datasets/demo", headers=h, json={"template_id": "gas-sense", "per_class": 20}).json()
        assert c.post("/api/models/fitcheck", headers=h, json={"dataset_id": ds["id"], "layers": [{"op": "conv3d"}]}).status_code == 400
        j = c.post("/api/models/fitcheck", headers=h, json={"dataset_id": ds["id"], "layers": BASE, "targets": ["md6605"]})
        assert j.status_code == 200 and j.json()["kind"] == "fitcheck"
        if not sai_engine.available():
            c.post(f"/api/jobs/{j.json()['id']}/cancel", headers=h)
            return
        r = _wait(c, j.json()["id"], h)
        assert r["status"] == "succeeded", r["error"]
        assert r["result"]["spec"]["input"]["channels"] == 8 and r["result"]["plans"]["recommended"]["target"] == "md6605"


@pytest.mark.skipif(not sai_engine.available(), reason="no SAI compiler (set SAI_REPO)")
def test_train_a_customized_baseline_end_to_end():
    with _client() as c:
        h, _ = _engineer(c)
        ds = c.post("/api/datasets/demo", headers=h, json={"template_id": "gas-sense", "per_class": 30}).json()
        edited = [{"op": "conv", "out": 12, "k": 3, "stride": 2}, {"op": "dwconv", "k": 3},
                  {"op": "pwconv", "out": 16}, {"op": "gap"}]
        r = c.post("/api/projects", headers=h, json={"name": "edited", "template_id": "gas-sense", "dataset_id": ds["id"],
                                                     "targets": ["md6605"], "sizes": [], "epochs": 3,
                                                     "custom_variants": {"c_my_net_1": edited},
                                                     "custom_labels": {"c_my_net_1": "My net"}}).json()
        t = _wait(c, r["job"]["id"], h, timeout=900)
        assert t["status"] == "succeeded", t["error"]
        res = t["result"]
        assert res["selected"]["size"] == "c_my_net_1" and res["verify"]["passed"] == res["verify"]["total"]
        v = next(x for x in res["variants"] if x["size"] == "c_my_net_1")
        assert v["summary"]["params"] == c.post("/api/models/preview", headers=h, json={
            "dataset_id": ds["id"], "layers": edited}).json()["summary"]["params"]
