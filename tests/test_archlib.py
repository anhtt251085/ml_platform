# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Architecture library: built-ins, compatibility with real data, custom entries and their permissions, projects
that train library architectures, and (with SAI) verification on every board."""

from __future__ import annotations

import os
import tempfile
import time
import uuid

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import pytest  # noqa: E402

from backend import archlib, archspec, sai_engine  # noqa: E402


def test_builtins_are_valid_and_cover_every_input_kind():
    ids = [a["id"] for a in archlib.BUILTIN]
    assert len(ids) == len(set(ids)) >= 15
    for a in archlib.BUILTIN:
        assert set(a["inputs"]) <= set(archlib.INPUT_KINDS) and set(a["tasks"]) <= set(archlib.TASKS)
        assert all(layer["op"] in archspec.OPS for layer in a["layers"])
        for kind in a["inputs"]:
            archspec.normalize({"input": archlib.REFERENCE[kind], "layers": a["layers"], "outputs": 4})
    for kind in archlib.INPUT_KINDS:
        assert any(kind in a["inputs"] for a in archlib.BUILTIN)
    assert any(a["tasks"] == ["anomaly"] for a in archlib.BUILTIN)


def test_compatibility_is_decided_on_the_real_input():
    get = {a["id"]: a for a in archlib.BUILTIN}
    ok = archlib.compatible(get["ds-tcn-m"], (3, 1, 256), "classification", 4)
    assert ok["ok"] and ok["params"] > 0 and ok["receptive_field"] > 1
    assert "2d" in archlib.compatible(get["ds-tcn-m"], (1, 49, 10), "classification", 4)["reason"]
    assert archlib.compatible(get["ds-cnn-2d-s"], (1, 49, 10), "classification", 12)["ok"]
    assert "anomaly" in archlib.compatible(get["ds-tcn-s"], (1, 1, 64), "anomaly", 64)["reason"]
    assert archlib.compatible(get["ae-s"], (1, 1, 64), "anomaly", 64)["ok"]
    assert not archlib.compatible(get["ae-s"], (1, 1, 64), "classification", 4)["ok"]
    # too short for the wide 31-tap stride-4 stem plus three poolings
    assert not archlib.compatible(get["wide-kernel-cnn"], (1, 1, 16), "classification", 2)["ok"]


@pytest.mark.parametrize("body, msg", [
    ({"name": "x", "inputs": ["timeseries"], "tasks": ["classification"], "layers": [{"op": "gap"}]}, "name"),
    ({"name": "Bad", "inputs": [], "tasks": ["classification"], "layers": [{"op": "gap"}]}, "input kind"),
    ({"name": "Bad", "inputs": ["timeseries"], "tasks": ["classification"], "layers": [{"op": "conv3d"}]}, "unknown op"),
    ({"name": "Bad", "inputs": ["timeseries"], "tasks": ["classification"], "layers": []}, "layers")])
def test_custom_entries_are_validated(body, msg):
    with pytest.raises(archlib.LibError, match=msg):
        archlib.validate(body)


def _client():
    from fastapi.testclient import TestClient

    from backend.app import app
    return TestClient(app)


def _h(c, email, pw):
    r = c.post("/api/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _admin(c):
    return _h(c, os.environ["SAI_PLATFORM_ADMIN_EMAIL"], os.environ["SAI_PLATFORM_ADMIN_PASSWORD"])


def _user(c, ah, role, org_id=None):
    email = f"{role}-{uuid.uuid4().hex[:6]}@lib.test"
    body = {"email": email, "name": role, "role": role, "password": "password-123"}
    if org_id:
        body["org_id"] = org_id
    assert c.post("/api/admin/users", headers=ah, json=body).status_code == 200
    return _h(c, email, "password-123")


NEW = {"name": "Tiny stem net", "family": "Custom", "inputs": ["timeseries"], "tasks": ["classification"],
       "layers": [{"op": "conv", "out": 6, "k": 5, "stride": 4}, {"op": "gap"}], "summary": "test"}


def test_library_api_permissions_and_scopes():
    with _client() as c:
        ah = _admin(c)
        a = c.post("/api/admin/orgs", headers=ah, json={"name": "LibA", "tier": "professional"}).json()
        b = c.post("/api/admin/orgs", headers=ah, json={"name": "LibB", "tier": "professional"}).json()
        eng, res_a, res_b = _user(c, ah, "engineer", a["id"]), _user(c, ah, "researcher", a["id"]), _user(c, ah, "researcher", b["id"])
        lib = c.get("/api/architectures", headers=eng).json()
        assert {x["id"] for x in lib["architectures"]} >= {x["id"] for x in archlib.BUILTIN} and not lib["can_add"]
        assert c.post("/api/architectures", headers=eng, json=NEW).status_code == 403
        # a customer researcher adds a private one; the other organisation does not see it
        assert c.post("/api/architectures", headers=res_a, json={**NEW, "scope": "global"}).status_code == 403
        mine = c.post("/api/architectures", headers=res_a, json=NEW).json()
        assert mine["org_id"] == a["id"] and mine["layers"][0]["stride"] == 4
        assert mine["id"] in {x["id"] for x in c.get("/api/architectures", headers=eng).json()["architectures"]}
        assert mine["id"] not in {x["id"] for x in c.get("/api/architectures", headers=res_b).json()["architectures"]}
        assert c.put(f"/api/architectures/{mine['id']}", headers=res_b, json=NEW).status_code == 404
        assert c.delete("/api/architectures/ds-tcn-s", headers=ah).status_code == 403            # built-in
        # SAI admins add global ones that everybody sees
        glob = c.post("/api/architectures", headers=ah, json={**NEW, "name": "Global stem", "scope": "global"}).json()
        assert glob["org_id"] is None
        assert glob["id"] in {x["id"] for x in c.get("/api/architectures", headers=res_b).json()["architectures"]}
        upd = c.put(f"/api/architectures/{mine['id']}", headers=res_a, json={**NEW, "name": "Renamed net"}).json()
        assert upd["name"] == "Renamed net"
        assert c.delete(f"/api/architectures/{mine['id']}", headers=res_a).json()["ok"]
        # verifying the whole library is for SAI researchers and admins
        assert c.post("/api/architectures/verify", headers=res_a, json={}).status_code == 403


def test_projects_train_library_architectures():
    with _client() as c:
        c.headers.update(_admin(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 20}).json()
        fits = {x["id"]: x["fit"] for x in c.get(f"/api/architectures?dataset_id={ds['id']}").json()["architectures"]}
        assert fits["ds-tcn-xs"]["ok"] and not fits["ds-cnn-2d-s"]["ok"]
        base = {"name": "lib", "template_id": "gas-sense", "dataset_id": ds["id"], "targets": ["md6605"], "sizes": []}
        assert c.post("/api/projects", json={**base, "architectures": ["nope"]}).status_code == 400
        bad = c.post("/api/projects", json={**base, "architectures": ["ds-cnn-2d-s"]})
        assert bad.status_code == 400 and "does not suit" in bad.text
        r = c.post("/api/projects", json={**base, "architectures": ["ds-tcn-xs"], "epochs": 2})
        assert r.status_code == 200, r.text
        jid = r.json()["job"]["id"]
        p = c.get(f"/api/jobs/{jid}").json()["params"]
        assert p["sizes"] == ["ds-tcn-xs"] and p["custom_variants"]["ds-tcn-xs"][0]["op"] == "conv"
        assert p["architecture_names"]["ds-tcn-xs"] == "DS-TCN XS"
        c.post(f"/api/jobs/{jid}/cancel")


def _wait(c, jid, timeout=900):
    for _ in range(timeout):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(1)
    raise AssertionError("job timeout")


@pytest.mark.skipif(not sai_engine.available(), reason="no SAI compiler (set SAI_REPO)")
def test_sai_verifies_library_and_trains_a_library_architecture():
    with _client() as c:
        c.headers.update(_admin(c))
        j = _wait(c, c.post("/api/architectures/verify", json={"ids": ["ds-tcn-xs", "ae-s", "ds-cnn-2d-s"]}).json()["id"])
        assert j["status"] == "succeeded" and not j["result"]["failed"], j
        assert all(set(v) == {"md6605", "cortex-m7"} for v in j["result"]["fits"].values())
        lib = {x["id"]: x for x in c.get("/api/architectures").json()["architectures"]}
        chk = lib["ds-tcn-xs"]["checks"]["md6605"]
        assert chk["fits"] and chk["ram"] > 0 and chk["flash"] > 0 and chk["sai_version"]
        ds = c.post("/api/datasets/demo", json={"template_id": "gas-sense", "per_class": 30}).json()
        t = _wait(c, c.post("/api/projects", json={"name": "lib", "template_id": "gas-sense", "dataset_id": ds["id"],
                                                   "targets": ["md6605"], "sizes": [], "epochs": 3,
                                                   "architectures": ["ds-tcn-xs"]}).json()["job"]["id"])
        assert t["status"] == "succeeded", t["error"]
        assert t["result"]["selected"]["size"] == "ds-tcn-xs" and t["result"]["verify"]["passed"] == t["result"]["verify"]["total"]
        rec = _wait(c, c.post("/api/recommend", json={"template_id": "gas-sense", "dataset_id": ds["id"],
                                                      "targets": ["md6605"], "max_candidates": 2}).json()["id"])
        assert rec["status"] == "succeeded" and any(x["origin"] == "library" for x in rec["result"]["candidates"])
