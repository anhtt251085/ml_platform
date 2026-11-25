# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Four roles (engineer, researcher, manager, admin) and the manager approval of compiled-package downloads."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import pytest  # noqa: E402

from backend import auth, db  # noqa: E402


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


def _org(c, ah, name):
    """A customer organisation with one user of every role; returns {role: headers} and the org."""
    org = c.post("/api/admin/orgs", headers=ah, json={"name": name, "tier": "professional"}).json()
    hs = {}
    for role in auth.ROLES:
        email = f"{role}-{uuid.uuid4().hex[:6]}@{name.lower()}.test"
        r = c.post("/api/admin/users", headers=ah, json={"email": email, "name": role.title(), "role": role,
                                                         "password": "password-123", "org_id": org["id"]})
        assert r.status_code == 200, r.text
        hs[role] = _h(c, email, "password-123")
    return org, hs


def _finished_project(org_id: str, name="Motor monitor") -> str:
    """A succeeded project with a compiled package on disk (what the pipeline leaves behind)."""
    from backend import config
    jid = "job_" + uuid.uuid4().hex[:10]
    result = {"selected": {"target": "md6605", "schedule": "row-stream-t16", "size": "tiny", "ram": 572, "flash": 3720,
                           "latency": {"ms": 0.98}}, "exact": {"headline": "accuracy", "value": 0.97},
              "signature": {"key_id": "cb7e86d1"}, "deployable": True}
    with db.conn() as c:
        c.execute("INSERT INTO jobs (id, kind, project_id, status, stage, progress, stages, result, error, params,"
                  " created, updated, org_id, user_id, priority, attempts, cancel_requested)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)",
                  (jid, "train", None, "succeeded", "package", 1.0, "[]", db.j(result), None,
                   db.j({"name": name, "template_id": "motor-current"}), db.now(), db.now(), org_id, None, 0))
    d = config.DATA_DIR / "jobs" / jid
    d.mkdir(parents=True, exist_ok=True)
    (d / "package.zip").write_bytes(b"PK-compiled-sources")
    (d / "model.onnx").write_bytes(b"onnx")
    (d / "report.json").write_text("{}")
    (d / "MODEL_CARD.md").write_text("# card")
    return jid


def test_role_permissions():
    with _client() as c:
        ah = _admin(c)
        org, hs = _org(c, ah, "Roles")
        perms = {role: c.get("/api/me", headers=h).json()["perms"] for role, h in hs.items()}
        assert [r for r in auth.ROLES if perms[r]["research"]] == ["researcher", "manager", "admin"]
        assert [r for r in auth.ROLES if perms[r]["approve"]] == ["manager", "admin"]
        assert [r for r in auth.ROLES if perms[r]["org_admin"]] == ["admin"]
        # platform powers stay with the SAI organisation's admins
        assert not any(perms[r]["platform_admin"] or perms[r]["farm_admin"] or perms[r]["studio"] for r in auth.ROLES)
        assert c.get("/api/me", headers=ah).json()["perms"]["platform_admin"]
        spec = {"input": {"channels": 1, "width": 64}, "layers": [{"op": "gap"}], "outputs": 2}
        # engineers cannot use research pages
        assert c.post("/api/studio/validate", headers=hs["engineer"], json=spec).status_code == 403
        assert c.post("/api/studio/fitcheck", headers=hs["engineer"], json={"spec": spec}).status_code == 403
        assert c.post("/api/explorer/analyze", headers=hs["engineer"],
                      files={"file": ("m.onnx", b"x")}).status_code == 403
        for role in ("researcher", "manager", "admin"):
            assert c.post("/api/studio/validate", headers=hs[role], json=spec).json()["ok"]
        # only admins open the Admin console; a customer admin sees only their organisation
        for role in ("engineer", "researcher", "manager"):
            assert c.get("/api/admin/users", headers=hs[role]).status_code == 403
            assert c.get("/api/admin/audit", headers=hs[role]).status_code == 403
        users = c.get("/api/admin/users", headers=hs["admin"]).json()
        assert {u["org_id"] for u in users} == {org["id"]}
        assert c.get("/api/admin/queue", headers=hs["admin"]).status_code == 403          # platform only
        # a customer admin can add any of the four roles in their own organisation
        r = c.post("/api/admin/users", headers=hs["admin"], json={"email": "new@roles.test", "name": "N",
                                                                  "role": "manager", "password": "password-123"})
        assert r.status_code == 200 and r.json()["role"] == "manager" and r.json()["org_id"] == org["id"]
        bad = c.post("/api/admin/users", headers=hs["admin"], json={"email": "x@roles.test", "name": "X",
                                                                    "role": "superuser", "password": "password-123"})
        assert bad.status_code == 400


def test_legacy_role_names_are_mapped():
    assert auth.norm_role("member") == "engineer" and auth.norm_role("lab") == "engineer"
    assert auth.norm_role("org_admin") == "admin" and auth.norm_role("platform_admin") == "admin"
    with pytest.raises(ValueError):
        auth.norm_role("owner")
    # the migration rewrites stored roles
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE users (id TEXT, role TEXT)")
    con.executemany("INSERT INTO users VALUES (?,?)", [(r, r) for r in ("platform_admin", "org_admin", "member",
                                                                       "lab", "researcher")])
    for stmt in db._statements(db.MIGRATIONS[6]):
        con.execute(stmt)
    assert dict(con.execute("SELECT id, role FROM users")) == {
        "platform_admin": "admin", "org_admin": "admin", "member": "engineer", "lab": "engineer",
        "researcher": "researcher"}


def test_package_download_needs_a_managers_approval():
    with _client() as c:
        ah = _admin(c)
        org, hs = _org(c, ah, "Acme")
        other, ho = _org(c, ah, "Globex")
        jid = _finished_project(org["id"])
        eng, mgr = hs["engineer"], hs["manager"]
        dl = lambda h: c.get(f"/api/jobs/{jid}/download/package.zip", headers=h)  # noqa: E731

        r = dl(eng)
        assert r.status_code == 403 and "approval" in r.text
        for name in ("model.onnx", "report.json"):              # the model and the report need the approval too
            assert c.get(f"/api/jobs/{jid}/download/{name}", headers=eng).status_code == 403
        row = next(p for p in c.get("/api/packages", headers=eng).json()["packages"] if p["job_id"] == jid)
        assert row["approval"]["status"] == "not_requested" and row["ram"] == 572 and row["plan"] == "row-stream-t16"

        r = c.post(f"/api/packages/{jid}/request", headers=eng, json={"note": "release 1.2 for the field trial"})
        assert r.status_code == 200 and r.json()["status"] == "pending"
        assert c.post(f"/api/packages/{jid}/request", headers=eng, json={}).status_code == 409
        assert "waiting" in dl(eng).text
        # engineers and researchers cannot approve; other organisations cannot see it
        for role in ("engineer", "researcher"):
            assert c.post(f"/api/packages/{jid}/approve", headers=hs[role], json={}).status_code == 403
        assert c.post(f"/api/packages/{jid}/approve", headers=ho["manager"], json={}).status_code == 404
        assert jid not in {p["job_id"] for p in c.get("/api/packages", headers=ho["manager"]).json()["packages"]}
        assert c.get("/api/packages?status=pending", headers=mgr).json()["packages"][0]["job_id"] == jid
        # a rejection needs a reason
        assert c.post(f"/api/packages/{jid}/reject", headers=mgr, json={"note": " "}).status_code == 400
        r = c.post(f"/api/packages/{jid}/approve", headers=mgr, json={"note": "trial approved"})
        assert r.status_code == 200 and r.json()["status"] == "approved" and r.json()["decided_by_name"] == "Manager"
        assert dl(eng).status_code == 200 and dl(eng).content == b"PK-compiled-sources"
        for name in ("model.onnx", "report.json"):
            assert c.get(f"/api/jobs/{jid}/download/{name}", headers=eng).status_code == 200
        assert c.get(f"/api/jobs/{jid}/download/MODEL_CARD.md", headers=eng).status_code == 200   # documentation
        assert c.get(f"/api/packages/{jid}", headers=eng).json()["downloads"] == 4

        # revoke (reason required) stops downloads; a new request starts over
        assert c.post(f"/api/packages/{jid}/revoke", headers=mgr, json={}).status_code == 400
        assert c.post(f"/api/packages/{jid}/revoke", headers=mgr, json={"note": "found a bug"}).json()["status"] == "revoked"
        r = dl(eng)
        assert r.status_code == 403 and "found a bug" in r.text
        assert c.get(f"/api/jobs/{jid}/download/model.onnx", headers=eng).status_code == 403
        assert c.post(f"/api/packages/{jid}/request", headers=eng, json={"note": "fixed"}).json()["status"] == "pending"
        r = c.post(f"/api/packages/{jid}/reject", headers=hs["admin"], json={"note": "wait for 1.3"})
        assert r.json()["status"] == "rejected" and "wait for 1.3" in dl(eng).text

        hist = [h["action"] for h in c.get(f"/api/packages/{jid}", headers=eng).json()["history"]]
        for a in ("package_requested", "package_approved", "download", "package_revoked", "package_rejected"):
            assert a in hist


def test_nobody_approves_their_own_request():
    with _client() as c:
        ah = _admin(c)
        org, hs = _org(c, ah, "Initrode")
        jid = _finished_project(org["id"])
        assert c.post(f"/api/packages/{jid}/request", headers=hs["manager"], json={"note": "mine"}).status_code == 200
        r = c.post(f"/api/packages/{jid}/approve", headers=hs["manager"], json={})
        assert r.status_code == 403 and "own request" in r.text
        assert c.post(f"/api/packages/{jid}/approve", headers=hs["admin"], json={}).json()["status"] == "approved"


def test_unfinished_jobs_have_no_package_to_request():
    with _client() as c:
        ah = _admin(c)
        org, hs = _org(c, ah, "Hooli")
        jid = _finished_project(org["id"])
        with db.conn() as con:
            con.execute("UPDATE jobs SET status='failed' WHERE id=?", (jid,))
        assert c.post(f"/api/packages/{jid}/request", headers=hs["engineer"], json={}).status_code == 400
