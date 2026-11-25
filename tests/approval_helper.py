# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Test helper: get a compiled package released the way users do (request, then another person approves)."""

from __future__ import annotations

import os
import uuid


def release(c, jid: str, headers: dict | None = None) -> dict:
    kw = {"headers": headers} if headers else {}
    r = c.post(f"/api/packages/{jid}/request", json={"note": "test"}, **kw)
    assert r.status_code == 200, r.text
    org_id = c.get(f"/api/jobs/{jid}", **kw).json()["org_id"]
    admin = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                            "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]}).json()["token"]
    ah = {"Authorization": f"Bearer {admin}"}
    email = f"mgr-{uuid.uuid4().hex[:8]}@approvals.test"
    assert c.post("/api/admin/users", headers=ah, json={"email": email, "name": "Manager", "role": "manager",
                                                        "password": "manager-pass-1", "org_id": org_id}).status_code == 200
    mh = {"Authorization": "Bearer " + c.post("/api/auth/login", json={"email": email, "password": "manager-pass-1"}).json()["token"]}
    r = c.post(f"/api/packages/{jid}/approve", headers=mh, json={"note": "ok"})
    assert r.status_code == 200, r.text
    return r.json()
