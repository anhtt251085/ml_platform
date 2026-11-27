# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Customer-VPC workers: data access from inside the customer's own network.

A connection marked "through a VPC worker" keeps no credentials on the platform. Browsing it, testing it or
importing from it becomes a task; a worker the customer runs in their network (tools/vpc_worker.py, outbound
HTTPS only, no inbound port) claims the task with its token, talks to the private endpoint (MinIO, an internal
InfluxDB or PostgreSQL) with credentials from its own local secrets file, and sends back the listing or the
imported file. The platform then ingests the file exactly like a direct import.

Workers belong to one organisation and only see that organisation's tasks. Their tokens are stored hashed and can
be disabled; tasks a worker does not finish within the lease go back to the queue."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid

from . import db

LEASE_S = 300
WAIT_S = 20.0               # how long an API call waits for the worker before answering 202 (pending)
KINDS = ("browse", "test", "import")


def _th(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_worker(org_id: str, name: str) -> tuple[dict, str]:
    wid = "vpcw_" + uuid.uuid4().hex[:8]
    token = "sai_vpc_" + secrets.token_urlsafe(32)
    with db.conn() as c:
        c.execute("INSERT INTO vpc_workers (id, org_id, name, token_hash, created, last_seen, disabled, info)"
                  " VALUES (?,?,?,?,?,?,0,?)", (wid, org_id, (name or wid)[:80], _th(token), db.now(), None, "{}"))
    return {"id": wid, "name": name, "org_id": org_id}, token


def workers(org_id: str | None) -> list[dict]:
    q, a = "SELECT id, org_id, name, created, last_seen, disabled, info FROM vpc_workers", ()
    if org_id:
        q, a = q + " WHERE org_id=?", (org_id,)
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute(q + " ORDER BY created", a)]
    for r in rows:
        r["info"] = db.uj(r["info"], {})
        r["online"] = bool(r["last_seen"] and db.now() - r["last_seen"] < 90 and not r["disabled"])
    return rows


def disable_worker(wid: str):
    with db.conn() as c:
        c.execute("UPDATE vpc_workers SET disabled=1 WHERE id=?", (wid,))


def worker_from_token(token: str) -> dict | None:
    with db.conn() as c:
        r = c.execute("SELECT id, org_id, disabled FROM vpc_workers WHERE token_hash=?", (_th(token or ""),)).fetchone()
        if not r or r["disabled"]:
            return None
        c.execute("UPDATE vpc_workers SET last_seen=? WHERE id=?", (db.now(), r["id"]))
        return {"id": r["id"], "org_id": r["org_id"]}


def heartbeat(wid: str, info: dict):
    with db.conn() as c:
        c.execute("UPDATE vpc_workers SET last_seen=?, info=? WHERE id=?", (db.now(), json.dumps(info)[:2000], wid))


def submit(org_id: str, connection_id: str, kind: str, payload: dict) -> dict:
    if kind not in KINDS:
        raise ValueError(kind)
    tid = "vpct_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO vpc_tasks (id, org_id, connection_id, kind, payload, status, created, updated)"
                  " VALUES (?,?,?,?,?,?,?,?)", (tid, org_id, connection_id, kind, json.dumps(payload), "queued",
                                               db.now(), db.now()))
    return get(tid)


def get(tid: str) -> dict | None:
    with db.conn() as c:
        r = c.execute("SELECT * FROM vpc_tasks WHERE id=?", (tid,)).fetchone()
    if not r:
        return None
    r = db.row(r)
    r["payload"] = db.uj(r["payload"], {})
    r["result"] = db.uj(r["result"], None)
    return r


def claim(worker: dict) -> dict | None:
    """Oldest queued task of the worker's organisation (or one whose lease expired)."""
    with db.conn() as c:
        r = c.execute("SELECT id FROM vpc_tasks WHERE org_id=? AND (status='queued' OR (status='running' AND updated<?))"
                      " ORDER BY created LIMIT 1", (worker["org_id"], db.now() - LEASE_S)).fetchone()
        if not r:
            return None
        n = c.execute("UPDATE vpc_tasks SET status='running', worker_id=?, updated=? WHERE id=? AND "
                      "(status='queued' OR (status='running' AND updated<?))",
                      (worker["id"], db.now(), r[0], db.now() - LEASE_S)).rowcount
    return get(r[0]) if n else None


def finish(tid: str, ok: bool, result: dict | None = None, error: str | None = None):
    with db.conn() as c:
        c.execute("UPDATE vpc_tasks SET status=?, result=?, error=?, updated=? WHERE id=?",
                  ("succeeded" if ok else "failed", json.dumps(result) if result is not None else None,
                   (error or "")[:1000] or None, db.now(), tid))


def wait(tid: str, timeout: float = 20.0) -> dict:
    """Long-poll for a task's end (the API answers when the worker is quick, else returns the pending task)."""
    end = time.time() + timeout
    while True:
        t = get(tid)
        if t["status"] in ("succeeded", "failed") or time.time() >= end:
            return t
        time.sleep(0.25)
