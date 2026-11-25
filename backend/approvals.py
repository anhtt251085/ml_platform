# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Compiled packages and their release approval.

Every successful project produces release files: the compiled package (package.zip: SAI-generated C sources,
runtime library, headers, manifest, memory certificate, signature), the trained INT8 model (model.onnx) and the
full report (report.json). Downloading any of them (GATED) needs a manager's approval of the project's release:

  not requested -> pending (an engineer or anyone with access requests it, with a note)
  pending -> approved | rejected (a manager or admin of the organisation decides, with a note; never their own
             request - separation of duties)
  approved -> revoked (a manager or admin withdraws it; downloads stop)
  rejected / revoked -> pending (requested again)

Every transition and every download is in the audit log; downloads are counted per package.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import auth, config, db

STATUSES = ("not_requested", "pending", "approved", "rejected", "revoked")
GATED = ("package.zip", "model.onnx", "report.json")     # MODEL_CARD.md (documentation) stays freely downloadable


class ApprovalError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def get(job_id: str) -> dict:
    with db.conn() as c:
        r = c.execute("SELECT a.*, ru.name AS requested_by_name, ru.email AS requested_by_email,"
                      " du.name AS decided_by_name, du.email AS decided_by_email FROM package_approvals a"
                      " LEFT JOIN users ru ON ru.id = a.requested_by LEFT JOIN users du ON du.id = a.decided_by"
                      " WHERE a.job_id=?", (job_id,)).fetchone()
    return db.row(r) if r else {"job_id": job_id, "status": "not_requested", "downloads": 0}


def _package_exists(job_id: str) -> bool:
    return (config.DATA_DIR / "jobs" / job_id / "package.zip").exists()


def request(u: dict, job: dict, note: str = "") -> dict:
    if job["kind"] != "train" or job["status"] != "succeeded" or not _package_exists(job["id"]):
        raise ApprovalError(400, "only a finished project with a compiled package can be requested")
    cur = get(job["id"])
    if cur["status"] in ("pending", "approved"):
        raise ApprovalError(409, f"already {cur['status']}")
    with db.conn() as c:
        c.execute("INSERT OR REPLACE INTO package_approvals (job_id, org_id, status, requested_by, requested_at,"
                  " request_note, decided_by, decided_at, decision_note, downloads, last_download)"
                  " VALUES (?,?,?,?,?,?,NULL,NULL,NULL,?,?)",
                  (job["id"], job["org_id"], "pending", u["id"], db.now(), (note or "")[:1000],
                   cur.get("downloads", 0), cur.get("last_download")))
    auth.audit(u, "package_requested", job["id"], {"note": note})
    return get(job["id"])


def decide(u: dict, job: dict, approve: bool, note: str = "") -> dict:
    cur = get(job["id"])
    if cur["status"] != "pending":
        raise ApprovalError(409, f"nothing to decide: the package is {cur['status'].replace('_', ' ')}")
    if cur["requested_by"] == u["id"]:
        raise ApprovalError(403, "you cannot approve or reject your own request: another manager or an admin must")
    if not approve and not (note or "").strip():
        raise ApprovalError(400, "a rejection needs a reason")
    status = "approved" if approve else "rejected"
    with db.conn() as c:
        c.execute("UPDATE package_approvals SET status=?, decided_by=?, decided_at=?, decision_note=? WHERE job_id=?",
                  (status, u["id"], db.now(), (note or "")[:1000], job["id"]))
    auth.audit(u, f"package_{status}", job["id"], {"note": note, "requested_by": cur["requested_by"]})
    return get(job["id"])


def revoke(u: dict, job: dict, note: str = "") -> dict:
    cur = get(job["id"])
    if cur["status"] != "approved":
        raise ApprovalError(409, "only an approved package can be revoked")
    if not (note or "").strip():
        raise ApprovalError(400, "a revocation needs a reason")
    with db.conn() as c:
        c.execute("UPDATE package_approvals SET status='revoked', decided_by=?, decided_at=?, decision_note=?"
                  " WHERE job_id=?", (u["id"], db.now(), note[:1000], job["id"]))
    auth.audit(u, "package_revoked", job["id"], {"note": note})
    return get(job["id"])


def check_download(job_id: str):
    a = get(job_id)
    if a["status"] != "approved":
        msg = {"not_requested": "the compiled package, model and report need a manager's approval: request it on "
                                "the Compiled packages page",
               "pending": "the download request is waiting for a manager's approval",
               "rejected": f"the download was rejected: {a.get('decision_note') or 'no reason given'}",
               "revoked": f"the approval was revoked: {a.get('decision_note') or 'no reason given'}"}[a["status"]]
        raise ApprovalError(403, msg)


def count_download(job_id: str):
    with db.conn() as c:
        c.execute("UPDATE package_approvals SET downloads = downloads + 1, last_download=? WHERE job_id=?",
                  (db.now(), job_id))


def _summary(job_dir: Path, result: dict) -> dict:
    sel = result.get("selected") or {}
    sig = result.get("signature") or {}
    pz = job_dir / "package.zip"
    return {"target": sel.get("target"), "plan": sel.get("schedule"), "size": sel.get("size"), "ram": sel.get("ram"),
            "flash": sel.get("flash"), "latency_ms": (sel.get("latency") or {}).get("ms"),
            "accuracy": (result.get("exact") or {}).get("value"), "metric": (result.get("exact") or {}).get("headline"),
            "sai": (result.get("provenance") or {}).get("sai_version") or (result.get("sai") or {}).get("sai"),
            "signed_key": sig.get("key_id"), "deployable": result.get("deployable", True),
            "package_bytes": pz.stat().st_size if pz.exists() else None}


def packages(org_id: str | None, status: str | None = None, limit: int = 500) -> list[dict]:
    """Every compiled package (succeeded project) visible to the caller, with its approval state."""
    where, args = ["j.kind='train'", "j.status='succeeded'"], []
    if org_id:
        where.append("j.org_id=?"); args.append(org_id)
    q = ("SELECT j.id, j.org_id, j.params, j.result, j.created, j.updated, o.name AS org_name, a.status AS a_status"
         " FROM jobs j LEFT JOIN orgs o ON o.id = j.org_id LEFT JOIN package_approvals a ON a.job_id = j.id"
         " WHERE " + " AND ".join(where) + " ORDER BY j.updated DESC LIMIT ?")
    with db.conn() as c:
        rows = [dict(r) for r in c.execute(q, (*args, limit))]
    out = []
    for r in rows:
        st = r["a_status"] or "not_requested"
        if status and st != status:
            continue
        jd = config.DATA_DIR / "jobs" / r["id"]
        if not (jd / "package.zip").exists():
            continue
        params, result = json.loads(r["params"] or "{}"), json.loads(r["result"] or "{}")
        out.append({"job_id": r["id"], "org_id": r["org_id"], "org_name": r["org_name"],
                    "project": params.get("name"), "template_id": params.get("template_id"),
                    "built": r["updated"], **_summary(jd, result), "approval": get(r["id"])})
    return out
