"""Model pool (built-in + researcher templates), dataset registry and projects."""

from __future__ import annotations

import re
import uuid

from . import archspec, catalog, db

_BUILTIN = catalog.builtin()


def templates(include_drafts: bool = False) -> list[dict]:
    """Built-ins + published research templates; drafts only for researchers (include_drafts). Deprecated
    templates stay readable for old projects but are hidden from new ones unless drafts are included."""
    out = list(_BUILTIN.values())
    with db.conn() as c:
        for r in c.execute("SELECT body FROM templates ORDER BY updated DESC"):
            t = db.uj(r[0])
            if include_drafts or t.get("status") == "published":
                out.append(t)
    return out


def template(tid: str) -> dict | None:
    if tid in _BUILTIN:
        return _BUILTIN[tid]
    with db.conn() as c:
        r = c.execute("SELECT body FROM templates WHERE id=?", (tid,)).fetchone()
    return db.uj(r[0]) if r else None


def save_template(body: dict, org_id: str | None = None) -> dict:
    """Researcher create/update. Validates every variant against the input contract."""
    name = (body.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    tid = body.get("id") or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] + "-" + uuid.uuid4().hex[:4]
    if tid in _BUILTIN:
        raise ValueError("built-in templates are read-only; save a copy under a new name")
    task = body.get("task", "classification")
    if body.get("status", "draft") not in ("draft", "published", "deprecated"):
        raise ValueError("status must be draft, published or deprecated")
    if task not in ("classification", "regression", "anomaly"):
        raise ValueError("task must be classification, regression or anomaly")
    inp = body.get("input") or {}
    labels = [s.strip() for s in body.get("labels", []) if str(s).strip()] or (["value"] if task == "regression" else [])
    if task == "classification" and len(labels) < 2:
        raise ValueError("a classifier needs at least two labels")
    variants = {}
    for size, layers in (body.get("variants") or {}).items():
        if not layers:
            continue
        n_out = len(labels) if task == "classification" else 1 if task == "regression" else \
            int(inp.get("channels", 1)) * int(inp.get("height", 1)) * int(inp.get("width", 1))
        spec = archspec.normalize({"input": inp, "layers": layers, "outputs": n_out})   # raises SpecError
        variants[size] = spec["layers"]
    if not variants:
        raise ValueError("add at least one architecture variant")
    t = {
        "id": tid, "name": name, "domain": body.get("domain") or "Custom", "icon": body.get("icon") or "🧪",
        "task": task, "summary": body.get("summary") or "", "use_cases": body.get("use_cases") or [],
        "input": {"kind": inp.get("kind", "timeseries"), "channels": int(inp.get("channels", 1)),
                  "height": int(inp.get("height", 1)), "width": int(inp.get("width", 128)),
                  "sample_rate_hz": inp.get("sample_rate_hz"), "channel_names": inp.get("channel_names") or []},
        "labels": labels, "window": int(inp.get("width", 128)) if int(inp.get("height", 1)) == 1 else None,
        "hop": body.get("hop") or max(1, int(inp.get("width", 128)) // 2), "synth": body.get("synth") or None,
        "variants": variants, "targets": body.get("targets") or ["md6605", "cortex-m7"],
        "datasets_hint": body.get("datasets_hint") or "", "status": body.get("status", "draft"),
        "author": body.get("author") or "researcher", "source": "studio",
        "licence": body.get("licence") or "",
    }
    prev = template(tid)
    t["version"] = int((prev or {}).get("version", 0)) + (1 if t["status"] == "published" else 0) or 1
    t["history"] = ((prev or {}).get("history") or [])[-19:] + [{"version": t["version"], "status": t["status"],
                                                                "ts": db.now(), "author": t["author"]}]
    with db.conn() as c:
        c.execute("INSERT OR REPLACE INTO templates (id, body, status, author, updated, org_id) VALUES (?,?,?,?,?,?)",
                  (tid, db.j(t), t["status"], t["author"], db.now(), org_id))
    return t


def delete_template(tid: str):
    if tid in _BUILTIN:
        raise ValueError("built-in templates cannot be deleted")
    with db.conn() as c:
        c.execute("DELETE FROM templates WHERE id=?", (tid,))


def add_dataset(meta: dict, org_id: str | None = None):
    meta["org_id"] = org_id
    with db.conn() as c:
        c.execute("INSERT OR REPLACE INTO datasets (id, body, created, org_id) VALUES (?,?,?,?)",
                  (meta["id"], db.j(meta), db.now(), org_id))


def datasets(template_id: str | None = None, org_id: str | None = None) -> list[dict]:
    with db.conn() as c:
        q = "SELECT body FROM datasets" + (" WHERE org_id=?" if org_id else "") + " ORDER BY created DESC"
        rows = [db.uj(r[0]) for r in c.execute(q, (org_id,) if org_id else ())]
    rows = [r for r in rows if not template_id or r.get("template") == template_id]
    for r in rows:
        r.pop("preview", None)
    return rows


def dataset(did: str) -> dict | None:
    with db.conn() as c:
        r = c.execute("SELECT body FROM datasets WHERE id=?", (did,)).fetchone()
    return db.uj(r[0]) if r else None


def create_project(name, template_id, dataset_id, cfg, job_id, org_id=None) -> dict:
    pid = "prj_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO projects (id, name, template_id, dataset_id, config, job_id, created, org_id)"
                  " VALUES (?,?,?,?,?,?,?,?)", (pid, name, template_id, dataset_id, db.j(cfg), job_id, db.now(), org_id))
    return project(pid)


def project(pid: str) -> dict | None:
    with db.conn() as c:
        r = db.row(c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone())
    if r:
        r["config"] = db.uj(r["config"], {})
    return r


def projects(org_id: str | None = None) -> list[dict]:
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute(
            "SELECT p.*, j.status AS job_status, j.progress AS job_progress, j.stage AS job_stage "
            "FROM projects p LEFT JOIN jobs j ON j.id = p.job_id"
            + (" WHERE p.org_id=?" if org_id else "") + " ORDER BY p.created DESC", (org_id,) if org_id else ())]
    for r in rows:
        r["config"] = db.uj(r["config"], {})
    return rows
