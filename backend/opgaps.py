# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Log of models refused for a missing operator: which operators customers need that SAI cannot lower yet.

Every refusal is recorded where it happens:
  * SAI (`graph.unsupported` / status `unsupported` from any `sai` command: inspect, estimate, explore, compile)
    for uploaded models (Plan Explorer), fit checks, training, rebuilds;
  * SAI's front end (sai-front) refusing a layer of a sequence model;
  * the platform's own model editor, for layer types it has no SAI lowering for.

Each row keeps the operator, the target, SAI's capability class and reason, the organisation and job, and the
model's full operator list (for ONNX uploads) so the compiler team can see what else those models need.
The Admin > Operator gaps page groups them by operator (how often, how many organisations, which targets)."""

from __future__ import annotations

import contextvars
import json
import re
from collections import Counter
from pathlib import Path

from . import db

# set by the job runner: (job_id, kind, org_id)
JOB: contextvars.ContextVar[tuple | None] = contextvars.ContextVar("sai_job", default=None)


def _model_ops(model: str | Path | None) -> list[str]:
    try:
        p = Path(model) if model else None
        if not p or p.suffix.lower() != ".onnx" or not p.is_file():
            return []
        import onnx
        m = onnx.load(str(p), load_external_data=False)
        return sorted(Counter(n.op_type for n in m.graph.node))
    except Exception:          # noqa: BLE001 - the log must never break the job
        return []


def record(op: str | None, source: str, target: str | None = None, capability: str | None = None,
           reason: str | None = None, model: str | Path | None = None, org_id: str | None = None,
           job_id: str | None = None, kind: str | None = None):
    ctx = JOB.get()
    if ctx:
        job_id, kind, org_id = job_id or ctx[0], kind or ctx[1], org_id or ctx[2]
    try:
        with db.conn() as c:
            c.execute("INSERT INTO operator_gaps (created, org_id, job_id, kind, source, op, target, capability,"
                      " reason, model_ops) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (db.now(), org_id, job_id, kind, source, (op or "unknown")[:80], target, capability,
                       (reason or "")[:400], json.dumps(_model_ops(model))))
    except Exception:          # noqa: BLE001
        pass


def from_sai_error(err: dict, args: list[str]):
    """Called by sai_engine.run for every structured SAI error; records the operator refusals."""
    if not isinstance(err, dict):
        return
    if err.get("code") != "graph.unsupported" and err.get("status") != "unsupported":
        return
    reason = (err.get("reason") or "").lower()
    if "floating-point" in reason or "quantiz" in reason:
        return                       # a float graph (export problem), not a missing operator
    loc = err.get("location") or {}
    model = next((a for a in args[1:] if not a.startswith("-") and a.lower().endswith((".onnx", ".tflite"))), None)
    target = err.get("target")
    if "--target" in args:
        i = args.index("--target")
        target = args[i + 1] if i + 1 < len(args) else target
    record(loc.get("op") or loc.get("operator") or err.get("capability"), "sai", target, err.get("capability"),
           err.get("reason"), model)


FRONT_RE = re.compile(r"(?:unsupported|not supported|no lowering)[^\n]*?(?:layer|op(?:erator)?)\s*[:'\"]?\s*([A-Za-z_][\w.]*)", re.I)


def from_text(text: str, source: str, target: str | None = None, model: str | Path | None = None) -> bool:
    """Best-effort parse of a refusal in a tool's text output (SAI's front end prints them)."""
    m = FRONT_RE.search(text or "")
    if m:
        record(m.group(1), source, target, "operator_or_layer", m.group(0)[:300], model)
    return bool(m)


def summary(since: float | None = None, org_id: str | None = None) -> dict:
    q, a = "SELECT * FROM operator_gaps WHERE 1=1", []
    if since:
        q += " AND created>=?"; a.append(since)
    if org_id:
        q += " AND org_id=?"; a.append(org_id)
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute(q + " ORDER BY created DESC", a)]
    ops: dict[str, dict] = {}
    for r in rows:
        e = ops.setdefault(r["op"], {"op": r["op"], "count": 0, "orgs": set(), "targets": Counter(),
                                     "sources": Counter(), "first": r["created"], "last": r["created"],
                                     "capability": r["capability"], "reason": r["reason"], "with": Counter()})
        e["count"] += 1
        e["orgs"].add(r["org_id"])
        e["targets"][r["target"] or "any"] += 1
        e["sources"][r["source"]] += 1
        e["first"] = min(e["first"], r["created"])
        e["last"] = max(e["last"], r["created"])
        for o in json.loads(r["model_ops"] or "[]"):
            if o != r["op"]:
                e["with"][o] += 1
    out = []
    for e in ops.values():
        out.append({**e, "orgs": len(e["orgs"] - {None}) or (1 if None in e["orgs"] else 0),
                    "targets": dict(e["targets"]), "sources": dict(e["sources"]),
                    "with": [k for k, _ in e["with"].most_common(8)]})
    out.sort(key=lambda e: (-e["orgs"], -e["count"]))
    return {"operators": out, "total": len(rows),
            "recent": [{k: r[k] for k in ("created", "op", "target", "source", "kind", "job_id", "org_id", "reason")}
                       for r in rows[:50]]}
