# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Retention: keep what customers and reproducibility need, delete what can be rebuilt.

Kept for every finished project: model.onnx, build/ (the compiled package), package.zip, report.json,
MODEL_CARD.md, generated headers, hil_vectors.npz — enough to download, rebuild byte-identically and run on boards.

Deleted right away by the pipeline / board farm: per-size ONNX files, QAT drafts, the host build used for the
bit-exact evaluation, HIL firmware of finished board runs.

Deleted by the sweep (hourly in every worker, or Admin -> Job queue -> Clean up now):
  uploads/ older than UPLOAD_TTL_DAYS (explorer models and staged dataset uploads)
  tmp/ and regression/ work folders older than a day
  jobs/<id>/hil_build_<target>/ older than RETENTION_DAYS (recompiled on demand for the next board run)
  jobs/<id>/ of failed or cancelled jobs older than RETENTION_DAYS (the job row and its log stay)
  expired sessions
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import config, db

RETENTION_DAYS = float(os.environ.get("SAI_PLATFORM_RETENTION_DAYS", "30"))
UPLOAD_TTL_DAYS = float(os.environ.get("SAI_PLATFORM_UPLOAD_TTL_DAYS", "7"))
DAY = 86400.0


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _age(p: Path, now: float) -> float:
    return now - p.stat().st_mtime


def _remove(p: Path, report: dict, kind: str):
    try:
        n = _size(p)
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except OSError:
        return
    report["freed_bytes"] += n
    report["removed"][kind] = report["removed"].get(kind, 0) + 1


def job_intermediates(job_dir: Path, keep: set[str] | None = None) -> int:
    """Called by the pipeline when a training job finishes: drop what the package and rebuilds do not need."""
    report = {"freed_bytes": 0, "removed": {}}
    keep = keep or set()
    for p in list(job_dir.glob("model_*.onnx")):
        if p.name not in keep:
            _remove(p, report, "intermediate")
    for name in ("exact",):
        if (job_dir / name).exists():
            _remove(job_dir / name, report, "intermediate")
    for p in list(job_dir.glob("seq_*")):         # SAI front-end builds per size and board, calibration / test data
        _remove(p, report, "intermediate")
    return report["freed_bytes"]


def sweep(now: float | None = None) -> dict:
    now = now or time.time()
    report = {"freed_bytes": 0, "removed": {}, "at": now}
    root = config.DATA_DIR
    up = root / "uploads"
    for p in up.iterdir() if up.exists() else []:
        if _age(p, now) > UPLOAD_TTL_DAYS * DAY:
            _remove(p, report, "uploads")
    for sub in ("tmp", "regression"):
        d = root / sub
        for p in d.iterdir() if d.exists() else []:
            if p.name != "latest.json" and _age(p, now) > DAY:
                _remove(p, report, sub)
    with db.conn() as c:
        done = {r[0]: r[1] for r in c.execute("SELECT id, status FROM jobs WHERE status IN ('succeeded','failed','cancelled')"
                                              " AND updated<?", (now - RETENTION_DAYS * DAY,))}
        c.execute("DELETE FROM sessions WHERE expires<?", (now,))
    jobs = root / "jobs"
    for jid, status in done.items():
        jd = jobs / jid
        if not jd.exists():
            continue
        if status in ("failed", "cancelled"):
            _remove(jd, report, "failed_job_dirs")
            continue
        for p in jd.glob("hil_build_*"):
            _remove(p, report, "hil_builds")
        if (jd / "hil").exists():
            _remove(jd / "hil", report, "hil_firmware")
    db.set_setting("retention_last", {"at": now, "freed_bytes": report["freed_bytes"], "removed": report["removed"]})
    return report
