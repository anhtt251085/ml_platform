# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Runtime configuration from environment variables (12-factor; the same image runs locally and in the cloud)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SAI_PLATFORM_DATA", ROOT / "var")).resolve()
DB_PATH = DATA_DIR / "platform.sqlite3"
WORK_DIR = DATA_DIR
# The SAI compiler is never installed or copied into the platform: it runs in place from its own repository
# checkout (SAI_REPO) through bin/sai. SAI_BIN overrides the command entirely (e.g. a separately managed install).
SAI_REPO = os.environ.get("SAI_REPO", "")
SAI_LAUNCHER = str(ROOT / "bin" / "sai")
SAI_BIN = os.environ.get("SAI_BIN") or (SAI_LAUNCHER if SAI_REPO else shutil.which("sai") or "sai")
SAI_MD6605_GCC = os.environ.get("SAI_MD6605_GCC", "")
SAI_CORTEX_M7_GCC = os.environ.get("SAI_CORTEX_M7_GCC", "")
HOST_CC = os.environ.get("CC", "gcc")
MAX_UPLOAD_BYTES = int(os.environ.get("SAI_PLATFORM_MAX_UPLOAD", str(2 * 1024 * 1024 * 1024)))
# Uploads up to this size are parsed in the request (a few seconds); larger ones become an `ingest` job.
INLINE_INGEST_BYTES = int(os.environ.get("SAI_PLATFORM_INLINE_INGEST", str(16 * 1024 * 1024)))

# Deployment mode: "multi" = multi-tenant SaaS (SAI admins create customer orgs); "single" = one customer per
# install (on-prem / customer VPC): one customer org, no org management.
MODE = os.environ.get("SAI_PLATFORM_MODE", "multi")
SINGLE_ORG_NAME = os.environ.get("SAI_PLATFORM_ORG_NAME", "Customer")

# Jobs: a DB-backed queue. The API process runs INPROC_WORKERS worker threads (0 = run `python -m backend.worker`
# processes separately, e.g. GPU train workers and CPU compile workers).
INPROC_WORKERS = int(os.environ.get("SAI_PLATFORM_INPROC_WORKERS", os.environ.get("SAI_PLATFORM_WORKERS", "2")))
WORKER_KINDS = [k for k in os.environ.get("SAI_PLATFORM_WORKER_KINDS", "").split(",") if k]   # empty = all

SIM_BOARDS = os.environ.get("SAI_PLATFORM_SIM_BOARDS", "1") == "1"      # in-process simulated board agent
AGENT_TOKEN = os.environ.get("SAI_PLATFORM_AGENT_TOKEN", "dev-agent-token")
ALLOW_DEV_AGENT_TOKEN = os.environ.get("SAI_PLATFORM_ALLOW_DEV_AGENT_TOKEN", "0") == "1"
# auto (default): Secure cookie whenever the request came over HTTPS (behind a proxy: uvicorn --proxy-headers);
# 1: always (HTTPS-only deployments); 0: never (not recommended).
COOKIE_SECURE = os.environ.get("SAI_PLATFORM_COOKIE_SECURE", "auto").lower()
BRAND_NAME = os.environ.get("SAI_PLATFORM_BRAND", "SAI Platform")         # white-label (M6)
SIGNING_KEY = os.environ.get("SAI_PLATFORM_SIGNING_KEY", "")                # Ed25519 PEM; generated when unset


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


PLATFORM_VERSION = os.environ.get("SAI_PLATFORM_VERSION") or f"0.2.0+{_git_sha()}"

for sub in ("datasets", "jobs", "studio", "uploads", "pool"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
