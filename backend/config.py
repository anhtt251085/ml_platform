"""Runtime configuration from environment variables (12-factor; the same image runs locally and in the cloud)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SAI_PLATFORM_DATA", ROOT / "var")).resolve()
DB_PATH = DATA_DIR / "platform.sqlite3"
WORK_DIR = DATA_DIR
SAI_BIN = os.environ.get("SAI_BIN") or shutil.which("sai") or "sai"
SAI_MD6605_GCC = os.environ.get("SAI_MD6605_GCC", "")
SAI_CORTEX_M7_GCC = os.environ.get("SAI_CORTEX_M7_GCC", "")
HOST_CC = os.environ.get("CC", "gcc")
WORKERS = int(os.environ.get("SAI_PLATFORM_WORKERS", "2"))
MAX_UPLOAD_BYTES = int(os.environ.get("SAI_PLATFORM_MAX_UPLOAD", str(200 * 1024 * 1024)))
SIM_BOARDS = os.environ.get("SAI_PLATFORM_SIM_BOARDS", "1") == "1"      # in-process simulated board agent
AGENT_TOKEN = os.environ.get("SAI_PLATFORM_AGENT_TOKEN", "dev-agent-token")

for sub in ("datasets", "jobs", "studio", "uploads"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
