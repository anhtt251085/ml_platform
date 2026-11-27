#!/usr/bin/env python3
# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Customer-VPC worker: reach data sources inside your own network for the SAI Platform.

Runs next to your data (any machine or container that can reach your MinIO, InfluxDB or PostgreSQL). It only makes
outbound HTTPS requests to the platform: it asks for work, does it against your private endpoint and sends back
the listing or the imported file. Credentials for your data sources stay in a local secrets file and are never
sent to the platform.

    1. Admin > VPC workers: create a worker, copy its token.
    2. Data > Cloud & database: create the connection with "through a VPC worker" ticked (no secrets there).
    3. On the worker machine (with this repository's backend/ next to it, or inside the platform image):

         export SAI_VPC_TOKEN=sai_vpc_...
         python tools/vpc_worker.py --server https://platform.example.com --secrets vpc-secrets.json

       vpc-secrets.json maps a connection *name* to its secrets, e.g.
         {"plant MinIO": {"secret_key": "..."}, "historian": {"password": "..."}}
       (or set SAI_VPC_SECRET_<NAME> to a JSON object, NAME upper-cased with non-alphanumerics as _).

Private and plain-http endpoints are allowed here (this process runs inside your network)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("SAI_PLATFORM_ALLOW_PRIVATE_CONNECTORS", "1")
os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="sai-vpc-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import connectors  # noqa: E402

VERSION = "1"


class Platform:
    def __init__(self, server: str, token: str):
        self.server, self.token = server.rstrip("/"), token

    def call(self, path: str, body: bytes | None = None, ctype: str = "application/json", headers: dict | None = None):
        req = urllib.request.Request(self.server + path, data=body if body is not None else b"{}", method="POST",
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": ctype,
                                              **(headers or {})})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)


def secrets_for(name: str, table: dict) -> dict:
    env = os.environ.get("SAI_VPC_SECRET_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper())
    if env:
        return json.loads(env)
    return table.get(name, {})


def handle(task: dict, table: dict) -> tuple[str, object]:
    """Returns ("json", {ok, result | error}) or ("file", (bytes, filename))."""
    c = task["connection"]
    kind, cfg, sec = c["kind"], c["config"], secrets_for(c["name"], table)
    if task["kind"] == "browse":
        return "json", {"ok": True, "result": {"items": connectors.browse_with(kind, cfg, sec, task["payload"].get("prefix", ""))}}
    if task["kind"] == "test":
        return "json", {"ok": True, "result": connectors.test_with(kind, cfg, sec)}
    if task["kind"] == "import":
        o = task["payload"]["options"]
        if o.get("key"):
            raw, name = connectors.fetch_with(kind, cfg, sec, o["key"])
        else:
            raw, name = connectors.time_to_seconds(connectors.query_with(kind, cfg, sec, o["query"])), "query.csv"
        return "file", (raw, name)
    return "json", {"ok": False, "error": f"unknown task kind {task['kind']}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", default=os.environ.get("SAI_VPC_TOKEN"))
    ap.add_argument("--secrets", help="JSON file: connection name -> secrets")
    ap.add_argument("--once", action="store_true", help="handle the queued tasks, then exit")
    a = ap.parse_args()
    if not a.token:
        sys.exit("give --token or set SAI_VPC_TOKEN")
    table = json.loads(Path(a.secrets).read_text()) if a.secrets else {}
    pf = Platform(a.server, a.token)
    info = {"host": platform.node(), "worker": VERSION, "python": platform.python_version(),
            "connections_with_secrets": sorted(table)}
    idle = 0
    while True:
        try:
            st, task = pf.call("/api/vpc/claim", json.dumps({"info": info}).encode())
        except urllib.error.HTTPError as e:
            if e.code == 401:
                sys.exit("the platform refused the token (disabled or wrong)")
            time.sleep(5)
            continue
        except urllib.error.URLError as e:
            print(f"platform unreachable: {e.reason}", file=sys.stderr)
            time.sleep(10)
            continue
        if st == 204 or not task:
            if a.once:
                return
            idle = min(idle + 1, 5)
            time.sleep(0.5 * idle)
            continue
        idle = 0
        tid = task["id"]
        try:
            kind, out = handle(task, table)
            if kind == "file":
                raw, name = out
                pf.call(f"/api/vpc/tasks/{tid}/file", raw, "application/octet-stream", {"X-Filename": name})
                print(f"{tid}: imported {name} ({len(raw)} bytes)")
            else:
                pf.call(f"/api/vpc/tasks/{tid}/complete", json.dumps(out).encode())
                print(f"{tid}: {task['kind']} done")
        except urllib.error.HTTPError as e:
            print(f"{tid}: platform answered {e.code}: {e.read()[:300].decode(errors='replace')}", file=sys.stderr)
        except Exception as e:                       # noqa: BLE001 - report every failure to the platform
            try:
                pf.call(f"/api/vpc/tasks/{tid}/complete", json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:900]}).encode())
            except Exception:                        # noqa: BLE001
                pass
            print(f"{tid}: failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
