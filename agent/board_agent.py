"""SAI Platform lab agent: connects USB-attached MCU boards to the platform's board farm.

    python agent/board_agent.py --server https://platform.example --token $TOKEN --boards boards.json [--simulate]

boards.json:
    [{"id": "LAB1-H753-01", "name": "NUCLEO-H753ZI #1", "target": "cortex-m7", "serial": "0668FF...", "driver": "pyocd"},
     {"id": "LAB1-MD6605-01", "name": "MD6605 EVB", "target": "md6605", "serial": "123", "driver": "jlink"}]

Loop: register boards -> poll every 2 s -> for a run: download the SAI package, build the HIL harness firmware
for the board (harness/<target>/: the package's library + a runner that feeds tests vectors over UART and reports
outputs, DWT/mcycle cycle counts and painted stack), flash, collect, post the result. The server compares outputs
bit-exactly and calibrates latency from real runs. See docs/BOARD_AGENT_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path


class BoardDriver:
    """One physical board. Implementations: PyOcdDriver (ST-LINK/CMSIS-DAP), JLinkDriver, OpenOcdDriver."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build(self, package_dir: Path, target: str) -> Path:
        """Build the HIL harness firmware against the package; returns the ELF. (toolchain per target)"""
        raise NotImplementedError

    def flash(self, elf: Path) -> None:
        raise NotImplementedError

    def run_vectors(self, vectors: bytes, n: int, out_bytes: int, timeout_s: float = 30) -> dict:
        """Send vectors over the harness UART; returns {"outputs": [[...]...], "cycles": int, "stack_painted": int}."""
        raise NotImplementedError


class PyOcdDriver(BoardDriver):
    def build(self, package_dir, target):   # pragma: no cover - needs a cross toolchain and the harness sources
        raise NotImplementedError("harness build: arm-none-eabi-gcc + harness/cortex-m7 (plan milestone M4)")

    def flash(self, elf):                    # pragma: no cover
        from pyocd.core.helpers import ConnectHelper  # noqa: F401  (pyocd flash <elf> --uid <serial>)
        raise NotImplementedError

    def run_vectors(self, vectors, n, out_bytes, timeout_s=30):  # pragma: no cover
        raise NotImplementedError


class SimDriver(BoardDriver):
    """Loopback for testing the agent protocol without hardware: echoes the expected outputs."""

    def build(self, package_dir, target):
        return package_dir / "sim.elf"

    def flash(self, elf):
        time.sleep(0.5)

    def run_vectors(self, vectors, n, out_bytes, timeout_s=30):
        return {"outputs": None, "cycles": None, "stack_painted": None}


DRIVERS = {"pyocd": PyOcdDriver, "sim": SimDriver}


class Agent:
    def __init__(self, server: str, token: str, agent_id: str, boards: list[dict], simulate: bool):
        self.server, self.token, self.agent_id = server.rstrip("/"), token, agent_id
        self.boards = boards
        self.drivers = {b["id"]: (SimDriver if simulate else DRIVERS[b.get("driver", "pyocd")])(b) for b in boards}

    def _req(self, method: str, path: str, body: dict | None = None, raw: bool = False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.server + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:      # noqa: S310 - operator-configured server URL
            return r.read() if raw else json.loads(r.read() or b"{}")

    def loop(self):
        print("registered", self._req("POST", "/api/agent/register", {"agent_id": self.agent_id, "boards": self.boards}))
        while True:
            run = self._req("POST", "/api/agent/poll", {"agent_id": self.agent_id}).get("run")
            if run:
                self.execute(run)
            time.sleep(2)

    def execute(self, run: dict):
        rid, board = run["id"], run["board_id"]
        drv = self.drivers[board]
        work = Path("agent_work") / rid
        work.mkdir(parents=True, exist_ok=True)
        try:
            pkg = self._req("GET", f"/api/agent/runs/{rid}/package", raw=True)
            zipfile.ZipFile(io.BytesIO(pkg)).extractall(work)
            elf = drv.build(work / "sai_package", run["target"])
            drv.flash(elf)
            req = run["request"]
            expected = req.get("expected_outputs") or []
            res = drv.run_vectors(b"", len(expected), len(expected[0]) if expected else 0)
            if res["outputs"] is None:          # SimDriver: protocol test only, no measurement
                res = {"outputs": expected, "cycles": req.get("estimated_cycles"), "stack_painted": None}
                simulated = True
            else:
                simulated = False
            clock = {"cortex-m7": 480e6, "md6605": 66.666666e6}.get(run["target"])
            self._req("POST", f"/api/agent/runs/{rid}/result", {
                "ok": True, "simulated": simulated, "outputs": res["outputs"], "cycles": res["cycles"],
                "latency_ms": round(res["cycles"] / clock * 1000, 4) if res["cycles"] and clock else None,
                "stack_painted": res["stack_painted"], "board": board, "n_vectors": len(res["outputs"] or []),
                "firmware": "sai-hil-harness 0.1", "log": [f"agent {self.agent_id}"]})
        except Exception as e:                   # report, never crash the agent
            self._req("POST", f"/api/agent/runs/{rid}/result", {"ok": False, "error": str(e), "board": board})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--agent-id", default="lab-agent-1")
    ap.add_argument("--boards", required=True, help="JSON file listing the boards on this machine")
    ap.add_argument("--simulate", action="store_true", help="protocol test without hardware")
    a = ap.parse_args()
    Agent(a.server, a.token, a.agent_id, json.loads(Path(a.boards).read_text()), a.simulate).loop()


if __name__ == "__main__":
    main()
