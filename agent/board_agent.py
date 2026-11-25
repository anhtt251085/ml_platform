"""SAI Platform lab agent: connects MCU boards (USB) or QEMU boards to the platform's board farm.

    python agent/board_agent.py --server https://platform.example --token $AGENT_TOKEN --boards boards.json

boards.json (one entry per board on this PC):
    [{"id": "LAB1-H753-01", "name": "NUCLEO-H753ZI #1", "target": "cortex-m7", "port": "stm32h753-nucleo",
      "driver": "pyocd", "uid": "0668FF...", "serial": "/dev/ttyACM0", "pyocd_target": "stm32h753zitx",
      "power": {"uhubctl": ["-l", "1-1", "-p", "2"]}},
     {"id": "LAB1-MD6605-01", "name": "MD6605 EVB", "target": "md6605", "port": "md6605-evb",
      "driver": "jlink", "jlink_device": "MD6605", "serial": "/dev/ttyUSB0"},
     {"id": "CI-QEMU-M7", "name": "QEMU Cortex-M7", "target": "cortex-m7", "port": "qemu-mps2-an500", "driver": "qemu"}]

Loop: register boards -> poll every 2 s (heartbeat + health) -> for a run: download the SAI build, build the HIL
harness for the board's port (backend/hil.py, SAI's own target flags), flash, read the line protocol from the
UART, post outputs + cycles + painted stack. The server checks outputs bit-exactly; real boards calibrate latency
and turn plan rows into `measured`. A board whose run fails is power-cycled once (uhubctl) and retried.
Protocol: docs/BOARD_AGENT_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import hil  # noqa: E402

import numpy as np  # noqa: E402


class BoardDriver:
    emulated = False

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def health(self) -> dict:
        return {"probe": "unknown"}

    def run(self, elf: str, timeout: int = 60) -> dict:
        """Flash (or load) the firmware and return the parsed harness result."""
        raise NotImplementedError

    def power_cycle(self) -> bool:
        p = (self.cfg.get("power") or {}).get("uhubctl")
        if not p:
            return False
        return subprocess.run(["uhubctl", *p, "-a", "cycle"], capture_output=True).returncode == 0


class QemuDriver(BoardDriver):
    emulated = True

    def health(self):
        return {"probe": "ok" if hil.available(self.cfg["port"]) else "missing qemu/toolchain", "kind": "qemu"}

    def run(self, elf, timeout=60):
        return hil.run_emulated(elf, self.cfg["port"], timeout)


class _SerialDriver(BoardDriver):
    """Flash with a probe tool, then read the harness's UART lines until DONE."""

    def _read_uart(self, timeout: int) -> dict:
        import serial                                        # pyserial
        lines = []
        with serial.Serial(self.cfg["serial"], self.cfg.get("baud", 115200), timeout=1) as s:
            self._reset()
            t0 = time.time()
            while time.time() - t0 < timeout:
                raw = s.readline().decode(errors="replace").strip()
                if raw:
                    lines.append(raw)
                    if raw.startswith("DONE"):
                        break
        res = hil.parse("\n".join(lines))
        if res["status"] != "ok":
            raise hil.HilError(f"board did not finish: {lines[-3:]}")
        return res

    def _reset(self):
        pass


class PyOcdDriver(_SerialDriver):
    """ST-LINK / CMSIS-DAP boards (NUCLEO-H753ZI)."""

    def _pyocd(self, *args, timeout=120):
        return subprocess.run(["pyocd", *args, "-t", self.cfg.get("pyocd_target", "stm32h753zitx"),
                               *(["-u", self.cfg["uid"]] if self.cfg.get("uid") else [])],
                              capture_output=True, text=True, timeout=timeout)

    def health(self):
        r = subprocess.run(["pyocd", "list"], capture_output=True, text=True)
        return {"probe": "ok" if self.cfg.get("uid", "") in r.stdout else "not found"}

    def run(self, elf, timeout=60):
        f = self._pyocd("flash", elf)
        if f.returncode:
            raise hil.HilError("pyocd flash failed: " + f.stderr[-400:])
        return self._read_uart(timeout)

    def _reset(self):
        self._pyocd("reset", timeout=30)


class JLinkDriver(_SerialDriver):
    """SEGGER J-Link boards (MD6605 EVB)."""

    def run(self, elf, timeout=60):
        bin_ = Path(elf).with_suffix(".hex")
        subprocess.run([hil.cc_for(self.cfg["port"]).replace("gcc", "objcopy"), "-O", "ihex", elf, str(bin_)], check=True)
        script = Path(elf).with_suffix(".jlink")
        script.write_text(f"loadfile {bin_}\nr\ng\nq\n")
        f = subprocess.run(["JLinkExe", "-device", self.cfg.get("jlink_device", "MD6605"), "-if", "JTAG", "-speed", "4000",
                            "-autoconnect", "1", "-CommandFile", str(script)], capture_output=True, text=True, timeout=120)
        if f.returncode:
            raise hil.HilError("J-Link flash failed: " + f.stdout[-400:])
        return self._read_uart(timeout)


DRIVERS = {"qemu": QemuDriver, "pyocd": PyOcdDriver, "jlink": JLinkDriver}


class Agent:
    def __init__(self, server: str, token: str, boards: list[dict], work: Path = Path("agent_work")):
        self.server, self.token, self.boards = server.rstrip("/"), token, boards
        self.drivers = {b["id"]: DRIVERS[b.get("driver", "qemu")](b) for b in boards}
        self.work = work

    def _req(self, method: str, path: str, body: dict | None = None, raw: bool = False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.server + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:      # noqa: S310 - operator-configured server URL
            return r.read() if raw else json.loads(r.read() or b"{}")

    def register(self):
        info = [{k: v for k, v in b.items() if k not in ("power",)} | {"emulated": self.drivers[b["id"]].emulated}
                for b in self.boards]
        return self._req("POST", "/api/agent/register", {"boards": info})

    def poll_once(self) -> bool:
        health = {bid: d.health() for bid, d in self.drivers.items()}
        run = self._req("POST", "/api/agent/poll", {"health": health}).get("run")
        if run:
            self.execute(run)
        return bool(run)

    def loop(self):
        print("registered", self.register())
        while True:
            if not self.poll_once():
                time.sleep(2)

    def execute(self, run: dict):
        rid, board = run["id"], run["board_id"]
        drv = self.drivers[board]
        work = self.work / rid
        try:
            pkg = self._req("GET", f"/api/agent/runs/{rid}/package", raw=True)
            zipfile.ZipFile(io.BytesIO(pkg)).extractall(work)
            vec = np.array(json.loads((work / "sai_package" / "hil_vectors.json").read_text()), np.int8)
            fw = hil.build(work / "sai_package", drv.cfg["port"], vec, work / "fw")
            try:
                res = drv.run(fw["elf"])
            except hil.HilError:
                if not drv.power_cycle():
                    raise
                res = drv.run(fw["elf"])
            clock = hil.PORTS[drv.cfg["port"]]["clock_hz"]
            self._req("POST", f"/api/agent/runs/{rid}/result", {
                "ok": True, "emulated": drv.emulated, "outputs": res["outputs"], "cycles": res["cycles"],
                "latency_ms": None if drv.emulated or not clock else round(res["cycles"] / clock * 1000, 4),
                "stack_painted": res["stack_painted"], "board": board, "n_vectors": len(res["outputs"]),
                "firmware": f"sai-hil-harness 1 ({drv.cfg['port']}, text {fw['text']} B)",
                "log": [f"agent driver {type(drv).__name__}"]})
        except Exception as e:                   # report, never crash the agent
            self._req("POST", f"/api/agent/runs/{rid}/result", {"ok": False, "error": str(e)[-800:], "board": board,
                                                                "emulated": drv.emulated})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--boards", required=True, help="JSON file listing the boards on this machine")
    a = ap.parse_args()
    Agent(a.server, a.token, json.loads(Path(a.boards).read_text())).loop()


if __name__ == "__main__":
    main()
