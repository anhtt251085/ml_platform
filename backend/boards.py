"""Board farm: a registry of physical (or simulated) MCU boards served by lab agents, a queue of
hardware-in-the-loop (HIL) runs, and the latency calibration those runs feed.

Protocol (docs/BOARD_AGENT_PROTOCOL.md): an agent on a lab PC owns some USB-attached boards. It registers them,
polls for runs targeting their MCU, downloads the SAI package, builds and flashes the board firmware, runs the
package's test vectors, and posts outputs + cycles + painted stack. The server checks outputs against the
expected ones and, for real (non-simulated) runs, adds a calibration point (MACs, cycles) for that target/plan.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from . import config, db, sai_engine

STALE_S = 30


def register(agent_id: str, boards: list[dict], simulated: bool = False) -> list[dict]:
    out = []
    with db.conn() as c:
        for b in boards:
            if b.get("target") not in sai_engine.TARGETS:
                continue
            bid = b.get("id") or f"{agent_id}:{b.get('serial', uuid.uuid4().hex[:6])}"
            c.execute("INSERT OR REPLACE INTO boards VALUES (?,?,?,?,?,?,?,?)",
                      (bid, b.get("name", bid), b["target"], agent_id, "idle", db.now(), int(simulated),
                       db.j({k: v for k, v in b.items() if k not in ("id", "name", "target")})))
            out.append({"id": bid, "target": b["target"]})
    return out


def board_agent(board_id: str | None) -> str | None:
    with db.conn() as c:
        r = c.execute("SELECT agent_id FROM boards WHERE id=?", (board_id,)).fetchone()
    return r[0] if r else None


def heartbeat(agent_id: str):
    with db.conn() as c:
        c.execute("UPDATE boards SET last_seen=? WHERE agent_id=?", (db.now(), agent_id))


def list_boards() -> list[dict]:
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute("SELECT * FROM boards ORDER BY target, name")]
    t = db.now()
    for r in rows:
        r["info"] = db.uj(r["info"], {})
        r["online"] = t - (r["last_seen"] or 0) < STALE_S
        r["simulated"] = bool(r["simulated"])
        if not r["online"]:
            r["status"] = "offline"
    return rows


def request_run(job_id: str, target: str, board_id: str | None, request: dict, org_id: str | None = None) -> dict:
    rid = "hil_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO hil_runs (id, job_id, board_id, target, status, created, updated, request, result, org_id)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (rid, job_id, board_id, target, "queued", db.now(), db.now(), db.j(request), None, org_id))
    return get_run(rid)


def get_run(rid: str) -> dict | None:
    with db.conn() as c:
        r = db.row(c.execute("SELECT * FROM hil_runs WHERE id=?", (rid,)).fetchone())
    if r:
        r["request"], r["result"] = db.uj(r["request"], {}), db.uj(r["result"])
    return r


def list_runs(job_id: str | None = None, limit: int = 50, org_id: str | None = None) -> list[dict]:
    where, args = [], []
    if job_id:
        where.append("job_id=?"); args.append(job_id)
    if org_id:
        where.append("org_id=?"); args.append(org_id)
    with db.conn() as c:
        q = "SELECT id FROM hil_runs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created DESC LIMIT ?"
        ids = [r[0] for r in c.execute(q, (*args, limit))]
    return [get_run(i) for i in ids]


def claim(agent_id: str) -> dict | None:
    """Give the agent the oldest queued run for an idle online board it owns."""
    with db.conn() as c:
        boards = [db.row(b) for b in c.execute("SELECT * FROM boards WHERE agent_id=? AND status='idle'", (agent_id,))]
        for b in boards:
            r = c.execute("SELECT * FROM hil_runs WHERE status='queued' AND target=? AND (board_id IS NULL OR board_id=?)"
                          " ORDER BY created LIMIT 1", (b["target"], b["id"])).fetchone()
            if r:
                c.execute("UPDATE hil_runs SET status='running', board_id=?, updated=? WHERE id=?", (b["id"], db.now(), r["id"]))
                c.execute("UPDATE boards SET status='busy' WHERE id=?", (b["id"],))
                run = db.row(r)
                run["board_id"] = b["id"]
                run["request"] = db.uj(run["request"], {})
                return run
    return None


def complete(rid: str, result: dict):
    run = get_run(rid)
    if not run:
        raise KeyError(rid)
    req = run["request"]
    expected = req.get("expected_outputs")
    got = result.get("outputs")
    if expected is not None and got is not None:
        result["outputs_match"] = bool(np.array_equal(np.array(expected), np.array(got)))
    status = "succeeded" if result.get("ok", True) and result.get("outputs_match", True) else "failed"
    with db.conn() as c:
        c.execute("UPDATE hil_runs SET status=?, result=?, updated=? WHERE id=?", (status, db.j(result), db.now(), rid))
        c.execute("UPDATE boards SET status='idle' WHERE id=?", (run["board_id"],))
        if status == "succeeded" and not result.get("simulated") and result.get("cycles") and req.get("macs"):
            c.execute("INSERT INTO calibration VALUES (?,?,?,?,?,?)",
                      (run["target"], req.get("objective"), int(req["macs"]), int(result["cycles"]), rid, db.now()))
    return get_run(rid)


def calibration() -> dict:
    """Least-squares cycles = a * MACs + b per target:objective from real board runs (never simulated ones)."""
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute("SELECT * FROM calibration")]
    out = {}
    for key in {(r["target"], r["objective"]) for r in rows}:
        pts = [(r["macs"], r["cycles"]) for r in rows if (r["target"], r["objective"]) == key]
        m = np.array(pts, float)
        if len(pts) >= 2 and np.ptp(m[:, 0]) > 0:
            a, b = np.polyfit(m[:, 0], m[:, 1], 1)
        else:
            a, b = float(m[:, 1].mean() / max(m[:, 0].mean(), 1)), 0.0
        out[f"{key[0]}:{key[1]}"] = {"cycles_per_mac": float(max(a, 0.01)), "overhead": float(max(b, 0)),
                                     "samples": len(pts)}
    return out


# ------------------------------------------------------------------------------------------------------------------
class SimulatedAgent(threading.Thread):
    """In-process stand-in for a lab agent until real boards are connected. It runs the build's bit-exact host
    library on the run's vectors (so output checking is real), and reports cycles from the planning model with
    +-4% jitter, flagged simulated=True so they never enter calibration."""

    AGENT = "sim-lab-01"
    BOARDS = [{"id": "SIM-H753-01", "name": "NUCLEO-H753ZI (simulated)", "target": "cortex-m7", "serial": "sim1",
               "probe": "ST-LINK V3 (sim)"},
              {"id": "SIM-H753-02", "name": "NUCLEO-H753ZI #2 (simulated)", "target": "cortex-m7", "serial": "sim2",
               "probe": "ST-LINK V3 (sim)"},
              {"id": "SIM-MD6605-01", "name": "MD6605 EVB (simulated)", "target": "md6605", "serial": "sim3",
               "probe": "J-Link (sim)"}]

    def __init__(self):
        super().__init__(daemon=True, name="sim-agent")
        self.stop = threading.Event()

    def run(self):
        register(self.AGENT, self.BOARDS, simulated=True)
        while not self.stop.is_set():
            heartbeat(self.AGENT)
            run = None
            try:
                run = claim(self.AGENT)
                if run:
                    self.execute(run)
            except Exception as e:                       # keep the agent alive; the run is marked failed
                if run:
                    complete(run["id"], {"ok": False, "error": str(e), "simulated": True})
            self.stop.wait(1.0)

    def execute(self, run):
        from . import exact_eval
        req = run["request"]
        time.sleep(random.uniform(1.5, 3.0))             # flash + run time
        jobdir = config.DATA_DIR / "jobs" / run["job_id"]
        vec = np.load(jobdir / "hil_vectors.npz")
        outs = exact_eval.run_exact(Path(req["model_path"]), jobdir / "exact", vec["x"])
        est = req.get("estimated_cycles") or 0
        cycles = int(est * random.uniform(0.96, 1.04)) if est else None
        clock = sai_engine.TARGETS[run["target"]]["clock_hz"]
        complete(run["id"], {
            "ok": True, "simulated": True, "outputs": outs.tolist(), "cycles": cycles,
            "latency_ms": round(cycles / clock * 1000, 4) if cycles else None,
            "stack_painted": int((req.get("stack_bound") or 512) * random.uniform(0.6, 0.9)),
            "board": run["board_id"], "firmware": "sai-hil-harness 0.1 (sim)", "n_vectors": int(len(outs)),
            "log": ["flash: ok (sim)", f"vectors: {len(outs)}", "check: outputs compared server-side"]})
