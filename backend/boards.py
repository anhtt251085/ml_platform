# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
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
            bid = str(b.get("id") or f"{agent_id}:{b.get('serial', uuid.uuid4().hex[:6])}")[:64]
            info = db.j({k: v for k, v in b.items() if k not in ("id", "name", "target")})
            own = c.execute("SELECT agent_id FROM boards WHERE id=?", (bid,)).fetchone()
            if own and own[0] != agent_id:
                out.append({"id": bid, "error": "board id belongs to another agent"})
                continue
            if own:        # re-register: keep status, failures and quarantine; refresh description only
                c.execute("UPDATE boards SET name=?, target=?, info=?, last_seen=?, simulated=? WHERE id=?",
                          (str(b.get("name", bid))[:120], b["target"], info, db.now(), int(simulated), bid))
            else:
                c.execute("INSERT INTO boards (id, name, target, agent_id, status, last_seen, simulated, info,"
                          " fail_count, health) VALUES (?,?,?,?,?,?,?,?,0,NULL)",
                          (bid, str(b.get("name", bid))[:120], b["target"], agent_id, "idle", db.now(), int(simulated), info))
            out.append({"id": bid, "target": b["target"]})
    return out


def board_agent(board_id: str | None) -> str | None:
    with db.conn() as c:
        r = c.execute("SELECT agent_id FROM boards WHERE id=?", (board_id,)).fetchone()
    return r[0] if r else None


def heartbeat(agent_id: str, health: dict | None = None):
    """health: {board_id: {"probe": "ok"|..., "temp_c": .., "note": ..}} reported by the agent each poll."""
    with db.conn() as c:
        c.execute("UPDATE boards SET last_seen=? WHERE agent_id=?", (db.now(), agent_id))
        for bid, h in (health or {}).items():
            c.execute("UPDATE boards SET health=? WHERE id=? AND agent_id=?", (db.j(h), bid, agent_id))


def open_runs(job_id: str | None = None, board_id: str | None = None) -> list[dict]:
    """Queued or running board runs of a project and/or on a board."""
    where, args = ["status IN ('queued', 'running')"], []
    if job_id:
        where.append("job_id=?"); args.append(job_id)
    if board_id:
        where.append("board_id=?"); args.append(board_id)
    with db.conn() as c:
        return [db.row(r) for r in c.execute("SELECT id, job_id, board_id, status, updated FROM hil_runs WHERE "
                                             + " AND ".join(where), args)]


def reset_board(board_id: str, force: bool = False):
    """Lab action after fixing a board: fail its open run, clear the failure count, put it back in service.
    A board in the middle of a live run (lease not expired) is only reset with force."""
    live = [r for r in open_runs(board_id=board_id) if r["status"] == "running" and db.now() - r["updated"] < RUN_LEASE_S]
    b = next((x for x in list_boards() if x["id"] == board_id), None)
    if live and not force and (b or {}).get("status") != "quarantined":
        raise ValueError(f"board {board_id} is running {live[0]['id']} right now; wait for it, or reset with force "
                         "to fail that run")
    with db.conn() as c:
        c.execute("UPDATE hil_runs SET status='failed', result=?, updated=? WHERE board_id=? AND status='running'",
                  (db.j({"ok": False, "error": "board reset by the lab"}), db.now(), board_id))
        c.execute("UPDATE boards SET status='idle', fail_count=0 WHERE id=?", (board_id,))


RUN_LEASE_S = 30 * 60


def reap_runs():
    """Runs claimed longer than the lease (agent died, board hung) fail and free their board."""
    cutoff = db.now() - RUN_LEASE_S
    with db.conn() as c:
        stale = [db.row(r) for r in c.execute("SELECT id, board_id FROM hil_runs WHERE status='running' AND updated<?",
                                              (cutoff,))]
        for r in stale:
            c.execute("UPDATE hil_runs SET status='failed', result=?, updated=? WHERE id=?",
                      (db.j({"ok": False, "error": f"no result within {RUN_LEASE_S // 60} min (agent lost?)"}), db.now(), r["id"]))
            c.execute("UPDATE boards SET status='idle' WHERE id=? AND status='busy'", (r["board_id"],))


QUARANTINE_AFTER = 3


def list_boards() -> list[dict]:
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute("SELECT * FROM boards ORDER BY target, name")]
    t = db.now()
    for r in rows:
        r["info"] = db.uj(r["info"], {})
        r["health"] = db.uj(r.get("health"), {})
        r["online"] = t - (r["last_seen"] or 0) < STALE_S
        r["simulated"] = bool(r["simulated"])
        r["kind"] = "emulated" if r["info"].get("emulated") else "simulated" if r["simulated"] else "physical"
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
    """Give the agent the oldest queued run for an idle board it owns whose kind the run's plan allows."""
    reap_runs()
    with db.conn() as c:
        boards = [db.row(b) for b in c.execute("SELECT * FROM boards WHERE agent_id=? AND status='idle'", (agent_id,))]
        for b in boards:
            info = db.uj(b["info"], {})
            kind = "emulated" if info.get("emulated") else "simulated" if b["simulated"] else "physical"
            r = None
            for cand in c.execute("SELECT * FROM hil_runs WHERE status='queued' AND target=? AND (board_id IS NULL OR"
                                  " board_id=?) ORDER BY created LIMIT 20", (b["target"], b["id"])).fetchall():
                allowed = db.uj(cand["request"], {}).get("allowed_kinds")
                if not allowed or kind in allowed:
                    r = cand
                    break
            if r:
                c.execute("UPDATE hil_runs SET status='running', board_id=?, updated=? WHERE id=?", (b["id"], db.now(), r["id"]))
                c.execute("UPDATE boards SET status='busy' WHERE id=?", (b["id"],))
                run = db.row(r)
                run["board_id"] = b["id"]
                run["request"] = db.uj(run["request"], {})
                return run
    return None


RESULT_KEYS = {"ok": bool, "simulated": bool, "emulated": bool, "outputs": list, "cycles": int, "cycles_min": int,
               "cycles_max": int, "latency_ms": float, "stack_painted": int, "energy_uj": float, "active_power_mw": float,
               "sleep_power_uw": float, "board": str,
               "n_vectors": int, "firmware": str, "log": list, "error": str, "config": dict}


def _clean(result: dict) -> dict:
    """Keep known fields with the right types (agents are not trusted with free-form JSON)."""
    out = {}
    for k, typ in RESULT_KEYS.items():
        v = result.get(k)
        if v is None:
            continue
        try:
            if typ is list:
                out[k] = list(v)[:4096] if k == "outputs" else [str(x)[:300] for x in list(v)[:50]]
            elif typ is dict:
                out[k] = {str(a)[:40]: str(b)[:80] for a, b in dict(v).items()}
            elif typ is str:
                out[k] = str(v)[:2000]
            else:
                out[k] = typ(v)
        except (TypeError, ValueError):
            continue
    return out


def complete(rid: str, result: dict, agent_id: str | None = None):
    """Record a run's result. With agent_id (an HTTP agent) the run must be running on one of its boards.
    outputs_match is always computed here, never taken from the agent."""
    run = get_run(rid)
    if not run:
        raise KeyError(rid)
    if agent_id is not None and (run["status"] != "running" or board_agent(run["board_id"]) != agent_id):
        raise PermissionError("this run is not running on one of your boards")
    result = _clean(result)
    req = run["request"]
    expected = req.get("expected_outputs")
    got = result.get("outputs")
    if expected is not None:
        try:
            result["outputs_match"] = got is not None and bool(np.array_equal(np.array(expected), np.array(got, dtype=np.int64)))
        except (TypeError, ValueError):
            result["outputs_match"] = False
    status = "succeeded" if result.get("ok", True) and result.get("outputs_match", expected is None) else "failed"
    with db.conn() as c:
        c.execute("UPDATE hil_runs SET status=?, result=?, updated=? WHERE id=?", (status, db.j(result), db.now(), rid))
        if status == "succeeded":
            c.execute("UPDATE boards SET status='idle', fail_count=0 WHERE id=?", (run["board_id"],))
        else:   # a board that fails QUARANTINE_AFTER runs in a row leaves service until the lab resets it
            c.execute("UPDATE boards SET fail_count=COALESCE(fail_count,0)+1, status=CASE WHEN COALESCE(fail_count,0)+1>=?"
                      " THEN 'quarantined' ELSE 'idle' END WHERE id=?", (QUARANTINE_AFTER, run["board_id"]))
        # calibration and `measured` only from physical boards at the reference clock SAI plans for
        ref_clock = sai_engine.TARGETS.get(run["target"], {}).get("clock_hz")
        clock = (result.get("config") or {}).get("clock")
        real = (not result.get("simulated") and not result.get("emulated")
                and (clock is None or str(clock) == str(ref_clock)))
        if status == "succeeded" and real and result.get("cycles") and req.get("macs"):
            c.execute("INSERT INTO calibration VALUES (?,?,?,?,?,?)",
                      (run["target"], req.get("objective"), int(req["macs"]), int(result["cycles"]), rid, db.now()))
        if status == "succeeded" and real and result.get("cycles") and req.get("plan_digest"):
            c.execute("INSERT INTO measurements VALUES (?,?,?,?,?,?,?)",
                      (run["target"], req["plan_digest"], int(result["cycles"]), result.get("stack_painted"),
                       run["board_id"], rid, db.now()))
    return get_run(rid)


def measurements() -> dict:
    """`target:plan_digest` -> median cycles of real board runs of that exact program (never simulated ones)."""
    with db.conn() as c:
        rows = [db.row(r) for r in c.execute("SELECT * FROM measurements ORDER BY created")]
    out: dict[str, dict] = {}
    for r in rows:
        k = f"{r['target']}:{r['plan_digest']}"
        e = out.setdefault(k, {"samples": [], "board": r["board"], "stack": []})
        e["samples"].append(r["cycles"])
        if r["stack_painted"]:
            e["stack"].append(r["stack_painted"])
        e["board"] = r["board"]
    return {k: {"cycles": float(np.median(v["samples"])), "runs": len(v["samples"]), "board": v["board"],
                "stack_painted": max(v["stack"]) if v["stack"] else None} for k, v in out.items()}


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
class EmulatedAgent(threading.Thread):
    """In-process lab of QEMU boards: the real HIL harness firmware, cross-compiled with SAI's target flags and
    linked with the job's SAI build, runs on an emulated Cortex-M7 (MPS2-AN500) and RV32IMAFC (virt). Outputs are
    checked bit-exactly server side; cycles are QEMU's instruction clock, flagged emulated (never calibration)."""

    AGENT = "emu-lab-01"
    BOARDS = [{"id": "QEMU-M7-01", "name": "QEMU MPS2-AN500 Cortex-M7 (emulated)", "target": "cortex-m7",
               "port": "qemu-mps2-an500", "probe": "qemu", "emulated": True},
              {"id": "QEMU-RV32-01", "name": "QEMU virt RV32IMAFC (emulated)", "target": "md6605",
               "port": "qemu-virt-rv32", "probe": "qemu", "emulated": True}]

    def __init__(self):
        super().__init__(daemon=True, name="emu-agent")
        self.stop = threading.Event()

    @classmethod
    def usable(cls) -> bool:
        from . import hil
        return all(hil.available(b["port"]) for b in cls.BOARDS)

    def run(self):
        register(self.AGENT, self.BOARDS, simulated=False)
        while not self.stop.is_set():
            heartbeat(self.AGENT, {b["id"]: {"probe": "ok", "kind": "qemu"} for b in self.BOARDS})
            run = None
            try:
                run = claim(self.AGENT)
                if run:
                    self.execute(run)
            except Exception as e:                        # keep the lab alive; the run is marked failed
                if run:
                    complete(run["id"], {"ok": False, "error": str(e)[-800:], "emulated": True})
            self.stop.wait(1.0)

    def execute(self, run):
        from . import hil
        req = run["request"]
        port = next(b["port"] for b in self.BOARDS if b["id"] == run["board_id"])
        jobdir = config.DATA_DIR / "jobs" / run["job_id"]
        vec = np.load(jobdir / "hil_vectors.npz")
        fwdir = jobdir / "hil" / run["id"]
        try:
            fw = hil.build(Path(req["build_dir"]), port, vec["x"], fwdir)
            res = hil.run_emulated(fw["elf"], port)
        finally:
            import shutil
            shutil.rmtree(fwdir, ignore_errors=True)              # firmware is rebuilt per run; keep no copies
        complete(run["id"], {
            "ok": True, "emulated": True, "outputs": res["outputs"], "cycles": res["cycles"],
            "cycles_min": res.get("cycles_min"), "cycles_max": res.get("cycles_max"), "latency_ms": None,
            "stack_painted": res["stack_painted"], "board": run["board_id"], "n_vectors": len(res["outputs"]),
            "config": res.get("config") or {},
            "firmware": f"sai-hil-harness 1 ({port}, text {fw['text']} B, bss {fw['bss']} B)",
            "log": [f"built {Path(fw['elf']).name}", f"qemu {port}: DONE {res['status']}",
                    "cycles: QEMU -icount instruction clock (emulated, not latency evidence)"]})


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
