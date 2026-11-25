"""The platform's only door into the SAI compiler: the supported `sai` command, run as a subprocess with JSON output.

Nothing here imports compiler internals, so the platform tracks SAI releases through the CLI contract
(`sai estimate`, `sai explore`, `sai compile`, `sai verify`, `sai inspect`, `sai targets`).

Evidence discipline: SAI labels every number (static_exact, linked_exact, proven_upper_bound, measured, estimate,
unavailable). The platform keeps those labels and adds its own, clearly separated:
  * "sai_cost_model"  - SAI's kernel cost model rank score on a measured kernel catalog (uncalibrated cycles);
  * "heuristic"       - MACs x cycles-per-MAC for the target/plan family, a planning number only;
  * "calibrated"      - the same fit after board-farm measurements of similar programs;
  * "measured"        - a board run of this exact build.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from . import config

# Board budgets come from the compiler's own profiles (profiles/*.json); clocks from the measured sessions.
TARGETS = {
    "md6605": {
        "id": "md6605", "name": "Sanken MD6605", "core": "RV32IMAFC", "clock_hz": 66_666_666,
        "flash_total": 131072, "ram_total": 8192,
        "budget": {"flash": 65536, "workspace": 4096, "stack": 2048},
        "notes": "8 KB SRAM power-control MCU. AI arena 4 KB, AI ROM 64 KB (half of ReRAM). Softmax forbidden.",
        "status": "available",
    },
    "cortex-m7": {
        "id": "cortex-m7", "name": "STM32H753 (Cortex-M7)", "core": "Cortex-M7 + DSP ext.", "clock_hz": 480_000_000,
        "flash_total": 2097152, "ram_total": 524288,
        "budget": {"flash": 1048576, "workspace": 262144, "stack": 4096},
        "notes": "NUCLEO-H753ZI reference. SMLAD/SXTB16 kernels in SAI Professional.",
        "status": "available",
    },
    "host-cpu": {
        "id": "host-cpu", "name": "Host CPU (x86-64/arm64)", "core": "hosted C", "clock_hz": None,
        "flash_total": None, "ram_total": None, "budget": {}, "notes": "Functional reference and exact accuracy check.",
        "status": "reference",
    },
}
ROADMAP_TARGETS = [
    {"id": "cortex-m4", "name": "Cortex-M4F (STM32L4/G4)", "status": "roadmap"},
    {"id": "cortex-m33", "name": "Cortex-M33 (nRF54, STM32U5)", "status": "roadmap"},
    {"id": "cortex-m55", "name": "Cortex-M55 + Ethos-U55", "status": "roadmap"},
    {"id": "esp32-s3", "name": "ESP32-S3 (Xtensa LX7)", "status": "roadmap"},
    {"id": "rv32-generic", "name": "Generic RV32IMC", "status": "roadmap"},
]
MCU_TARGETS = ("md6605", "cortex-m7")
OBJECTIVES = ("fast", "min-ram", "min-code")

# Planning heuristic, cycles per MAC by target and plan family (replaced by calibration when boards report).
# cortex-m7 fast ~1.0: MLPerf Tiny ResNet-8 (12.5 M MACs) measured 12.96 M cycles on the H753 (2026-11-12 session).
CYCLES_PER_MAC = {
    ("cortex-m7", "fast"): 1.05, ("cortex-m7", "min-ram"): 1.6, ("cortex-m7", "min-code"): 3.2,
    ("md6605", "fast"): 3.0, ("md6605", "min-ram"): 4.0, ("md6605", "min-code"): 6.0,
}
CALL_OVERHEAD_CYCLES = 1500


class SaiFailure(RuntimeError):
    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message)
        self.detail = detail or {}


def _env() -> dict:
    env = dict(os.environ)
    if config.SAI_MD6605_GCC:
        env["SAI_MD6605_GCC"] = config.SAI_MD6605_GCC
    if config.SAI_CORTEX_M7_GCC:
        env["SAI_CORTEX_M7_GCC"] = config.SAI_CORTEX_M7_GCC
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run(args: list[str], timeout: int = 300) -> dict:
    proc = subprocess.run([config.SAI_BIN, *args, "--format", "json"], capture_output=True, text=True,
                          timeout=timeout, env=_env(), cwd=config.WORK_DIR)
    try:
        data = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        data = {}
    if "error" in data:
        err = data["error"]
        raise SaiFailure(f"{err.get('code')}: {err.get('reason')}", err)
    if proc.returncode not in (0,) and not data:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        raise SaiFailure("sai " + args[0] + " failed: " + " | ".join(tail), {"exit": proc.returncode})
    return data


def version() -> dict:
    out = subprocess.run([config.SAI_BIN, "version"], capture_output=True, text=True, env=_env())
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"sai": "unknown"}


def target_status() -> list[dict]:
    try:
        caps = run(["targets"])
        live = {t["id"]: t for t in caps.get("targets", [])}
    except Exception:          # the platform still renders if the CLI is missing
        live = {}
    rows = []
    for t in TARGETS.values():
        row = dict(t)
        info = live.get(t["id"], {})
        row["toolchain"] = (info.get("toolchain") or {}).get("status", "unknown") if isinstance(info, dict) else "unknown"
        rows.append(row)
    return rows + ROADMAP_TARGETS


def inspect(model: Path) -> dict:
    return run(["inspect", str(model)])


def estimate(model: Path, target: str, objective: str, tier: str | None = None) -> dict:
    return run(["estimate", str(model), "--target", target, "--objective", objective,
                *(["--tier", tier] if tier else [])])


def tier_compare(model: Path, targets: list[str]) -> dict:
    """SAI Standard (tier core) vs Professional (pro) FAST plan per target: the edition upsell, from SAI itself.
    Standard is estimated (SAI builds MCU packages at the Professional tier today)."""
    out = {}
    for t in targets:
        if t not in MCU_TARGETS:
            continue
        row = {}
        for tier, key in (("core", "standard"), ("pro", "professional")):
            try:
                e = estimate(model, t, "fast", tier)
            except SaiFailure as err:
                row[key] = {"error": str(err)}
                continue
            res = e.get("resources", {})
            row[key] = {"schedule": ((e.get("provenance") or {}).get("plan") or {}).get("schedule"),
                        "workspace": (res.get("tensor_workspace") or {}).get("value"),
                        "flash": (res.get("linked_sai_flash") or {}).get("value"),
                        "stack": (res.get("stack") or {}).get("value")}
        out[t] = row
    return out


def explore(model: Path, target: str, spatial_axes: int, max_seconds: int = 20, top: int = 25) -> dict | None:
    """SAI decides the lowering (a strided 1-D graph lowers on the 2-D path), so try the expected mode first and
    switch when SAI says the graph has the other number of axes."""
    args = ["explore", str(model), "--target", target, "--max-seconds", str(max_seconds), "--top", str(top)]
    modes = [True, False] if spatial_axes == 1 else [False, True]
    last = None
    for one_axis in modes:
        try:
            return run(args + (["--one-axis"] if one_axis else []), timeout=max_seconds + 120)
        except SaiFailure as e:
            last = e
            if e.detail.get("code") not in ("explore.two_axes", "explore.one_axis"):
                break
    return {"unavailable": str(last)}


def compile_build(model: Path, target: str, objective: str, out: Path) -> dict:
    return run(["compile", str(model), "--target", target, "--output", str(out), "--objective", objective], timeout=600)


def verify(build: Path, model: Path) -> dict:
    return run(["verify", str(build), "--model", str(model)], timeout=600)


# ------------------------------------------------------------------------------------------------------------------
def _val(resources: dict, key: str):
    r = resources.get(key) or {}
    return r.get("value"), r.get("evidence", "unavailable")


def latency(target: str, objective: str, macs: int, calibration: dict | None, cost_scale: float | None) -> dict:
    """Cycles and milliseconds with an explicit evidence label."""
    clock = TARGETS[target]["clock_hz"]
    cal = (calibration or {}).get(f"{target}:{objective}")
    if cal and cal.get("samples", 0) >= 2:
        cycles, evidence = macs * cal["cycles_per_mac"] + cal.get("overhead", CALL_OVERHEAD_CYCLES), "calibrated"
        basis = f"fit of {cal['samples']} board measurements"
    elif cost_scale:
        cpm = CYCLES_PER_MAC.get((target, objective), 4.0) * cost_scale
        cycles, evidence = macs * cpm + CALL_OVERHEAD_CYCLES, "sai_cost_model"
        basis = f"heuristic scaled x{cost_scale:.2f} to SAI's kernel cost model for this graph (uncalibrated)"
    else:
        cpm = CYCLES_PER_MAC.get((target, objective), 4.0)
        cycles, evidence = macs * cpm + CALL_OVERHEAD_CYCLES, "heuristic"
        basis = f"{macs:,} MACs x {cpm} cycles/MAC (planning heuristic)"
    return {"cycles": int(cycles), "ms": round(cycles / clock * 1000, 4) if clock else None, "evidence": evidence,
            "basis": basis, "clock_hz": clock}


def plan_search(model: Path, macs: int, spatial_axes: int, targets: list[str], goal: str = "balanced",
                budgets: dict | None = None, calibration: dict | None = None, explore_seconds: int = 15,
                log=lambda m: None) -> dict:
    """Ask SAI for its plan under every objective on every target, add the explored plan space, check board and
    user budgets, and recommend one plan for the user's goal.

    goal: fastest | min-ram | min-flash | balanced. budgets: {max_ram, max_flash, max_latency_ms} (optional)."""
    budgets = {k: v for k, v in (budgets or {}).items() if v not in (None, "", 0)}
    candidates, spaces, errors = [], {}, []
    for target in targets:
        if target not in MCU_TARGETS:
            continue
        log(f"plan search on {target}: exploring plan space")
        ex = explore(model, target, spatial_axes, max_seconds=explore_seconds)
        cost_scale = None
        space = []
        if ex and "ranking" in ex:
            for r in ex["ranking"]:
                rs = r.get("rank_score") or {}
                space.append({"name": r.get("name"), "workspace": (r.get("workspace") or {}).get("value"),
                              "stack_estimate": (r.get("stack_estimate") or {}).get("value"),
                              "rank_score": rs.get("value"), "basis": rs.get("basis")})
            kernel_model = [s for s in space if s["basis"] and s["basis"] != "MAC count" and s["rank_score"]]
            if kernel_model:
                # one per-graph scale: SAI's best FAST rank score over the FAST heuristic, applied to every plan
                best = min(s["rank_score"] for s in kernel_model)
                heur = macs * CYCLES_PER_MAC.get((target, "fast"), 4.0) + CALL_OVERHEAD_CYCLES
                cost_scale = max(best / heur, 0.05)
            spaces[target] = {"evaluated": (ex.get("search") or {}).get("evaluated"),
                              "complete": (ex.get("search") or {}).get("complete"), "plans": space[:40],
                              "measured_frontier": ex.get("measured_frontier", [])}
        elif ex:
            spaces[target] = {"unavailable": ex.get("unavailable")}
        for objective in OBJECTIVES:
            log(f"plan search on {target}: sai estimate --objective {objective}")
            try:
                est = estimate(model, target, objective)
            except SaiFailure as e:
                errors.append({"target": target, "objective": objective, "error": str(e)})
                continue
            res = est.get("resources", {})
            ws, ws_ev = _val(res, "tensor_workspace")
            flash, flash_ev = _val(res, "linked_sai_flash")
            stack, stack_ev = _val(res, "stack")
            sram, sram_ev = _val(res, "static_ram")
            if stack is None:
                guess = [s["stack_estimate"] for s in space if s.get("stack_estimate")]
                stack, stack_ev = (max(guess), "estimate") if guess else (512, "assumed")
            measured = (est.get("metrics") or {}).get("latency_cycles") or {}
            if measured.get("value"):
                clock = TARGETS[target]["clock_hz"]
                lat = {"cycles": measured["value"], "ms": round(measured["value"] / clock * 1000, 4),
                       "evidence": measured.get("evidence", "measured"), "basis": "SAI measured row", "clock_hz": clock}
            else:
                lat = latency(target, objective, macs, calibration, cost_scale)
            plan = (est.get("provenance") or {}).get("plan") or {}
            cand = {
                "id": f"{target}:{objective}", "target": target, "objective": objective,
                "schedule": plan.get("schedule"), "tier": plan.get("tier"), "program_id": plan.get("program_id"),
                "workspace": ws, "workspace_evidence": ws_ev, "stack": stack, "stack_evidence": stack_ev,
                "static_ram": sram or 0, "static_ram_evidence": sram_ev,
                "ram": (ws or 0) + (stack or 0) + (sram or 0), "flash": flash, "flash_evidence": flash_ev,
                "latency": lat, "plan_digest": est.get("plan_digest"), "status": est.get("status"),
            }
            cand["violations"] = _violations(cand, budgets)
            cand["fits"] = not cand["violations"]
            candidates.append(cand)
    # the same program can be chosen by two objectives; keep one row, remember every objective that chose it
    uniq: dict[tuple, dict] = {}
    for c in candidates:
        key = (c["target"], c["schedule"], c["workspace"], c["flash"])
        if key in uniq:
            uniq[key]["chosen_by"].append(c["objective"])
        else:
            c["chosen_by"] = [c["objective"]]
            uniq[key] = c
    rows = list(uniq.values())
    feasible = [c for c in rows if c["fits"]]
    rec = _pick(feasible, goal) if feasible else None
    for c in rows:
        c["recommended"] = rec is not None and c["id"] == rec["id"]
    return {"goal": goal, "budgets": budgets, "candidates": rows, "recommended": rec, "spaces": spaces,
            "errors": errors, "feasible": len(feasible),
            "advice": None if rec else _advice(rows, budgets)}


def _violations(c: dict, budgets: dict) -> list[str]:
    v = []
    b = TARGETS[c["target"]]["budget"]
    if c["workspace"] is not None and c["workspace"] + c["static_ram"] > b.get("workspace", 1 << 62):
        v.append(f"workspace {c['workspace'] + c['static_ram']:,} B > board AI arena {b['workspace']:,} B")
    if c["stack"] is not None and c["stack"] > b.get("stack", 1 << 62):
        v.append(f"stack {c['stack']:,} B > board stack reserve {b['stack']:,} B")
    if c["flash"] is not None and c["flash"] > b.get("flash", 1 << 62):
        v.append(f"flash {c['flash']:,} B > board AI ROM budget {b['flash']:,} B")
    if budgets.get("max_ram") and c["ram"] > float(budgets["max_ram"]):
        v.append(f"RAM {c['ram']:,} B > your limit {int(budgets['max_ram']):,} B")
    if budgets.get("max_flash") and c["flash"] is not None and c["flash"] > float(budgets["max_flash"]):
        v.append(f"flash {c['flash']:,} B > your limit {int(budgets['max_flash']):,} B")
    if budgets.get("max_latency_ms") and c["latency"]["ms"] is not None and c["latency"]["ms"] > float(budgets["max_latency_ms"]):
        v.append(f"latency {c['latency']['ms']} ms > your limit {budgets['max_latency_ms']} ms")
    return v


def _pick(rows: list[dict], goal: str) -> dict:
    flash = lambda c: c["flash"] if c["flash"] is not None else 1 << 40  # noqa: E731
    if goal == "fastest":
        return min(rows, key=lambda c: (c["latency"]["ms"] or 1e9, c["ram"]))
    if goal == "min-ram":
        return min(rows, key=lambda c: (c["ram"], c["latency"]["ms"] or 1e9))
    if goal == "min-flash":
        return min(rows, key=lambda c: (flash(c), c["ram"]))
    lo_ms = min(c["latency"]["ms"] or 1e9 for c in rows) or 1e-9
    lo_ram = min(c["ram"] for c in rows) or 1
    lo_fl = min(flash(c) for c in rows) or 1
    return min(rows, key=lambda c: (c["latency"]["ms"] or 1e9) / lo_ms + c["ram"] / lo_ram + flash(c) / lo_fl)


def _advice(rows: list[dict], budgets: dict) -> str:
    if not rows:
        return "SAI produced no plan for the selected targets; see the errors."
    best = min(rows, key=lambda c: len(c["violations"]))
    return ("No plan fits. Closest: " + f"{best['target']} {best['objective']} ({'; '.join(best['violations'])}). "
            "Try a smaller model size, a shorter input window, fewer channels, or a larger board.")
