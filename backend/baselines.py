# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Side-by-side: the same INT8 model under SAI and under TensorFlow Lite for Microcontrollers (TFLM), with
CMSIS-NN kernels on Arm Cortex-M and TFLM's reference kernels elsewhere (CMSIS-NN is Arm-only).

SAI's numbers come from the project (exact / linked / proven where SAI says so). The TFLM numbers are a model of
what TFLM would need for this exact graph, computed the way TFLM computes them where that is deterministic:

  * tensor arena: TFLM's GreedyMemoryPlanner replayed on the graph's int8 activations (largest first, first fit
    in time-overlap order, 16-byte alignment), plus the largest CMSIS-NN kernel scratch buffer
    (arm_convolve_s8_get_buffer_size / arm_depthwise_conv_wrapper_s8_get_buffer_size formulas for a DSP core),
    plus persistent per-op data (per-channel output multipliers and shifts, node and eval-tensor structs);
  * flash: int8 weights, int32 biases, the flatbuffer's per-channel scales / zero points (float32 + int64) and
    per-tensor metadata, plus runtime and kernel code (assumed sizes for an -Os Cortex-M build; your linker map
    replaces them);
  * latency: cycles per MAC by kernel class plus per-op interpreter overhead (assumed; a measured run replaces
    them).

Every number carries its evidence. A measured TFLM build (RAM, flash, cycles from the customer's own map file
and board) can be stored per project and is then shown instead of the model."""

from __future__ import annotations

import json

from . import archspec, config, sai_engine

ALIGN = 16
# code sizes for an -Os Cortex-M build (bytes): interpreter, allocator, flatbuffer reader, error reporter, ...
RUNTIME_CODE = 22_000
KERNEL_CODE = {  # per used op type, CMSIS-NN optimized / reference
    "conv": (4_200, 2_600), "dwconv": (3_800, 2_200), "pwconv": (0, 0), "fc": (1_600, 1_100),
    "maxpool": (900, 700), "avgpool": (900, 700), "gap": (1_300, 1_000), "residual": (700, 500),
    "softmax": (1_400, 1_400),
}
PER_OP_PERSISTENT = 96          # TfLiteNode + registration + op data
PER_TENSOR_PERSISTENT = 16      # TfLiteEvalTensor
PER_TENSOR_FLASH = 96           # flatbuffer tensor table entry, shape, name
STACK = 1_200                   # interpreter Invoke() + kernel frames
# effective cycles per MAC (memory stalls, im2col, requantization included) and per-op dispatch overhead.
# Order of magnitude of published TFLM results: MLPerf Tiny KWS DS-CNN (~2.5 M MACs) runs in the order of
# 150-200 ms on an 80 MHz Cortex-M4 with CMSIS-NN (~5-6 cycles/MAC); a Cortex-M7 issues about twice as fast.
CYCLES = {
    "cmsis-nn": {"conv": 2.4, "pwconv": 2.0, "dwconv": 3.5, "fc": 1.8, "residual": 2.4, "pool": 2.0, "op": 1_400},
    "reference": {"conv": 14.0, "pwconv": 12.0, "dwconv": 16.0, "fc": 9.0, "residual": 14.0, "pool": 6.0, "op": 900},
}
ARM_TARGETS = {"cortex-m7"}


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def tensors(spec: dict) -> tuple[list[dict], list[dict]]:
    """Activation tensors with [first, last] op lifetimes, and the op list, as TFLM would see the graph
    (a residual block is Conv -> Conv -> Add, its input alive until the Add)."""
    rows = archspec.shapes(spec)
    inp = spec["input"]
    size0 = inp["channels"] * inp["height"] * inp["width"]
    ts = [{"name": "input", "size": size0, "first": 0, "last": 0}]
    ops = []
    cur = 0                                   # index of the tensor feeding the next op
    shape_in = [inp["channels"], inp["height"], inp["width"]]
    for r in rows:
        op = r["op"]
        if op == "dropout":
            continue
        k = len(ops)
        if op == "residual":
            a = len(ts)
            ts.append({"name": f"res{r['index']}_a", "size": r["activation_bytes"], "first": k, "last": k + 1})
            ts.append({"name": f"res{r['index']}_b", "size": r["activation_bytes"], "first": k + 1, "last": k + 2})
            ts.append({"name": f"t{r['index']}", "size": r["activation_bytes"], "first": k + 2, "last": k + 3})
            ts[cur]["last"] = max(ts[cur]["last"], k + 2)
            for j, (kind, macs) in enumerate((("conv", (r["macs"] - r["activation_bytes"]) // 2),
                                              ("conv", (r["macs"] - r["activation_bytes"]) // 2),
                                              ("add", r["activation_bytes"]))):
                ops.append({"op": kind, "src": "residual", "macs": macs, "in": shape_in, "out": r["shape"],
                            "k": None, "params": r["params"] // 2 if kind == "conv" else 0, "index": r["index"]})
            cur = a + 2
        else:
            ts[cur]["last"] = max(ts[cur]["last"], k)
            ts.append({"name": f"t{r['index']}", "size": r["activation_bytes"], "first": k, "last": k + 1})
            lay = (spec["layers"] + [{"op": "fc"}])[r["index"]] if r["index"] < len(spec["layers"]) + 1 else {}
            ops.append({"op": op, "macs": r["macs"], "in": shape_in, "out": r["shape"], "k": lay.get("k"),
                        "params": r["params"], "index": r["index"], "head": r["head"]})
            cur = len(ts) - 1
        shape_in = r["shape"]
    ts[cur]["last"] = len(ops)                  # the output stays until the application reads it
    return ts, ops


def greedy_plan(ts: list[dict]) -> int:
    """TFLM GreedyMemoryPlanner: place buffers largest first at the lowest offset that does not overlap (in time
    and in memory) any buffer already placed. Returns the arena high-water mark."""
    placed: list[tuple[int, int, int, int]] = []           # offset, size, first, last
    for t in sorted(ts, key=lambda x: -x["size"]):
        size = _align(t["size"])
        live = sorted((o, s) for o, s, f, la in placed if not (la < t["first"] or f > t["last"]))
        off = 0
        for o, s in live:
            if off + size <= o:
                break
            off = max(off, o + s)
        placed.append((off, size, t["first"], t["last"]))
    return max((o + s for o, s, _, _ in placed), default=0)


def scratch(op: dict) -> int:
    """CMSIS-NN scratch buffer for a DSP core (bytes)."""
    cin = op["in"][0] if op.get("in") else 0
    if op["op"] in ("fc", "maxpool", "avgpool", "gap", "add", "pwconv"):
        return 0
    k = op.get("k") or 1
    kk = k * k if (op.get("in") and len(op["in"]) > 1 and op["in"][1] > 1) else k
    if op["op"] == "conv" and op.get("src") != "residual":
        return 2 * cin * kk * 2
    if op["op"] == "conv":
        return 2 * cin * 9 * 2
    if op["op"] == "dwconv":
        return cin * kk * 2 if k != 3 else 0                  # 3x3 has a dedicated kernel without scratch
    return 0


def tflm(spec: dict, target: str) -> dict:
    kernels = "cmsis-nn" if target in ARM_TARGETS else "reference"
    ts, ops = tensors(spec)
    arena_act = greedy_plan(ts)
    scr = max([scratch(o) for o in ops] + [0]) if kernels == "cmsis-nn" else 0
    out_ch = sum(o["out"][0] for o in ops if o["op"] in ("conv", "dwconv", "pwconv", "fc"))
    persistent = len(ops) * PER_OP_PERSISTENT + len(ts) * PER_TENSOR_PERSISTENT + out_ch * 8
    arena = _align(arena_act) + _align(scr) + _align(persistent)
    weights = sum(o["params"] for o in ops)
    biases = sum(o["out"][0] for o in ops if o["op"] in ("conv", "dwconv", "pwconv", "fc"))
    flat_meta = out_ch * 12 + (len(ts) + len(ops) * 2) * PER_TENSOR_FLASH
    used = sorted({("pwconv" if o["op"] == "pwconv" else o["op"]) for o in ops} - {"add"}) + (["residual"] if any(
        o.get("src") == "residual" for o in ops) else [])
    code = RUNTIME_CODE + sum(KERNEL_CODE.get(u, (1_000, 800))[0 if kernels == "cmsis-nn" else 1] for u in set(used))
    flash = weights + biases * 3 + flat_meta + code        # int32 bias = weight byte + 3
    cyc = CYCLES[kernels]
    cycles = 0
    for o in ops:
        cls = "pool" if o["op"] in ("maxpool", "avgpool", "gap", "add") else o["op"]
        cycles += o["macs"] * cyc.get(cls, cyc["conv"]) + cyc["op"]
    clock = sai_engine.TARGETS.get(target, {}).get("clock_hz") or 1
    return {"runtime": f"TFLM + {'CMSIS-NN' if kernels == 'cmsis-nn' else 'reference kernels'}", "kernels": kernels,
            "arena": arena, "arena_parts": {"activations": _align(arena_act), "scratch": _align(scr),
                                            "persistent": _align(persistent)},
            "stack": STACK, "ram": arena + STACK, "flash": int(flash),
            "flash_parts": {"weights_and_biases": weights + biases * 3, "flatbuffer_metadata": flat_meta, "code": code},
            "cycles": int(cycles), "latency_ms": round(cycles / clock * 1000, 4), "ops": len(ops),
            "evidence": {"arena": "replayed", "flash": "assumed", "latency": "assumed"},
            "note": None if kernels == "cmsis-nn" else "CMSIS-NN targets Arm Cortex-M only; this board runs TFLM's "
                                                       "portable reference kernels"}


def _measured_path(job_id: str):
    return config.DATA_DIR / "jobs" / job_id / "baseline_measured.json"


def save_measured(job_id: str, target: str, m: dict) -> dict:
    p = _measured_path(job_id)
    cur = json.loads(p.read_text()) if p.exists() else {}
    clean = {k: (float(m[k]) if m.get(k) not in (None, "") else None) for k in ("ram", "flash", "latency_ms", "cycles")}
    clean["runtime"] = str(m.get("runtime") or "TFLM")[:60]
    clean["note"] = str(m.get("note") or "")[:300]
    cur[target] = clean
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, indent=1))
    return cur


def compare(res: dict, job_id: str | None = None, targets: list[str] | None = None) -> dict:
    """SAI (the project's numbers) vs TFLM for the selected model, on its board and the other MCU boards."""
    sel = res["selected"]
    v = next(x for x in res["variants"] if x["size"] == sel["size"])
    spec = archspec.normalize(v["spec"])
    if spec.get("layout") == "sequence":
        raise ValueError("TFLM comparison is for SAI Core (CNN) models; TFLM's LSTM support is a different runtime path")
    measured = {}
    if job_id and _measured_path(job_id).exists():
        measured = json.loads(_measured_path(job_id).read_text())
    rows = []
    for t in targets or sai_engine.MCU_TARGETS:
        cands = [c for c in v["plans"]["candidates"] if c["target"] == t and c.get("fits", True)] if v.get("plans") else []
        best = sel if t == sel["target"] else (min(cands, key=lambda c: c["ram"]) if cands else None)
        tf = tflm(spec, t)
        budget = sai_engine.TARGETS[t]
        row = {"target": t, "board": budget["name"], "ram_total": budget.get("ram_total"),
               "flash_total": budget.get("flash_total"), "tflm": tf, "measured": measured.get(t)}
        if best:
            row["sai"] = {"ram": best["ram"], "flash": best["flash"], "latency_ms": (best.get("latency") or {}).get("ms"),
                          "workspace": best.get("workspace"), "stack": best.get("stack"),
                          "evidence": {"ram": best.get("workspace_evidence"), "flash": best.get("flash_evidence"),
                                       "latency": (best.get("latency") or {}).get("evidence")},
                          "schedule": best.get("schedule")}
            ref = row["measured"] or tf
            row["ratio"] = {k: round(ref[k] / row["sai"][k], 2) if ref.get(k) and row["sai"].get(k) else None
                            for k in ("ram", "flash", "latency_ms")}
        row["tflm_fits"] = (tf["ram"] <= (budget.get("ram_total") or 1e12)) and (tf["flash"] <= (budget.get("flash_total") or 1e12))
        rows.append(row)
    return {"rows": rows, "selected_target": sel["target"], "size": sel["size"],
            "method": "SAI: the project's compiled numbers. TFLM: arena replayed with TFLM's greedy memory planner on "
                      "this graph + CMSIS-NN scratch + persistent op data; flash and latency from assumed code sizes "
                      "and cycles/MAC (replace with a measured build)."}
