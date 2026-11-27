# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Sequence models (LSTM, GRU, SimpleRNN, attention, LayerNorm) through SAI's unified front end.

SAI Core compiles CNNs; recurrent and attention models go through `sai_front` (branch product/sai-unified of the SAI
repository), which quantizes the float model itself (SAI integer contract) and routes it to the XModels backend.
The platform trains in PyTorch, exports the float ONNX graph in the [N, T, F] layout (archspec.to_sequence) and runs
`bin/sai-front compile` per board, in place from $SAI_REPO, through its command line only.

The front end's own README states its output is not qualified yet (EVALUATION_ONLY, verified on the host and on
QEMU, not on a board); packages made this way say so.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import config, sai_engine

LAUNCHER = str(config.ROOT / "bin" / "sai-front")
GOAL_OBJECTIVE = {"fastest": "fast", "min-ram": "min-ram", "min-flash": "balanced", "balanced": "balanced",
                  "accuracy": "balanced"}


def available() -> bool:
    return bool(config.SAI_REPO) and (Path(config.SAI_REPO) / "sai_front" / "cli.py").is_file() \
        and (Path(config.SAI_REPO) / "extensions" / "sai_xmodels").is_dir()


def why_unavailable() -> str:
    return ("sequence models (LSTM, GRU, RNN, attention) need SAI's unified front end: set SAI_REPO to a checkout of "
            "the SAI repository's product/sai-unified branch")


def export_float(net, spec: dict, path: Path):
    """Float ONNX of a sequence SpecNet with input [1, T, F] (what sai_front imports)."""
    import torch

    from .archspec import seq_dims
    t, f = seq_dims(spec["input"])
    net.eval()
    torch.onnx.export(net, torch.zeros(1, t, f), str(path), input_names=["x"], output_names=["y"], opset_version=17,
                      dynamo=False)


def write_data(path: Path, x: np.ndarray, y: np.ndarray | None, task: str):
    d = {"x": np.ascontiguousarray(x, dtype=np.float32)}
    if y is not None and task == "classification":
        d["y"] = np.asarray(y, dtype=np.int64)
    np.savez(path, **d)


def compile(model: Path, target: str, out: Path, data: Path, test: Path | None, objective: str,
            ram_limit: int | None = None, timeout: int = 1200) -> dict:
    """Run the front end for one board; returns {report, compile_report, library}. Raises sai_engine.SaiFailure."""
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [LAUNCHER, "compile", str(model), "--target", target, "--output", str(out), "--data", str(data),
           "--objective", objective, "--format", "json"]
    if test:
        cmd += ["--test-data", str(test)]
    if ram_limit:
        cmd += ["--ram-limit", str(int(ram_limit))]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=sai_engine._env(), cwd=out.parent)
    rep_path = out / "sai-front-report.json"
    if p.returncode != 0 or not rep_path.exists():
        txt = (p.stdout or "") + (p.stderr or "")
        reason = txt.strip().splitlines()[-1] if txt.strip() else f"exit {p.returncode}"
        from . import opgaps
        try:
            err = json.loads(txt[txt.index("{"):]).get("error") or {}
            reason = err.get("reason") or err.get("message") or reason
            if err.get("code") == "graph.unsupported" or err.get("status") == "unsupported":
                opgaps.from_sai_error({**err, "target": target}, ["compile", str(model)])
            else:
                opgaps.from_text(txt, "sai-front", target, model)
        except (ValueError, AttributeError):
            opgaps.from_text(txt, "sai-front", target, model)
        raise sai_engine.SaiFailure(f"sai_front on {target}: {reason}", {"stdout": txt[-2000:]})
    rep = json.loads(rep_path.read_text())
    lib = out / "library"
    cr = json.loads((lib / "compile-report.json").read_text()) if (lib / "compile-report.json").exists() else {}
    return {"report": rep, "compile_report": cr, "library": str(lib), "dir": str(out)}


def _flash(cr: dict) -> tuple[int | None, str]:
    code = cr.get("code") or {}
    for key, label in (("rv32_object", "RV32 object"), ("cortex_m7_object", "Cortex-M7 object"),
                       ("host_object", "host object")):
        o = code.get(key)
        if o:
            return int(o.get("text", o.get(".text", 0)) + o.get("rodata", o.get(".rodata", 0))
                       + o.get(".data", 0)), f"code + constants ({label})"
    mem = cr.get("memory") or {}
    const = sum((mem.get("constant_bytes") or {}).values()) if mem.get("constant_bytes") else None
    return const, "constants only (no object size)"


def row(res: dict, target: str, budgets: dict) -> dict:
    """One plan row in the platform's format (sai_engine.plan_search candidates) from a front-end build."""
    rep, cr = res["report"], res["compile_report"]
    mem, dev = cr.get("memory") or {}, cr.get("device_ram") or {}
    ws = int(mem.get("workspace_bytes") or (rep.get("backend") or {}).get("workspace_bytes") or 0)
    io = int((mem.get("input_bytes") or 0) + (mem.get("output_bytes") or 0))
    stack = dev.get("stack_static_bound")
    flash, flash_ev = _flash(cr)
    plan = cr.get("plan") or {}
    sel = next((c for c in plan.get("measured_shortlist") or [] if c.get("label") == plan.get("selected")), None)
    instr = (sel or {}).get("instructions") or (rep.get("backend") or {}).get("cortex_m7_instructions")
    clock = sai_engine.TARGETS[target]["clock_hz"]
    lat = {"cycles": int(instr) if instr else None, "ms": round(instr / clock * 1000, 4) if instr and clock else None,
           "evidence": "qemu_instructions" if instr else "unavailable",
           "basis": (f"{int(instr):,} instructions on QEMU ({(sel or {}).get('isa', 'cortex-m7')}), counted as cycles"
                     if instr else "not measured (QEMU not available)"), "clock_hz": clock}
    c = {"id": f"{target}:xmodels", "target": target, "objective": "xmodels:" + (rep.get("objective") or "balanced"),
         "schedule": (rep.get("backend") or {}).get("plan") or plan.get("selected"), "tier": "SAI Professional (XModels)",
         "backend": "xmodels", "program_id": None,
         "workspace": ws, "workspace_evidence": "exact", "static_ram": io, "static_ram_evidence": "exact (input + output)",
         "stack": stack if stack is not None else 512, "stack_evidence": "proven bound" if stack is not None else "assumed",
         "ram": ws + io + (stack if stack is not None else 512), "flash": flash, "flash_evidence": flash_ev,
         "latency": lat, "plan_digest": None, "status": "evaluation_only", "chosen_by": ["xmodels"],
         "accuracy": rep.get("accuracy"), "verification": cr.get("verification"), "build_dir": res["dir"]}
    c["violations"] = sai_engine._violations(c, budgets or {})
    c["fits"] = not c["violations"]
    return c


def plans(model: Path, targets, goal: str, budgets: dict | None, workdir: Path, data: Path, test: Path | None,
          log=lambda m: None) -> dict:
    """Front-end builds on every board, in the shape of sai_engine.plan_search's result."""
    budgets = {k: v for k, v in (budgets or {}).items() if v not in (None, "", 0)}
    rows, errors = [], []
    for t in targets:
        b = sai_engine.TARGETS[t]["budget"]
        limit = int(budgets.get("max_ram") or 0) or b.get("workspace")
        log(f"sai_front on {t}: XModels, objective {GOAL_OBJECTIVE.get(goal, 'balanced')}, RAM limit {limit} B")
        try:
            res = compile(model, t, workdir / f"xm_{t}", data, test, GOAL_OBJECTIVE.get(goal, "balanced"), limit)
        except sai_engine.SaiFailure as e:
            errors.append({"target": t, "objective": "xmodels", "error": str(e)})
            log(f"sai_front on {t}: {e}")
            continue
        r = row(res, t, budgets)
        acc = r["accuracy"] or {}
        log(f"sai_front on {t}: plan {r['schedule']}, workspace {r['workspace']} B, RAM {r['ram']} B, flash "
            f"{r['flash']} B, {r['latency']['basis']}; integer top-1 agreement {acc.get('top1_agreement')}")
        rows.append(r)
    feasible = [r for r in rows if r["fits"]]
    pick_goal = goal if goal in ("fastest", "min-ram", "min-flash") else "balanced"
    rec = sai_engine._pick(feasible, pick_goal) if feasible else None
    for r in rows:
        r["recommended"] = rec is not None and r is rec
    return {"goal": goal, "budgets": budgets, "candidates": rows, "recommended": rec, "spaces": {}, "errors": errors,
            "feasible": len(feasible), "backend": "xmodels",
            "advice": None if rec else (sai_engine._advice(rows, budgets) if rows else
                                        "; ".join(e["error"] for e in errors) or "no build")}


# ---- bit-exact evaluation of the generated C on the host --------------------------------------------------------
HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include "sai_xm_model.h"
static uint8_t ws[SAI_XM_WORKSPACE_BYTES + 16] __attribute__((aligned(16)));
int main(int argc, char **argv) {
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    int8_t in[SAI_XM_INPUT_BYTES], out[SAI_XM_OUTPUT_BYTES];
    if (!fi || !fo) return 2;
    while (fread(in, 1, SAI_XM_INPUT_BYTES, fi) == SAI_XM_INPUT_BYTES) {
        if (sai_xm_invoke(in, out, ws)) return 3;
        fwrite(out, 1, SAI_XM_OUTPUT_BYTES, fo);
    }
    fclose(fi); fclose(fo); return 0;
}
"""


def io_quant(library: Path) -> dict:
    """Input / output quantization of a front-end library (from sai_xm_model.h)."""
    import re
    h = (Path(library) / "sai_xm_model.h").read_text()
    g = lambda k: float(re.search(rf"#define SAI_XM_{k} (\S+)", h).group(1).rstrip("u"))  # noqa: E731
    return {"input": {"scale": g("INPUT_SCALE"), "zero_point": int(g("INPUT_ZERO_POINT"))},
            "output": {"scale": g("OUTPUT_SCALE"), "zero_point": int(g("OUTPUT_ZERO_POINT"))},
            "input_bytes": int(g("INPUT_BYTES")), "output_bytes": int(g("OUTPUT_BYTES")),
            "workspace_bytes": int(g("WORKSPACE_BYTES"))}


def run_exact(library: Path, x_seq: np.ndarray, work: Path) -> tuple[np.ndarray, dict]:
    """Compile the library with the host C compiler and run every input through sai_xm_invoke.
    Returns (dequantized outputs [N, out], quantization)."""
    q = io_quant(library)
    work.mkdir(parents=True, exist_ok=True)
    (work / "harness.c").write_text(HARNESS)
    exe = work / "xm_eval"
    src = [str(Path(library) / "sai_xm_model.c"), str(work / "harness.c")]
    p = subprocess.run([config.HOST_CC, "-O2", "-std=c99", "-I", str(library), *src, "-lm", "-o", str(exe)],
                       capture_output=True, text=True)
    if p.returncode:
        raise sai_engine.SaiFailure("host build of the XModels library failed", {"stderr": p.stderr[-2000:]})
    s, z = q["input"]["scale"], q["input"]["zero_point"]
    xq = np.clip(np.rint(x_seq.reshape(len(x_seq), -1) / s) + z, -128, 127).astype(np.int8)
    assert xq.shape[1] == q["input_bytes"], (xq.shape, q["input_bytes"])
    xq.tofile(work / "in.bin")
    r = subprocess.run([str(exe), str(work / "in.bin"), str(work / "out.bin")], capture_output=True, text=True)
    if r.returncode:
        raise sai_engine.SaiFailure(f"XModels library run failed (exit {r.returncode})")
    oq = np.fromfile(work / "out.bin", dtype=np.int8).reshape(len(x_seq), q["output_bytes"])
    return (oq.astype(np.float32) - q["output"]["zero_point"]) * q["output"]["scale"], q


EXAMPLE_MAIN = r"""/* Generated by SAI Platform: calling the sequence model (SAI front end, XModels backend). */
#include <stdint.h>
#include <stdio.h>
#include "sai_xm_model.h"
#include "sai_app_prepare.h"
#include "sai_app.h"

static uint8_t workspace[SAI_XM_WORKSPACE_BYTES] __attribute__((aligned(SAI_XM_WORKSPACE_ALIGN)));

int main(void) {
    static float raw[SAI_APP_RAW_CHANNELS * SAI_APP_RAW_SAMPLES];  /* one window from your sensor, channel-major */
    static int8_t in[SAI_XM_INPUT_BYTES];
    int8_t out[SAI_XM_OUTPUT_BYTES];
    /* fill raw[] from the sensor here */
    if (sai_app_prepare(raw, in)) return 1;          /* preprocessing + [time][features] layout + INT8 quantization */
    if (sai_xm_invoke(in, out, workspace)) return 2;
    for (unsigned i = 0; i < SAI_XM_OUTPUT_BYTES; i++)
        printf("%u: %f\n", i, ((int)out[i] - SAI_XM_OUTPUT_ZERO_POINT) * SAI_XM_OUTPUT_SCALE);
    return 0;
}
"""
