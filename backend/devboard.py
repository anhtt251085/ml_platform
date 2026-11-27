# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Virtual dev board: try a finished project's package on an emulated board with your own inputs, from the
browser, before any hardware is on the desk.

What runs is the real thing: the HIL harness firmware cross-compiled with SAI's target flags and linked with the
project's SAI library for that board, booted in QEMU (MPS2-AN500 Cortex-M7, or virt RV32IMAFC for the MD6605
class). The inputs are the user's: windows of the project's dataset, a stretch of a recording, or values pasted
in; they go through the package's own preprocessing (normalization, INT8 quantization) first. The session shows
what the board printed on its UART, its int8 outputs and the decoded prediction per window, the expected outputs
of the bit-exact host program (they must match), and QEMU's instruction count (emulated: it shows the program
runs on that ISA, not how fast a real board is)."""

from __future__ import annotations

import shutil

import numpy as np

from . import config, data, exact_eval, hil, inference, sai_engine

PORT = {"cortex-m7": "qemu-mps2-an500", "md6605": "qemu-virt-rv32"}
MAX_WINDOWS = 16


def available() -> dict:
    return {t: hil.available(p) for t, p in PORT.items()}


def _raw_windows(job: dict, m: inference.ProjectModel, inputs: dict) -> tuple[np.ndarray, list, list[str], bool]:
    """Windows in the project's input layout, true labels where known, display names, and whether the windows
    already hold front-end features (dataset windows of an MFCC / band-energy model type)."""
    src = inputs.get("source", "dataset")
    C, T = m.raw_channels, m.raw_window
    if src == "dataset":
        X, y, labels, _ = data.load(inputs.get("dataset_id") or job["params"]["dataset_id"])
        idx = [int(i) for i in (inputs.get("indices") or [])][:MAX_WINDOWS] or list(
            np.random.default_rng(int(inputs.get("seed", 0))).choice(len(X), min(6, len(X)), replace=False))
        if any(i < 0 or i >= len(X) for i in idx):
            raise ValueError(f"window index out of range 0..{len(X) - 1}")
        truth = [labels[int(y[i])] if job["result"]["template"]["task"] == "classification" else None for i in idx]
        return X[idx], truth, [f"dataset window #{i}" for i in idx], True
    if src == "recording":
        from . import recordings
        rec = recordings.get(inputs["recording_id"])
        if not rec:
            raise ValueError("unknown recording")
        if len(rec["channels"]) != C:
            raise ValueError(f"the model reads {C} channels; the recording has {len(rec['channels'])}")
        sig = np.asarray(recordings.signal(rec), np.float32)
        start, hop = int(inputs.get("start", 0)), int(inputs.get("hop") or T)
        n = min(int(inputs.get("count", 4)), MAX_WINDOWS)
        starts = [start + k * hop for k in range(n) if start + k * hop + T <= len(sig)]
        if not starts:
            raise ValueError(f"the recording has no {T}-sample window from sample {start}")
        truth = []
        for s0 in starts:
            lab = [sp["label"] for sp in rec.get("spans", []) if sp["start"] <= s0 + T // 2 < sp["end"]]
            truth.append(lab[0] if lab else None)
        return np.stack([sig[s0:s0 + T].T for s0 in starts])[:, :, None, :], truth, [
            f"{rec['name']} @ {s0 / rec['sample_rate']:.3f} s" for s0 in starts], False
    if src == "values":
        rows = inputs.get("windows") or []
        arr = np.asarray(rows, np.float32)
        if arr.ndim == 2 and C == 1:
            arr = arr[:, None, :]
        if arr.ndim != 3 or arr.shape[1:] != (C, T):
            raise ValueError(f"each window must be {C} channel(s) x {T} samples")
        return arr[:MAX_WINDOWS, :, None, :], [None] * min(len(arr), MAX_WINDOWS), [
            f"pasted window {k + 1}" for k in range(min(len(arr), MAX_WINDOWS))], False
    raise ValueError("source must be dataset, recording or values")


def build_dir_for(job: dict, target: str) -> "Path":
    from pathlib import Path
    jobdir = config.DATA_DIR / "jobs" / job["id"]
    sel = job["result"]["selected"]
    if target == sel["target"]:
        return jobdir / "build"
    d = jobdir / f"hil_build_{target}"
    if not (d / "manifest.json").exists():
        sai_engine.compile_build(jobdir / "model.onnx", target, "fast", d)
    return Path(d)


def run(ctx, p: dict):
    from . import pipeline
    job = pipeline.get(p["job_id"])
    target = p["target"]
    if target not in PORT:
        raise ValueError(f"virtual boards: {', '.join(PORT)}")
    if not hil.available(PORT[target]):
        raise ValueError(f"the {target} emulator or its cross compiler is not installed on this server")
    ctx.stage("prepare")
    m = inference.ProjectModel(job["id"])
    X, truth, names, ready = _raw_windows(job, m, p.get("inputs") or {})
    xq = m.quantized(X, features_ready=ready)
    ctx.log(f"{len(xq)} window(s) -> int8 input {list(xq.shape[1:])} (the package's preprocessing)")
    jobdir = config.DATA_DIR / "jobs" / job["id"]
    expected = exact_eval.run_exact(jobdir / "model.onnx", jobdir / "exact", xq)
    build = build_dir_for(job, target)
    ctx.stage("prepare", "done")
    ctx.stage("run")
    port = PORT[target]
    fw_dir = ctx.dir / "fw"
    try:
        fw = hil.build(build, port, xq, fw_dir)
        ctx.log(f"firmware {port}: text {fw['text']} B, data {fw['data']} B, bss {fw['bss']} B")
        res = hil.run_emulated(fw["elf"], port, timeout=180)
    finally:
        shutil.rmtree(fw_dir, ignore_errors=True)
    outs = np.asarray(res["outputs"], np.int64)
    match = [bool(np.array_equal(outs[i], expected[i])) for i in range(len(outs))]
    deq = m.dequantize(outs)
    rows = []
    for i in range(len(outs)):
        row = {"name": names[i], "truth": truth[i], "output_int8": outs[i].tolist(),
               "expected_int8": expected[i].astype(int).tolist(), "bit_exact": match[i],
               "input_preview": X[i, :, 0, ::max(1, X.shape[-1] // 160)][:3].round(4).tolist()}
        if m.task == "classification":
            o = deq[i].astype(np.float64)
            pr = np.exp(o - o.max())
            pr /= pr.sum()
            k = int(np.argmax(pr))
            row.update(prediction=m.labels[k], confidence=round(float(pr[k]), 4),
                       probabilities=dict(zip(m.labels, np.round(pr, 4).tolist())))
        else:
            row["value"] = deq[i].round(4).tolist()
        rows.append(row)
    ctx.log(f"board printed {len(outs)} result(s); bit-exact with the host program: {sum(match)}/{len(match)}")
    ctx.stage("run", "done")
    ctx.result({"target": target, "board": f"QEMU {port} (emulated)", "port": port, "windows": rows,
                "all_bit_exact": all(match), "cycles": {"avg": res.get("cycles"), "min": res.get("cycles_min"),
                                                        "max": res.get("cycles_max"), "evidence": "emulated"},
                "stack_painted": res.get("stack_painted"), "firmware": {k: fw.get(k) for k in ("text", "data", "bss")},
                "uart": res.get("uart", ""), "config": res.get("config"), "labels": m.labels, "task": m.task,
                "job_id": job["id"]})
