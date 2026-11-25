"""Bit-exact evaluation: compile the model for host-cpu with SAI, link a batch runner, run the test set.

The INT8 accuracy the customer sees is the deployed C program's, not a framework's approximation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import config, sai_engine

RUNNER = Path(__file__).resolve().parent / "runner_c" / "batch_runner.c"


def build_runner(model: Path, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    build = workdir / "host_build"
    if not (build / "manifest.json").exists():
        sai_engine.compile_build(model, "host-cpu", "min-code", build)
    exe = workdir / "batch_runner"
    if not exe.exists():
        proc = subprocess.run([config.HOST_CC, "-O2", "-std=c11", "-I", str(build / "model"), str(RUNNER),
                               str(build / "lib" / "libsai_model.a"), "-o", str(exe)], capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError("batch runner link failed: " + proc.stderr[-800:])
    return exe


def run_exact(model: Path, workdir: Path, xq: np.ndarray) -> np.ndarray:
    """xq: [N, ...] int8 in the model's input layout. Returns [N, out_elems] int8."""
    exe = build_runner(model, workdir)
    n = len(xq)
    proc = subprocess.run([str(exe), str(n)], input=np.ascontiguousarray(xq, dtype=np.int8).tobytes(),
                          capture_output=True, timeout=600)
    if proc.returncode:
        raise RuntimeError("exact run failed: " + proc.stderr.decode(errors="replace")[-500:])
    return np.frombuffer(proc.stdout, dtype=np.int8).reshape(n, -1)
