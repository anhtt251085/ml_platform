# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Benchmark corpus: the four MLPerf Tiny (v1.x) reference tasks, written in the platform's SAI-compatible spec
language, plus the platform's own built-in model types, compiled by SAI for every MCU board.

This is a *resource* benchmark (RAM, flash, latency per board, every number with its evidence) with random
weights; it is not an MLPerf submission and makes no accuracy claim (that needs the reference datasets and the
EEMBC runner). Where the spec language cannot express a reference layer exactly, the closest SAI-expressible
variant is used and the difference is stated next to the model.

Each run is a `benchmark` job; results are kept per SAI version, so the corpus doubles as a regression suite
(RAM / flash / latency of every model on every board, run after run), and each row also carries the TFLM +
CMSIS-NN model of the same graph (baselines.py) for the side-by-side."""

from __future__ import annotations

from . import archspec


def _dwsep(out, stride=1):
    return [{"op": "dwconv", "k": 3, "stride": stride}, {"op": "pwconv", "out": out}]


MOBILENET_V1_025 = ([{"op": "conv", "out": 8, "k": 3, "stride": 2}] + _dwsep(16) + _dwsep(32, 2) + _dwsep(32)
                    + _dwsep(64, 2) + _dwsep(64) + _dwsep(128, 2) + _dwsep(128) * 5 + _dwsep(256, 2) + _dwsep(256)
                    + [{"op": "gap"}])

CORPUS = [
    {"id": "mlperf-kws-dscnn", "suite": "MLPerf Tiny", "task": "Keyword spotting (Speech Commands, 12 classes)",
     "reference": "DS-CNN (S): 64-ch 10x4 stem stride 2, 4 x (dw 3x3 + pw 64), avgpool, fc 12",
     "deviation": "square kernels only: 5x5 stem instead of 10x4",
     "spec": {"input": {"channels": 1, "height": 49, "width": 10},
              "layers": [{"op": "conv", "out": 64, "k": 5, "stride": 2}] + _dwsep(64) * 4 + [{"op": "gap"}],
              "outputs": 12}},
    {"id": "mlperf-vww-mbv1", "suite": "MLPerf Tiny", "task": "Visual wake words (person / no person, 96x96 RGB)",
     "reference": "MobileNetV1 alpha 0.25, 96x96x3, 2 classes",
     "deviation": None,
     "spec": {"input": {"channels": 3, "height": 96, "width": 96}, "layers": MOBILENET_V1_025, "outputs": 2}},
    {"id": "mlperf-ic-resnet8", "suite": "MLPerf Tiny", "task": "Image classification (CIFAR-10, 32x32 RGB)",
     "reference": "ResNet-8: conv 16, residual stacks 16 / 32 (s2) / 64 (s2), avgpool, fc 10",
     "deviation": "channel-changing shortcuts (1x1 conv projection) written as a strided conv before an "
                  "identity residual block",
     "spec": {"input": {"channels": 3, "height": 32, "width": 32},
              "layers": [{"op": "conv", "out": 16, "k": 3}, {"op": "residual", "k": 3},
                         {"op": "conv", "out": 32, "k": 3, "stride": 2}, {"op": "residual", "k": 3},
                         {"op": "conv", "out": 64, "k": 3, "stride": 2}, {"op": "residual", "k": 3},
                         {"op": "gap"}], "outputs": 10}},
    {"id": "mlperf-ad-fcae", "suite": "MLPerf Tiny", "task": "Anomaly detection (ToyADMOS, 5 x 128 log-mel frames)",
     "reference": "FC autoencoder 640 -> 128 x4 -> 8 -> 128 x4 -> 640",
     "deviation": None,
     "spec": {"input": {"channels": 1, "height": 1, "width": 640},
              "layers": [{"op": "fc", "out": 128}] * 4 + [{"op": "fc", "out": 8}] + [{"op": "fc", "out": 128}] * 4,
              "outputs": 640}},
]


def corpus(templates: list[dict] | None = None) -> list[dict]:
    """MLPerf Tiny entries + one entry per built-in model type (its small variant)."""
    out = []
    for e in CORPUS:
        spec = archspec.normalize(e["spec"])
        out.append({**e, "spec": spec, "summary": _summ(spec)})
    for t in templates or []:
        var = (t.get("variants") or {}).get("small")
        if not var or t.get("frontend") or t["input"].get("kind") not in ("timeseries", "image"):
            continue
        inp = t["input"]
        spec = {"input": {"channels": inp["channels"], "height": inp["height"], "width": inp["width"]},
                "layers": var, "outputs": len(t.get("labels") or []) or 1}
        try:
            spec = archspec.normalize(spec)
        except archspec.SpecError:
            continue
        if spec.get("layout") == "sequence":
            continue
        out.append({"id": f"tpl-{t['id']}", "suite": "SAI Platform model types", "task": t["name"],
                    "reference": "small variant of the built-in model type", "deviation": None, "spec": spec,
                    "summary": _summ(spec)})
    return out


def _summ(spec):
    s = archspec.summary(spec)
    return {k: s[k] for k in ("macs", "params", "peak_activation_bytes", "input_bytes")}
