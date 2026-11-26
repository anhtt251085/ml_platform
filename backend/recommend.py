# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Architecture recommender: derive candidate architectures from the customer's data and objective, cost every one
exactly with SAI before any training, and recommend one.

1. Profile the data: input shape, classes and imbalance, amount of data, and (for time series) where the signal's
   energy sits in frequency, which sets how long a receptive field the network needs.
2. Turn the profile into rules: model family, minimum receptive field, parameter cap for the amount of data.
3. Generate candidates around those rules (stem stride and kernel, depth, width, kernel size) in the families SAI
   streams best, plus the pool's own tiny / small / base, and keep a shortlist on the capacity-per-cost frontier.
4. Export each shortlisted candidate with random weights to INT8 and run SAI's plan search on every board: RAM,
   flash and the plan are fixed by the architecture, not by the weights, so these numbers are what the trained
   model will get. Candidates that do not fit a board or the user's budgets are dropped.
5. Recommend for the goal: the cheapest candidate with enough capacity (fastest / min-ram / min-flash), the
   largest that fits (accuracy), or the best capacity-for-cost trade-off (balanced).
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from . import archspec

RF_MIN = 8
MAX_LIBRARY = 6


def _r4(x: float) -> int:
    return max(4, int(round(x / 4)) * 4)


# ---- 1. data profile ----------------------------------------------------------------------------------------------
def profile(X: np.ndarray, y: np.ndarray, task: str, labels: list, sample_rate_hz: float | None,
            n_train: int | None = None) -> dict:
    n, c, h, w = (int(v) for v in X.shape)
    p = {"n": n, "n_train": int(n_train or round(0.7 * n)), "channels": c, "height": h, "width": w,
         "kind": "timeseries" if h == 1 else "2d", "task": task, "classes": len(labels) if task == "classification" else None}
    if task == "classification" and len(y):
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=len(labels))
        p["class_counts"] = {str(lb): int(k) for lb, k in zip(labels, counts)}
        present = counts[counts > 0]
        p["imbalance"] = round(float(present.max() / present.min()), 2) if len(present) else None
    if h == 1 and w >= 16:
        rng = np.random.default_rng(0)
        sel = X[rng.choice(n, size=min(n, 256), replace=False), :, 0, :].astype(np.float64)
        sel = sel - sel.mean(axis=-1, keepdims=True)
        power = (np.abs(np.fft.rfft(sel * np.hanning(w), axis=-1)) ** 2).mean(axis=(0, 1))[1:]   # drop DC
        total = float(power.sum())
        if total > 0:
            cum = np.cumsum(power) / total
            b10 = int(np.searchsorted(cum, 0.10)) + 1           # rfft bin index (1 = one cycle per window)
            peak = int(np.argmax(power)) + 1
            flat = float(np.exp(np.mean(np.log(power + 1e-30))) / (power.mean() + 1e-30))
            fs = sample_rate_hz or None
            p["spectrum"] = {
                "peak_cycles_per_window": peak, "low10_cycles_per_window": b10, "flatness": round(flat, 3),
                "peak_hz": round(peak * fs / w, 2) if fs else None, "low10_hz": round(b10 * fs / w, 2) if fs else None}
    return p


# ---- 2. rules -----------------------------------------------------------------------------------------------------
def rules(prof: dict, conv: bool) -> dict:
    """Receptive-field target, parameter cap and the reasons, in words a customer can check."""
    why = []
    n_train = prof["n_train"]
    cap = int(min(60000, max(1500, 25 * n_train)))
    why.append(f"{n_train:,} training windows: parameters capped at {cap:,} (about 25 per window) so the model "
               "cannot simply memorise the data")
    floor = int(max(300, 0.08 * cap, 40 * (prof.get("classes") or 2)))
    why.append(f"{prof.get('classes') or 'the'} output(s): at least {floor:,} parameters to separate them reliably")
    if prof.get("imbalance") and prof["imbalance"] > 3:
        why.append(f"classes are imbalanced ({prof['imbalance']}:1): training uses class weights; prefer the "
                   "recording-level split and check per-class recall")
    rf, max_stride = None, 2
    if conv and prof["kind"] == "timeseries":
        w = prof["width"]
        sp = prof.get("spectrum")
        if sp:
            per_cycle = w / sp["peak_cycles_per_window"]
            max_stride = int(max(1, min(4, per_cycle // 4)))
            hz = f" ({sp['peak_hz']} Hz)" if sp.get("peak_hz") else ""
            why.append(f"the strongest component{hz} lasts {per_cycle:.0f} samples per cycle: the first layer "
                       f"downsamples by at most {max_stride} so it stays resolved (4+ samples per cycle)")
        else:
            max_stride = 4
        if sp and sp["flatness"] > 0.5:
            rf = min(w, max(16, w // 8))
            why.append(f"broadband signal (spectral flatness {sp['flatness']}): short kernels; receptive field "
                       f">= {rf} samples")
        elif sp:
            period = w / sp["low10_cycles_per_window"]
            rf = int(min(w, max(RF_MIN, math.ceil(2 * period))))
            hz = f" ({sp['low10_hz']} Hz)" if sp.get("low10_hz") else ""
            why.append(f"the slowest significant component repeats every {period:.0f} samples{hz}: receptive field "
                       f">= {rf} samples to see two cycles of it")
        else:
            rf = min(w, max(RF_MIN, w // 4))
            why.append(f"short window: receptive field >= {rf} samples")
    elif conv:
        rf = max(3, math.ceil(0.5 * min(prof["height"], prof["width"])))
        why.append(f"{prof['height']}x{prof['width']} input: receptive field >= {rf} pixels (half the shorter side)")
    else:
        why.append("feature-vector input: fully connected family")
    return {"param_cap": cap, "param_floor": floor, "rf_target": rf, "max_stride": max_stride, "reasons": why}


# ---- 3. candidates ------------------------------------------------------------------------------------------------
def receptive_field(layers: list[dict]) -> int:
    rf, jump = 1, 1
    for layer in layers:
        op = layer["op"]
        if op in ("conv", "dwconv"):
            rf += (layer.get("k", 3) - 1) * layer.get("dilation", 1) * jump
            jump *= layer.get("stride", 1)
        elif op == "residual":
            rf += 2 * (layer.get("k", 3) - 1) * jump
        elif op in ("maxpool", "avgpool"):
            rf += (layer.get("k", 2) - 1) * jump
            jump *= layer.get("k", 2)
        elif op in ("gap", "fc"):
            break
    return rf


def _conv_family(prof: dict, max_stride: int = 4) -> list[tuple[str, list[dict]]]:
    one_d = prof["kind"] == "timeseries"
    w, h = prof["width"], prof["height"]
    short = w if one_d else min(w, h)
    out = []
    stems = [(s, k) for s in (1, 2, 4) for k in ((3, 5, 7) if one_d else (3,)) if s * 8 <= short and k > s - 1] \
        if one_d else [(s, 3) for s in (1, 2) if s * 6 <= short]
    stems = [(s, k) for s, k in stems if s <= max_stride] or [(1, 3)]
    for (s0, k0), depth, width, k in itertools.product(stems, (1, 2, 3, 4), (8, 12, 16, 24), (3, 5) if one_d else (3,)):
        layers = [{"op": "conv", "out": width, "k": k0, "stride": s0}]
        size = math.ceil(short / s0)
        cw = width
        for b in range(depth):
            cw = _r4(width * (1.5 ** b)) if b else width
            cw = min(cw, 64)
            layers += [{"op": "dwconv", "k": k}, {"op": "pwconv", "out": _r4(cw * 1.5) if b == depth - 1 else cw}]
            if b < depth - 1 and size >= 8:
                layers.append({"op": "maxpool", "k": 2})
                size //= 2
        layers.append({"op": "gap"})
        name = f"s{s0}k{k0}-d{depth}-w{width}-k{k}"
        out.append((name, layers))
    return out


def _fc_family(prof: dict, task: str) -> list[tuple[str, list[dict]]]:
    out = []
    if task == "anomaly":
        for a, b in itertools.product((16, 24, 32, 48, 64), (4, 6, 8, 12, 16)):
            if b < a:
                out.append((f"ae{a}-{b}", [{"op": "fc", "out": a, "relu": True}, {"op": "fc", "out": b, "relu": True},
                                          {"op": "fc", "out": a, "relu": True}]))
    else:
        for a, depth in itertools.product((16, 24, 32, 48, 64, 96), (1, 2, 3)):
            out.append((f"mlp{a}x{depth}", [{"op": "fc", "out": max(8, a // (2 ** i)), "relu": True}
                                              for i in range(depth)]))
    return out


def candidates(prof: dict, rl: dict, template: dict, n_outputs: int, max_candidates: int = 8,
               library: list | None = None) -> list[dict]:
    conv = any(layer["op"] in ("conv", "dwconv", "pwconv") for v in template["variants"].values() for layer in v)
    gen = _conv_family(prof, rl.get("max_stride", 4)) if conv else _fc_family(prof, template["task"])
    inp = {"channels": prof["channels"], "height": prof["height"], "width": prof["width"]}
    rows = []

    def add(name, layers, origin):
        try:
            spec = archspec.normalize({"input": inp, "layers": layers, "outputs": n_outputs})
        except archspec.SpecError:
            return
        s = archspec.summary(spec)
        rows.append({"id": name, "origin": origin, "layers": spec["layers"], "params": s["params"], "macs": s["macs"],
                     "peak_activation": s["peak_activation_bytes"],
                     "receptive_field": receptive_field(spec["layers"]) if conv else None})

    for name, layers in gen:
        add(name, layers, "generated")
    pool = []
    for size, layers in template["variants"].items():
        before = len(rows)
        add(f"pool:{size}", layers, "pool")
        if len(rows) > before:
            pool.append(rows.pop())
    lib = []
    for aid, layers in library or []:         # architecture library entries that build on this input
        before = len(rows)
        add(f"lib:{aid}", layers, "library")
        if len(rows) > before:
            r = rows.pop()
            if rl["param_floor"] * 0.5 <= r["params"] <= rl["param_cap"] and \
                    (not (conv and rl["rf_target"]) or (r["receptive_field"] or 0) >= rl["rf_target"] * 0.5):
                lib.append(r)
    lib.sort(key=lambda r: r["params"])
    if len(lib) > MAX_LIBRARY:                # at most a handful, spread over the capacity range (each costs a SAI run)
        lib = [lib[i] for i in np.unique(np.linspace(0, len(lib) - 1, MAX_LIBRARY).round().astype(int))]
    pool += lib
    ok = [r for r in rows if r["params"] <= rl["param_cap"] and r["params"] >= rl["param_floor"] * 0.5]
    if conv and rl["rf_target"]:
        reach = [r for r in ok if r["receptive_field"] >= rl["rf_target"]]
        ok = reach or sorted(ok, key=lambda r: -r["receptive_field"])[:max(1, len(ok) // 4)]
    # capacity-per-cost frontier: no other candidate has more parameters for fewer MACs and less activation memory
    front = [r for r in ok if not any(q["params"] >= r["params"] and q["macs"] <= r["macs"]
                                      and q["peak_activation"] <= r["peak_activation"] and q is not r
                                      and (q["params"], q["macs"], q["peak_activation"]) != (r["params"], r["macs"], r["peak_activation"])
                                      for q in ok)]
    front.sort(key=lambda r: r["params"])
    if len(front) > max_candidates:                    # spread over the capacity range
        idx = np.unique(np.linspace(0, len(front) - 1, max_candidates).round().astype(int))
        front = [front[i] for i in idx]
    for r in front + pool:
        r["over_cap"] = r["params"] > rl["param_cap"]
    return front + pool


# ---- 4. exact cost with SAI ---------------------------------------------------------------------------------------
def cost(cands: list[dict], prof: dict, n_outputs: int, targets: list[str], goal: str, budgets: dict | None,
         workdir: Path, calibration: dict | None = None, measurements: dict | None = None, log=lambda m: None,
         cancelled=lambda: False) -> None:
    import torch

    from . import qexport, sai_engine
    inp = {"channels": prof["channels"], "height": prof["height"], "width": prof["width"]}
    calib = np.random.default_rng(0).normal(0, 1, (32, inp["channels"], inp["height"], inp["width"])).astype(np.float32)
    plan_goal = goal if goal in ("fastest", "min-ram", "min-flash") else "balanced"
    for i, c in enumerate(cands):
        if cancelled():
            raise RuntimeError("cancelled")
        spec = archspec.normalize({"input": inp, "layers": c["layers"], "outputs": n_outputs})
        torch.manual_seed(0)
        net = archspec.SpecNet(spec)
        net.eval()
        net.fold_bn()
        path = workdir / f"rec_{i}.onnx"
        qexport.export_qdq(net, calib, path)
        summ = archspec.summary(spec)
        plans = sai_engine.plan_search(path, summ["macs"], summ["spatial_axes"], targets, plan_goal, budgets,
                                       calibration, explore_seconds=4, measurements=measurements, explore_builds=0)
        path.unlink(missing_ok=True)
        c["boards"] = {}
        for t in targets:
            rows = [r for r in plans["candidates"] if r["target"] == t]
            fit = [r for r in rows if r["fits"]]
            best = sai_engine._pick(fit, plan_goal) if fit else None
            pick = best or (min(rows, key=lambda r: len(r["violations"])) if rows else None)
            c["boards"][t] = None if pick is None else {
                "fits": best is not None, "ram": pick["ram"], "flash": pick["flash"], "latency_ms": pick["latency"]["ms"],
                "latency_evidence": pick["latency"]["evidence"], "schedule": pick["schedule"],
                "objective": pick["objective"], "violations": pick["violations"]}
        c["fits"] = all(b and b["fits"] for b in c["boards"].values())
        worst = [b for b in c["boards"].values() if b]
        c["ram"] = max((b["ram"] for b in worst), default=None)
        c["flash"] = max((b["flash"] or 0 for b in worst), default=None)
        c["latency_ms"] = max((b["latency_ms"] or 0 for b in worst), default=None)
        log(f"{c['id']}: {c['params']:,} params -> "
            + ", ".join(f"{t} {b['ram']} B RAM / {b['flash']} B flash / {b['latency_ms']} ms{'' if b['fits'] else ' (does not fit)'}"
                        for t, b in c["boards"].items() if b))


# ---- 5. recommend -------------------------------------------------------------------------------------------------
def pick(cands: list[dict], rl: dict, goal: str) -> tuple[dict | None, str]:
    fit = [c for c in cands if c.get("fits") and not c.get("over_cap")]
    if not fit:
        return None, ("No candidate fits every selected board and your limits within the parameter cap: choose a "
                      "larger board, relax the limits, shorten the input window or add data.")
    adequate = [c for c in fit if c["params"] >= rl["param_floor"]] or [max(fit, key=lambda c: c["params"])]
    if goal == "fastest":
        best = min(adequate, key=lambda c: (c["latency_ms"] or 1e9, c["ram"]))
        why = "the lowest latency among the candidates with enough capacity"
    elif goal == "min-ram":
        best = min(adequate, key=lambda c: (c["ram"], c["latency_ms"] or 1e9))
        why = "the least RAM among the candidates with enough capacity"
    elif goal == "min-flash":
        best = min(adequate, key=lambda c: (c["flash"] or 1 << 40, c["ram"]))
        why = "the least flash among the candidates with enough capacity"
    elif goal == "accuracy":
        best = max(fit, key=lambda c: (c["params"], -c["ram"]))
        why = "the most capacity that still fits every board and stays under the parameter cap"
    else:
        lo = {k: min((c[k] or 0) for c in fit) or 1e-9 for k in ("ram", "flash", "latency_ms")}
        cap = math.log(rl["param_cap"])

        def score(c):
            capacity = min(1.0, math.log(max(c["params"], 2)) / cap)
            cost_ = sum((c[k] or 0) / lo[k] for k in ("ram", "flash", "latency_ms")) / 3
            return capacity * 3 - math.log(cost_)
        best = max(adequate, key=score)
        why = "the best capacity for its RAM, flash and latency (balanced)"
    return best, why


def recommend(X, y, task, labels, template, n_outputs, goal, targets, budgets, workdir, sample_rate_hz=None,
              n_train=None, max_candidates=8, calibration=None, measurements=None, log=lambda m: None,
              cancelled=lambda: False, stage=lambda name, status="running": None, library=None) -> dict:
    stage("profile")
    prof = profile(X, y, task, labels, sample_rate_hz, n_train)
    conv = any(layer["op"] in ("conv", "dwconv", "pwconv") for v in template["variants"].values() for layer in v)
    rl = rules(prof, conv)
    for r in rl["reasons"]:
        log("rule: " + r)
    stage("profile", "done")
    stage("generate")
    cands = candidates(prof, rl, template, n_outputs, max_candidates, library)
    log(f"{len(cands)} candidates to cost with SAI ({sum(c['origin'] == 'generated' for c in cands)} generated, "
        f"{sum(c['origin'] == 'pool' for c in cands)} from the pool, "
        f"{sum(c['origin'] == 'library' for c in cands)} from the architecture library)")
    stage("generate", "done")
    stage("cost")
    cost(cands, prof, n_outputs, targets, goal, budgets, workdir, calibration, measurements, log, cancelled)
    stage("cost", "done")
    stage("pick")
    best, why = pick(cands, rl, goal)
    for c in cands:
        c["recommended"] = best is not None and c is best
        c["adequate"] = c["params"] >= rl["param_floor"]
    stage("pick", "done")
    return {"profile": prof, "rules": rl, "goal": goal, "targets": targets, "candidates": cands,
            "recommended": best and {k: best[k] for k in ("id", "layers", "params", "macs", "receptive_field", "ram",
                                                           "flash", "latency_ms", "boards")},
            "why": why}
