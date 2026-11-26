# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Architecture library: named, reusable model architectures built only from SAI-supported operators.

Customers pick from it in the new-project wizard (next to a model type's own tiny / small / base sizes); the
recommender also costs the compatible ones. Built-in entries live in this file; SAI researchers add global ones,
and researchers, managers and admins of a customer organisation add private ones (stored in the database).

Every entry declares which inputs it suits (timeseries = [C, 1, T]; 2d = [C, H, W] spectrograms and images;
features = a vector) and which tasks. Whether it works for a given dataset is decided on the real input shape
(archspec.normalize), never assumed. `verify` exports each entry with random weights at a reference input and
runs SAI's plan search on every board, recording RAM, flash, latency and whether it fits, with the SAI version:
that is what "SAI-verified" means in the UI.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import numpy as np

from . import archspec, db

INPUT_KINDS = ("timeseries", "2d", "features")
TASKS = ("classification", "regression", "anomaly")
# reference inputs for SAI verification (typical of the pool: 3-axis 256-sample window, 49x10 MFCC, 64 features)
REFERENCE = {"timeseries": {"channels": 3, "height": 1, "width": 256},
             "2d": {"channels": 1, "height": 49, "width": 10},
             "features": {"channels": 1, "height": 1, "width": 64}}

C = lambda out, k=3, stride=1, dilation=1: {"op": "conv", "out": out, "k": k, "stride": stride, "dilation": dilation}  # noqa: E731
DW = lambda k=3, stride=1, dilation=1: {"op": "dwconv", "k": k, "stride": stride, "dilation": dilation}  # noqa: E731
PW = lambda out: {"op": "pwconv", "out": out}  # noqa: E731
MP = lambda k=2: {"op": "maxpool", "k": k}  # noqa: E731
AP = lambda k=2: {"op": "avgpool", "k": k}  # noqa: E731
GAP = {"op": "gap"}
FC = lambda out, relu=True: {"op": "fc", "out": out, "relu": relu}  # noqa: E731
RES = lambda k=3: {"op": "residual", "k": k}  # noqa: E731
DROP = {"op": "dropout", "p": 0.1}

TS, D2, FV = ["timeseries"], ["2d"], ["features"]
CR = ["classification", "regression"]


def _a(aid, name, family, size, inputs, tasks, layers, summary, notes=""):
    return {"id": aid, "name": name, "family": family, "size": size, "inputs": inputs, "tasks": tasks,
            "layers": layers, "summary": summary, "notes": notes, "source": "builtin", "status": "published",
            "org_id": None}


BUILTIN = [
    # ---- 1-D time series -------------------------------------------------------------------------------------
    _a("ds-tcn-xs", "DS-TCN XS", "Depthwise-separable TCN", "xs", TS, CR,
       [C(8, 5, 4), DW(3), PW(12), GAP],
       "The smallest useful temporal CNN: one strided stem, one depthwise-separable block.",
       "For very small MCUs and simple signals (thresholds, a few classes)."),
    _a("ds-tcn-s", "DS-TCN S", "Depthwise-separable TCN", "s", TS, CR,
       [C(8, 5, 2), DW(3), PW(12), MP(2), DW(3), PW(16), GAP],
       "Strided stem and two depthwise-separable blocks: the shape SAI streams row by row with the least RAM."),
    _a("ds-tcn-m", "DS-TCN M", "Depthwise-separable TCN", "m", TS, CR,
       [C(12, 5, 2), DW(5), PW(16), MP(2), DW(3), PW(24), MP(2), DW(3), PW(32), GAP, DROP],
       "Three depthwise-separable blocks with pooling between them; the default for vibration, current and IMU."),
    _a("ds-tcn-l", "DS-TCN L", "Depthwise-separable TCN", "l", TS, CR,
       [C(16, 7, 2), DW(5), PW(24), RES(3), MP(2), DW(3), PW(32), MP(2), DW(3), PW(48), GAP, FC(32), DROP],
       "Wider, with a residual block and a hidden dense layer, for harder problems on Cortex-M7 class parts."),
    _a("dilated-tcn-s", "Dilated TCN S", "Dilated TCN", "s", TS, CR,
       [C(12, 3), DW(3, 1, 1), PW(12), DW(3, 1, 2), PW(16), DW(3, 1, 4), PW(16), DW(3, 1, 8), PW(24), GAP],
       "Dilations 1-2-4-8 reach a 31-sample receptive field without pooling, keeping full time resolution.",
       "For short events whose timing matters (arcs, transients)."),
    _a("dilated-tcn-m", "Dilated TCN M", "Dilated TCN", "m", TS, CR,
       [C(16, 5, 2), DW(3, 1, 1), PW(16), DW(3, 1, 2), PW(24), DW(3, 1, 4), PW(24), DW(3, 1, 8), PW(32),
        DW(3, 1, 16), PW(32), GAP, DROP],
       "Strided stem and five dilated blocks (1-16): a long receptive field for slow, periodic patterns."),
    _a("wide-kernel-cnn", "Wide-kernel CNN", "Wide-kernel CNN", "m", TS, CR,
       [C(16, 31, 4), MP(2), C(24, 3), MP(2), C(32, 3), MP(2), C(32, 3), GAP],
       "A 31-tap first layer acts as a learned filter bank on raw vibration, then small standard convolutions.",
       "Raw high-rate vibration without a separate front end."),
    _a("res-tcn-m", "Residual TCN M", "Residual TCN", "m", TS, CR,
       [C(16, 5, 2), RES(3), MP(2), PW(24), RES(3), MP(2), PW(32), RES(3), GAP, DROP],
       "Residual blocks between channel expansions: trains reliably when the dataset is large."),
    _a("cnn-1d-s", "CNN-1D S", "Standard CNN-1D", "s", TS, CR,
       [C(8, 5, 2), MP(2), C(16, 3), MP(2), C(24, 3), GAP],
       "Plain convolutions: more MACs than depthwise-separable at the same width, very robust baseline."),
    # ---- 2-D: spectrograms, MFCC, thermal / low-resolution images ---------------------------------------------
    _a("ds-cnn-2d-s", "DS-CNN 2D S", "Depthwise-separable CNN-2D", "s", D2, CR,
       [C(16, 3, 2), DW(3), PW(16), DW(3), PW(24), GAP],
       "Keyword-spotting style: strided stem and depthwise-separable blocks over a spectrogram or MFCC map."),
    _a("ds-cnn-2d-m", "DS-CNN 2D M", "Depthwise-separable CNN-2D", "m", D2, CR,
       [C(24, 3, 2), DW(3), PW(24), DW(3), PW(32), DW(3), PW(32), DW(3), PW(48), GAP, DROP],
       "Four depthwise-separable blocks: the common accuracy / size sweet spot for audio events and words."),
    _a("cnn-2d-s", "CNN-2D S", "Standard CNN-2D", "s", D2, CR,
       [C(8, 3), MP(2), C(16, 3), MP(2), C(24, 3), GAP],
       "Plain 3x3 convolutions with pooling, for small images such as 8x8 to 32x32 thermal arrays."),
    _a("mobile-2d-m", "Mobile-style 2D M", "Inverted bottleneck CNN-2D", "m", D2, CR,
       [C(16, 3, 2), PW(48), DW(3), PW(16), RES(3), PW(64), DW(3, 2), PW(24), GAP],
       "Expand (1x1) - depthwise - project blocks with a residual, the mobile-network pattern at MCU scale."),
    # ---- feature vectors ---------------------------------------------------------------------------------------
    _a("mlp-s", "MLP S", "Dense network", "s", FV, CR, [FC(32), FC(16)],
       "Two dense layers for hand-crafted or front-end features (band energies, statistics)."),
    _a("mlp-m", "MLP M", "Dense network", "m", FV, CR, [FC(64), DROP, FC(32), FC(16)],
       "Three dense layers with dropout."),
    _a("ae-s", "Autoencoder S", "Dense autoencoder", "s", FV, ["anomaly"], [FC(24), FC(8), FC(24)],
       "Compress to 8 values and reconstruct: the reconstruction error is the anomaly score.",
       "Trained on normal data only."),
    _a("ae-m", "Autoencoder M", "Dense autoencoder", "m", FV, ["anomaly"], [FC(48), FC(16), FC(8), FC(16), FC(48)],
       "A deeper bottleneck for richer feature vectors."),
]
_BUILTIN = {a["id"]: a for a in BUILTIN}


class LibError(ValueError):
    pass


# ---- storage ------------------------------------------------------------------------------------------------------
def _row(r) -> dict:
    d = db.uj(r["data"], {})
    return {**d, "id": r["id"], "status": r["status"], "org_id": r["org_id"], "source": "custom",
            "created_by": r["created_by"], "created": r["created"], "updated": r["updated"]}


def all_visible(org_id: str | None, platform: bool = False) -> list[dict]:
    """Built-ins plus custom entries: global ones and the caller's organisation's (platform admins see all)."""
    with db.conn() as c:
        rows = c.execute("SELECT * FROM architectures" + ("" if platform else " WHERE org_id IS NULL OR org_id=?")
                         + " ORDER BY created", () if platform else (org_id,)).fetchall()
    return [dict(a) for a in BUILTIN] + [_row(r) for r in rows]


def get(aid: str, org_id: str | None = None, platform: bool = False) -> dict | None:
    if aid in _BUILTIN:
        return dict(_BUILTIN[aid])
    with db.conn() as c:
        r = c.execute("SELECT * FROM architectures WHERE id=?", (aid,)).fetchone()
    if not r or not (platform or r["org_id"] is None or r["org_id"] == org_id):
        return None
    return _row(r)


def validate(body: dict) -> dict:
    name = str(body.get("name") or "").strip()
    if not 2 <= len(name) <= 60:
        raise LibError("name: 2 to 60 characters")
    inputs = [x for x in body.get("inputs") or [] if x in INPUT_KINDS]
    tasks = [x for x in body.get("tasks") or [] if x in TASKS]
    if not inputs or not tasks:
        raise LibError(f"choose at least one input kind ({', '.join(INPUT_KINDS)}) and task ({', '.join(TASKS)})")
    layers = body.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 64:
        raise LibError("layers: a list of 1 to 64 layers")
    ok = False
    for kind in inputs:                         # must build at its reference input for at least one declared kind
        try:
            archspec.normalize({"input": REFERENCE[kind], "layers": layers, "outputs": 4})
            ok = True
        except archspec.SpecError as e:
            err = e
    if not ok:
        raise LibError(f"layers do not build at the reference input: {err}")
    norm = archspec.normalize({"input": REFERENCE[inputs[0]], "layers": layers, "outputs": 4})["layers"] \
        if ok else layers
    return {"name": name, "family": str(body.get("family") or "Custom")[:60], "size": str(body.get("size") or "")[:8],
            "inputs": inputs, "tasks": tasks, "layers": norm, "summary": str(body.get("summary") or "")[:400],
            "notes": str(body.get("notes") or "")[:400]}


def save(body: dict, user: dict, global_: bool = False, aid: str | None = None) -> dict:
    d = validate(body)
    org = None if global_ else user["org_id"]
    now = db.now()
    with db.conn() as c:
        if aid:
            r = c.execute("SELECT * FROM architectures WHERE id=?", (aid,)).fetchone()
            if not r:
                raise LibError("unknown architecture")
            c.execute("UPDATE architectures SET data=?, updated=? WHERE id=?", (db.j(d), now, aid))
        else:
            slug = re.sub(r"[^a-z0-9]+", "-", d["name"].lower()).strip("-")[:24] or "arch"
            aid = f"{slug}-{uuid.uuid4().hex[:4]}"
            c.execute("INSERT INTO architectures VALUES (?,?,?,?,?,?,?)",
                      (aid, org, "published", db.j(d), user["id"], now, now))
    return get(aid, user["org_id"], platform=True)


def delete(aid: str):
    with db.conn() as c:
        c.execute("DELETE FROM architectures WHERE id=?", (aid,))
        c.execute("DELETE FROM architecture_checks WHERE arch_id=?", (aid,))


# ---- compatibility with a real dataset -----------------------------------------------------------------------------
def input_kind(shape) -> str:
    c, h, w = (int(v) for v in shape)
    return "timeseries" if h == 1 and w > 1 and c >= 1 and w >= 8 else "2d" if h > 1 else "features"


def compatible(arch: dict, shape, task: str, n_out: int) -> dict:
    """{ok, reason, params, macs, receptive_field} for this architecture on this (preprocessed) input."""
    c, h, w = (int(v) for v in shape)
    kind = input_kind(shape)
    if task not in arch["tasks"]:
        return {"ok": False, "reason": f"made for {', '.join(arch['tasks'])}, not {task}"}
    kinds = arch["inputs"]
    feature_ok = "features" in kinds and h == 1       # a dense network takes any flat vector
    if kind not in kinds and not feature_ok:
        return {"ok": False, "reason": f"made for {', '.join(kinds)} input; this data is {kind} {c}x{h}x{w}"}
    try:
        spec = archspec.normalize({"input": {"channels": c, "height": h, "width": w}, "layers": arch["layers"],
                                   "outputs": n_out})
    except archspec.SpecError as e:
        return {"ok": False, "reason": str(e)}
    s = archspec.summary(spec)
    from .recommend import receptive_field
    conv = any(layer["op"] in ("conv", "dwconv", "pwconv", "residual") for layer in arch["layers"])
    return {"ok": True, "reason": "", "params": s["params"], "macs": s["macs"],
            "receptive_field": receptive_field(spec["layers"]) if conv else None}


# ---- SAI verification ----------------------------------------------------------------------------------------------
def checks(aid: str | None = None) -> dict:
    """Latest SAI check per architecture and target: {arch_id: {target: row}}."""
    q = "SELECT * FROM architecture_checks" + (" WHERE arch_id=?" if aid else "") + " ORDER BY checked"
    with db.conn() as c:
        rows = [dict(r) for r in c.execute(q, (aid,) if aid else ())]
    out: dict = {}
    for r in rows:
        r["detail"] = db.uj(r["detail"], {})
        out.setdefault(r["arch_id"], {})[r["target"]] = r
    return out


def verify(arch: dict, targets, workdir: Path, log=lambda m: None) -> dict:
    """Export at the reference input with random weights and run SAI's plan search per board; store the result."""
    import torch

    from . import qexport, sai_engine
    kind = arch["inputs"][0]
    inp = REFERENCE[kind]
    n_out = int(np.prod([inp["channels"], inp["height"], inp["width"]])) if arch["tasks"] == ["anomaly"] else 4
    spec = archspec.normalize({"input": inp, "layers": arch["layers"], "outputs": n_out})
    torch.manual_seed(0)
    net = archspec.SpecNet(spec)
    net.eval()
    net.fold_bn()
    calib = np.random.default_rng(0).normal(0, 1, (32, inp["channels"], inp["height"], inp["width"])).astype(np.float32)
    path = workdir / f"arch_{arch['id']}.onnx"
    qexport.export_qdq(net, calib, path)
    summ = archspec.summary(spec)
    sai_v = (sai_engine.version() or {}).get("sai", "unknown")
    out = {}
    try:
        plans = sai_engine.plan_search(path, summ["macs"], summ["spatial_axes"], list(targets), "balanced",
                                       explore_seconds=3, explore_builds=0)
    finally:
        path.unlink(missing_ok=True)
    for t in targets:
        rows = [r for r in plans["candidates"] if r["target"] == t]
        fit = [r for r in rows if r["fits"]]
        best = sai_engine._pick(fit, "balanced") if fit else (min(rows, key=lambda r: len(r["violations"])) if rows else None)
        err = next((e["error"] for e in plans.get("errors", []) if e["target"] == t), None)
        row = {"arch_id": arch["id"], "target": t, "sai_version": sai_v, "ok": int(best is not None),
               "fits": int(bool(fit)), "ram": best and best["ram"], "flash": best and best["flash"],
               "latency_ms": best and best["latency"]["ms"], "plan": best and best["schedule"],
               "detail": db.j({"input": inp, "params": summ["params"], "macs": summ["macs"],
                               "violations": best["violations"] if best else [], "error": err}),
               "checked": db.now()}
        with db.conn() as c:
            c.execute("INSERT INTO architecture_checks VALUES (:arch_id,:target,:sai_version,:ok,:fits,:ram,:flash,"
                      ":latency_ms,:plan,:detail,:checked)", row)
        out[t] = {**row, "detail": json.loads(row["detail"])}
        log(f"{arch['id']} on {t}: " + (f"{row['ram']} B RAM, {row['flash']} B flash, {row['latency_ms']} ms"
                                       f"{'' if row['fits'] else ' (does not fit this board)'}" if best else
                                       f"SAI produced no plan ({err})"))
    return out
