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
LSTM = lambda u, bi=False, seq=False: {"op": "lstm", "units": u, "bidirectional": bi, "sequences": seq}  # noqa: E731
GRU = lambda u, bi=False, seq=False: {"op": "gru", "units": u, "bidirectional": bi, "sequences": seq}  # noqa: E731
RNN = lambda u: {"op": "rnn", "units": u, "bidirectional": False, "sequences": False}  # noqa: E731
ATT = lambda h=2: {"op": "attention", "heads": h}  # noqa: E731
LN = {"op": "layernorm"}
CV = lambda out, k=5, stride=2: {"op": "conv", "out": out, "k": k, "stride": stride}  # noqa: E731
CR = ["classification", "regression"]


def _a(aid, name, family, size, inputs, tasks, layers, summary, notes="", ref=""):
    """ref: the published design this entry follows, and how it was adapted to the platform's operator set."""
    return {"id": aid, "name": name, "family": family, "size": size, "inputs": inputs, "tasks": tasks,
            "layers": layers, "summary": summary, "notes": notes, "ref": ref, "source": "builtin",
            "status": "published", "org_id": None}


FA = ["anomaly"]
CRA = ["classification", "regression"]


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

    # ==== research additions: published TinyML designs, adapted to the platform's operator set ===================
    # (identity residual = conv-conv-add with equal channels; square kernels <= 31; dense layers flatten)
    # ---- 1-D ----
    _a("ib-tcn-s", "Inverted-bottleneck TCN S", "Inverted-bottleneck TCN", "s", TS, CRA,
       [C(16, 3, 2), PW(48), DW(9), PW(16), PW(48), DW(9, 2), PW(24), PW(72), DW(9), PW(24), GAP],
       "Expand (1x1) - depthwise k9 - project blocks: wide temporal context at very few parameters.",
       "Good for keyword-like events in 1-D signals and sensor bursts.",
       "TENet (Li et al., Interspeech 2020) inverted bottleneck blocks; block skip connections removed "
       "(the platform's residual only wraps two equal-width convolutions)."),
    _a("sep-largek-tcn", "Large-kernel separable TCN", "Time-channel separable TCN", "m", TS, CRA,
       [C(32, 11, 2), DW(13), PW(32), DW(15), PW(32), DW(17), PW(32), DW(21, 1, 2), PW(32), PW(32), GAP, DROP],
       "Depthwise kernels 13-21 (the last dilated) on a 32-channel stem: receptive field of ~150 samples with "
       "few multiplies.", "Slow, long patterns: battery cycles, ECG beats, machine cycles. Cortex-M7 class parts: at a "
       "3x256 window it needs more than MD6605's 4 KB AI arena.",
       "MatchboxNet / QuartzNet (NVIDIA, 2020) time-channel separable blocks; single repeats, no residuals."),
    _a("strided-ds-tcn", "Strided DS-TCN (no pooling)", "Depthwise-separable TCN", "m", TS, CRA,
       [C(12, 5, 2), DW(3, 2), PW(16), DW(3, 2), PW(24), DW(3, 2), PW(32), GAP],
       "Downsamples with stride-2 depthwise convolutions instead of max-pooling: fewer layers, learned "
       "downsampling.", "", "MobileNet-style strided depthwise downsampling applied to 1-D."),
    _a("raw-cnn-m5", "Raw-signal CNN (M5-style)", "Raw-waveform CNN", "m", TS, CRA,
       [C(32, 31, 4), MP(4), C(32, 3), MP(4), C(64, 3), MP(4), C(64, 3), GAP],
       "A long strided first kernel on the raw waveform, then three small convolutions with 4x pooling.",
       "Raw audio or vibration at high sample rates (windows of 1,024+ samples).",
       "M5 (Dai et al., ICASSP 2017); first kernel 80 -> 31 (the platform's maximum)."),
    _a("res-tcn-s", "Residual TCN S", "Residual TCN", "s", TS, CRA,
       [C(12, 5, 2), RES(3), MP(2), PW(16), RES(3), GAP],
       "Two identity residual blocks: the smallest residual network, trains stably on small datasets."),
    _a("tcn-dense-head", "TCN + dense head", "Depthwise-separable TCN", "m", TS, CRA,
       [C(12, 5, 4), DW(3), PW(16), MP(2), DW(3), PW(24), MP(2), FC(32), DROP],
       "Keeps the temporal order to the end (flatten instead of global average), then a dense layer.",
       "When *where* in the window something happens matters (e.g. fault position in a cycle).",
       "Classic CNN-1D + dense classifier head used in many HAR and fault-diagnosis papers."),
    # ---- 2-D ----
    _a("ds-cnn-2d-l", "DS-CNN 2D L (64 ch)", "Depthwise-separable CNN-2D", "l", D2, CRA,
       [C(64, 3, 2), DW(3), PW(64), DW(3), PW(64), DW(3), PW(64), DW(3), PW(64), GAP, DROP],
       "Four 64-channel depthwise-separable blocks: the reference keyword-spotting size.",
       "Cortex-M7 class parts; large for MD6605.",
       "DS-CNN (Zhang et al., 'Hello Edge', 2017; MLPerf Tiny KWS); stem 10x4 kernel -> square 3x3."),
    _a("mobilenet-v1-025", "MobileNetV1-style 0.25", "MobileNet", "m", D2, CRA,
       [C(8, 3, 2), DW(3), PW(16), DW(3, 2), PW(32), DW(3), PW(32), DW(3, 2), PW(64), DW(3), PW(64), GAP],
       "Width-0.25 MobileNet body: strided depthwise downsampling and doubling channels.",
       "Small images (thermal, low-resolution grayscale).",
       "MobileNetV1 (Howard et al., 2017) at width 0.25 as in MLPerf Tiny visual wake words; truncated depth."),
    _a("resnet8-style", "ResNet-style 2D S", "Residual CNN-2D", "s", D2, CRA,
       [C(8, 3), RES(3), C(16, 3, 2), RES(3), C(32, 3, 2), RES(3), GAP],
       "Three stages of identity residual blocks with strided convolutions between them.",
       "", "ResNet-8 (MLPerf Tiny image classification) at half width (8-16-32); projection shortcuts replaced "
           "by a strided convolution before each identity residual block."),
    _a("tinyconv", "TinyConv (micro-speech style)", "Single-layer CNN", "xs", D2, CR,
       [C(8, 9, 2), FC(16)],
       "One strided 9x9 convolution and a small dense layer: the smallest spectrogram model that works.",
       "First bring-up on a new board; 2-4 keyword classes.",
       "TensorFlow Lite Micro 'micro_speech' tiny_conv (8 filters 10x8, stride 2); kernel made square."),
    _a("lenet5-style", "LeNet-5 style", "Classic CNN-2D", "s", D2, CR,
       [C(6, 5), AP(2), C(16, 5), AP(2), FC(64), FC(32)],
       "Two 5x5 convolutions with average pooling and two dense layers.",
       "Tiny images where spatial layout matters (thermal arrays, 8x8 to 32x32).",
       "LeNet-5 (LeCun et al., 1998) with SAME padding and smaller dense layers."),
    _a("vgg-tiny-2d", "VGG-tiny 2D", "Classic CNN-2D", "s", D2, CR,
       [C(8, 3), C(8, 3), MP(2), C(16, 3), C(16, 3), MP(2), GAP],
       "Pairs of 3x3 convolutions before each pooling: simple, regular, compiler-friendly.",
       "", "VGG (Simonyan & Zisserman, 2014) pattern at MCU scale."),
    # ---- feature vectors ----
    _a("linear", "Linear baseline", "Linear model", "xs", FV + TS, CR, [DROP],
       "No hidden layer: a linear (logistic) model on the flattened input. The baseline every other model "
       "must beat.", "Use it to check that a deeper model earns its RAM and flash."),
    _a("mlp-l", "MLP L", "Dense network", "l", FV, CR, [FC(128), DROP, FC(64), FC(32)],
       "Three wider dense layers for rich feature vectors (spectral statistics, many sensors)."),
    # ---- sequence models: SAI front end, XModels backend (evaluation only until the front end is qualified) ----
    _a("lstm-s", "LSTM S", "Recurrent (XModels)", "s", TS + D2, CRA, [LSTM(16)],
       "One 16-unit LSTM over the window, classifying from its last state.",
       "Sequence model: compiled by SAI's front end (XModels); packages are evaluation-only for now.",
       "Classic LSTM classifier (Hochreiter & Schmidhuber, 1997) at MCU scale."),
    _a("gru-s", "GRU S", "Recurrent (XModels)", "s", TS + D2, CRA, [GRU(24)],
       "One 24-unit GRU: fewer gates than an LSTM, often as accurate on sensor data.",
       "Sequence model (XModels); evaluation-only packages.", "GRU (Cho et al., 2014)."),
    _a("rnn-s", "SimpleRNN S", "Recurrent (XModels)", "xs", TS + D2, CRA, [RNN(16)],
       "The smallest recurrent model: a 16-unit tanh RNN.", "Sequence model (XModels); evaluation-only packages."),
    _a("bilstm-s", "BiLSTM S", "Recurrent (XModels)", "s", TS + D2, CRA, [LSTM(12, bi=True)],
       "Reads the window forwards and backwards (12 units each).",
       "More RAM than one direction; Cortex-M7 class parts. Sequence model (XModels); evaluation-only packages.",
       "Bidirectional RNN (Schuster & Paliwal, 1997)."),
    _a("lstm-stack", "Stacked LSTM M", "Recurrent (XModels)", "m", TS + D2, CRA,
       [LSTM(24, seq=True), LSTM(24)], "Two stacked LSTMs (24 units): the first passes its full sequence on.",
       "Sequence model (XModels); evaluation-only packages."),
    _a("crnn-s", "CRNN S", "Convolutional-recurrent (XModels)", "s", TS, CRA, [CV(16, 5, 2), GRU(16)],
       "A strided convolution shortens and enriches the sequence, then a GRU reads it.",
       "Long windows (the convolution halves the steps). Sequence model (XModels); evaluation-only packages.",
       "CRNN (e.g. DeepConvLSTM-style conv + recurrent layers) with valid padding."),
    _a("tiny-transformer", "Tiny transformer encoder", "Attention (XModels)", "s", TS + D2, CRA,
       [{"op": "fc", "out": 16, "relu": True}, ATT(2), {"op": "fc", "out": 16, "relu": True}, {"op": "gap"}],
       "Per-step projection to 16 features, one 2-head self-attention block with LayerNorm, average over time.",
       "Cortex-M7 class parts (attention RAM grows with the square of the steps; a 256-step window needs ~20 KB). "
       "Sequence model (XModels); evaluation-only packages.",
       "Transformer encoder (Vaswani et al., 2017) at KWT-style MCU scale (Berg et al., 2021)."),
    _a("lstm-ln", "LSTM + LayerNorm", "Recurrent (XModels)", "s", TS + D2, CRA, [LSTM(16, seq=True), LN, {"op": "gap"}],
       "LSTM over the whole window, LayerNorm, then the average over time (uses every step, not just the last).",
       "Keeps every step: Cortex-M7 class parts for long windows. Sequence model (XModels); evaluation-only packages."),
    _a("ae-l", "Autoencoder L (MLPerf-style)", "Dense autoencoder", "l", FV, FA,
       [FC(128), FC(128), FC(8), FC(128), FC(128)],
       "Two 128-unit layers on each side of an 8-value bottleneck.", "Trained on normal data only.",
       "MLPerf Tiny anomaly-detection FC autoencoder (ToyADMOS), 4 hidden layers per side reduced to 2."),
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


def random_plans(spec: dict, targets, workdir: Path, goal: str = "balanced", budgets: dict | None = None,
                 log=lambda m: None, name: str = "model") -> dict:
    """Plans per board for an untrained architecture (random weights; RAM, flash and plan do not depend on them):
    SAI Core's plan search for CNNs, SAI's front end (XModels) for sequence models."""
    import torch

    from . import qexport, sai_engine, seqfront
    inp = spec["input"]
    torch.manual_seed(0)
    net = archspec.SpecNet(spec)
    net.eval()
    rng = np.random.default_rng(0)
    summ = archspec.summary(spec)
    if spec.get("layout") == "sequence":
        if not seqfront.available():
            raise RuntimeError(seqfront.why_unavailable())
        t, f = archspec.seq_dims(inp)
        path = workdir / f"{name}_float.onnx"
        seqfront.export_float(net, spec, path)
        data = workdir / f"{name}_calib.npz"
        seqfront.write_data(data, rng.normal(0, 1, (32, t, f)).astype(np.float32), None, "regression")
        try:
            return seqfront.plans(path, list(targets), goal, budgets, workdir / f"{name}_xm", data, None, log=log)
        finally:
            path.unlink(missing_ok=True)
            data.unlink(missing_ok=True)
    net.fold_bn()
    calib = rng.normal(0, 1, (32, inp["channels"], inp["height"], inp["width"])).astype(np.float32)
    path = workdir / f"{name}.onnx"
    qexport.export_qdq(net, calib, path)
    try:
        return sai_engine.plan_search(path, summ["macs"], summ["spatial_axes"], list(targets), goal, budgets,
                                      explore_seconds=3, explore_builds=0, log=log)
    finally:
        path.unlink(missing_ok=True)


def verify(arch: dict, targets, workdir: Path, log=lambda m: None) -> dict:
    """Export at the reference input with random weights and run SAI's plan search per board; store the result."""
    from . import sai_engine
    kind = arch["inputs"][0]
    inp = REFERENCE[kind]
    n_out = int(np.prod([inp["channels"], inp["height"], inp["width"]])) if arch["tasks"] == ["anomaly"] else 4
    spec = archspec.normalize({"input": inp, "layers": arch["layers"], "outputs": n_out})
    summ = archspec.summary(spec)
    sai_v = (sai_engine.version() or {}).get("sai", "unknown")
    if spec.get("layout") == "sequence":
        sai_v = f"{sai_v} + sai_front (XModels)"
    out = {}
    plans = random_plans(spec, targets, workdir, name=f"arch_{arch['id']}")
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
