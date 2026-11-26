# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Architecture specs restricted to what SAI compiles.

A spec is plain JSON so researchers can author it in the Studio UI and customers never see it:

    {"input": {"channels": 3, "height": 1, "width": 128},          # height 1 = a 1-D signal (time on width)
     "layers": [{"op": "conv", "out": 16, "k": 5, "relu": true},
                {"op": "dwconv", "k": 3, "relu": true},
                {"op": "pwconv", "out": 24, "relu": true},
                {"op": "maxpool", "k": 2},
                {"op": "residual", "k": 3},
                {"op": "gap"},
                {"op": "fc", "out": 32, "relu": true},
                {"op": "dropout", "p": 0.1}],
     "outputs": 4}                                                 # appended head: fc(outputs), no activation

Every layer maps onto SAI's 12 operators (Conv, Depthwise Conv, FC, Mean, Add, MaxPool, AveragePool, Reshape ...).
Dropout is training-only and is not exported. One code path: a 1-D signal is NCHW with H = 1 and kernels (1, k),
which is how SAI's ONNX front end reads a one-spatial-axis graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    "conv": {"params": {"out": int, "k": int, "stride": int, "dilation": int, "relu": bool}, "sai": "Conv"},
    "dwconv": {"params": {"k": int, "stride": int, "dilation": int, "relu": bool}, "sai": "Depthwise Conv"},
    "pwconv": {"params": {"out": int, "relu": bool}, "sai": "Conv (1x1)"},
    "maxpool": {"params": {"k": int}, "sai": "MaxPool"},
    "avgpool": {"params": {"k": int}, "sai": "AveragePool"},
    "gap": {"params": {}, "sai": "Mean"},
    "fc": {"params": {"out": int, "relu": bool}, "sai": "Fully Connected"},
    "residual": {"params": {"k": int}, "sai": "Conv + Conv + Add"},
    "dropout": {"params": {"p": float}, "sai": "(training only)"},
    # sequence layers: compiled by SAI's front end through the XModels backend (SAI_REPO on product/sai-unified)
    "lstm": {"params": {"units": int, "bidirectional": bool, "sequences": bool}, "sai": "LSTM (XModels)"},
    "gru": {"params": {"units": int, "bidirectional": bool, "sequences": bool}, "sai": "GRU (XModels)"},
    "rnn": {"params": {"units": int, "bidirectional": bool, "sequences": bool}, "sai": "SimpleRNN (XModels)"},
    "attention": {"params": {"heads": int}, "sai": "Self-attention + LayerNorm (XModels)"},
    "layernorm": {"params": {}, "sai": "LayerNorm (XModels)"},
}
RECURRENT = {"lstm": 4, "gru": 3, "rnn": 1}            # op -> gates
SEQ_OPS = set(RECURRENT) | {"attention", "layernorm"}
# operators a sequence model may use: in it, `conv` is a 1-D convolution over time with valid padding, `fc` before
# the time reduction is applied to every time step, `gap` averages over time
SEQ_ALLOWED = SEQ_OPS | {"conv", "fc", "gap", "dropout"}


def is_sequence(spec: dict) -> bool:
    return any(layer["op"] in SEQ_OPS for layer in spec.get("layers") or [])


def seq_dims(inp: dict) -> tuple[int, int]:
    """(time steps, features per step) a sequence model sees: 1-D [C, 1, T] -> T steps of C; 2-D [C, H, W] -> H
    steps of C*W (rows of a spectrogram or image)."""
    c, h, w = inp["channels"], inp["height"], inp["width"]
    return (w, c) if h == 1 else (h, c * w)


def to_sequence(X):
    """Data [N, C, H, W] -> [N, T, F] in the layout of seq_dims (what sequence models are trained and fed with)."""
    n, c, h, w = X.shape
    if h == 1:
        return X[:, :, 0, :].transpose(0, 2, 1).copy()
    return X.transpose(0, 2, 1, 3).reshape(n, h, c * w).copy()


class SpecError(ValueError):
    pass


@dataclass
class Shape:
    c: int
    h: int
    w: int
    flat: bool = False      # after fc / flatten: [N, c]

    def tuple(self):
        return (self.c,) if self.flat else (self.c, self.h, self.w)

    @property
    def size(self):
        return self.c if self.flat else self.c * self.h * self.w


def _same_pads(size: int, k: int, s: int, d: int) -> tuple[int, int, int]:
    """TFLite SAME: out = ceil(size / s). Returns (begin, end, out)."""
    out = math.ceil(size / s)
    span = (k - 1) * d + 1
    total = max((out - 1) * s + span - size, 0)
    return total // 2, total - total // 2, out


def normalize(spec: dict) -> dict:
    """Validate and fill defaults. Raises SpecError with a user-facing message."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be an object")
    inp = spec.get("input") or {}
    try:
        c, h, w = int(inp["channels"]), int(inp.get("height", 1)), int(inp["width"])
    except (KeyError, TypeError, ValueError):
        raise SpecError("input needs integer channels and width (height defaults to 1)") from None
    if not (1 <= c <= 512 and 1 <= h <= 512 and 1 <= w <= 8192):
        raise SpecError("input dimensions out of range")
    outputs = int(spec.get("outputs", 2))
    if not 1 <= outputs <= 1024:
        raise SpecError("outputs must be 1..1024")
    layers = []
    for i, raw in enumerate(spec.get("layers") or []):
        op = raw.get("op")
        if op not in OPS:
            raise SpecError(f"layer {i}: unknown op {op!r}; SAI-compatible ops are {', '.join(OPS)}")
        layer = {"op": op}
        for name, typ in OPS[op]["params"].items():
            if name in raw and raw[name] is not None and raw[name] != "":
                try:
                    layer[name] = typ(raw[name]) if typ is not bool else bool(raw[name])
                except (TypeError, ValueError):
                    raise SpecError(f"layer {i} ({op}): {name} must be {typ.__name__}") from None
        layer.setdefault("stride", 1) if op in ("conv", "dwconv") else None
        layer.setdefault("dilation", 1) if op in ("conv", "dwconv") else None
        layer.setdefault("relu", True) if op in ("conv", "dwconv", "pwconv") else None
        layer.setdefault("relu", False) if op == "fc" else None
        if op in ("conv", "dwconv", "maxpool", "avgpool", "residual"):
            layer.setdefault("k", 3 if op != "maxpool" and op != "avgpool" else 2)
            if not 1 <= layer["k"] <= 31:
                raise SpecError(f"layer {i} ({op}): k must be 1..31")
        if op in ("conv", "pwconv", "fc"):
            if "out" not in layer or not 1 <= layer["out"] <= 1024:
                raise SpecError(f"layer {i} ({op}): out must be 1..1024")
        if op == "dropout":
            layer.setdefault("p", 0.1)
        if op in RECURRENT:
            layer.setdefault("units", 32)
            layer.setdefault("bidirectional", False)
            layer.setdefault("sequences", False)
            if not 1 <= layer["units"] <= 512:
                raise SpecError(f"layer {i} ({op}): units must be 1..512")
        if op == "attention":
            layer.setdefault("heads", 2)
            if not 1 <= layer["heads"] <= 16:
                raise SpecError(f"layer {i} (attention): heads must be 1..16")
        layers.append(layer)
    norm = {"input": {"channels": c, "height": h, "width": w}, "layers": layers, "outputs": outputs}
    if any(layer["op"] in SEQ_OPS for layer in layers):
        bad = sorted({layer["op"] for layer in layers if layer["op"] not in SEQ_ALLOWED})
        if bad:
            raise SpecError(f"a sequence model (LSTM, GRU, RNN, attention, LayerNorm) can also use conv, fc, gap and "
                            f"dropout, not {', '.join(bad)}")
        norm["layout"] = "sequence"
    shapes(norm)    # raises on an impossible graph
    return norm


def _seq_shapes(spec: dict) -> list[dict]:
    """Shapes of a sequence model: [T, F] while it is a sequence, [F] after the time reduction."""
    t, f = seq_dims(spec["input"])
    seq, prev = True, None
    rows = []
    for i, layer in enumerate(spec["layers"] + [{"op": "fc", "out": spec["outputs"], "relu": False, "head": True}]):
        op = layer["op"]
        macs = params = 0
        if op in ("conv", "lstm", "gru", "rnn", "attention", "layernorm", "gap") and not seq:
            raise SpecError(f"layer {i} ({op}): needs the sequence, but the time axis was already reduced (by a "
                            "recurrent layer with sequences off, or by gap)")
        if op == "conv":
            k, st, d = layer["k"], layer["stride"], layer["dilation"]
            nt = (t - (k - 1) * d - 1) // st + 1
            if nt < 1:
                raise SpecError(f"layer {i} (conv): {t} time steps are shorter than the kernel (valid padding)")
            macs, params = nt * layer["out"] * f * k, layer["out"] * f * k + layer["out"]
            t, f = nt, layer["out"]
        elif op in RECURRENT:
            g, u, dirs = RECURRENT[op], layer["units"], 2 if layer["bidirectional"] else 1
            params = dirs * g * (f * u + u * u + 2 * u)
            macs = dirs * g * t * (f * u + u * u)
            f = dirs * u
            seq = layer["sequences"]
        elif op == "attention":
            if f % layer["heads"]:
                raise SpecError(f"layer {i} (attention): {f} features do not split into {layer['heads']} heads")
            params = 4 * f * f + 4 * f + 2 * f
            macs = 4 * t * f * f + 2 * t * t * f
        elif op == "layernorm":
            params, macs = 2 * f, 2 * t * f
        elif op == "gap":
            if prev in RECURRENT:
                raise SpecError(f"layer {i} (gap): averaging over time right after a recurrent layer is not "
                                "supported; put a fc or layernorm layer in between, or turn off 'sequences' to use "
                                "the last state")
            macs, seq = t * f, False
        elif op == "fc":
            if layer.get("head") and seq:
                raise SpecError("end the sequence part before the output: a recurrent layer with sequences off "
                                "(last state) or gap (average over time)")
            n = t if seq else 1
            macs, params = n * f * layer["out"], f * layer["out"] + layer["out"]
            f = layer["out"]
        if op != "dropout":
            prev = op
        shape = [t, f] if seq else [f]
        rows.append({"index": i, "op": op, "head": bool(layer.get("head")), "shape": shape, "macs": int(macs),
                     "params": int(params), "activation_bytes": int(t * f if seq else f)})
    return rows


def shapes(spec: dict) -> list[dict]:
    """Per-layer output shapes, MACs and parameters (head included). Raises SpecError."""
    if spec.get("layout") == "sequence" or is_sequence(spec):
        return _seq_shapes(spec)
    inp = spec["input"]
    s = Shape(inp["channels"], inp["height"], inp["width"])
    one_d = s.h == 1
    rows = []
    for i, layer in enumerate(spec["layers"] + [{"op": "fc", "out": spec["outputs"], "relu": False, "head": True}]):
        op = layer["op"]
        macs = params = 0
        if op in ("conv", "dwconv", "pwconv", "residual") and s.flat:
            raise SpecError(f"layer {i} ({op}): convolution after a flatten/fc is not possible")
        if op in ("conv", "dwconv"):
            k, st, d = layer["k"], layer["stride"], layer["dilation"]
            _, _, ow = _same_pads(s.w, k, st, d)
            if one_d:
                oh, kk = 1, k
            else:
                _, _, oh = _same_pads(s.h, k, st, d)
                kk = k * k
            cout = layer["out"] if op == "conv" else s.c
            per = s.c if op == "conv" else 1
            macs = cout * oh * ow * per * kk
            params = cout * per * kk + cout
            s = Shape(cout, oh, ow)
        elif op == "pwconv":
            macs = layer["out"] * s.h * s.w * s.c
            params = layer["out"] * s.c + layer["out"]
            s = Shape(layer["out"], s.h, s.w)
        elif op == "residual":
            kk = layer["k"] if one_d else layer["k"] ** 2
            macs = 2 * s.c * s.h * s.w * s.c * kk + s.c * s.h * s.w
            params = 2 * (s.c * s.c * kk + s.c)
        elif op in ("maxpool", "avgpool"):
            if s.flat:
                raise SpecError(f"layer {i} ({op}): pooling needs a spatial tensor")
            k = layer["k"]
            ow, oh = s.w // k, (1 if one_d else s.h // k)
            if ow < 1 or oh < 1:
                raise SpecError(f"layer {i} ({op}): input {s.tuple()} is smaller than the pool k={k}")
            macs = s.c * oh * ow * (k if one_d else k * k)
            s = Shape(s.c, oh, ow)
        elif op == "gap":
            if s.flat:
                raise SpecError(f"layer {i} (gap): needs a spatial tensor")
            macs = s.c * s.h * s.w
            s = Shape(s.c, 1, 1)
        elif op == "fc":
            macs = s.size * layer["out"]
            params = s.size * layer["out"] + layer["out"]
            s = Shape(layer["out"], 1, 1, flat=True)
        rows.append({"index": i, "op": op, "head": bool(layer.get("head")), "shape": list(s.tuple()),
                     "macs": int(macs), "params": int(params), "activation_bytes": int(s.size)})
    return rows


def summary(spec: dict) -> dict:
    rows = shapes(spec)
    inp = spec["input"]
    return {"layers": rows, "macs": sum(r["macs"] for r in rows), "params": sum(r["params"] for r in rows),
            "weight_bytes_int8": sum(r["params"] for r in rows),   # ~1 B per weight, biases are int32 (small)
            "input_bytes": inp["channels"] * inp["height"] * inp["width"],
            "peak_activation_bytes": max([r["activation_bytes"] for r in rows] + [inp["channels"] * inp["height"] * inp["width"]]),
            "spatial_axes": 1 if inp["height"] == 1 or spec.get("layout") == "sequence" else 2,
            "backend": "xmodels" if spec.get("layout") == "sequence" else "core"}


def fake_quant(x, scale, zp):
    """INT8 fake quantization with a straight-through gradient."""
    q = (torch.clamp(torch.round(x / scale) + zp, -128, 127) - zp) * scale
    return x + (q - x).detach()


def fake_quant_weight(w):
    """Symmetric per-output-channel INT8, as the exporter quantizes weights."""
    s = w.detach().abs().reshape(w.shape[0], -1).amax(1).clamp_min(1e-8) / 127.0
    s = s.reshape((-1,) + (1,) * (w.ndim - 1))
    q = torch.clamp(torch.round(w / s), -127, 127) * s
    return w + (q - w).detach()


class _Conv(nn.Module):
    """Conv2d with explicit SAME padding (asymmetric where needed, as TFLite/ONNX export it) and a BatchNorm that
    is folded into the weights before quantization (fold_bn), so the exported graph is a plain Conv."""

    def __init__(self, cin, cout, k, stride, dilation, groups, one_d):
        super().__init__()
        self.one_d = one_d
        ks = (1, k) if one_d else (k, k)
        st = (1, stride) if one_d else (stride, stride)
        dl = (1, dilation) if one_d else (dilation, dilation)
        self.k, self.stride, self.dilation = k, stride, dilation
        self.conv = nn.Conv2d(cin, cout, ks, st, 0, dl, groups=groups, bias=True)
        self.bn = nn.BatchNorm2d(cout)
        self.qat = False

    def pads(self, h, w):
        bw, ew, _ = _same_pads(w, self.k, self.stride, self.dilation)
        if self.one_d:
            return 0, 0, bw, ew
        bh, eh, _ = _same_pads(h, self.k, self.stride, self.dilation)
        return bh, eh, bw, ew

    @torch.no_grad()
    def fold_bn(self):
        if isinstance(self.bn, nn.Identity):
            return
        bn = self.bn
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        self.conv.weight.mul_(g.reshape(-1, 1, 1, 1))
        self.conv.bias.copy_((self.conv.bias - bn.running_mean) * g + bn.bias)
        self.bn = nn.Identity()

    def forward(self, x):
        bh, eh, bw, ew = self.pads(x.shape[2], x.shape[3])
        self._export_pads = [bh, bw, eh, ew]          # ONNX order: h_begin, w_begin, h_end, w_end
        w = fake_quant_weight(self.conv.weight) if self.qat else self.conv.weight
        y = F.conv2d(F.pad(x, (bw, ew, bh, eh)), w, self.conv.bias, self.conv.stride, 0, self.conv.dilation,
                     self.conv.groups)
        return self.bn(y)


class _Linear(nn.Linear):
    qat = False

    def forward(self, x):
        return F.linear(x, fake_quant_weight(self.weight) if self.qat else self.weight, self.bias)


class _SelfAttention(nn.Module):
    """Self-attention block: LayerNorm(x + MultiHeadAttention(x, x, x)), the pattern SAI's front end rebuilds."""

    def __init__(self, d, heads):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.att(x, x, x, need_weights=False)[0])


class SpecNet(nn.Module):
    """A torch module built from a normalized spec. forward(x, record) fills `record[name] = tensor` for every
    tensor the exporter quantizes (names: "x", "L{i}", "L{i}_a" for the residual's inner tensor)."""

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec
        inp = spec["input"]
        self.one_d = inp["height"] == 1
        c = inp["channels"]
        flat_size = None
        self.mods = nn.ModuleList()
        self.layers = spec["layers"] + [{"op": "fc", "out": spec["outputs"], "relu": False, "head": True}]
        rows = shapes(spec)
        self.sequence = spec.get("layout") == "sequence"
        if self.sequence:
            self._build_sequence(rows)
            return
        prev_shape = [inp["channels"], inp["height"], inp["width"]]
        for layer, row in zip(self.layers, rows):
            op = layer["op"]
            if op == "conv":
                self.mods.append(_Conv(c, layer["out"], layer["k"], layer["stride"], layer["dilation"], 1, self.one_d))
                c = layer["out"]
            elif op == "dwconv":
                self.mods.append(_Conv(c, c, layer["k"], layer["stride"], layer["dilation"], c, self.one_d))
            elif op == "pwconv":
                self.mods.append(_Conv(c, layer["out"], 1, 1, 1, 1, self.one_d))
                c = layer["out"]
            elif op == "residual":
                self.mods.append(nn.ModuleList([_Conv(c, c, layer["k"], 1, 1, 1, self.one_d),
                                                _Conv(c, c, layer["k"], 1, 1, 1, self.one_d)]))
            elif op == "fc":
                fin = int(math.prod(prev_shape))
                self.mods.append(_Linear(fin, layer["out"]))
            elif op == "dropout":
                self.mods.append(nn.Dropout(layer["p"]))
            else:
                self.mods.append(nn.Identity())
            prev_shape = row["shape"]

    def _build_sequence(self, rows):
        t, f = seq_dims(self.spec["input"])
        seq = True
        for layer, row in zip(self.layers, rows):
            op = layer["op"]
            if op == "conv":
                self.mods.append(nn.Conv1d(f, layer["out"], layer["k"], layer["stride"], 0, layer["dilation"]))
            elif op in RECURRENT:
                cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[op]
                self.mods.append(cls(f, layer["units"], batch_first=True, bidirectional=layer["bidirectional"]))
            elif op == "attention":
                self.mods.append(_SelfAttention(f, layer["heads"]))
            elif op == "layernorm":
                self.mods.append(nn.LayerNorm(f))
            elif op == "fc":
                self.mods.append(_Linear(f, layer["out"]))
            elif op == "dropout":
                self.mods.append(nn.Dropout(layer["p"]))
            else:
                self.mods.append(nn.Identity())
            f = row["shape"][-1]

    def _forward_sequence(self, x):
        """x: [N, T, F] (archspec.to_sequence)."""
        for layer, mod in zip(self.layers, self.mods):
            op = layer["op"]
            if op == "conv":
                x = mod(x.transpose(1, 2)).transpose(1, 2)
                if layer.get("relu"):
                    x = F.relu(x)
            elif op in RECURRENT:
                y = mod(x)[0]
                x = y if layer["sequences"] else y[:, -1, :]
            elif op in ("attention", "layernorm", "dropout"):
                x = mod(x)
            elif op == "gap":
                x = x.mean(1)
            elif op == "fc":
                x = mod(x)
                if layer.get("relu"):
                    x = F.relu(x)
        return x

    # ---- quantization support -------------------------------------------------------------------------------
    act_q: dict | None = None            # name -> (scale, zero_point); set for quantization-aware fine-tuning

    def fold_bn(self):
        for m in self.modules():
            if isinstance(m, _Conv):
                m.fold_bn()

    def set_qat(self, act_q: dict | None):
        self.act_q = act_q
        for m in self.modules():
            if isinstance(m, (_Conv, _Linear)):
                m.qat = act_q is not None

    def _t(self, name, x, record):
        if self.act_q is not None and name in self.act_q:
            x = fake_quant(x, *self.act_q[name])
        if record is not None:
            record[name] = x
        return x

    def forward(self, x, record: dict | None = None):
        if self.sequence:
            return self._forward_sequence(x)
        x = self._t("x", x, record)
        for i, (layer, mod) in enumerate(zip(self.layers, self.mods)):
            op = layer["op"]
            if op in ("conv", "dwconv", "pwconv"):
                x = mod(x)
                if layer.get("relu"):
                    x = F.relu(x)
            elif op == "residual":
                a = self._t(f"L{i}_a", F.relu(mod[0](x)), record)
                inner = self._t(f"L{i}__inner", mod[1](a), record)
                x = x + inner
            elif op == "maxpool":
                k = (1, layer["k"]) if self.one_d else layer["k"]
                x = F.max_pool2d(x, k, k)
            elif op == "avgpool":
                k = (1, layer["k"]) if self.one_d else layer["k"]
                x = F.avg_pool2d(x, k, k)
            elif op == "gap":
                x = x.mean(dim=(2, 3), keepdim=True)
            elif op == "fc":
                x = mod(x.flatten(1))
                if layer.get("relu"):
                    x = F.relu(x)
            elif op == "dropout":
                x = mod(x)
                continue                       # not a tensor of the exported graph
            x = self._t(f"L{i}", x, record)
        return x
