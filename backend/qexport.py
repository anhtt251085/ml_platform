"""Post-training INT8 quantization of a SpecNet and emission of the QDQ ONNX graph SAI's ONNX front end imports.

Contract (the one SAI's front end checks, see the compiler's frontend_onnx.py):
  * graph input and output INT8; activations asymmetric per-tensor INT8, calibrated on real training windows;
  * weights symmetric per-output-channel INT8 (zero point 0), biases INT32 at scale x_scale * w_scale;
  * Conv/Gemm read DequantizeLinear operands and feed a QuantizeLinear, optionally through a Relu;
  * pools keep their input's scale; intermediate INT8 tensors carry value_info shapes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

OPSET = 13


def act_q(lo: float, hi: float) -> tuple[float, int]:
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = float(np.float32((hi - lo) / 255.0 or 1.0))
    zp = int(np.clip(round(-128 - lo / scale), -128, 127))
    return scale, zp


def _range(t: torch.Tensor, pct: float) -> tuple[float, float]:
    v = t.detach().float().reshape(-1).numpy()
    if pct >= 100.0 or v.size < 1000:
        return float(v.min()), float(v.max())
    return float(np.percentile(v, 100 - pct)), float(np.percentile(v, pct))


def _wq(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = w.shape[0]
    ws = (np.maximum(np.abs(w).reshape(o, -1).max(axis=1), 1e-8) / 127.0).astype(np.float32)
    wq = np.clip(np.round(w / ws.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int8)
    return wq, ws


class _Builder:
    def __init__(self):
        self.nodes, self.inits, self.vinfo = [], [], {}
        self.q: dict[str, tuple[float, int]] = {}
        self.n = 0

    def const(self, arr, stem):
        self.n += 1
        name = f"{stem}_{self.n}"
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def sz(self, t):
        s, z = self.q[t]
        return self.const(np.float32(s), f"{t}_scale"), self.const(np.int8(z), f"{t}_zp")

    def dq(self, t, tag):
        s, z = self.sz(t)
        out = f"{tag}__dq_{t}"
        self.nodes.append(helper.make_node("DequantizeLinear", [t, s, z], [out]))
        return out

    def qout(self, f, y):
        s, z = self.sz(y)
        self.nodes.append(helper.make_node("QuantizeLinear", [f, s, z], [y]))

    def weight_bias(self, y, x, w, b, axis_w=0):
        wq, ws = _wq(w)
        cout = w.shape[0]
        wf = f"{y}__wf"
        self.nodes.append(helper.make_node("DequantizeLinear", [self.const(wq, "w"), self.const(ws, "ws"),
                                                               self.const(np.zeros(cout, np.int8), "wz")], [wf], axis=axis_w))
        bs = (np.float32(self.q[x][0]) * ws).astype(np.float32)
        bq = np.clip(np.round(b / bs), -2**31, 2**31 - 1).astype(np.int32)
        bf = f"{y}__bf"
        self.nodes.append(helper.make_node("DequantizeLinear", [self.const(bq, "b"), self.const(bs, "bs"),
                                                               self.const(np.zeros(cout, np.int32), "bz")], [bf], axis=0))
        return wf, bf


def calibrate(net, calib: np.ndarray, percentile: float = 99.99) -> tuple[dict, dict]:
    """Activation (scale, zero point) for every tensor the exported graph quantizes, and their shapes.
    Pools keep their input's quantization (as the INT8 kernels do)."""
    net.eval()
    act = net.act_q
    net.act_q = None
    rec: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        net(torch.from_numpy(calib.astype(np.float32)), rec)
    net.act_q = act
    shapes = {k: tuple(v.shape[1:]) for k, v in rec.items()}
    q = {"x": act_q(*_range(rec["x"], 100.0))}        # the input range is the data's, never clipped
    prev = "x"
    for i, layer in enumerate(net.layers):
        y, op = f"L{i}", layer["op"]
        if op == "dropout":
            continue
        if op in ("maxpool", "avgpool"):
            q[y] = q[prev]
        else:
            if op == "residual":
                q[f"L{i}_a"] = act_q(*_range(rec[f"L{i}_a"], percentile))
                q[f"L{i}__inner"] = act_q(*_range(rec[f"L{i}__inner"], percentile))
            full = op == "gap" or layer.get("head")
            q[y] = act_q(*_range(rec[y], 100.0 if full else percentile))
        prev = y
    return q, shapes


def export_qdq(net, calib: np.ndarray, path: Path, percentile: float = 99.99, q: dict | None = None) -> dict:
    """Write the QDQ ONNX model (BatchNorm must already be folded). `q` reuses ranges from a QAT run; otherwise
    they are calibrated on `calib` [N, C, H, W]. Returns the input/output quantization record."""
    cq, shapes = calibrate(net, calib, percentile)
    b = _Builder()
    b.q.update(q or cq)
    cur = "x"
    one_d = net.one_d
    for i, (layer, mod) in enumerate(zip(net.layers, net.mods)):
        op, y = layer["op"], f"L{i}"
        if op == "dropout":
            continue
        if op in ("conv", "dwconv", "pwconv"):
            _conv(b, mod, cur, y, layer.get("relu", False), one_d)
        elif op == "residual":
            a, inner = f"L{i}_a", f"L{i}__inner"
            _conv(b, mod[0], cur, a, True, one_d)
            _conv(b, mod[1], a, inner, False, one_d)
            fa, fb = b.dq(cur, y), b.dq(inner, y)
            b.nodes.append(helper.make_node("Add", [fa, fb], [f"{y}__f"]))
            b.qout(f"{y}__f", y)
        elif op in ("maxpool", "avgpool"):
            k = [1, layer["k"]] if one_d else [layer["k"], layer["k"]]
            kind = "MaxPool" if op == "maxpool" else "AveragePool"
            b.nodes.append(helper.make_node(kind, [b.dq(cur, y)], [f"{y}__f"], kernel_shape=k, strides=k,
                                            pads=[0, 0, 0, 0]))
            b.qout(f"{y}__f", y)
        elif op == "gap":
            b.nodes.append(helper.make_node("GlobalAveragePool", [b.dq(cur, y)], [f"{y}__f"]))
            b.qout(f"{y}__f", y)
        elif op == "fc":
            src = b.dq(cur, y)
            if len(shapes[cur]) > 1:
                flat = f"{y}__flat"
                b.nodes.append(helper.make_node("Flatten", [src], [flat], axis=1))
                src = flat
            w = mod.weight.detach().numpy().astype(np.float32)
            bias = mod.bias.detach().numpy().astype(np.float32)
            wf, bf = b.weight_bias(y, cur, w, bias)
            pre = f"{y}__pre" if layer.get("relu") else f"{y}__f"
            b.nodes.append(helper.make_node("Gemm", [src, wf, bf], [pre], transB=1))
            if layer.get("relu"):
                b.nodes.append(helper.make_node("Relu", [pre], [f"{y}__f"]))
            b.qout(f"{y}__f", y)
        cur = y
    out = cur
    io = {"x", out}
    vinfo = [helper.make_tensor_value_info(n, TensorProto.INT8, [1, *shapes[n]]) for n in b.q if n not in io and n in shapes]
    graph = helper.make_graph(b.nodes, "sai_platform_model",
                              [helper.make_tensor_value_info("x", TensorProto.INT8, [1, *shapes["x"]])],
                              [helper.make_tensor_value_info(out, TensorProto.INT8, [1, *shapes[out]])],
                              b.inits, value_info=vinfo)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="sai-platform")
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    path = Path(path)
    path.write_bytes(model.SerializeToString(deterministic=True))
    return {"input": {"name": "x", "shape": [1, *shapes["x"]], "scale": b.q["x"][0], "zero_point": b.q["x"][1]},
            "output": {"name": out, "shape": [1, *shapes[out]], "scale": b.q[out][0], "zero_point": b.q[out][1]}}


def _conv(b: _Builder, conv_mod, x, y, relu, one_d):
    c = conv_mod.conv
    assert isinstance(conv_mod.bn, torch.nn.Identity), "fold BatchNorm before export"
    w = c.weight.detach().numpy().astype(np.float32)
    bias = c.bias.detach().numpy().astype(np.float32)
    wf, bf = b.weight_bias(y, x, w, bias)
    # SAME pads depend on the input size; the module recorded them on the calibration pass.
    attrs = dict(kernel_shape=list(c.kernel_size), strides=list(c.stride), group=c.groups)
    if c.dilation != (1, 1):
        attrs["dilations"] = list(c.dilation)
    attrs["pads"] = conv_mod._export_pads
    pre = f"{y}__pre" if relu else f"{y}__f"
    b.nodes.append(helper.make_node("Conv", [b.dq(x, y), wf, bf], [pre], **attrs))
    if relu:
        b.nodes.append(helper.make_node("Relu", [pre], [f"{y}__f"]))
    b.qout(f"{y}__f", y)


def quantize_input(x: np.ndarray, qin: dict) -> np.ndarray:
    return np.clip(np.round(x / qin["scale"]) + qin["zero_point"], -128, 127).astype(np.int8)


def dequantize_output(q: np.ndarray, qout: dict) -> np.ndarray:
    return (q.astype(np.float32) - qout["zero_point"]) * qout["scale"]


def ort_int8_outputs(path: Path, xq: np.ndarray) -> np.ndarray:
    """Run the QDQ graph in onnxruntime (framework reference, not SAI's exact contract) on int8 inputs [N,...]."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: xq[i:i + 1]})[0] for i in range(len(xq))])
