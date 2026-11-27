# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Run a finished project's delivered model on new raw windows, in the platform (not on a board).

The steps are the package's own (preprocessing.json): C front end features if the model type has one, the C window
steps (channel pick, decimation, demean), the per-channel normalization fitted at training, INT8 input
quantization, then the QDQ model.onnx in onnxruntime and output dequantization. onnxruntime is the framework
reference; the bit-exact SAI program's numbers are in the project's report.

Used for label suggestions in the labelling tool (and by anything else that needs predictions on fresh data)."""

from __future__ import annotations

import json

import numpy as np

from . import config, features, preprocess, qexport


class ProjectModel:
    def __init__(self, job_id: str):
        d = config.DATA_DIR / "jobs" / job_id
        if not (d / "preprocessing.json").exists() or not (d / "model.onnx").exists():
            raise ValueError("that project has no finished model yet")
        self.pre = json.loads((d / "preprocessing.json").read_text())
        if "sequence" in (self.pre.get("layout") or ""):
            raise ValueError("predictions in the platform are available for Core (CNN) projects; sequence models "
                             "run as the SAI front end's C library only")
        self.path = d / "model.onnx"
        self.task = self.pre["task"]
        self.labels = list(self.pre.get("labels") or [])
        fe = self.pre.get("frontend")
        shape = self.pre["raw_shape"]
        self.raw_channels = int(shape[0])
        self.raw_window = int(fe["raw_length"]) if fe else int(shape[-1])

    def outputs(self, X: np.ndarray) -> np.ndarray:
        """X: raw windows [N, C, 1, T] float -> dequantized outputs [N, n_out]."""
        X = np.asarray(X, np.float32)
        fe = self.pre.get("frontend")
        if fe:
            X = features.apply(fe, X)
        cfg = self.pre["preprocess"]
        X = preprocess.window_ops(X, cfg)
        off = np.asarray(self.pre["mean"], np.float32)
        sc = np.asarray(self.pre["std"], np.float32)
        Xn = preprocess.apply_norm(X, off, sc, cfg.get("clip"))
        q = self.pre["input"]
        xq = preprocess.quantize(Xn, q["scale"], q["zero_point"])
        out = qexport.ort_int8_outputs(self.path, xq)
        return qexport.dequantize_output(out.reshape(len(X), -1), self.pre["output"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        o = self.outputs(X).astype(np.float64)
        e = np.exp(o - o.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
