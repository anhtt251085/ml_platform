"""Preprocessing stage: what happens to a dataset between upload and training, and on the MCU before inference.

Config (all optional; `normalize_cfg` fills defaults and validates against the dataset):
    {"channels": [0, 2],            # keep these raw channels (1-D time series only)
     "decimate": 2,                 # block-average every 2 samples (1-D time series only)
     "demean": true,                # remove each window's per-channel mean (DC / gravity / bias)
     "normalization": "zscore",     # zscore | minmax | robust | none   (per channel, fit on the train split)
     "clip": 4.0,                   # clip normalized values to +-4 (outliers), or null
     "augment": {"gain": 0.05, "noise": 0.0, "shift": 0},   # training only
     "class_weights": true,         # classification only
     "split": {"train": 0.7, "val": 0.15, "test": 0.15, "mode": "auto"}}   # auto = by recording when possible

Deployability: the window steps run in C (frontends/sai_prep.c) both here (ctypes) and on the MCU; normalization,
clipping and quantization use one float32 formula mirrored exactly by the generated `sai_app_prepare`
(contraction off), so the MCU quantizes a raw window to the very int8 tensor the model was evaluated on.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import threading

import numpy as np

from . import config

SRC = config.ROOT / "frontends"
NORMS = ("zscore", "minmax", "robust", "none")
_lib = None
_lock = threading.Lock()


class PrepError(ValueError):
    pass


def defaults() -> dict:
    return {"channels": None, "decimate": 1, "demean": False, "normalization": "zscore", "clip": None,
            "augment": {"gain": 0.05, "noise": 0.0, "shift": 0}, "class_weights": True,
            "split": {"train": 0.7, "val": 0.15, "test": 0.15, "mode": "auto"}}


def window_ops_allowed(meta: dict, shape) -> bool:
    """Channel selection / decimation / demean apply to raw 1-D time series, not to front-end features or images."""
    return not meta.get("frontend") and int(shape[1]) == 1


def normalize_cfg(cfg: dict | None, meta: dict, shape, task: str) -> dict:
    out = defaults()
    cfg = cfg or {}
    for k in ("channels", "decimate", "demean", "normalization", "clip", "class_weights"):
        if k in cfg and cfg[k] is not None and cfg[k] != "":
            out[k] = cfg[k]
    out["augment"].update({k: v for k, v in (cfg.get("augment") or {}).items() if v is not None})
    out["split"].update({k: v for k, v in (cfg.get("split") or {}).items() if v is not None})
    C, H, W = int(shape[0]), int(shape[1]), int(shape[2])
    ops = window_ops_allowed(meta, (C, H, W))
    if out["channels"] is not None:
        ch = sorted({int(c) for c in out["channels"]})
        if not ch or min(ch) < 0 or max(ch) >= C:
            raise PrepError(f"channels must be indices in 0..{C - 1}")
        out["channels"] = None if ch == list(range(C)) else ch
    out["decimate"] = int(out["decimate"])
    out["demean"] = bool(out["demean"])
    if not ops and (out["channels"] is not None or out["decimate"] != 1 or out["demean"]):
        raise PrepError("channel selection, decimation and mean removal apply to raw 1-D signals only "
                        "(this dataset is images or front-end features)")
    if not 1 <= out["decimate"] <= 64 or W // out["decimate"] < 8:
        raise PrepError(f"decimate must keep at least 8 samples per window (window {W})")
    if out["normalization"] not in NORMS:
        raise PrepError(f"normalization must be one of {NORMS}")
    if out["clip"] is not None:
        out["clip"] = float(out["clip"])
        if not 0.5 <= out["clip"] <= 100:
            raise PrepError("clip must be between 0.5 and 100 (normalized units) or empty")
    a = out["augment"]
    a["gain"], a["noise"], a["shift"] = float(a.get("gain", 0)), float(a.get("noise", 0)), int(a.get("shift", 0))
    if not (0 <= a["gain"] <= 0.5 and 0 <= a["noise"] <= 1 and 0 <= a["shift"] <= W // 2):
        raise PrepError("augmentation: gain 0..0.5, noise 0..1 (normalized units), shift 0..half the window")
    sp = out["split"]
    fr = [float(sp["train"]), float(sp["val"]), float(sp["test"])]
    if min(fr) < 0.05 or abs(sum(fr) - 1) > 1e-6:
        raise PrepError("split fractions must each be >= 0.05 and sum to 1")
    sp["train"], sp["val"], sp["test"] = fr
    if sp["mode"] not in ("auto", "window"):
        raise PrepError("split mode must be auto (by recording when possible) or window")
    out["class_weights"] = bool(out["class_weights"]) and task == "classification"
    out["window_ops"] = ops and (out["channels"] is not None or out["decimate"] != 1 or out["demean"])
    return out


# ---- C window steps --------------------------------------------------------------------------------------------
def lib():
    global _lib
    with _lock:
        if _lib is None:
            c, h = (SRC / "sai_prep.c").read_bytes(), (SRC / "sai_prep.h").read_bytes()
            so = config.DATA_DIR / f"libsai_prep_{hashlib.sha256(c + h).hexdigest()[:12]}.so"
            if not so.exists():
                tmp = so.with_suffix(".tmp.so")
                proc = subprocess.run([config.HOST_CC, "-O2", "-std=c11", "-fPIC", "-shared", "-ffp-contract=off",
                                       "-I", str(SRC), str(SRC / "sai_prep.c"), "-o", str(tmp)],
                                      capture_output=True, text=True)
                if proc.returncode:
                    raise RuntimeError("preprocessing build failed: " + proc.stderr[-800:])
                tmp.replace(so)
            L = ctypes.CDLL(str(so))
            fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
            L.sai_prep_window.argtypes = [fp, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int, ctypes.c_int, fp]
            L.sai_prep_window.restype = ctypes.c_int
            _lib = L
    return _lib


def window_ops(X: np.ndarray, cfg: dict) -> np.ndarray:
    """[N, C, 1, T] raw -> [N, C', 1, T'] through the C window steps (identity when no step is configured)."""
    if not cfg.get("window_ops"):
        return X
    L = lib()
    n, c_raw, _, t_raw = X.shape
    chans = cfg["channels"] or list(range(c_raw))
    c_out, t_out = len(chans), t_raw // cfg["decimate"]
    ch = (ctypes.c_int * c_out)(*chans)
    out = np.zeros((n, c_out, 1, t_out), np.float32)
    buf = np.zeros(c_out * t_out, np.float32)
    fp = ctypes.POINTER(ctypes.c_float)
    for i in range(n):
        x = np.ascontiguousarray(X[i, :, 0, :], dtype=np.float32)
        if L.sai_prep_window(x.ctypes.data_as(fp), c_raw, t_raw, ch, c_out, cfg["decimate"], int(cfg["demean"]),
                             buf.ctypes.data_as(fp)) != t_out:
            raise PrepError("window preprocessing failed")
        out[i, :, 0, :] = buf.reshape(c_out, t_out)
    return out


# ---- normalization ------------------------------------------------------------------------------------------------
def fit_norm(X: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel offset and scale (float32) such that xn = (x - offset) / scale. A constant channel gets scale 1
    (it carries no information; dividing by ~0 would explode it)."""
    C = X.shape[1]
    flat = np.moveaxis(X, 1, 0).reshape(C, -1).astype(np.float64)
    kind = cfg["normalization"]
    if kind == "zscore":
        off, sc = flat.mean(1), flat.std(1)
    elif kind == "minmax":
        lo, hi = flat.min(1), flat.max(1)
        off, sc = (lo + hi) / 2, (hi - lo) / 2
    elif kind == "robust":
        q25, q50, q75 = np.percentile(flat, [25, 50, 75], axis=1)
        off, sc = q50, (q75 - q25) / 1.349
    else:
        off, sc = np.zeros(C), np.ones(C)
    sc = np.where(np.abs(sc) < 1e-6, 1.0, sc)
    return off.astype(np.float32), sc.astype(np.float32)


def apply_norm(X: np.ndarray, off: np.ndarray, sc: np.ndarray, clip: float | None) -> np.ndarray:
    """float32, element-wise, in the order the generated C uses: (x - off) / sc, then clip."""
    s = (1, -1, 1, 1)
    Xn = (X.astype(np.float32) - off.reshape(s)) / sc.reshape(s)
    if clip is not None:
        Xn = np.clip(Xn, np.float32(-clip), np.float32(clip))
    return Xn.astype(np.float32)


def quantize(Xn: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """Same formula as the generated C: rint((x / s)) + zp, saturated (rint = round half to even, like np.round)."""
    return np.clip(np.round(Xn / np.float32(scale)) + zp, -128, 127).astype(np.int8)


# ---- preview ----------------------------------------------------------------------------------------------------
def _stats(X):
    C = X.shape[1]
    flat = np.moveaxis(X, 1, 0).reshape(C, -1)
    return [{"mean": float(f.mean()), "std": float(f.std()), "min": float(f.min()), "max": float(f.max()),
             "p1": float(np.percentile(f, 1)), "p99": float(np.percentile(f, 99))} for f in flat]


def preview(X, y, labels, meta, cfg, task) -> dict:
    """Stats and example windows before/after, on a sample of the data (normalization fit on that sample)."""
    rng = np.random.default_rng(0)
    pick = rng.choice(len(X), size=min(len(X), 2000), replace=False)
    Xs = X[pick]
    Xw = window_ops(Xs, cfg)
    off, sc = fit_norm(Xw, cfg)
    Xn = apply_norm(Xw, off, sc, cfg["clip"])
    examples = []
    classes = sorted(set(int(v) for v in y)) if task != "regression" else [None]
    for k in classes[:4]:
        i = int(np.argmax(y[pick] == k)) if k is not None else 0
        step_b = max(1, Xs.shape[3] // 160)
        step_a = max(1, Xn.shape[3] // 160)
        ex = {"label": labels[k] if k is not None else f"{float(y[pick][i]):.2f}"}
        if Xs.shape[2] == 1:
            ex["before"] = Xs[i, :4, 0, ::step_b].round(4).tolist()
            ex["after"] = Xn[i, :4, 0, ::step_a].round(4).tolist()
        else:
            ex["before_image"] = Xs[i, 0].round(3).tolist()
            ex["after_image"] = Xn[i, 0].round(3).tolist()
        examples.append(ex)
    clipped = float(np.mean(np.abs(Xn) >= cfg["clip"])) if cfg["clip"] else 0.0
    warn = []
    if clipped > 0.01:
        warn.append(f"{clipped:.1%} of normalized values hit the clip limit: raise it or check outliers")
    if cfg["decimate"] > 1:
        warn.append(f"decimation by {cfg['decimate']} keeps content below {0.5 / cfg['decimate']:.2f} x the sample "
                    "rate's Nyquist band: check the signal of interest survives")
    return {"input_shape": list(X.shape[1:]), "model_input_shape": list(Xn.shape[1:]),
            "input_bytes": int(np.prod(Xn.shape[1:])), "before": _stats(Xs), "after": _stats(Xn),
            "offset": off.tolist(), "scale": sc.tolist(), "clipped_fraction": clipped, "examples": examples,
            "warnings": warn, "window_ops": cfg["window_ops"]}


# ---- generated C ------------------------------------------------------------------------------------------------
def c_prepare(cfg: dict, off, sc, qin: dict, raw_shape) -> str:
    """`sai_app_prepare(raw, q)`: raw window (channel-major, raw channels x raw samples, float) -> model int8 input."""
    C_raw, T_raw = int(raw_shape[0]), int(raw_shape[1]) * int(raw_shape[2])     # H x W flattened per channel
    chans = cfg["channels"] or list(range(C_raw))
    t_out = T_raw // cfg["decimate"] if cfg["window_ops"] else T_raw
    def f(v):          # a float literal that round-trips the float32 value exactly ("3" -> "3.0f")
        t = f"{float(np.float32(v)):.9g}"
        return (t if any(ch in t for ch in ".en") else t + ".0") + "f"
    lines = ["/* Generated by SAI Platform: preprocessing the model was trained with (sai_app_prepare). */",
             "#ifndef SAI_APP_PREPARE_H", "#define SAI_APP_PREPARE_H", "#include <math.h>", "#include <stdint.h>",
             '#include "sai_prep.h"', "",
             f"#define SAI_APP_RAW_CHANNELS {C_raw}", f"#define SAI_APP_RAW_SAMPLES {T_raw}",
             f"#define SAI_APP_IN_CHANNELS {len(chans)}", f"#define SAI_APP_IN_SAMPLES {t_out}",
             f"/* window steps: channels {chans}, decimate {cfg['decimate']}, demean {int(cfg['demean'])}; "
             f"normalization {cfg['normalization']}, clip {cfg['clip']} */",
             f"static const int sai_app_chans[] = {{ {', '.join(map(str, chans))} }};",
             f"static const float sai_app_off[] = {{ {', '.join(f(v) for v in off)} }};",
             f"static const float sai_app_sc[] = {{ {', '.join(f(v) for v in sc)} }};", "",
             "#if defined(__GNUC__) && !defined(__clang__)",
             '__attribute__((optimize("fp-contract=off")))', "#endif",
             "static inline int sai_app_prepare(const float *raw, int8_t *q) {",
             "    static float buf[SAI_APP_IN_CHANNELS * SAI_APP_IN_SAMPLES];"]
    if cfg["window_ops"]:
        lines += [f"    if (sai_prep_window(raw, SAI_APP_RAW_CHANNELS, SAI_APP_RAW_SAMPLES, sai_app_chans, "
                  f"SAI_APP_IN_CHANNELS, {cfg['decimate']}, {int(cfg['demean'])}, buf) < 0) return -1;"]
    else:
        lines += ["    for (int i = 0; i < SAI_APP_IN_CHANNELS * SAI_APP_IN_SAMPLES; i++) buf[i] = raw[i];"]
    clip = (f"        v = v < -{f(cfg['clip'])} ? -{f(cfg['clip'])} : v > {f(cfg['clip'])} ? {f(cfg['clip'])} : v;"
            if cfg["clip"] is not None else "")
    lines += ["    for (int c = 0; c < SAI_APP_IN_CHANNELS; c++)",
              "      for (int t = 0; t < SAI_APP_IN_SAMPLES; t++) {",
              "        float v = (buf[c * SAI_APP_IN_SAMPLES + t] - sai_app_off[c]) / sai_app_sc[c];",
              *([clip] if clip else []),
              f"        float r = rintf(v / {f(qin['scale'])}) + {int(qin['zero_point'])}.0f;",
              "        q[c * SAI_APP_IN_SAMPLES + t] = (int8_t)(r < -128.0f ? -128 : r > 127.0f ? 127 : (int)r);",
              "      }", "    return 0;", "}", "", "#endif", ""]
    return "\n".join(lines)
