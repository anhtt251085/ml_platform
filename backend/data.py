"""Dataset ingestion: every source ends as data.npz (X float32 [N, C, H, W], y, labels) plus meta.json.

Sources today: file upload (CSV or NPZ), HTTPS URL (CSV or NPZ), synthetic demo.
Planned connectors (see docs/IMPLEMENTATION_PLAN.md): S3/GCS/Azure bucket, MQTT/OPC-UA live capture, board-farm
data logger, Edge Impulse / Kaggle import.

CSV contract (time series): one row per sample; numeric columns are channels; a `label` column; optional
`segment` (or `file`/`recording`) column so windows never cross recordings; optional `time`/`timestamp` ignored.
Windows of `window` samples every `hop` samples; a window's label is its majority label (classification) or
mean (regression).
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from . import config, synth

IGNORED = {"time", "timestamp", "t", "index", "idx", "sample"}
SEGMENT = ("segment", "file", "recording", "session", "id")


class DataError(ValueError):
    pass


def _save(X, y, labels, meta: dict) -> dict:
    did = "ds_" + uuid.uuid4().hex[:10]
    d = config.DATA_DIR / "datasets" / did
    d.mkdir(parents=True)
    np.savez_compressed(d / "data.npz", X=X.astype(np.float32), y=y, labels=np.array(labels))
    meta = {**meta, "id": did, **describe(X, y, labels, meta.get("task", "classification"))}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(did: str):
    d = config.DATA_DIR / "datasets" / did
    if not (d / "data.npz").exists():
        raise DataError(f"dataset {did} not found")
    with np.load(d / "data.npz", allow_pickle=False) as z:
        return z["X"], z["y"], [str(s) for s in z["labels"]], json.loads((d / "meta.json").read_text())


def describe(X, y, labels, task) -> dict:
    info = {"n": int(len(X)), "shape": list(X.shape[1:]), "channels": int(X.shape[1]), "height": int(X.shape[2]),
            "width": int(X.shape[3]), "labels": list(labels)}
    if task == "regression":
        info["target_stats"] = {"min": float(np.min(y)), "max": float(np.max(y)), "mean": float(np.mean(y))}
    else:
        cnt = Counter(int(v) for v in y)
        info["class_counts"] = {labels[k] if k < len(labels) else str(k): cnt[k] for k in sorted(cnt)}
    # a small preview for the UI: first example of each class, channel 0..2, decimated to <= 128 points
    prev = []
    for k in (sorted(set(int(v) for v in y))[:6] if task != "regression" else [0]):
        idx = int(np.argmax(y == k)) if task != "regression" else 0
        ex = X[idx]
        if ex.shape[1] == 1:
            step = max(1, ex.shape[2] // 128)
            prev.append({"label": labels[k] if task != "regression" else f"{float(y[idx]):.1f}",
                         "kind": "series", "series": ex[:3, 0, ::step].round(4).tolist()})
        else:
            prev.append({"label": labels[k], "kind": "image", "image": ex[0].round(3).tolist()})
    info["preview"] = prev
    warnings = []
    if task != "regression":
        counts = list(info["class_counts"].values())
        if min(counts) < 20:
            warnings.append(f"some classes have very few windows (min {min(counts)}); aim for 100+ per class")
        if max(counts) > 5 * max(min(counts), 1):
            warnings.append("strong class imbalance; the trainer uses class weights, but more data is better")
    if not np.isfinite(X).all():
        warnings.append("non-finite values were replaced by 0")
    info["warnings"] = warnings
    return info


def from_synthetic(template: dict, n_per_class: int = 300, seed: int = 7) -> dict:
    X, y, labels = synth.generate(template["synth"], n_per_class, seed)
    return _save(X, y, labels, {"source": "synthetic", "template": template["id"], "task": template["task"],
                                "name": f"Synthetic demo - {template['name']}", "synthetic": True})


def _windows(arr, lab, seg, window, hop, regression):
    Xs, ys = [], []
    bounds = []
    start = 0
    for i in range(1, len(seg) + 1):
        if i == len(seg) or seg[i] != seg[start]:
            bounds.append((start, i))
            start = i
    for a, b in bounds:
        for s in range(a, b - window + 1, hop):
            Xs.append(arr[s:s + window].T)
            w = lab[s:s + window]
            ys.append(float(np.mean(np.asarray(w, dtype=float))) if regression else Counter(w).most_common(1)[0][0])
    return Xs, ys


def from_csv_bytes(raw: bytes, template: dict, name: str, window: int | None = None, hop: int | None = None) -> dict:
    task = template["task"]
    if template["input"]["kind"] in ("image", "spectrogram"):
        raise DataError("this model type takes 2-D frames; upload an NPZ with X [N, 1, H, W] and y [N]")
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        raise DataError("CSV has no data rows")
    header = [h.strip() for h in rows[0]]
    low = [h.lower() for h in header]
    if "label" not in low:
        raise DataError("CSV needs a 'label' column (class name, or the numeric target for regression)")
    li = low.index("label")
    si = next((low.index(s) for s in SEGMENT if s in low), None)
    chans = [i for i, h in enumerate(low) if i not in (li, si) and h not in IGNORED]
    if not chans:
        raise DataError("CSV has no numeric channel columns")
    data = np.zeros((len(rows) - 1, len(chans)), np.float32)
    labs, segs = [], []
    for r, row in enumerate(rows[1:]):
        if len(row) < len(header):
            raise DataError(f"row {r + 2} has {len(row)} fields, header has {len(header)}")
        try:
            data[r] = [float(row[i]) for i in chans]
        except ValueError:
            raise DataError(f"row {r + 2}: a channel value is not numeric") from None
        labs.append(row[li].strip())
        segs.append(row[si] if si is not None else "")
    window = int(window or template.get("window") or 128)
    hop = int(hop or template.get("hop") or window // 2)
    regression = task == "regression"
    if task == "anomaly":
        labs = ["normal" if v.lower() in ("normal", "0", "ok", "healthy") else "anomaly" for v in labs]
    Xs, ys = _windows(data, labs if not regression else [float(v) for v in labs], segs, window, hop, regression)
    if len(Xs) < 10:
        raise DataError(f"only {len(Xs)} windows of {window} samples; provide longer recordings or a shorter window")
    X = np.nan_to_num(np.array(Xs, np.float32))[:, :, None, :]
    if regression:
        y, labels = np.array(ys, np.float32), template["labels"]
    else:
        labels = sorted(set(ys), key=lambda v: (v != "normal", v)) if task == "anomaly" else sorted(set(ys))
        y = np.array([labels.index(v) for v in ys])
    return _save(X, y, labels, {"source": "csv", "name": name, "template": template["id"], "task": task,
                                "channel_names": [header[i] for i in chans], "window": window, "hop": hop})


def from_npz_bytes(raw: bytes, template: dict, name: str) -> dict:
    try:
        z = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        raise DataError("not a readable .npz (pickled objects are refused)") from None
    if "X" not in z or "y" not in z:
        raise DataError("NPZ needs arrays X and y")
    try:                                   # arrays load lazily: a pickled array is refused here
        X = np.nan_to_num(np.asarray(z["X"], np.float32))
        y = np.asarray(z["y"])
        names = [str(s) for s in z["labels"]] if "labels" in z else None
    except (ValueError, TypeError):
        raise DataError("NPZ arrays must be numeric (pickled objects are refused)") from None
    if X.ndim == 2:
        X = X[:, None, None, :]
    elif X.ndim == 3:
        X = X[:, :, None, :]
    elif X.ndim != 4:
        raise DataError("X must be [N, T], [N, C, T] or [N, C, H, W]")
    if len(X) != len(y):
        raise DataError("X and y have different lengths")
    task = template["task"]
    if task == "regression":
        labels, y = template["labels"], y.astype(np.float32)
    else:
        if names:
            labels = names
            y = y.astype(int)
        elif y.dtype.kind in "iu":
            labels = [str(i) for i in range(int(y.max()) + 1)]
        else:
            labels = sorted(set(str(v) for v in y))
            y = np.array([labels.index(str(v)) for v in y])
    return _save(X, y, labels, {"source": "npz", "name": name, "template": template["id"], "task": task})


def ingest_bytes(raw: bytes, filename: str, template: dict, **kw) -> dict:
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise DataError("file too large")
    if filename.lower().endswith(".npz"):
        return from_npz_bytes(raw, template, filename)
    if filename.lower().endswith((".csv", ".txt")):
        return from_csv_bytes(raw, template, filename, kw.get("window"), kw.get("hop"))
    raise DataError("supported files: .csv (time series) and .npz (arrays)")


def ingest_url(url: str, template: dict, **kw) -> dict:
    if not url.lower().startswith("https://"):
        raise DataError("only https:// URLs are accepted")
    req = urllib.request.Request(url, headers={"User-Agent": "sai-platform-ingest/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:        # noqa: S310 (scheme checked above)
        raw = resp.read(config.MAX_UPLOAD_BYTES + 1)
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "download.csv"
    meta = ingest_bytes(raw, name, template, **kw)
    return meta
