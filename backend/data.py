"""Dataset ingestion: every source ends as data.npz (X float32 [N, C, H, W], y, labels, groups) plus meta.json.

Sources: file upload (CSV, NPZ, ZIP of WAV folders), HTTPS URL, synthetic demo data.
Planned connectors (docs/IMPLEMENTATION_PLAN.md): S3/GCS/Azure, MQTT/OPC-UA live capture, board-farm logger.

CSV (time series): one row per sample. Columns are mapped explicitly (`mapping`: channels, label, segment, time)
or guessed (numeric columns = channels, `label`, `segment`/`file`/`recording`, `time`). Windows of `window`
samples every `hop` samples never cross a segment; a window's label is its majority label (classification) or
mean (regression); its group is its segment, so train/val/test splits are by recording, never by window.

Templates with a `frontend` (MFCC, band energies) receive raw signals here and store the C front end's features.

Every dataset records a licence; non-commercial or research-only data cannot be packaged for deployment.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
import uuid
import wave
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from . import config, features, synth

IGNORED = {"time", "timestamp", "t", "ts", "time_s", "index", "idx", "sample"}
SEGMENT = ("segment", "file", "recording", "session", "id", "run", "rec", "recording_id")
TIME = ("time", "timestamp", "t", "ts", "time_s")
LICENCES = {
    "proprietary": {"label": "Own / proprietary data", "commercial": True},
    "cc-by-4.0": {"label": "CC BY 4.0", "commercial": True},
    "cc0": {"label": "CC0 / public domain", "commercial": True},
    "odc-by": {"label": "ODC-By", "commercial": True},
    "cc-by-nc": {"label": "CC BY-NC (non-commercial)", "commercial": False},
    "research-only": {"label": "Research / evaluation only", "commercial": False},
    "unknown": {"label": "Unknown - must be cleared before deployment", "commercial": False},
    "synthetic": {"label": "Synthetic demo data", "commercial": False},
}


class DataError(ValueError):
    pass


# ---- storage -----------------------------------------------------------------------------------------------------
MAX_VALUES = 400_000_000          # 1.6 GB of float32 once decoded: larger datasets go through a worker import


def _save(X, y, labels, meta: dict, groups=None, template: dict | None = None) -> dict:
    X = np.asarray(X, np.float32)
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise DataError(f"{bad} sample values are NaN or infinite: clean the recording (or drop those rows)")
    if meta.get("task") == "regression" and not np.isfinite(np.asarray(y, np.float64)).all():
        raise DataError("some regression targets are NaN or infinite")
    if len(X) < 10:
        raise DataError(f"only {len(X)} windows: at least 10 are needed (100+ per class recommended)")
    fe = (template or {}).get("frontend")
    if fe and X.shape[2] == 1 and X.shape[3] == int(fe["raw_length"]):
        X = features.apply(fe, X)                     # raw signal -> the C front end's features
        meta["frontend"] = fe
    elif fe and list(X.shape[1:]) != list(features.output_shape(fe, X.shape[1])):
        raise DataError(f"this model type expects raw windows of {fe['raw_length']} samples or features of shape "
                        f"{list(features.output_shape(fe, X.shape[1]))}; got {list(X.shape[1:])}")
    elif fe:
        meta["frontend"] = fe
    if groups is None:
        groups = np.arange(len(X))
    groups = np.asarray([str(g) for g in groups])
    lic = meta.get("licence") or ("synthetic" if meta.get("synthetic") else "unknown")
    if lic not in LICENCES:
        raise DataError(f"licence must be one of {', '.join(LICENCES)}")
    meta["licence"] = lic
    meta["commercial_use"] = LICENCES[lic]["commercial"]
    did = "ds_" + uuid.uuid4().hex[:10]
    d = config.DATA_DIR / "datasets" / did
    d.mkdir(parents=True)
    np.savez_compressed(d / "data.npz", X=X, y=y, labels=np.array(labels), groups=groups)
    meta = {**meta, "id": did, **describe(X, y, labels, meta.get("task", "classification"), groups)}
    meta["quality"] = quality(X, y, groups, meta.get("task", "classification"), meta.pop("_raw_stats", None))
    meta["warnings"] = meta.get("warnings", []) + meta["quality"]["warnings"]
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(did: str):
    d = config.DATA_DIR / "datasets" / did
    if not (d / "data.npz").exists():
        raise DataError(f"dataset {did} not found")
    with np.load(d / "data.npz", allow_pickle=False) as z:
        return z["X"], z["y"], [str(s) for s in z["labels"]], json.loads((d / "meta.json").read_text())


def load_groups(did: str) -> np.ndarray | None:
    with np.load(config.DATA_DIR / "datasets" / did / "data.npz", allow_pickle=False) as z:
        return z["groups"] if "groups" in z else None


def describe(X, y, labels, task, groups=None) -> dict:
    info = {"n": int(len(X)), "shape": list(X.shape[1:]), "channels": int(X.shape[1]), "height": int(X.shape[2]),
            "width": int(X.shape[3]), "labels": list(labels),
            "recordings": int(len(set(groups.tolist()))) if groups is not None else None}
    if task == "regression":
        info["target_stats"] = {"min": float(np.min(y)), "max": float(np.max(y)), "mean": float(np.mean(y))}
    else:
        cnt = Counter(int(v) for v in y)
        info["class_counts"] = {labels[k] if k < len(labels) else str(k): cnt[k] for k in sorted(cnt)}
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
    info["warnings"] = warnings
    return info


def quality(X, y, groups, task, raw_stats: dict | None = None) -> dict:
    """Data quality report: duplicates, clipping, recordings per class, sample-rate jitter."""
    h = [hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest() for x in X]
    dup = len(h) - len(set(h))
    first: dict[str, str] = {}
    cross = 0
    for hh, g in zip(h, groups):
        if hh in first and first[hh] != g:
            cross += 1
        first.setdefault(hh, g)
    lo, hi = X.min(axis=(0, 2, 3)), X.max(axis=(0, 2, 3))
    clip = [float(np.mean((X[:, c] == lo[c]) | (X[:, c] == hi[c]))) for c in range(X.shape[1])]
    rep = {"duplicate_windows": int(dup), "duplicates_across_recordings": int(cross),
           "clipping_fraction_per_channel": [round(v, 4) for v in clip],
           "recordings": int(len(set(groups.tolist())))}
    if task != "regression":
        rep["recordings_per_class"] = {str(int(k)): int(len(set(groups[y == k].tolist()))) for k in np.unique(y)}
    if raw_stats:
        rep.update(raw_stats)
    w = []
    if dup:
        w.append(f"{dup} duplicate windows" + (f", {cross} of them across recordings (split leakage risk)" if cross else ""))
    if max(clip) > 0.01:
        w.append(f"channel {int(np.argmax(clip))} sits at its min/max in {max(clip):.1%} of samples: sensor clipping?")
    if task != "regression" and min(rep["recordings_per_class"].values()) < 3 and rep["recordings"] < len(X):
        w.append("some classes come from fewer than 3 recordings: test accuracy will not generalise")
    if raw_stats and raw_stats.get("sample_period_jitter", 0) > 0.05:
        w.append(f"sample period jitter {raw_stats['sample_period_jitter']:.1%}: check the logger's sample rate")
    rep["warnings"] = w
    return rep


# ---- sources -----------------------------------------------------------------------------------------------------
def from_synthetic(template: dict, n_per_class: int = 300, seed: int = 7) -> dict:
    X, y, labels = synth.generate(template["synth"], n_per_class, seed)
    groups = np.arange(len(X)) // 10              # synthetic "recordings" of 10 windows
    return _save(X, y, labels, {"source": "synthetic", "template": template["id"], "task": template["task"],
                                "name": f"Synthetic demo - {template['name']}", "synthetic": True,
                                "licence": "synthetic"}, groups, template)


def _windows(arr, lab, seg, window, hop, regression):
    Xs, ys, gs = [], [], []
    bounds, start = [], 0
    for i in range(1, len(seg) + 1):
        if i == len(seg) or seg[i] != seg[start]:
            bounds.append((start, i))
            start = i
    for a, b in bounds:
        for s in range(a, b - window + 1, hop):
            Xs.append(arr[s:s + window].T)
            w = lab[s:s + window]
            ys.append(float(np.mean(np.asarray(w, dtype=float))) if regression else Counter(w).most_common(1)[0][0])
            gs.append(seg[a])
    return Xs, ys, gs


def csv_preview(raw: bytes, rows: int = 12) -> dict:
    """Header, first rows and a guessed column mapping, for the column-mapping step of the wizard."""
    text = raw[:2_000_000].decode("utf-8-sig", errors="replace")
    data = list(csv.reader(io.StringIO(text)))
    if len(data) < 2:
        raise DataError("CSV has no data rows")
    header = [h.strip() for h in data[0]]
    low = [h.lower() for h in header]
    numeric = []
    for i in range(len(header)):
        try:
            [float(r[i]) for r in data[1:min(len(data), 50)] if len(r) > i]
            numeric.append(True)
        except ValueError:
            numeric.append(False)
    guess = {"label": next((h for h, l in zip(header, low) if l == "label"), None),
             "segment": next((h for h, l in zip(header, low) if l in SEGMENT), None),
             "time": next((h for h, l in zip(header, low) if l in TIME), None)}
    used = {guess["label"], guess["segment"], guess["time"]}
    guess["channels"] = [h for h, l, num in zip(header, low, numeric) if num and h not in used and l not in IGNORED]
    return {"header": header, "rows": data[1:rows + 1], "numeric": numeric, "guess": guess, "n_rows": len(data) - 1}


def from_csv_bytes(raw: bytes, template: dict, name: str, window: int | None = None, hop: int | None = None,
                   mapping: dict | None = None, licence: str | None = None) -> dict:
    task = template["task"]
    if template["input"]["kind"] in ("image",):
        raise DataError("this model type takes 2-D frames; upload an NPZ with X [N, 1, H, W] and y [N]")
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        raise DataError("CSV has no data rows")
    header = [h.strip() for h in rows[0]]
    m = mapping or csv_preview(raw)["guess"]
    if not m.get("label") or m["label"] not in header:
        raise DataError("choose the label column (a 'label' column, or map one)")
    missing = [c for c in m.get("channels") or [] if c not in header]
    if missing or not m.get("channels"):
        raise DataError(f"channel columns not found: {missing or 'none selected'}")
    li = header.index(m["label"])
    si = header.index(m["segment"]) if m.get("segment") in header else None
    ti = header.index(m["time"]) if m.get("time") in header else None
    chans = [header.index(c) for c in m["channels"]]
    data = np.zeros((len(rows) - 1, len(chans)), np.float32)
    labs, segs, times = [], [], []
    for r, row in enumerate(rows[1:]):
        if len(row) < len(header):
            raise DataError(f"row {r + 2} has {len(row)} fields, header has {len(header)}")
        try:
            data[r] = [float(row[i]) for i in chans]
            if ti is not None:
                times.append(float(row[ti]))
        except ValueError:
            raise DataError(f"row {r + 2}: a channel or time value is not numeric") from None
        labs.append(row[li].strip())
        segs.append(row[si] if si is not None else "")
    fe = template.get("frontend")
    window = int(window or (fe or {}).get("raw_length") or template.get("window") or 128)
    hop = int(hop or template.get("hop") or window // 2)
    regression = task == "regression"
    if task == "anomaly":
        labs = ["normal" if v.lower() in ("normal", "0", "ok", "healthy") else "anomaly" for v in labs]
    if regression:
        try:
            labs = [float(v) for v in labs]
        except ValueError:
            raise DataError("regression labels must be numeric") from None
    if not np.isfinite(data).all():
        raise DataError("some channel values are NaN or infinite: clean the recording")
    Xs, ys, gs = _windows(data, labs, segs, window, hop, regression)
    if len(Xs) < 10:
        raise DataError(f"only {len(Xs)} windows of {window} samples; provide longer recordings or a shorter window")
    raw_stats = None
    if len(times) > 2:
        dt = np.diff(np.asarray(times))
        dt = dt[dt > 0]
        if len(dt):
            raw_stats = {"sample_period": float(np.median(dt)),
                         "sample_period_jitter": float(np.std(dt) / max(np.median(dt), 1e-12))}
    X = np.array(Xs, np.float32)[:, :, None, :]
    if regression:
        y, labels = np.array(ys, np.float32), template["labels"]
    else:
        labels = sorted(set(ys), key=lambda v: (v != "normal", v)) if task == "anomaly" else sorted(set(ys))
        y = np.array([labels.index(v) for v in ys])
    expected = (template.get("input") or {}).get("channels")
    warn = [f"{len(m['channels'])} channel columns mapped, the model type was designed for {expected}"] \
        if expected and not fe and len(m["channels"]) != expected else []
    meta = {"source": "csv", "name": name, "template": template["id"], "task": task, "warnings": warn,
            "channel_names": list(m["channels"]), "window": window, "hop": hop, "mapping": m,
            "licence": licence, "_raw_stats": raw_stats}
    return _save(X, y, labels, meta, np.array(gs) if si is not None else None, template)


def _npz_guard(raw: bytes):
    """Refuse decompression bombs: the declared uncompressed size of every member and the element count in each
    .npy header must stay within MAX_VALUES before numpy inflates anything."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise DataError("not a readable .npz") from None
    total = 0
    for info in zf.infolist():
        total += info.file_size
        if total > MAX_VALUES * 8:
            raise DataError("the arrays in this .npz are too large to import here (ask for a worker import)")
        if info.filename.endswith(".npy"):
            with zf.open(info) as f:
                try:
                    ver = np.lib.format.read_magic(f)
                    shape, _, _ = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                                   else np.lib.format.read_array_header_2_0(f))
                except ValueError as e:
                    raise DataError(f"{info.filename}: {e}") from None
            if int(np.prod(shape, dtype=np.int64)) > MAX_VALUES:
                raise DataError(f"{info.filename} has {int(np.prod(shape, dtype=np.int64)):,} values: too large")


def from_npz_bytes(raw: bytes, template: dict, name: str, licence: str | None = None) -> dict:
    _npz_guard(raw)
    try:
        z = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        raise DataError("not a readable .npz (pickled objects are refused)") from None
    if "X" not in z or "y" not in z:
        raise DataError("NPZ needs arrays X and y")
    try:                                   # arrays load lazily: a pickled array is refused here
        X = np.asarray(z["X"], np.float32)
        y = np.asarray(z["y"])
        names = [str(s) for s in z["labels"]] if "labels" in z else None
        groups = np.asarray(z["groups"]) if "groups" in z else None
    except (ValueError, TypeError):
        raise DataError("NPZ arrays must be numeric (pickled objects are refused)") from None
    if X.ndim == 2:
        X = X[:, None, None, :]
    elif X.ndim == 3:
        X = X[:, :, None, :]
    elif X.ndim != 4:
        raise DataError("X must be [N, T], [N, C, T] or [N, C, H, W]")
    if len(X) != len(y) or (groups is not None and len(groups) != len(y)):
        raise DataError("X, y (and groups) have different lengths")
    task = template["task"]
    try:
        y, labels = _npz_labels(y, names, task, template)
    except (ValueError, TypeError) as e:
        raise DataError(f"labels: {e}") from None
    return _save(X, y, labels, {"source": "npz", "name": name, "template": template["id"], "task": task,
                                "licence": licence}, groups, template)


def _npz_labels(y, names, task, template):
    if task == "regression":
        return y.astype(np.float32), template["labels"]
    if names:
        if y.dtype.kind not in "iu":
            raise ValueError("with a `labels` array, y must hold class indices")
        if len(y) and (y.min() < 0 or y.max() >= len(names)):
            raise ValueError(f"class indices must be in 0..{len(names) - 1}")
        return y.astype(int), names
    if y.dtype.kind in "iu":
        if len(y) and y.min() < 0:
            raise ValueError("class indices must be >= 0")
        return y.astype(int), [str(i) for i in range(int(y.max()) + 1)]
    labels = sorted(set(str(v) for v in y))
    return np.array([labels.index(str(v)) for v in y]), labels


SPEAKER = re.compile(r"^([0-9a-f]{8})_nohash_", re.I)     # Speech Commands naming: <speaker>_nohash_<n>.wav


def from_wav_zip(raw: bytes, template: dict, name: str, licence: str | None = None) -> dict:
    """ZIP of <label>/<file>.wav (16-bit PCM mono). Clips are trimmed/zero-padded to the front end's raw length.
    Groups: the speaker id in Speech Commands names, else the file name."""
    fe = template.get("frontend")
    if not fe:
        raise DataError("WAV folders are accepted by model types with an audio front end (e.g. keyword spotting)")
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise DataError("not a readable .zip") from None
    L = int(fe["raw_length"])
    Xs, labs, gs, rates = [], [], [], set()
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".wav") or "/" not in info.filename.strip("/"):
            continue
        parts = info.filename.strip("/").split("/")
        label, fname = parts[-2], parts[-1]
        if label.startswith("_") or label.startswith("."):
            continue
        if info.file_size > 50 * 1024 * 1024:
            raise DataError(f"{info.filename} is larger than 50 MB")
        try:
            with zf.open(info) as f, wave.open(f) as w:
                if w.getsampwidth() != 2 or w.getnchannels() != 1:
                    raise DataError(f"{info.filename}: need 16-bit mono PCM")
                rates.add(w.getframerate())
                a = np.frombuffer(w.readframes(min(w.getnframes(), L)), dtype="<i2").astype(np.float32) / 32768.0
        except (wave.Error, EOFError) as e:
            raise DataError(f"{info.filename}: not a valid WAV file ({e})") from None
        a = a[:L] if len(a) >= L else np.pad(a, (0, L - len(a)))
        Xs.append(a)
        labs.append(label)
        mm = SPEAKER.match(fname)
        gs.append(mm.group(1) if mm else fname)
    if len(Xs) < 10:
        raise DataError("found fewer than 10 WAV files in <label>/<file>.wav folders")
    if rates != {int(fe["sample_rate"])}:
        raise DataError(f"sample rate {sorted(rates)} Hz; this model type expects {fe['sample_rate']} Hz")
    labels = sorted(set(labs))
    X = np.array(Xs, np.float32)[:, None, None, :]
    y = np.array([labels.index(v) for v in labs])
    return _save(X, y, labels, {"source": "wav-zip", "name": name, "template": template["id"],
                                "task": template["task"], "licence": licence}, np.array(gs), template)


def ingest_bytes(raw: bytes, filename: str, template: dict, **kw) -> dict:
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise DataError("file too large")
    low = filename.lower()
    lic = kw.get("licence")
    if low.endswith(".npz"):
        return from_npz_bytes(raw, template, filename, lic)
    if low.endswith(".zip"):
        return from_wav_zip(raw, template, filename, lic)
    if low.endswith((".csv", ".txt")):
        return from_csv_bytes(raw, template, filename, kw.get("window"), kw.get("hop"), kw.get("mapping"), lic)
    raise DataError("supported files: .csv (time series), .npz (arrays), .zip (label folders of WAV files)")


def safe_fetch(url: str, limit: int) -> bytes:
    """HTTPS GET that cannot reach internal services: the host must resolve only to public addresses, the
    connection goes to the address that was checked (no DNS rebinding), redirects are refused."""
    import http.client
    import ipaddress
    import socket
    import ssl
    from urllib.parse import urlsplit

    u = urlsplit(url)
    if u.scheme != "https" or not u.hostname:
        raise DataError("only https:// URLs are accepted")
    if u.port not in (None, 443):
        raise DataError("only the standard HTTPS port is allowed")
    try:
        infos = socket.getaddrinfo(u.hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise DataError("the host name does not resolve") from None
    addrs = {i[4][0] for i in infos}
    for a in addrs:
        ip = ipaddress.ip_address(a)
        if not ip.is_global or ip.is_multicast:
            raise DataError("the URL points to a private or reserved network address")
    ip = sorted(addrs)[0]
    ctx = ssl.create_default_context()

    class _Pinned(http.client.HTTPSConnection):
        def connect(self):
            sock = socket.create_connection((ip, 443), timeout=60)
            self.sock = ctx.wrap_socket(sock, server_hostname=u.hostname)

    conn = _Pinned(u.hostname, 443, timeout=120, context=ctx)
    try:
        conn.request("GET", (u.path or "/") + (f"?{u.query}" if u.query else ""),
                     headers={"User-Agent": "sai-platform-ingest/0.3", "Host": u.hostname})
        resp = conn.getresponse()
        if resp.status in (301, 302, 303, 307, 308):
            raise DataError("the URL redirects; give the final https:// address of the file")
        if resp.status != 200:
            raise DataError(f"download failed: HTTP {resp.status}")
        raw = resp.read(limit + 1)
    except (OSError, http.client.HTTPException) as e:
        raise DataError(f"download failed: {type(e).__name__}") from None
    finally:
        conn.close()
    if len(raw) > limit:
        raise DataError("file too large")
    return raw


def ingest_url(url: str, template: dict, **kw) -> dict:
    raw = safe_fetch(url, config.MAX_UPLOAD_BYTES)
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "download.csv"
    return ingest_bytes(raw, name, template, **kw)
