# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Recordings: continuous multi-channel signals that are labelled by time span and then cut into a dataset.

Where they come from: a CSV without (or with) a label column, a cloud / database connector, or live capture from a
device (a capture token lets a gateway, a dev board's UART bridge or tools/capture_client.py post samples while the
engineer watches the plot and marks what is happening right now).

Storage: DATA_DIR/recordings/{id}.npy, float32 [samples, channels]; a capture appends raw float32 rows to
{id}.part until it is stopped. Labels are spans {start, end, label} in samples, non-overlapping and sorted; setting a
span over existing ones trims them.

Labelling helpers: a min/max envelope view of any range (so a million-sample recording plots in ~1000 points per
channel) and label suggestions from a finished project's INT8 model run over sliding windows (the engineer accepts
or edits them; suggestions are never applied silently).

To a dataset: windows of the model type's length every `hop` samples; a window takes the label that covers at least
`min_cover` of it (otherwise it is skipped as unlabelled); its group is recording + span, so a span never ends up in
two splits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import secrets
import uuid
from pathlib import Path

import numpy as np

from . import config, data, db

MAX_VALUES = 50_000_000            # per recording (200 MB of float32)
MAX_PUSH_ROWS = 200_000
LABEL_RE_MAX = 40


def _dir() -> Path:
    d = config.DATA_DIR / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _row(r) -> dict | None:
    if not r:
        return None
    r = db.row(r)
    meta = db.uj(r.pop("meta"), {})
    r.pop("token_hash", None)
    return {**meta, **r}


def get(rid: str) -> dict | None:
    with db.conn() as c:
        return _row(c.execute("SELECT * FROM recordings WHERE id=?", (rid,)).fetchone())


def list_(org_id: str | None) -> list[dict]:
    q, a = "SELECT * FROM recordings", ()
    if org_id:
        q, a = q + " WHERE org_id=?", (org_id,)
    with db.conn() as c:
        rows = [_row(r) for r in c.execute(q + " ORDER BY created DESC", a)]
    for r in rows:
        r.pop("spans", None)
        r["samples"] = samples(r)
    return rows


def _save_meta(rid: str, meta: dict, status: str | None = None):
    keep = {k: v for k, v in meta.items() if k not in ("id", "org_id", "status", "created", "updated", "samples")}
    with db.conn() as c:
        if status:
            c.execute("UPDATE recordings SET meta=?, status=?, updated=? WHERE id=?", (db.j(keep), status, db.now(), rid))
        else:
            c.execute("UPDATE recordings SET meta=?, updated=? WHERE id=?", (db.j(keep), db.now(), rid))


def _labels_ok(lab: str) -> str:
    lab = str(lab).strip()
    if not lab or len(lab) > LABEL_RE_MAX or any(ch in lab for ch in "\n\r\t,"):
        raise data.DataError(f"label '{lab[:60]}' must be 1-{LABEL_RE_MAX} characters, no commas or line breaks")
    return lab


def create(org_id: str, user_id: str | None, name: str, channels: list[str], sample_rate: float,
           source: str = "upload", licence: str = "proprietary", template_id: str | None = None,
           status: str = "ready", token_hash: str | None = None, extra: dict | None = None) -> dict:
    if not channels or len(channels) > 64:
        raise data.DataError("1-64 channels")
    if not 0 < float(sample_rate) <= 10_000_000:
        raise data.DataError("sample rate must be between 0 and 10 MHz")
    if licence not in data.LICENCES:
        raise data.DataError(f"licence must be one of {', '.join(data.LICENCES)}")
    rid = "rec_" + uuid.uuid4().hex[:10]
    meta = {"name": (name or rid)[:120], "channels": [str(c)[:40] for c in channels], "sample_rate": float(sample_rate),
            "source": source, "licence": licence, "template_id": template_id, "labels": [], "spans": [],
            "created_by": user_id, **(extra or {})}
    with db.conn() as c:
        c.execute("INSERT INTO recordings (id, org_id, meta, status, token_hash, created, updated) VALUES (?,?,?,?,?,?,?)",
                  (rid, org_id, db.j(meta), status, token_hash, db.now(), db.now()))
    return get(rid)


def _npy(rid: str) -> Path:
    return _dir() / f"{rid}.npy"


def _part(rid: str) -> Path:
    return _dir() / f"{rid}.part"


def signal(rec: dict) -> np.ndarray:
    """[samples, channels] float32 (a capture in progress: what has arrived so far)."""
    C = len(rec["channels"])
    if rec["status"] == "capturing":
        p = _part(rec["id"])
        if not p.exists():
            return np.zeros((0, C), np.float32)
        raw = p.read_bytes()
        n = len(raw) // (4 * C)
        return np.frombuffer(raw[:n * 4 * C], np.float32).reshape(n, C)
    return np.load(_npy(rec["id"]), mmap_mode="r")


def samples(rec: dict) -> int:
    C = len(rec["channels"])
    if rec["status"] == "capturing":
        p = _part(rec["id"])
        return p.stat().st_size // (4 * C) if p.exists() else 0
    p = _npy(rec["id"])
    return int(np.load(p, mmap_mode="r").shape[0]) if p.exists() else 0


def store_array(rec: dict, arr: np.ndarray) -> dict:
    arr = np.asarray(arr, np.float32)
    if arr.ndim != 2 or arr.shape[1] != len(rec["channels"]):
        raise data.DataError(f"expected [samples, {len(rec['channels'])}] values")
    if arr.size > MAX_VALUES:
        raise data.DataError(f"recording too large ({arr.size} values; limit {MAX_VALUES})")
    if not np.isfinite(arr).all():
        raise data.DataError("the recording has NaN or infinite values")
    np.save(_npy(rec["id"]), arr)
    return get(rec["id"])


# ---- import ------------------------------------------------------------------------------------------------------
def from_csv(raw: bytes, org_id: str, user_id: str | None, name: str, sample_rate: float | None = None,
             licence: str = "proprietary", template_id: str | None = None, source: str = "upload") -> dict:
    """Numeric columns become channels; a time column gives the sample rate; a label column (optional) becomes
    spans (runs of the same non-empty label)."""
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if r and not r[0].startswith("#")]
    if len(rows) < 3:
        raise data.DataError("CSV has no data rows")
    header = [h.strip() for h in rows[0]]
    low = [h.lower() for h in header]
    ti = next((i for i, h in enumerate(low) if h in data.TIME), None)
    li = next((i for i, h in enumerate(low) if h == "label"), None)
    skip = {ti, li} | {i for i, h in enumerate(low) if h in data.IGNORED or h in ("result", "table", "")}
    chans = []
    for i in range(len(header)):
        if i in skip:
            continue
        try:
            [float(r[i]) for r in rows[1:min(len(rows), 50)] if len(r) > i]
            chans.append(i)
        except ValueError:
            pass
    if not chans:
        raise data.DataError("no numeric channel columns")
    arr = np.zeros((len(rows) - 1, len(chans)), np.float32)
    labs, times = [], []
    for r, row in enumerate(rows[1:]):
        try:
            arr[r] = [float(row[i]) for i in chans]
            if ti is not None:
                times.append(float(row[ti]))
        except (ValueError, IndexError):
            raise data.DataError(f"row {r + 2}: a channel or time value is missing or not numeric") from None
        labs.append(row[li].strip() if li is not None and li < len(row) else "")
    if not sample_rate and len(times) > 2:
        dt = float(np.median(np.diff(times)))
        if dt > 0:
            sample_rate = 1.0 / dt
    if not sample_rate:
        raise data.DataError("give the sample rate (no time column to infer it from)")
    rec = create(org_id, user_id, name, [header[i] for i in chans], float(sample_rate), source, licence, template_id)
    store_array(rec, arr)
    spans, start = [], 0
    for i in range(1, len(labs) + 1):
        if i == len(labs) or labs[i] != labs[start]:
            if labs[start]:
                spans.append({"start": start, "end": i, "label": labs[start]})
            start = i
    if spans:
        set_spans(rec["id"], spans)
    return get(rec["id"])


# ---- view --------------------------------------------------------------------------------------------------------
def view(rec: dict, start: int = 0, end: int | None = None, points: int = 1000, tail: int | None = None) -> dict:
    sig = signal(rec)
    n = sig.shape[0]
    if tail:
        start, end = max(0, n - int(tail)), n
    end = n if end is None else max(0, min(int(end), n))
    start = max(0, min(int(start), end))
    seg = np.asarray(sig[start:end], np.float32)
    points = max(10, min(int(points), 4000))
    if len(seg) <= points:
        lo = hi = seg.T
        idx = np.arange(start, end)
    else:
        b = len(seg) // points
        m = seg[:b * points].reshape(points, b, -1)
        lo, hi = m.min(1).T, m.max(1).T
        idx = start + np.arange(points) * b
    return {"start": start, "end": end, "samples": n, "x": idx.tolist(), "min": np.round(lo, 5).tolist(),
            "max": np.round(hi, 5).tolist(), "channels": rec["channels"], "sample_rate": rec["sample_rate"],
            "spans": [s for s in rec.get("spans", []) if s["end"] > start and s["start"] < end],
            "labels": rec.get("labels", []), "status": rec["status"], "live": rec.get("live")}


# ---- labels -------------------------------------------------------------------------------------------------------
def _merge(spans: list[dict], new: dict) -> list[dict]:
    """Put `new` on top: overlapping spans are trimmed or split; touching spans of the same label are joined."""
    out = []
    for s in spans:
        if s["end"] <= new["start"] or s["start"] >= new["end"]:
            out.append(s)
            continue
        if s["start"] < new["start"]:
            out.append({**s, "end": new["start"]})
        if s["end"] > new["end"]:
            out.append({**s, "start": new["end"]})
    if new.get("label"):
        out.append(new)
    out.sort(key=lambda s: s["start"])
    joined = []
    for s in out:
        if joined and joined[-1]["label"] == s["label"] and joined[-1]["end"] >= s["start"]:
            joined[-1] = {**joined[-1], "end": max(joined[-1]["end"], s["end"])}
        else:
            joined.append(dict(s))
    return joined


def set_spans(rid: str, spans: list[dict], replace: bool = True) -> dict:
    """replace=True: the recording's labels become exactly `spans`; False: paint them over the existing ones.
    A span with an empty label erases that range."""
    rec = get(rid)
    n = samples(rec)
    cur = [] if replace else list(rec.get("spans", []))
    for s in spans:
        try:
            a, b = int(s["start"]), int(s["end"])
        except (KeyError, TypeError, ValueError):
            raise data.DataError("each span needs integer start and end (samples)") from None
        a, b = max(0, a), min(n, b)
        if b <= a:
            continue
        lab = _labels_ok(s["label"]) if s.get("label") else ""
        cur = _merge(cur, {"start": a, "end": b, "label": lab, **({"by": s["by"]} if s.get("by") else {})})
    rec["spans"] = cur
    labels = list(rec.get("labels") or [])
    for s in cur:
        if s["label"] not in labels:
            labels.append(s["label"])
    rec["labels"] = labels
    _save_meta(rid, rec)
    return get(rid)


def set_labels(rid: str, labels: list[str]) -> dict:
    rec = get(rid)
    labels = [_labels_ok(x) for x in labels]
    if len(set(labels)) != len(labels):
        raise data.DataError("labels must be unique")
    used = {s["label"] for s in rec.get("spans", [])}
    missing = used - set(labels)
    if missing:
        raise data.DataError(f"labels in use cannot be removed: {', '.join(sorted(missing))}")
    rec["labels"] = labels
    _save_meta(rid, rec)
    return get(rid)


# ---- live capture ------------------------------------------------------------------------------------------------
def start_capture(org_id: str, user_id: str | None, name: str, channels: list[str], sample_rate: float,
                  template_id: str | None = None, licence: str = "proprietary") -> tuple[dict, str]:
    token = "cap_" + secrets.token_urlsafe(24)
    rec = create(org_id, user_id, name, channels, sample_rate, "live", licence, template_id, status="capturing",
                 token_hash=hashlib.sha256(token.encode()).hexdigest())
    _part(rec["id"]).write_bytes(b"")
    return rec, token


def check_token(rid: str, token: str) -> dict:
    with db.conn() as c:
        r = c.execute("SELECT token_hash, status FROM recordings WHERE id=?", (rid,)).fetchone()
    if not r or not r[0] or not secrets.compare_digest(r[0], hashlib.sha256((token or "").encode()).hexdigest()):
        raise PermissionError("bad capture token")
    if r[1] != "capturing":
        raise data.DataError("this capture has been stopped")
    return get(rid)


def parse_push(body: bytes, content_type: str, C: int) -> np.ndarray:
    """JSON {"samples": [[c0, c1, ...], ...]} or text lines "c0,c1,..." (CSV / serial log)."""
    if "json" in (content_type or ""):
        try:
            rows = json.loads(body).get("samples")
        except (ValueError, AttributeError):
            raise data.DataError('send JSON {"samples": [[...], ...]}') from None
        try:
            arr = np.asarray(rows, np.float32)
        except (TypeError, ValueError):
            raise data.DataError("samples must be a list of equally long numeric rows") from None
        if arr.ndim == 1 and C == 1:
            arr = arr[:, None]
    else:
        vals = []
        for line in body.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line[0] in "#;":
                continue
            try:
                vals.append([float(v) for v in line.replace(";", ",").replace(" ", ",").split(",") if v != ""])
            except ValueError:
                continue                      # a serial log line that is not data (boot messages, ...)
        arr = np.asarray([v for v in vals if len(v) == C], np.float32).reshape(-1, C)
    if arr.ndim != 2 or (len(arr) and arr.shape[1] != C):
        raise data.DataError(f"each sample needs {C} values")
    if len(arr) > MAX_PUSH_ROWS:
        raise data.DataError(f"at most {MAX_PUSH_ROWS} samples per request")
    if not np.isfinite(arr).all():
        raise data.DataError("NaN or infinite values")
    return arr


def push(rec: dict, arr: np.ndarray) -> int:
    C = len(rec["channels"])
    if (samples(rec) + len(arr)) * C > MAX_VALUES:
        raise data.DataError("capture is full; stop it and start a new one")
    with open(_part(rec["id"]), "ab") as f:
        f.write(np.ascontiguousarray(arr, np.float32).tobytes())
    return samples(rec)


def mark(rid: str, label: str | None) -> dict:
    """Live labelling: from now on the incoming samples are `label` (None = unlabelled). Closes the open span."""
    rec = get(rid)
    if rec["status"] != "capturing":
        raise data.DataError("the capture has been stopped; edit its spans instead")
    now = samples(rec)
    live = rec.get("live") or {}
    if live.get("label") and now > live["start"]:
        rec = set_spans(rid, [{"start": live["start"], "end": now, "label": live["label"]}], replace=False)
    rec["live"] = {"label": _labels_ok(label), "start": now} if label else None
    if label and label not in rec["labels"]:
        rec["labels"] = rec["labels"] + [label]
    _save_meta(rid, rec)
    return get(rid)


def stop_capture(rid: str) -> dict:
    rec = get(rid)
    if rec["status"] != "capturing":
        return rec
    rec = mark(rid, None)
    arr = np.array(signal(rec))
    np.save(_npy(rid), arr)
    _part(rid).unlink(missing_ok=True)
    rec.pop("live", None)
    _save_meta(rid, rec, status="ready")
    with db.conn() as c:
        c.execute("UPDATE recordings SET token_hash=NULL WHERE id=?", (rid,))
    return get(rid)


def delete(rid: str):
    for p in (_npy(rid), _part(rid)):
        p.unlink(missing_ok=True)
    with db.conn() as c:
        c.execute("DELETE FROM recordings WHERE id=?", (rid,))


# ---- suggestions from a trained model ---------------------------------------------------------------------------
def suggest(rec: dict, job_id: str, hop: int | None = None, min_conf: float = 0.6, start: int = 0,
            end: int | None = None) -> dict:
    """Run a finished project's INT8 model over sliding windows; return proposed spans (not saved)."""
    from . import inference
    m = inference.ProjectModel(job_id)
    if m.task != "classification":
        raise data.DataError("label suggestions need a classification project")
    T = m.raw_window
    C = m.raw_channels
    if C != len(rec["channels"]):
        raise data.DataError(f"the project's model reads {C} channels; this recording has {len(rec['channels'])}")
    sig = np.asarray(signal(rec), np.float32)
    end = len(sig) if end is None else min(int(end), len(sig))
    hop = int(hop or max(1, T // 2))
    starts = list(range(max(0, int(start)), end - T + 1, hop))[:20000]
    if not starts:
        raise data.DataError(f"the range is shorter than one window ({T} samples)")
    X = np.stack([sig[s:s + T].T for s in starts])[:, :, None, :]
    probs = m.predict_proba(X)
    spans = []
    for s, p in zip(starts, probs):
        k = int(np.argmax(p))
        if p[k] < min_conf:
            continue
        lab = m.labels[k]
        if spans and spans[-1]["label"] == lab and spans[-1]["end"] >= s:
            spans[-1]["end"] = s + T
            spans[-1]["conf"] = round(min(spans[-1]["conf"], float(p[k])), 3)
            continue
        a = s
        if spans and spans[-1]["end"] > s:            # overlapping windows disagree: split the overlap
            a = s + (spans[-1]["end"] - s) // 2
            spans[-1]["end"] = a
        spans.append({"start": int(a), "end": int(s + T), "label": lab, "conf": round(float(p[k]), 3), "by": job_id})
    return {"spans": spans, "windows": len(starts), "window": T, "hop": hop, "labels": m.labels,
            "note": "suggestions from the project's INT8 model (onnxruntime); review before accepting"}


# ---- to a dataset -----------------------------------------------------------------------------------------------
def to_dataset(recs: list[dict], template: dict, name: str, window: int | None = None, hop: int | None = None,
               min_cover: float = 0.5, licence: str | None = None, labels: list[str] | None = None) -> dict:
    if template.get("task") == "regression":
        raise data.DataError("recordings are labelled with classes; regression datasets come from CSV with a target")
    inp = template["input"]
    if inp.get("kind") != "timeseries":
        raise data.DataError("recordings make time-series datasets")
    fe = template.get("frontend")
    T = int(window or (fe or {}).get("raw_length") or template.get("window") or inp["width"])
    hop = int(hop or template.get("hop") or T // 2)
    if not 1 <= hop <= 10 * T:
        raise data.DataError("hop must be between 1 and 10 windows")
    chans = {tuple(r["channels"]) for r in recs}
    if len(chans) != 1:
        raise data.DataError("all recordings must have the same channels")
    labels = list(labels or template.get("labels") or [])
    if template["task"] == "anomaly":
        labels = ["normal", "anomaly"]
    if not labels:
        for r in recs:
            labels += [x for x in r.get("labels", []) if x not in labels]
    Xs, ys, gs, skipped, unknown = [], [], [], 0, set()
    for r in recs:
        sig = np.asarray(signal(r), np.float32)
        spans = r.get("spans", [])
        if not spans:
            continue
        cover = np.full(len(sig), -1, np.int32)
        sid = np.full(len(sig), -1, np.int32)
        for i, s in enumerate(spans):
            lab = s["label"]
            if template["task"] == "anomaly":
                lab = "normal" if lab.lower() in ("normal", "ok", "healthy", "0") else "anomaly"
            if lab not in labels:
                unknown.add(s["label"])
                continue
            cover[s["start"]:s["end"]] = labels.index(lab)
            sid[s["start"]:s["end"]] = i
        for a in range(0, len(sig) - T + 1, hop):
            w = cover[a:a + T]
            vals, cnt = np.unique(w[w >= 0], return_counts=True)
            if not len(vals) or cnt.max() < min_cover * T:
                skipped += 1
                continue
            k = int(vals[np.argmax(cnt)])
            Xs.append(sig[a:a + T].T)
            ys.append(k)
            s = sid[a:a + T]
            gs.append(f"{r['id']}:{int(np.bincount(s[s >= 0]).argmax())}")
    if unknown:
        raise data.DataError(f"labels {sorted(unknown)} are not classes of this model type ({labels}); "
                             "rename them in the labelling tool")
    if not Xs:
        raise data.DataError(f"no window of {T} samples is at least {min_cover:.0%} labelled")
    X = np.array(Xs, np.float32)[:, :, None, :]
    if X.shape[1] != int(inp["channels"]) and not template.get("custom"):
        raise data.DataError(f"this model type reads {inp['channels']} channels; the recordings have {X.shape[1]}")
    lic = licence or (recs[0].get("licence") if len({r.get("licence") for r in recs}) == 1 else "unknown")
    meta = {"source": "recordings", "template": template["id"], "task": template["task"], "name": name,
            "licence": lic, "recordings_used": [r["id"] for r in recs], "window": T, "hop": hop,
            "unlabelled_windows_skipped": skipped}
    return data.save(X, np.array(ys), labels, meta, np.array(gs), template)
