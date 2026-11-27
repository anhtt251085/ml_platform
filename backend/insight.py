# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Project insight: what-if sliders, battery life and power, drift checks.

What-if (window length, channels)
  The selected model's architecture is rebuilt for another input (raw window length after the project's
  decimation, number of channels) and its MACs, weights and peak activation recomputed exactly. RAM, flash and
  latency are then *scaled from the project's own SAI numbers* (workspace by peak activation, flash by weight
  bytes, cycles by MACs): planning numbers, labelled "scaled". "Check with SAI" runs a real fit check of that
  input. Accuracy cannot be predicted without retraining; the UI says so.

Battery life and power
  energy per inference = active power x latency, or a board's measured energy per inference when a physical
  board with a power monitor has run this project (`energy_uj` in its result). Average power = sleep + rate x
  energy + sensor + radio; life = usable capacity / average power. Power profiles per target are typical values
  for that MCU class (evidence "assumed") and every number can be overridden.

Drift
  Production data (a recording or a dataset) is compared with the project's training data: per channel, the
  distributions of window mean, window RMS and window peak-to-peak (population stability index, 10 quantile bins
  of the reference), the model's predicted-class mix and its mean confidence. PSI < 0.1 stable, 0.1-0.25 watch,
  > 0.25 drift. The reference is computed once per project (drift_ref.json in its folder)."""

from __future__ import annotations

import copy
import json
import uuid

import numpy as np

from . import archspec, config, data, db, sai_engine

# ---- what-if -----------------------------------------------------------------------------------------------------
FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def _selected(res: dict) -> tuple[dict, dict]:
    sel = res["selected"]
    v = next(x for x in res["variants"] if x["size"] == sel["size"])
    return sel, v


def scale(res: dict, window: int, channels: int) -> dict:
    """Estimated resources of the selected model for a raw window of `window` samples and `channels` channels."""
    sel, v = _selected(res)
    pre = res.get("preprocessing") or {}
    pcfg = pre.get("preprocess") or {}
    dec = int(pcfg.get("decimate") or 1)
    base = v["spec"]
    seq = base.get("layout") == "sequence"
    spec = copy.deepcopy(base)
    if seq:
        raise ValueError("what-if for sequence models: change the window in a new project (SAI's front end plans them)")
    inp = spec["input"]
    if inp["height"] != 1:
        raise ValueError("what-if sliders are for 1-D signals")
    if pre.get("frontend"):
        raise ValueError("this model reads front-end features (MFCC / band energies); change the front end instead")
    inp["channels"], inp["width"] = int(channels), max(1, int(window) // dec)
    old = archspec.summary(archspec.normalize(base))      # stored summaries are slimmed; recompute exactly
    try:
        new = archspec.summary(archspec.normalize(spec))
    except archspec.SpecError as e:
        return {"window": int(window), "channels": int(channels), "ok": False, "reason": str(e)}
    ws0 = sel.get("workspace") or 0
    peak0 = max(1, old["peak_activation_bytes"])
    ws = int(round(ws0 * new["peak_activation_bytes"] / peak0)) if ws0 else None
    ram = (sel["ram"] - ws0 + ws) if (sel.get("ram") and ws is not None) else None
    flash = sel["flash"] + new["weight_bytes_int8"] - old["weight_bytes_int8"] if sel.get("flash") else None
    lat = (sel.get("latency") or {}).get("ms")
    ms = lat * new["macs"] / max(1, old["macs"]) if lat else None
    t = sai_engine.TARGETS.get(sel["target"], {})
    b = t.get("budget") or {}
    viol = []
    if ws is not None and b.get("workspace") and ws > b["workspace"]:
        viol.append(f"workspace {ws} B > {b['workspace']} B")
    if flash and b.get("flash") and flash > b["flash"]:
        viol.append(f"flash {flash} B > {b['flash']} B")
    if ram and t.get("ram_total") and ram > t["ram_total"]:
        viol.append(f"RAM {ram} B > {t['ram_total']} B")
    return {"window": int(window), "channels": int(channels), "ok": True, "model_input": [int(channels), 1, inp["width"]],
            "macs": new["macs"], "params": new["params"], "peak_activation": new["peak_activation_bytes"],
            "workspace": ws, "ram": ram, "flash": flash, "latency_ms": round(ms, 4) if ms else None,
            "fits": not viol, "violations": viol, "evidence": "scaled"}


def whatif_grid(res: dict, tpl: dict) -> dict:
    sel, v = _selected(res)
    pre = res.get("preprocessing") or {}
    raw = pre.get("raw_shape") or v["spec"]["input"] and [v["spec"]["input"]["channels"], 1, v["spec"]["input"]["width"]]
    W, C = int(raw[-1]), int(raw[0])
    if (pre.get("preprocess") or {}).get("channels"):
        C_used = len(pre["preprocess"]["channels"])
    else:
        C_used = C
    windows = sorted({max(16, int(round(W * f / 8)) * 8) for f in FACTORS})
    cmax = min(16, max(C, C_used) + 3)            # also: what if a sensor channel is added
    rows = [scale(res, w, c) for w in windows for c in range(1, cmax + 1)]
    fs = (tpl.get("input") or {}).get("sample_rate_hz")
    return {"base": {"window": W, "channels": C_used, "target": sel["target"], "ram": sel.get("ram"),
                     "flash": sel.get("flash"), "workspace": sel.get("workspace"),
                     "latency_ms": (sel.get("latency") or {}).get("ms"),
                     "latency_evidence": (sel.get("latency") or {}).get("evidence"),
                     "macs": v["summary"].get("macs"), "metric": sel.get("metric")},
            "sample_rate_hz": fs, "hop": tpl.get("hop"), "windows": windows, "max_channels": cmax, "raw_channels": C,
            "channel_names": (tpl.get("input") or {}).get("channel_names"), "grid": rows,
            "budget": sai_engine.TARGETS.get(sel["target"], {}).get("budget"),
            "note": "RAM, flash and latency are scaled from this project's SAI numbers; accuracy needs retraining."}


def whatif_spec(res: dict, window: int, channels: int) -> dict:
    """The selected architecture for another input, for a real SAI fit check."""
    sel, v = _selected(res)
    dec = int(((res.get("preprocessing") or {}).get("preprocess") or {}).get("decimate") or 1)
    spec = copy.deepcopy(v["spec"])
    spec["input"]["channels"], spec["input"]["width"] = int(channels), max(1, int(window) // dec)
    return archspec.normalize(spec)


# ---- battery and power --------------------------------------------------------------------------------------------
POWER = {
    # typical values for the MCU class at SAI's planning clock; override with your board's numbers
    "md6605": {"active_mw": 33.0, "sleep_uw": 10.0, "basis": "RV32 at 66.7 MHz, ~150 uA/MHz at 3.3 V; deep sleep ~3 uA"},
    "cortex-m7": {"active_mw": 420.0, "sleep_uw": 300.0, "basis": "STM32H7 at 480 MHz, ~263 uA/MHz at 3.3 V; Stop mode ~90 uA"},
}
BATTERIES = {
    "cr2032": {"label": "CR2032 coin cell", "mah": 225, "v": 3.0},
    "2xaa": {"label": "2 x AA alkaline", "mah": 2500, "v": 3.0},
    "li-18650": {"label": "Li-ion 18650", "mah": 3000, "v": 3.7},
    "lipo-500": {"label": "LiPo 500 mAh", "mah": 500, "v": 3.7},
    "4-20ma": {"label": "Loop-powered (no battery)", "mah": None, "v": None},
}


def measured_power(job_id: str, target: str) -> dict | None:
    """Median energy per inference / power from physical board runs of this project (never simulated ones)."""
    with db.conn() as c:
        rows = [db.uj(r[0], {}) for r in c.execute(
            "SELECT result FROM hil_runs WHERE job_id=? AND target=? AND status='succeeded'", (job_id, target))]
    rows = [r for r in rows if not r.get("simulated") and not r.get("emulated")]
    e = [r["energy_uj"] for r in rows if r.get("energy_uj")]
    a = [r["active_power_mw"] for r in rows if r.get("active_power_mw")]
    s = [r["sleep_power_uw"] for r in rows if r.get("sleep_power_uw")]
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    if not (e or a):
        return None
    return {"energy_uj": float(np.median(e)) if e else None, "active_mw": float(np.median(a)) if a else None,
            "sleep_uw": float(np.median(s)) if s else None, "latency_ms": float(np.median(lat)) if lat else None,
            "runs": len(rows), "evidence": "measured"}


def battery(res: dict, job_id: str, tpl: dict, p: dict) -> dict:
    sel = res["selected"]
    tgt = p.get("target") or sel["target"]
    prof = POWER.get(tgt, {"active_mw": 50.0, "sleep_uw": 50.0, "basis": "generic MCU"})
    meas = measured_power(job_id, tgt)
    lat = (sel.get("latency") or {}).get("ms")
    lat_ev = (sel.get("latency") or {}).get("evidence")
    if meas and meas.get("latency_ms"):
        lat, lat_ev = meas["latency_ms"], "measured"
    active = float(p.get("active_mw") or (meas or {}).get("active_mw") or prof["active_mw"])
    sleep_uw = float(p.get("sleep_uw") or (meas or {}).get("sleep_uw") or prof["sleep_uw"])
    if not lat:
        raise ValueError("this project has no latency estimate for that board")
    if meas and meas.get("energy_uj") and not p.get("active_mw"):
        e_inf, e_ev = meas["energy_uj"], "measured"
    else:
        e_inf = active * lat                                   # mW x ms = uJ
        e_ev = "measured" if (meas and meas.get("active_mw") and not p.get("active_mw")) else (
            "custom" if p.get("active_mw") else "assumed")
    fs = (tpl.get("input") or {}).get("sample_rate_hz")
    hop = tpl.get("hop") or (tpl.get("input") or {}).get("width")
    rate = float(p.get("inferences_per_s") or (fs / hop if fs and hop else 1.0))
    max_rate = 1000.0 / lat
    sensor_mw = float(p.get("sensor_mw") or 0)
    radio = float(p.get("radio_uj") or 0) * float(p.get("events_per_hour") or 0) / 3600.0 / 1000.0   # mW
    duty = min(1.0, rate * lat / 1000.0)
    infer_mw = rate * e_inf / 1000.0
    avg_mw = sleep_uw / 1000.0 * (1 - duty) + infer_mw + sensor_mw + radio
    bat = BATTERIES.get(p.get("battery") or "2xaa", BATTERIES["2xaa"])
    mah = float(p.get("mah") or bat["mah"] or 0)
    v = float(p.get("volts") or bat["v"] or 0)
    derate = float(p.get("derate") or 0.85)
    hours = (mah * v * derate / avg_mw) if mah and v and avg_mw > 0 else None
    return {"target": tgt, "latency_ms": round(lat, 4), "latency_evidence": lat_ev, "energy_uj": round(e_inf, 3),
            "energy_evidence": e_ev, "active_mw": active, "sleep_uw": sleep_uw, "profile_basis": prof["basis"],
            "inferences_per_s": rate, "max_inferences_per_s": round(max_rate, 1), "duty_cycle": round(duty, 6),
            "breakdown_mw": {"sleep": round(sleep_uw / 1000.0 * (1 - duty), 5), "inference": round(infer_mw, 5),
                             "sensor": sensor_mw, "radio": round(radio, 5)},
            "avg_mw": round(avg_mw, 5), "battery": {**bat, "mah": mah, "v": v, "derate": derate},
            "life_hours": round(hours, 1) if hours else None, "life_days": round(hours / 24, 1) if hours else None,
            "measured": meas, "too_fast": rate > max_rate,
            "batteries": BATTERIES}


# ---- drift ------------------------------------------------------------------------------------------------------
BINS = 10
FEATURES = ("mean", "rms", "p2p")


def _feats(X: np.ndarray) -> dict[str, np.ndarray]:
    """X [N, C, H, W] -> per feature [N, C]."""
    Xf = X.reshape(X.shape[0], X.shape[1], -1).astype(np.float64)
    return {"mean": Xf.mean(2), "rms": np.sqrt((Xf ** 2).mean(2)), "p2p": Xf.max(2) - Xf.min(2)}


def _edges(v: np.ndarray) -> list[float]:
    e = np.unique(np.quantile(v, np.linspace(0, 1, BINS + 1)[1:-1]))
    return e.tolist()


def _hist(v: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin shares with add-one-half smoothing: an empty bin in a small production sample must not read as
    infinite drift."""
    idx = np.searchsorted(np.asarray(edges), v, side="right")
    k = len(edges) + 1
    return (np.bincount(idx, minlength=k) + 0.5) / (len(v) + 0.5 * k)


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(np.asarray(ref, float), 1e-4, None), np.clip(np.asarray(cur, float), 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status_of(v: float) -> str:
    return "stable" if v < 0.1 else "watch" if v < 0.25 else "drift"


def _model(job_id):
    from . import inference
    try:
        return inference.ProjectModel(job_id)
    except ValueError:
        return None


def reference(job_id: str, dataset_id: str) -> dict:
    path = config.DATA_DIR / "jobs" / job_id / "drift_ref.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        X, y, labels, meta = data.load(dataset_id)
    except data.DataError:
        raise ValueError("the project's training dataset was deleted; drift needs it as the reference") from None
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(len(X), 2000), replace=False)
    X = X[idx]
    f = _feats(X)
    ref = {"n": int(len(X)), "channels": int(X.shape[1]), "shape": list(X.shape[1:]),
           "features": {k: {"edges": [_edges(f[k][:, c]) for c in range(X.shape[1])],
                            "hist": [], "mean": f[k].mean(0).tolist(), "std": f[k].std(0).tolist()} for k in FEATURES}}
    for k in FEATURES:
        ref["features"][k]["hist"] = [_hist(f[k][:, c], ref["features"][k]["edges"][c]).tolist()
                                      for c in range(X.shape[1])]
    m = _model(job_id)
    if m and m.task == "classification":
        pr = m.predict_proba(X[:500], features_ready=True)          # dataset windows hold front-end features
        ref["pred_mix"] = (np.bincount(pr.argmax(1), minlength=len(m.labels)) / len(pr)).tolist()
        ref["confidence"] = float(pr.max(1).mean())
        ref["labels"] = m.labels
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref))
    return ref


def check(job: dict, X: np.ndarray, source: str, channel_names: list[str] | None = None) -> dict:
    """X: production windows in the project's raw input layout [N, C, H, W]."""
    res = job["result"]
    ref = reference(job["id"], job["params"]["dataset_id"])
    if list(X.shape[1:]) != ref["shape"]:
        raise ValueError(f"production windows are {list(X.shape[1:])}; the project was trained on {ref['shape']}")
    if len(X) < 20:
        raise ValueError(f"only {len(X)} windows: at least 20 are needed for a drift check")
    f = _feats(X)
    names = channel_names or [f"ch{c}" for c in range(X.shape[1])]
    rows = []
    for k in FEATURES:
        for c in range(X.shape[1]):
            r = ref["features"][k]
            p = psi(r["hist"][c], _hist(f[k][:, c], r["edges"][c]))
            rows.append({"feature": k, "channel": names[c] if c < len(names) else f"ch{c}", "psi": round(p, 4),
                         "status": status_of(p), "ref_mean": round(r["mean"][c], 5),
                         "cur_mean": round(float(f[k][:, c].mean()), 5),
                         "shift_sd": round((float(f[k][:, c].mean()) - r["mean"][c]) / (r["std"][c] or 1e-9), 2)})
    out = {"windows": int(len(X)), "source": source, "inputs": rows,
           "input_psi": round(max(r["psi"] for r in rows), 4)}
    if ref.get("pred_mix"):
        m = _model(job["id"])
        pr = m.predict_proba(X[:2000])
        k = len(ref["labels"])
        mix = (np.bincount(pr.argmax(1), minlength=k) + 0.5) / (len(pr) + 0.5 * k)
        out["predictions"] = {"labels": ref["labels"], "reference": ref["pred_mix"], "current": mix.round(4).tolist(),
                              "psi": round(psi(ref["pred_mix"], mix), 4),
                              "confidence_ref": round(ref["confidence"], 4),
                              "confidence": round(float(pr.max(1).mean()), 4)}
    worst = max([out["input_psi"]] + ([out["predictions"]["psi"]] if "predictions" in out else []))
    out["psi"] = worst
    out["status"] = status_of(worst)
    if "predictions" in out and out["predictions"]["confidence"] < out["predictions"]["confidence_ref"] - 0.1:
        out["status"] = "drift" if out["status"] == "drift" else "watch"
        out["confidence_drop"] = True
    top = sorted(rows, key=lambda r: -r["psi"])[:3]
    out["summary"] = {"stable": "Production data looks like the training data.",
                      "watch": "Some inputs moved; keep an eye on accuracy.",
                      "drift": "Production data differs from the training data: collect and label new data and retrain."}[out["status"]]
    out["top"] = [f"{r['channel']} {r['feature']}: PSI {r['psi']} ({r['shift_sd']:+} sd)" for r in top if r["psi"] >= 0.1]
    return out


def windows_from_recording(rec: dict, raw_shape: list, frontend: dict | None = None) -> np.ndarray:
    """Windows in the layout the training data was stored in (front-end features for MFCC / band-energy types)."""
    from . import features, recordings
    C, T = int(raw_shape[0]), int(frontend["raw_length"]) if frontend else int(raw_shape[-1])
    if len(rec["channels"]) != C:
        raise ValueError(f"the project reads {C} channels; the recording has {len(rec['channels'])}")
    sig = np.asarray(recordings.signal(rec), np.float32)
    starts = range(0, len(sig) - T + 1, T)
    if not len(starts):
        raise ValueError(f"the recording is shorter than one window ({T} samples)")
    X = np.stack([sig[s:s + T].T for s in starts])[:, :, None, :][:5000]
    return features.apply(frontend, X) if frontend else X


def save(job: dict, out: dict) -> dict:
    did = "drf_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO drift_checks (id, job_id, org_id, created, source, status, result) VALUES (?,?,?,?,?,?,?)",
                  (did, job["id"], job.get("org_id"), db.now(), out["source"], out["status"], db.j(out)))
    return {"id": did, "created": db.now(), **out}


def history(job_id: str) -> list[dict]:
    with db.conn() as c:
        return [{"id": r[0], "created": r[1], "source": r[2], "status": r[3], **db.uj(r[4], {})}
                for r in c.execute("SELECT id, created, source, status, result FROM drift_checks WHERE job_id=?"
                                   " ORDER BY created", (job_id,))]
