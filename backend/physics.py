# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Physics-based data augmentation: windows simulated from the customer's own machine parameters.

A demo generator (synth.py) uses fixed, typical numbers. Here the engineer enters the numbers of *their* machine
(shaft speed, bearing geometry, supply frequency, slip, pole pairs, structural resonance, sensor noise...) and the
fault signatures follow from the physics:

* rolling bearings: the defect frequencies BPFO / BPFI / BSF / FTF from the kinematics; each impact rings the
  structure's resonance; inner-race impacts are amplitude-modulated by the shaft (the defect passes through the
  load zone once per turn), ball defects by the cage (FTF); 1-2 % random slip makes the impacts jitter.
* motor current (MCSA): broken rotor bars put sidebands at (1 +- 2ks) f; air-gap eccentricity at
  f (1 +- k (1 - s) / p); an inter-turn short unbalances the phases and adds a 3rd harmonic.
* DC series arcs: a current step at ignition, then broadband (1 kHz-fs/2), bursty arc noise on top of the
  converter's switching ripple.

The result is a dataset (licence "synthetic") or, with `mix_with`, a customer dataset plus simulated windows of the
classes it lacks (licence and recordings of the customer data are kept; simulated windows form their own groups so
they never leak across splits). `augment_real` adds physically meaningful variations of the real windows: sensor
gain and offset error, speed change (time stretch) and noise at a given SNR.

Every model lists its parameters with units and ranges (GET /api/physics/models) so the UI can render the form."""

from __future__ import annotations

import math

import numpy as np

from . import data


def _p(key, label, unit, default, lo, hi, step=None, kind="number"):
    return {"key": key, "label": label, "unit": unit, "default": default, "min": lo, "max": hi,
            "step": step, "kind": kind}


MODELS = {
    "bearing": {
        "name": "Rolling-element bearing", "classes": ["normal", "inner_race", "outer_race", "ball"],
        "params": [
            _p("rpm", "Shaft speed", "rpm", 1750, 60, 30000, 10),
            _p("n_balls", "Rolling elements", "", 9, 4, 40, 1),
            _p("ball_d_mm", "Ball diameter", "mm", 7.94, 1, 100, 0.01),
            _p("pitch_d_mm", "Pitch diameter", "mm", 39.04, 5, 600, 0.01),
            _p("contact_deg", "Contact angle", "deg", 0, 0, 45, 1),
            _p("resonance_hz", "Structural resonance", "Hz", 1100, 50, 20000, 10),
            _p("damping", "Damping ratio", "", 0.05, 0.005, 0.3, 0.005),
            _p("severity", "Defect severity", "x", 1.0, 0.1, 5, 0.1),
            _p("snr_db", "Signal-to-noise", "dB", 12, -10, 60, 1),
        ]},
    "motor_current": {
        "name": "Induction motor phase currents", "classes": ["healthy", "broken_bar", "eccentricity", "winding_short"],
        "params": [
            _p("supply_hz", "Supply frequency", "Hz", 50, 5, 400, 0.1),
            _p("slip", "Slip at load", "", 0.03, 0.001, 0.15, 0.001),
            _p("pole_pairs", "Pole pairs", "", 2, 1, 8, 1),
            _p("amp_a", "Current amplitude", "A", 1.0, 0.01, 1000, 0.01),
            _p("severity", "Fault severity", "x", 1.0, 0.1, 5, 0.1),
            _p("snr_db", "Signal-to-noise", "dB", 35, 0, 80, 1),
        ]},
    "arc_fault": {
        "name": "DC series arc", "classes": ["normal", "arc"],
        "params": [
            _p("i_dc", "DC current", "A", 8.0, 0.1, 200, 0.1),
            _p("f_sw_hz", "Converter switching frequency", "Hz", 16000, 1000, 49000, 100),
            _p("ripple_pct", "Switching ripple", "%", 2.0, 0, 30, 0.1),
            _p("arc_drop_pct", "Current drop at ignition", "%", 6, 0, 50, 0.5),
            _p("arc_noise_pct", "Arc noise (RMS)", "%", 4, 0.1, 50, 0.1),
            _p("snr_db", "Sensor signal-to-noise", "dB", 40, 0, 80, 1),
        ]},
}


def bearing_frequencies(rpm: float, n_balls: int, ball_d_mm: float, pitch_d_mm: float, contact_deg: float) -> dict:
    """The four classic defect frequencies (Hz) of a rolling-element bearing with a stationary outer race."""
    fr = rpm / 60.0
    r = ball_d_mm / pitch_d_mm * math.cos(math.radians(contact_deg))
    return {"shaft": fr, "ftf": fr / 2 * (1 - r), "bpfo": n_balls / 2 * fr * (1 - r),
            "bpfi": n_balls / 2 * fr * (1 + r), "bsf": pitch_d_mm / (2 * ball_d_mm) * fr * (1 - r * r)}


def motor_frequencies(supply_hz: float, slip: float, pole_pairs: int) -> dict:
    f, s, p = supply_hz, slip, pole_pairs
    return {"supply": f, "broken_bar": [f * (1 - 2 * s), f * (1 + 2 * s)],
            "eccentricity": [f * (1 - (1 - s) / p), f * (1 + (1 - s) / p)]}


def frequencies(model: str, params: dict) -> dict:
    p = resolve(model, params)
    if model == "bearing":
        return bearing_frequencies(p["rpm"], int(p["n_balls"]), p["ball_d_mm"], p["pitch_d_mm"], p["contact_deg"])
    if model == "motor_current":
        return motor_frequencies(p["supply_hz"], p["slip"], int(p["pole_pairs"]))
    return {"switching": p["f_sw_hz"]}


def resolve(model: str, params: dict | None) -> dict:
    if model not in MODELS:
        raise data.DataError(f"no physics model for '{model}'; available: {', '.join(MODELS)}")
    out = {}
    for spec in MODELS[model]["params"]:
        v = (params or {}).get(spec["key"], spec["default"])
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise data.DataError(f"{spec['label']} must be a number") from None
        if not spec["min"] <= v <= spec["max"]:
            raise data.DataError(f"{spec['label']} must be between {spec['min']} and {spec['max']} {spec['unit']}".strip())
        out[spec["key"]] = v
    unknown = set(params or {}) - set(out)
    if unknown:
        raise data.DataError(f"unknown parameter(s) {', '.join(sorted(unknown))}")
    if model == "bearing" and out["ball_d_mm"] * 2 >= out["pitch_d_mm"]:
        raise data.DataError("the pitch diameter must be more than twice the ball diameter")
    return out


def _noise(rng, sig, snr_db):
    pw = float(np.mean(sig ** 2)) or 1e-12
    return rng.normal(0, math.sqrt(pw / 10 ** (snr_db / 10)), sig.shape)


def _ring(fs, f_res, zeta, n):
    t = np.arange(n) / fs
    wn = 2 * math.pi * f_res
    return np.exp(-zeta * wn * t) * np.sin(wn * math.sqrt(max(1e-6, 1 - zeta * zeta)) * t)


def _bearing(rng, cls, p, C, T, fs):
    jit = rng.uniform(0.97, 1.03)                               # speed varies around the operating point
    fq = bearing_frequencies(p["rpm"] * jit, int(p["n_balls"]), p["ball_d_mm"], p["pitch_d_mm"], p["contact_deg"])
    t = np.arange(T) / fs
    ph = rng.uniform(0, 2 * math.pi)
    base = 0.3 * np.sin(2 * math.pi * fq["shaft"] * t + ph) + 0.1 * np.sin(4 * math.pi * fq["shaft"] * t + 2 * ph)
    x = np.zeros(T)
    if cls:
        fd = {1: fq["bpfi"], 2: fq["bpfo"], 3: 2 * fq["bsf"]}[cls]   # a ball defect hits both races per spin
        f_res = p["resonance_hz"] * rng.uniform(0.95, 1.05)
        n_ring = min(T, max(8, int(6 / (p["damping"] * 2 * math.pi * f_res) * fs)))
        ring = _ring(fs, f_res, p["damping"], n_ring)
        period = fs / fd
        k = rng.uniform(0, period)
        while k < T:
            ti = k / fs
            a = p["severity"] * rng.uniform(0.8, 1.2)
            if cls == 1:
                a *= 0.5 * (1 + math.cos(2 * math.pi * fq["shaft"] * ti + ph))     # load zone, once per turn
            elif cls == 3:
                a *= 0.5 * (1 + math.cos(2 * math.pi * fq["ftf"] * ti + ph))       # carried round by the cage
            i = int(k)
            m = min(n_ring, T - i)
            x[i:i + m] += a * ring[:m]
            k += period * rng.uniform(0.98, 1.02)               # slip between rolling elements and races
    sig = base + x
    gains = np.array([1.0, 0.7, 0.4, 0.3, 0.25, 0.2])[:C] if C <= 6 else np.linspace(1, 0.2, C)
    out = np.outer(gains, sig)
    return out + _noise(rng, out, p["snr_db"])


def _motor(rng, cls, p, C, T, fs):
    f = p["supply_hz"] * rng.uniform(0.995, 1.005)
    s = p["slip"] * rng.uniform(0.8, 1.2)
    pp = int(p["pole_pairs"])
    t = np.arange(T) / fs
    ph = rng.uniform(0, 2 * math.pi)
    A = p["amp_a"] * rng.uniform(0.9, 1.1)
    sev = p["severity"]
    out = []
    for k in range(C):
        w = 2 * math.pi * f * t + ph - k * 2 * math.pi / 3
        i = A * np.sin(w)
        if cls == 1:
            for m in (1, 2):
                a = A * 0.04 * sev / m
                i += a * (np.sin(2 * math.pi * f * (1 - 2 * m * s) * t + ph - k * 2 * math.pi / 3)
                          + np.sin(2 * math.pi * f * (1 + 2 * m * s) * t + ph - k * 2 * math.pi / 3))
        elif cls == 2:
            fe = f * (1 - s) / pp
            a = A * 0.05 * sev
            i += a * (np.sin(2 * math.pi * (f - fe) * t + ph) + np.sin(2 * math.pi * (f + fe) * t + ph))
        elif cls == 3:
            if k == 0:
                i *= 1 + 0.12 * sev
            i += A * 0.03 * sev * np.sin(3 * w)
        out.append(i)
    out = np.array(out)
    return out + _noise(rng, out, p["snr_db"])


def _arc(rng, cls, p, C, T, fs):
    t = np.arange(T) / fs
    I = p["i_dc"] * rng.uniform(0.9, 1.1)
    fsw = min(p["f_sw_hz"], fs / 2.2)
    x = I + I * p["ripple_pct"] / 100 * np.sign(np.sin(2 * math.pi * fsw * t + rng.uniform(0, 6.28)))
    if cls:
        start = int(rng.integers(0, T // 3))
        drop = I * p["arc_drop_pct"] / 100
        x[start:] -= drop
        burst = rng.normal(0, 1, T)
        # broadband, bursty: white noise high-passed (first difference) and gated by a random on/off envelope
        burst = np.diff(burst, prepend=0) / math.sqrt(2)
        gate = (rng.random(T // 16 + 1) < 0.7).repeat(16)[:T]
        x[start:] += (I * p["arc_noise_pct"] / 100 * burst * gate)[start:]
    out = np.tile(x, (C, 1))
    return out + _noise(rng, out - out.mean(), p["snr_db"])


SIM = {"bearing": _bearing, "motor_current": _motor, "arc_fault": _arc}


def model_for(template: dict) -> str:
    m = template.get("synth")
    if m not in MODELS:
        raise data.DataError("physics-based augmentation exists for bearings, motor currents and DC arcs; "
                             "this model type has none")
    if template["input"].get("kind") != "timeseries" or template.get("frontend"):
        raise data.DataError("physics-based augmentation needs a raw time-series model type")
    return m


def simulate(template: dict, params: dict | None, per_class: int, seed: int = 11, classes: list[str] | None = None):
    """[N, C, 1, T] windows at the template's sample rate and window length; labels in the template's order."""
    model = model_for(template)
    p = resolve(model, params)
    labels = list(template["labels"])
    names = MODELS[model]["classes"]
    if [x for x in labels if x not in names]:
        raise data.DataError(f"this model type's classes {labels} do not match the physics model's {names}")
    inp = template["input"]
    C, T, fs = int(inp["channels"]), int(inp["width"]), float(inp.get("sample_rate_hz") or 1000)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for li, lab in enumerate(labels):
        if classes and lab not in classes:
            continue
        for _ in range(per_class):
            X.append(SIM[model](rng, names.index(lab), p, C, T, fs))
            y.append(li)
    if not X:
        raise data.DataError("no classes to simulate")
    return np.array(X, np.float32)[:, :, None, :], np.array(y), labels


def augment_real(X, rng, gain_pct=5.0, offset_pct=2.0, speed_pct=3.0, snr_db=30.0):
    """One physically plausible variant per window: sensor gain/offset error, speed change, noise."""
    n, C, H, T = X.shape
    out = np.empty_like(X)
    for i in range(n):
        w = X[i, :, 0, :].astype(np.float64)
        k = 1 + rng.uniform(-speed_pct, speed_pct) / 100           # time stretch = shaft speed / supply change
        src = np.clip(np.arange(T) * k, 0, T - 1)
        w = np.stack([np.interp(src, np.arange(T), ch) for ch in w])
        rms = np.sqrt(np.mean(w ** 2, axis=1, keepdims=True)) + 1e-9
        w = w * (1 + rng.uniform(-gain_pct, gain_pct, (C, 1)) / 100) + rng.uniform(-offset_pct, offset_pct, (C, 1)) / 100 * rms
        out[i, :, 0, :] = w + _noise(rng, w, snr_db)
    return out.astype(np.float32)


def build(template: dict, params: dict | None, per_class: int = 200, mix_with: str | None = None,
          augment: dict | None = None, name: str | None = None, seed: int = 11) -> dict:
    """Simulated dataset, or a customer dataset + simulated windows (+ physical variants of its windows)."""
    per_class = max(10, min(int(per_class), 5000))
    model = model_for(template)
    p = resolve(model, params)
    Xs, ys, labels = simulate(template, p, per_class, seed)
    sim_groups = np.array([f"physics:{i // 10}" for i in range(len(Xs))])
    meta = {"source": "physics", "template": template["id"], "task": template["task"],
            "physics": {"model": model, "params": p, "per_class": per_class, "frequencies": frequencies(model, p)}}
    if not mix_with:
        meta.update(name=name or f"Physics - {MODELS[model]['name']}", synthetic=True, licence="synthetic")
        return data.save(Xs, ys, labels, meta, sim_groups, template)
    Xr, yr, lr, mr = data.load(mix_with)
    if list(lr) != labels:
        raise data.DataError(f"the dataset's classes {lr} differ from this model type's {labels}")
    if list(Xr.shape[1:]) != list(Xs.shape[1:]):
        raise data.DataError(f"the dataset's windows {list(Xr.shape[1:])} differ from the simulated "
                             f"{list(Xs.shape[1:])}; physics windows mix with raw windows of the same shape only")
    gr = data.load_groups(mix_with)
    gr = gr if gr is not None else np.arange(len(Xr))
    parts_X, parts_y, parts_g = [Xr, Xs], [yr, ys], [np.array([str(g) for g in gr]), sim_groups]
    if augment:
        rng = np.random.default_rng(seed + 1)
        Xa = augment_real(Xr, rng, float(augment.get("gain_pct", 5)), float(augment.get("offset_pct", 2)),
                          float(augment.get("speed_pct", 3)), float(augment.get("snr_db", 30)))
        parts_X.append(Xa); parts_y.append(yr)
        parts_g.append(np.array([str(g) for g in gr]))            # a variant stays in its recording's split
        meta["physics"]["augment"] = augment
    meta.update(name=name or f"{mr.get('name', mix_with)} + physics", licence=mr.get("licence", "unknown"),
                mixed_from=mix_with, physics_windows=int(len(Xs)))
    meta["notes"] = [f"{len(Xs)} of {sum(len(a) for a in parts_X)} windows are simulated ({model}); "
                        "check accuracy on real recordings only"]
    return data.save(np.concatenate(parts_X), np.concatenate(parts_y), labels, meta, np.concatenate(parts_g), template)
