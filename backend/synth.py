# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Synthetic demo datasets, one per template, so every pipeline can be demonstrated before a customer has data.

They are physics-flavoured (impulse trains at bearing defect frequencies, arc broadband bursts, PQRST beats...)
but they are demos: accuracy on them says nothing about field accuracy, and the UI labels them as synthetic."""

from __future__ import annotations

import numpy as np


def _t(n, fs):
    return np.arange(n) / fs


def bearing(rng, n_per, T=256, fs=3200):
    labels = ["normal", "inner_race", "outer_race", "ball"]
    X, y = [], []
    defect = {1: 162.0, 2: 107.0, 3: 141.0}
    for cls in range(4):
        for _ in range(n_per):
            t = _t(T, fs)
            f0 = rng.uniform(24, 30)
            base = 0.4 * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
            sig = np.stack([base, 0.6 * base, 0.3 * base]) + rng.normal(0, 0.25, (3, T))
            if cls:
                fd = defect[cls] * rng.uniform(0.95, 1.05)
                period = int(fs / fd)
                f_ring = {1: 1100, 2: 700, 3: 450}[cls] * rng.uniform(.93, 1.07)   # each defect excites its own mode
                ring = np.exp(-np.arange(24) / 5.0) * np.sin(2 * np.pi * f_ring / fs * np.arange(24))
                amp = rng.uniform(0.6, 1.4)
                for start in range(rng.integers(0, period), T - 24, period):
                    a = amp * (1 + 0.8 * np.sin(2 * np.pi * f0 * start / fs)) if cls == 1 else amp
                    a = a * (0.5 + rng.random()) if cls == 3 else a
                    sig[:, start:start + 24] += np.outer([1.0, 0.7, 0.4], a * ring)
            X.append(sig)
            y.append(cls)
    return np.array(X)[:, :, None, :], np.array(y), labels


def motor_current(rng, n_per, T=256, fs=2000):
    labels = ["healthy", "broken_bar", "eccentricity", "winding_short"]
    X, y = [], []
    for cls in range(4):
        for _ in range(n_per):
            t = _t(T, fs)
            f = rng.uniform(48, 52)
            s = rng.uniform(0.02, 0.05)
            ph = rng.uniform(0, 6.28)
            amp = rng.uniform(0.8, 1.2)
            phases = []
            for k in range(3):
                w = 2 * np.pi * f * t + ph - k * 2 * np.pi / 3
                i = amp * np.sin(w)
                if cls == 1:
                    i += 0.12 * np.sin(2 * np.pi * f * (1 - 2 * s) * t + ph) + 0.12 * np.sin(2 * np.pi * f * (1 + 2 * s) * t + ph)
                if cls == 2:
                    i += 0.15 * np.sin(2 * np.pi * (f + 25) * t + ph) + 0.15 * np.sin(2 * np.pi * (f - 25) * t + ph)
                if cls == 3 and k == 0:
                    i *= 1.25
                phases.append(i + rng.normal(0, 0.05, T))
            X.append(np.array(phases))
            y.append(cls)
    return np.array(X)[:, :, None, :], np.array(y), labels


def arc_fault(rng, n_per, T=512, fs=100000):
    labels = ["normal", "arc"]
    X, y = [], []
    for cls in range(2):
        for _ in range(n_per * 2):
            t = _t(T, fs)
            i = rng.uniform(4, 10) + 0.3 * np.sin(2 * np.pi * rng.uniform(15e3, 20e3) * t)   # DC + switching ripple
            i = i + rng.normal(0, 0.08, T)
            if cls:
                n_burst = rng.integers(2, 6)
                for _b in range(n_burst):
                    s = rng.integers(0, T - 60)
                    L = rng.integers(20, 60)
                    i[s:s + L] += rng.normal(0, rng.uniform(0.25, 0.6), L) - rng.uniform(0.2, 0.8)
            X.append((i - i.mean())[None])          # AC-coupled HF current sensor, as arc detectors read it
            y.append(cls)
    return np.array(X)[:, :, None, :], np.array(y), labels


def dcdc(rng, n_per, T=128, fs=20000):
    labels = ["normal", "overload", "short", "open_load", "oscillation"]
    X, y = [], []
    for cls in range(5):
        for _ in range(n_per):
            t = _t(T, fs)
            v = 12 + 0.05 * np.sin(2 * np.pi * 2000 * t + rng.uniform(0, 6))
            il = rng.uniform(2, 4) + 0.3 * np.sign(np.sin(2 * np.pi * 2000 * t))
            k = rng.integers(10, 60)
            if cls == 1:
                v[k:] -= rng.uniform(0.5, 1.5)
                il[k:] *= rng.uniform(1.6, 2.2)
            elif cls == 2:
                v[k:] *= np.exp(-np.arange(T - k) / rng.uniform(3, 8))
                il[k:] += rng.uniform(6, 10)
            elif cls == 3:
                v[k:] += rng.uniform(0.3, 0.8)
                il[k:] *= np.exp(-np.arange(T - k) / 5)
            elif cls == 4:
                v += rng.uniform(0.3, 0.8) * np.sin(2 * np.pi * rng.uniform(300, 700) * t)
            X.append(np.stack([v + rng.normal(0, .03, T), il + rng.normal(0, .05, T)]))
            y.append(cls)
    return np.array(X)[:, :, None, :], np.array(y), labels


def battery(rng, n, T=128):
    X, y = [], []
    for _ in range(n * 4):
        soh = rng.uniform(0.70, 1.0)
        r0 = 0.02 + (1 - soh) * 0.12
        cap = soh * 3.0
        i = np.zeros(T)
        for s in range(8, T, 32):
            i[s:s + 16] = rng.uniform(1.5, 3.0)
        q = np.cumsum(i) / 360.0
        soc = np.clip(rng.uniform(0.6, 1.0) - q / cap, 0.05, 1)
        ocv = 3.0 + 1.2 * soc - 0.1 * np.exp(-20 * soc)
        v = ocv - i * r0 + rng.normal(0, 0.003, T)
        tc = 25 + np.cumsum(i * i * r0) * 0.6 + rng.normal(0, 0.05, T)
        X.append(np.stack([v, i, tc]))
        y.append(soh * 100)
    return np.array(X)[:, :, None, :], np.array(y, dtype=np.float32), ["soh"]


def har(rng, n_per, T=128, fs=50):
    labels = ["walk", "run", "sit", "stand", "stairs", "fall"]
    X, y = [], []
    for cls in range(6):
        for _ in range(n_per):
            t = _t(T, fs)
            g = np.array([0, 0, 1.0]) if cls != 2 else np.array([0.7, 0, 0.7])
            acc = np.tile(g[:, None], (1, T)) + rng.normal(0, 0.03, (3, T))
            gyr = rng.normal(0, 0.03, (3, T))
            if cls in (0, 1, 4):
                f = {0: 1.9, 1: 2.9, 4: 1.5}[cls] * rng.uniform(.9, 1.1)
                a = {0: 0.3, 1: 0.9, 4: 0.4}[cls]
                acc[2] += a * np.sin(2 * np.pi * f * t)
                acc[0] += 0.5 * a * np.sin(2 * np.pi * f / 2 * t)
                gyr[1] += a * np.sin(2 * np.pi * f * t + 1)
                if cls == 4:
                    acc[2] += 0.15
            if cls == 5:
                k = rng.integers(30, 80)
                acc[:, k:k + 6] += rng.normal(0, 2.0, (3, 6))
                acc[:, k + 6:] = np.array([1.0, 0, 0])[:, None] + rng.normal(0, 0.02, (3, T - k - 6))
                gyr[:, k:k + 8] += rng.normal(0, 3.0, (3, 8))
            X.append(np.concatenate([acc, gyr]))
            y.append(cls)
    return np.array(X)[:, :, None, :], np.array(y), labels


def ecg(rng, n_per, T=256, fs=128):
    labels = ["normal", "pvc", "afib", "noise"]

    def beat(L, wide=False, p=True):
        x = np.arange(L)
        b = np.zeros(L)
        c = L // 3
        if p:
            b += 0.15 * np.exp(-((x - c + 12) / 3) ** 2)
        w = 3.5 if wide else 1.5
        b += (1.6 if wide else 1.0) * np.exp(-((x - c) / w) ** 2) - 0.2 * np.exp(-((x - c - 4) / 2) ** 2)
        b += 0.3 * np.exp(-((x - c - 22) / 6) ** 2)
        return b

    X, y = [], []
    for cls in range(4):
        for _ in range(n_per):
            sig = np.zeros(T + 200)
            pos = 0
            hr = rng.uniform(60, 90)
            while pos < T + 100:
                rr = int(fs * 60 / hr * (rng.uniform(0.6, 1.4) if cls == 2 else rng.uniform(0.97, 1.03)))
                wide = cls == 1 and rng.random() < 0.35
                L = min(rr, 100)
                sig[pos:pos + L] += beat(L, wide=wide, p=cls != 2)[:len(sig[pos:pos + L])]
                pos += rr
            sig = sig[:T]
            if cls == 2:
                sig += 0.05 * np.sin(2 * np.pi * rng.uniform(5, 8) * _t(T, fs))
            if cls == 3:
                sig += rng.normal(0, 0.4, T) + 0.5 * np.sin(2 * np.pi * 0.5 * _t(T, fs))
            X.append((sig + rng.normal(0, 0.03, T))[None])
            y.append(cls)
    return np.array(X)[:, :, None, :], np.array(y), labels


def gas(rng, n_per, T=32):
    labels = ["clean_air", "ethanol", "ammonia", "co", "smoke"]
    sig = rng.uniform(0.2, 1.0, (5, 8))
    sig[0] *= 0.1
    X, y = [], []
    for cls in range(5):
        for _ in range(n_per):
            conc = rng.uniform(0.5, 1.5)
            rise = 1 - np.exp(-np.arange(T) / rng.uniform(3, 7))
            resp = np.outer(sig[cls] * conc, rise) + rng.uniform(-0.05, 0.05, (8, 1))   # drift offset
            X.append(resp + rng.normal(0, 0.03, (8, T)))
            y.append(cls)
    return np.array(X)[:, :, None, :], np.array(y), labels


def vib_anomaly(rng, n_per, T=256, fs=3200):
    """Raw accelerometer windows; the band-energy front end turns them into the model input.
    Label 0 = normal (the autoencoder trains on it only), 1 = anomaly (test only)."""
    X, y = [], []
    t = np.arange(T) / fs
    for cls in (0, 1):
        for _ in range(n_per * (3 if cls == 0 else 1)):
            f0 = rng.uniform(24, 26)
            sig = sum(rng.uniform(0.6, 1.0) / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3)) for h in (1, 2, 3, 4))
            sig = sig + rng.normal(0, 0.05, T)
            if cls:
                kind = rng.integers(0, 3)
                if kind == 0:      # new resonance (looseness, bearing)
                    sig += rng.uniform(0.4, 0.8) * np.sin(2 * np.pi * rng.uniform(600, 1400) * t)
                elif kind == 1:    # raised broadband floor (cavitation, rubbing)
                    sig += rng.normal(0, rng.uniform(0.25, 0.5), T)
                else:              # imbalance: 1x grows
                    sig += rng.uniform(1.5, 2.5) * np.sin(2 * np.pi * f0 * t)
            X.append(sig[None])
            y.append(cls)
    return np.array(X)[:, :, None, :], np.array(y), ["normal", "anomaly"]


def kws(rng, n_per, L=16000, fs=16000):
    """Raw 1 s clips: each word is a sequence of formant-like tone glides (class-specific), with speaker pitch,
    timing and loudness variation plus background noise. The MFCC front end produces the model input."""
    labels = ["silence", "unknown", "yes", "no", "up", "down"]
    words = {2: [(700, 1100), (1100, 2300), (2300, 2600)], 3: [(400, 300), (300, 900)],
             4: [(600, 600), (600, 250)], 5: [(900, 500), (500, 400), (400, 1200)]}
    X, y = [], []
    for cls in range(6):
        for _ in range(n_per):
            a = rng.normal(0, rng.uniform(0.002, 0.01), L)
            if cls >= 1:
                segs = words.get(cls) or [(rng.uniform(250, 2500), rng.uniform(250, 2500)) for _ in range(rng.integers(1, 4))]
                pitch = rng.uniform(0.85, 1.15)
                pos = int(rng.uniform(0.05, 0.35) * L)
                seg_len = int(rng.uniform(0.12, 0.2) * L)
                for f_a, f_b in segs:
                    n = min(seg_len, L - pos)
                    if n <= 0:
                        break
                    f = np.linspace(f_a, f_b, n) * pitch
                    phase = 2 * np.pi * np.cumsum(f) / fs
                    env = np.sin(np.linspace(0, np.pi, n)) * rng.uniform(0.2, 0.5)
                    a[pos:pos + n] += env * (np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase))
                    pos += n
            X.append(np.clip(a, -1, 1)[None])
            y.append(cls)
    return np.array(X, np.float32)[:, :, None, :], np.array(y), labels


def thermal(rng, n_per, H=24, W=32):
    labels = ["empty", "one_person", "group", "fall"]
    yy, xx = np.mgrid[0:H, 0:W]
    X, y = [], []
    for cls in range(4):
        for _ in range(n_per):
            img = rng.uniform(20, 24) + rng.normal(0, 0.4, (H, W))
            n = {0: 0, 1: 1, 2: rng.integers(2, 5), 3: 1}[cls]
            for _p in range(n):
                cy, cx = rng.uniform(4, H - 4), rng.uniform(4, W - 4)
                sy, sx = (2.5, 1.5) if cls != 3 else (1.2, 4.0)
                if cls == 3:
                    cy = rng.uniform(H - 7, H - 3)
                img += rng.uniform(6, 10) * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
            X.append(img[None])
            y.append(cls)
    return np.array(X), np.array(y), labels


GENERATORS = {"bearing": bearing, "motor_current": motor_current, "arc_fault": arc_fault, "dcdc": dcdc,
              "battery": battery, "har": har, "ecg": ecg, "gas": gas, "vib_anomaly": vib_anomaly, "kws": kws,
              "thermal": thermal}


def generate(name: str, n_per_class: int = 300, seed: int = 7):
    rng = np.random.default_rng(seed)
    X, y, labels = GENERATORS[name](rng, n_per_class)
    return X.astype(np.float32), y, labels
