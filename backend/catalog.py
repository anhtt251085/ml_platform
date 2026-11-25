"""The model pool: built-in templates (SAI Research) plus researcher-published ones stored in the database.

A template is what a customer picks ("Bearing fault detection"); it owns the input contract, the default
labels, the data format the customer must provide, and size variants (tiny / small / base) whose layer lists are
SAI-compatible architecture specs. Input channels, window and outputs are filled from the customer's data.
"""

from __future__ import annotations

C = lambda out, k=3, stride=1, relu=True: {"op": "conv", "out": out, "k": k, "stride": stride, "relu": relu}  # noqa: E731
DW = lambda k=3, stride=1: {"op": "dwconv", "k": k, "stride": stride, "relu": True}  # noqa: E731
PW = lambda out: {"op": "pwconv", "out": out, "relu": True}  # noqa: E731
MP = lambda k=2: {"op": "maxpool", "k": k}  # noqa: E731
AP = lambda k=2: {"op": "avgpool", "k": k}  # noqa: E731
GAP = {"op": "gap"}
FC = lambda out, relu=True: {"op": "fc", "out": out, "relu": relu}  # noqa: E731
RES = lambda k=3: {"op": "residual", "k": k}  # noqa: E731
DROP = {"op": "dropout", "p": 0.1}

# Reusable 1-D families (depthwise-separable temporal CNNs, the shape SAI streams best)
TCN_TINY = [C(8, 5, 2), DW(3), PW(12), MP(2), DW(3), PW(16), GAP]
TCN_SMALL = [C(12, 5, 2), DW(5), PW(16), MP(2), DW(3), PW(24), MP(2), DW(3), PW(32), GAP, DROP]
TCN_BASE = [C(16, 7, 2), DW(5), PW(24), RES(3), MP(2), DW(3), PW(32), MP(2), DW(3), PW(48), GAP, FC(32), DROP]

BUILTIN = [
    {
        "id": "bearing-fault", "name": "Bearing fault detection", "domain": "Predictive maintenance", "icon": "⚙️",
        "task": "classification",
        "summary": "Classify rolling-bearing condition from a 3-axis accelerometer window.",
        "use_cases": ["Motors, pumps, fans, gearboxes", "Runs on the drive MCU, no gateway"],
        "input": {"kind": "timeseries", "channels": 3, "height": 1, "width": 256, "sample_rate_hz": 3200,
                  "channel_names": ["acc_x", "acc_y", "acc_z"]},
        "labels": ["normal", "inner_race", "outer_race", "ball"],
        "window": 256, "hop": 128, "synth": "bearing",
        "variants": {"tiny": TCN_TINY, "small": TCN_SMALL, "base": TCN_BASE},
        "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "Public: CWRU, Paderborn (check licences for commercial use). Own: 3-axis IMU at 1-10 kHz.",
    },
    {
        "id": "motor-current", "name": "Motor current signature analysis", "domain": "Predictive maintenance",
        "icon": "🔌", "task": "classification",
        "summary": "Broken rotor bar, eccentricity and winding faults from phase currents - no extra sensor.",
        "use_cases": ["Inverter / motor-control MCU", "Pumps and compressors"],
        "input": {"kind": "timeseries", "channels": 3, "height": 1, "width": 256, "sample_rate_hz": 2000,
                  "channel_names": ["i_a", "i_b", "i_c"]},
        "labels": ["healthy", "broken_bar", "eccentricity", "winding_short"],
        "window": 256, "hop": 128, "synth": "motor_current",
        "variants": {"tiny": TCN_TINY, "small": TCN_SMALL, "base": TCN_BASE}, "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "Own data from the drive's current sensors; label by maintenance records.",
    },
    {
        "id": "dc-arc-fault", "name": "DC arc-fault detection", "domain": "Power electronics", "icon": "⚡",
        "task": "classification",
        "summary": "Detect series arcs in PV strings, EV and DC buses from the current waveform.",
        "use_cases": ["PV inverters (UL 1699B / IEC 63027)", "DC breakers, EV chargers"],
        "input": {"kind": "timeseries", "channels": 1, "height": 1, "width": 512, "sample_rate_hz": 100000,
                  "channel_names": ["i_dc"]},
        "labels": ["normal", "arc"],
        "window": 512, "hop": 256, "synth": "arc_fault",
        "variants": {"tiny": [C(8, 7, 4), DW(5), PW(12), MP(2), DW(3), PW(16), GAP],
                     "small": [C(12, 7, 2), DW(5, 2), PW(16), MP(2), DW(3), PW(24), MP(2), DW(3), PW(32), GAP],
                     "base": [C(16, 9, 2), DW(5, 2), PW(24), RES(3), MP(2), DW(3), PW(32), MP(2), PW(48), GAP, FC(24)]},
        "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "Arc generator test bench per UL 1699B; include inverter switching noise as negatives.",
    },
    {
        "id": "dcdc-fault", "name": "DC/DC converter fault classifier", "domain": "Power electronics", "icon": "🔋",
        "task": "classification",
        "summary": "Classify converter state from output voltage and inductor current (the MD6605 reference task).",
        "use_cases": ["Server / telecom PSUs", "Automotive DC/DC"],
        "input": {"kind": "timeseries", "channels": 2, "height": 1, "width": 128, "sample_rate_hz": 20000,
                  "channel_names": ["v_out", "i_l"]},
        "labels": ["normal", "overload", "short", "open_load", "oscillation"],
        "window": 128, "hop": 64, "synth": "dcdc",
        "variants": {"tiny": [C(8, 3), DW(3), PW(8), DW(3), PW(16), GAP],
                     "small": TCN_TINY, "base": TCN_SMALL},
        "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "Fault-injection on the power stage; the controller's own ADC samples.",
    },
    {
        "id": "battery-soh", "name": "Battery state-of-health", "domain": "Energy / BMS", "icon": "🔋",
        "task": "regression",
        "summary": "Estimate cell SoH (%) from a voltage / current / temperature window of a load pulse.",
        "use_cases": ["BMS in e-bikes, ESS, power tools", "Warranty analytics"],
        "input": {"kind": "timeseries", "channels": 3, "height": 1, "width": 128, "sample_rate_hz": 10,
                  "channel_names": ["v_cell", "i_cell", "t_cell"]},
        "labels": ["soh"], "unit": "%",
        "window": 128, "hop": 32, "synth": "battery",
        "variants": {"tiny": TCN_TINY, "small": TCN_SMALL + [FC(16)], "base": TCN_BASE},
        "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "Cycling data with periodic capacity checks (label = measured capacity / nominal).",
    },
    {
        "id": "har-imu", "name": "Activity & fall detection", "domain": "Wearables", "icon": "🏃",
        "task": "classification",
        "summary": "Human activity recognition and fall detection from a 6-axis IMU.",
        "use_cases": ["Wearables, elderly care pendants", "Worker safety"],
        "input": {"kind": "timeseries", "channels": 6, "height": 1, "width": 128, "sample_rate_hz": 50,
                  "channel_names": ["ax", "ay", "az", "gx", "gy", "gz"]},
        "labels": ["walk", "run", "sit", "stand", "stairs", "fall"],
        "window": 128, "hop": 64, "synth": "har",
        "variants": {"tiny": TCN_TINY, "small": TCN_SMALL, "base": TCN_BASE}, "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "Public: UCI HAR (CC BY 4.0), WISDM, SisFall. Check licence terms per dataset.",
    },
    {
        "id": "ecg-rhythm", "name": "ECG rhythm classification", "domain": "Health", "icon": "❤️",
        "task": "classification",
        "summary": "Normal beat, PVC, atrial fibrillation and noise from a single-lead ECG window.",
        "use_cases": ["Patch monitors", "Smart watches (pre-screening, not diagnosis)"],
        "input": {"kind": "timeseries", "channels": 1, "height": 1, "width": 256, "sample_rate_hz": 128,
                  "channel_names": ["ecg"]},
        "labels": ["normal", "pvc", "afib", "noise"],
        "window": 256, "hop": 128, "synth": "ecg",
        "variants": {"tiny": TCN_TINY, "small": TCN_SMALL, "base": TCN_BASE}, "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "PhysioNet MIT-BIH / CinC 2017 - several are research-only; commercial models need cleared data.",
    },
    {
        "id": "gas-sense", "name": "Gas / odour classification", "domain": "Environment", "icon": "🌫️",
        "task": "classification",
        "summary": "Classify odours from a metal-oxide sensor array heater cycle.",
        "use_cases": ["Air quality, food spoilage", "Leak detection"],
        "input": {"kind": "timeseries", "channels": 8, "height": 1, "width": 32, "sample_rate_hz": 1,
                  "channel_names": [f"s{i}" for i in range(8)]},
        "labels": ["clean_air", "ethanol", "ammonia", "co", "smoke"],
        "window": 32, "hop": 8, "synth": "gas",
        "variants": {"tiny": [C(12, 3), PW(16), GAP],
                     "small": [C(16, 3), DW(3), PW(24), GAP, FC(16)],
                     "base": [C(24, 3), DW(3), PW(32), RES(3), GAP, FC(24)]},
        "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "Record each sensor array in your enclosure; drift needs data across weeks.",
    },
    {
        "id": "vibration-anomaly", "name": "Vibration anomaly (learns normal)", "domain": "Predictive maintenance",
        "icon": "📈", "task": "anomaly",
        "summary": "Autoencoder on band energies: train on healthy machines only, flag anything unusual.",
        "use_cases": ["Any rotating machine with no fault history", "Commissioning baseline"],
        "input": {"kind": "features", "channels": 1, "height": 1, "width": 64, "sample_rate_hz": 3200,
                  "channel_names": ["accel"]},
        "frontend": {"type": "band_energy", "raw_length": 256, "n_fft": 256, "bands": 64},
        "labels": ["normal", "anomaly"],
        "window": 64, "hop": 64, "synth": "vib_anomaly",
        "variants": {"tiny": [FC(24), FC(8), FC(24)], "small": [FC(32), FC(12), FC(32)],
                     "base": [FC(64), FC(16), FC(64)]},
        "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "Raw accelerometer windows of 256 samples (CSV or NPZ); the C front end computes 64 log band energies, on the MCU too.",
    },
    {
        "id": "kws", "name": "Keyword spotting", "domain": "Audio", "icon": "🎙️", "task": "classification",
        "summary": "Recognise a handful of command words from 49x10 MFCC frames (1 s of audio).",
        "use_cases": ["Appliances, toys, voice wake", "Hands-free industrial HMI"],
        "input": {"kind": "spectrogram", "channels": 1, "height": 49, "width": 10, "sample_rate_hz": 16000,
                  "channel_names": ["audio"]},
        "frontend": {"type": "mfcc", "raw_length": 16000, "sample_rate": 16000, "frame": 480, "stride": 320,
                     "n_fft": 512, "n_mels": 40, "n_coef": 10, "f_lo": 20.0, "f_hi": 4000.0},
        "labels": ["silence", "unknown", "yes", "no", "up", "down"],
        "window": None, "hop": None, "synth": "kws",
        "variants": {"tiny": [C(16, 3, 2), DW(3), PW(16), GAP],
                     "small": [C(32, 3, 2), DW(3), PW(32), DW(3), PW(32), GAP],
                     "base": [C(48, 3, 2), DW(3), PW(48), DW(3), PW(48), DW(3), PW(48), GAP]},
        "targets": ["cortex-m7"],
        "datasets_hint": "ZIP of <word>/<clip>.wav (16 kHz, 16-bit mono, 1 s), e.g. Google Speech Commands (CC BY 4.0). The C MFCC front end runs in training and on the MCU.",
    },
    {
        "id": "thermal-presence", "name": "Thermal presence & fall", "domain": "Smart building", "icon": "🌡️",
        "task": "classification",
        "summary": "Privacy-preserving people counting and fall detection on a 24x32 thermal array.",
        "use_cases": ["Elderly care rooms", "Meeting-room occupancy, HVAC control"],
        "input": {"kind": "image", "channels": 1, "height": 24, "width": 32, "sample_rate_hz": 4,
                  "channel_names": ["temp"]},
        "labels": ["empty", "one_person", "group", "fall"],
        "window": None, "hop": None, "synth": "thermal",
        "variants": {"tiny": [C(8, 3, 2), DW(3), PW(12), MP(2), DW(3), PW(16), GAP],
                     "small": [C(12, 3, 2), DW(3), PW(16), MP(2), DW(3), PW(24), DW(3), PW(32), GAP],
                     "base": [C(16, 3, 2), DW(3), PW(24), RES(3), MP(2), DW(3), PW(32), DW(3), PW(48), GAP, FC(24)]},
        "targets": ["md6605", "cortex-m7"],
        "datasets_hint": "MLX90640 / Grid-EYE frames in °C as NPZ X[N,1,24,32].",
    },
]


def builtin() -> dict[str, dict]:
    out = {}
    for t in BUILTIN:
        t = dict(t)
        t.setdefault("status", "published")
        t.setdefault("author", "SAI Research")
        t.setdefault("source", "builtin")
        out[t["id"]] = t
    return out
