# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Data features: physics-based augmentation, recordings + labelling tool, live capture, cloud / DB connectors.
None of these need the SAI compiler."""

from __future__ import annotations

import datetime as dt
import os
import tempfile

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import connectors, data, db, physics, recordings  # noqa: E402

db.init()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from backend.app import app
    with TestClient(app) as c:
        r = c.post("/api/auth/login", json={"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                            "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
        c.headers["Authorization"] = f"Bearer {r.json()['token']}"
        yield c


# ---- physics ------------------------------------------------------------------------------------------------------
def test_bearing_kinematics_match_the_textbook_values():
    # SKF 6205 (CWRU drive end): 9 balls, d = 7.94 mm, D = 39.04 mm, 1797 rpm -> BPFO 107.4 Hz, BPFI 162.2 Hz
    f = physics.bearing_frequencies(1797, 9, 7.94, 39.04, 0)
    assert abs(f["bpfo"] - 107.36) < 0.1 and abs(f["bpfi"] - 162.19) < 0.1
    assert abs(f["ftf"] - 11.93) < 0.05 and abs(f["bsf"] - 70.58) < 0.1
    m = physics.motor_frequencies(50, 0.03, 2)
    assert m["broken_bar"] == pytest.approx([47.0, 53.0]) and m["eccentricity"][1] == pytest.approx(74.25)


def _envelope_peak(x, fs, lo, hi):
    """Classic envelope analysis: high-pass around the resonance, rectify, spectrum of the envelope."""
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / fs) < 1000] = 0
    e = np.abs(np.fft.irfft(X, len(x)))
    e = e - e.mean()
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fr = np.fft.rfftfreq(len(e), 1 / fs)
    band = (fr > lo) & (fr < hi)
    return fr[band][np.argmax(spec[band])]


def test_simulated_outer_race_impacts_repeat_at_bpfo():
    tpl = {"id": "t", "synth": "bearing", "task": "classification", "labels": ["normal", "outer_race"],
           "input": {"kind": "timeseries", "channels": 1, "width": 16384, "sample_rate_hz": 12000}}
    X, y, labels = physics.simulate(tpl, {"rpm": 1797, "snr_db": 30, "damping": 0.08, "resonance_hz": 3000}, 3)
    assert X.shape == (6, 1, 1, 16384) and labels == ["normal", "outer_race"]
    bpfo = physics.bearing_frequencies(1797, 9, 7.94, 39.04, 0)["bpfo"]
    peaks = [_envelope_peak(X[i, 0, 0], 12000, 60, 400) for i in range(len(X)) if y[i] == 1]
    assert all(abs(p - bpfo) / bpfo < 0.05 for p in peaks), (peaks, bpfo)


def test_physics_parameters_are_validated():
    with pytest.raises(data.DataError, match="Shaft speed"):
        physics.resolve("bearing", {"rpm": 10})
    with pytest.raises(data.DataError, match="twice the ball"):
        physics.resolve("bearing", {"ball_d_mm": 30, "pitch_d_mm": 40})
    with pytest.raises(data.DataError, match="unknown parameter"):
        physics.resolve("bearing", {"rpmm": 1000})


def test_physics_api_preview_build_and_mix(client):
    assert "bearing" in client.get("/api/physics/models").json()
    pv = client.post("/api/datasets/physics", json={"template_id": "bearing-fault", "params": {"rpm": 3000},
                                                    "preview": True}).json()
    assert pv["model"] == "bearing" and len(pv["examples"]) == 4 and pv["frequencies"]["bpfo"] > 100
    ds = client.post("/api/datasets/physics", json={"template_id": "bearing-fault", "per_class": 20,
                                                    "params": {"rpm": 3000}}).json()
    assert ds["n"] == 80 and ds["licence"] == "synthetic" and ds["physics"]["params"]["rpm"] == 3000
    real = client.post("/api/datasets/demo", json={"template_id": "bearing-fault", "per_class": 30}).json()
    mix = client.post("/api/datasets/physics", json={"template_id": "bearing-fault", "per_class": 10,
                                                     "mix_with_dataset_id": real["id"],
                                                     "augment": {"speed_pct": 2, "snr_db": 25}}).json()
    assert mix["n"] == 120 + 40 + 120 and mix["mixed_from"] == real["id"] and mix["licence"] == real["licence"]
    # a model type without a physics model
    r = client.post("/api/datasets/physics", json={"template_id": "gas-sense"})
    assert r.status_code == 400 and "physics" in r.text


# ---- recordings / labelling ---------------------------------------------------------------------------------------
def _csv(n=4000, fs=1000.0, with_labels=True):
    t = np.arange(n) / fs
    lines = ["time,acc_x,acc_y,acc_z" + (",label" if with_labels else "")]
    for i in range(n):
        lab = ("normal" if i < n // 2 else "outer_race") if with_labels else None
        v = np.sin(2 * np.pi * 30 * t[i])
        lines.append(f"{t[i]:.6f},{v:.5f},{0.5 * v:.5f},{0.2 * v:.5f}" + (f",{lab}" if lab else ""))
    return ("\n".join(lines) + "\n").encode()


def test_recording_upload_view_and_label_editing(client):
    r = client.post("/api/recordings/upload", files={"file": ("rig.csv", _csv(with_labels=False))}).json()
    assert r["channels"] == ["acc_x", "acc_y", "acc_z"] and abs(r["sample_rate"] - 1000) < 1
    v = client.get(f"/api/recordings/{r['id']}/view", params={"points": 100}).json()
    assert v["samples"] == 4000 and len(v["x"]) == 100 and len(v["min"]) == 3
    assert all(lo <= hi for lo, hi in zip(v["min"][0], v["max"][0]))
    rid = r["id"]
    s = client.put(f"/api/recordings/{rid}/spans", json={"spans": [{"start": 0, "end": 2000, "label": "normal"},
                                                                   {"start": 2000, "end": 4000, "label": "ball"}]}).json()
    assert [x["label"] for x in s["spans"]] == ["normal", "ball"]
    # paint over the middle, erase a piece, touching same-label spans merge
    s = client.put(f"/api/recordings/{rid}/spans", json={"spans": [{"start": 1500, "end": 2500, "label": "ball"},
                                                                   {"start": 3800, "end": 4000, "label": ""}]}).json()
    assert [(x["start"], x["end"], x["label"]) for x in s["spans"]] == [(0, 1500, "normal"), (1500, 3800, "ball")]
    bad = client.put(f"/api/recordings/{rid}/spans", json={"spans": [{"start": 0, "end": 5, "label": "a,b"}]})
    assert bad.status_code == 400
    ds = client.post("/api/recordings/dataset", json={"recording_ids": [rid], "template_id": "bearing-fault",
                                                      "hop": 64, "licence": "proprietary"}).json()
    assert ds["shape"] == [3, 1, 256] and set(ds["class_counts"]) == {"normal", "ball"}
    assert ds["licence"] == "proprietary" and ds["unlabelled_windows_skipped"] >= 1
    assert client.delete(f"/api/recordings/{rid}").status_code == 200


def test_recording_csv_label_column_becomes_spans():
    rec = recordings.from_csv(_csv(), "org_x", None, "labelled")
    assert [(s["start"], s["end"], s["label"]) for s in rec["spans"]] == [(0, 2000, "normal"), (2000, 4000, "outer_race")]
    recordings.delete(rec["id"])


def test_unknown_labels_are_refused_for_a_model_type():
    rec = recordings.from_csv(_csv(), "org_x", None, "labelled")
    recordings.set_spans(rec["id"], [{"start": 0, "end": 1000, "label": "squeak"}], replace=False)
    from backend import store
    with pytest.raises(data.DataError, match="squeak"):
        recordings.to_dataset([recordings.get(rec["id"])], store.template("bearing-fault"), "x")
    recordings.delete(rec["id"])


def test_live_capture_with_token_and_live_labels(client):
    cap = client.post("/api/recordings/capture", json={"name": "bench", "channels": ["acc_x", "acc_y", "acc_z"],
                                                       "sample_rate": 3200}).json()
    rid, tok = cap["id"], cap["token"]
    assert cap["status"] == "capturing" and tok.startswith("cap_")
    url = f"/api/capture/{rid}/push"
    assert client.post(url, json={"samples": [[0, 0, 0]]}, headers={"X-Capture-Token": "nope"}).status_code == 401
    client.post(f"/api/recordings/{rid}/mark", json={"label": "normal"})
    rows = np.random.default_rng(0).normal(size=(600, 3)).round(4).tolist()
    r = client.post(url, json={"samples": rows}, headers={"X-Capture-Token": tok}).json()
    assert r == {"received": 600, "samples": 600}
    client.post(f"/api/recordings/{rid}/mark", json={"label": "ball"})
    text = "boot ok\n" + "\n".join("1,2,3" for _ in range(400)) + "\n"
    r = client.post(url, content=text, headers={"X-Capture-Token": tok, "Content-Type": "text/plain"}).json()
    assert r["samples"] == 1000
    tail = client.get(f"/api/recordings/{rid}/view", params={"tail": 500}).json()
    assert tail["end"] == 1000 and tail["live"]["label"] == "ball"
    assert client.delete(f"/api/recordings/{rid}").status_code == 409            # still capturing
    done = client.post(f"/api/recordings/{rid}/stop").json()
    assert done["status"] == "ready" and done["samples"] == 1000
    assert [(s["start"], s["end"], s["label"]) for s in done["spans"]] == [(0, 600, "normal"), (600, 1000, "ball")]
    assert client.post(url, json={"samples": [[0, 0, 0]]}, headers={"X-Capture-Token": tok}).status_code in (401, 409)


def test_label_suggestions_from_a_project_model(client, monkeypatch):
    from backend import db, inference

    class Fake:
        task, labels, raw_window, raw_channels = "classification", ["normal", "outer_race"], 256, 3

        def __init__(self, jid):
            pass

        def predict_proba(self, X):                    # loud windows = outer race
            loud = np.abs(X).mean(axis=(1, 2, 3)) > 0.5
            return np.stack([np.where(loud, 0.1, 0.9), np.where(loud, 0.9, 0.1)], 1)

    monkeypatch.setattr(inference, "ProjectModel", Fake)
    org = client.get("/api/me").json()["org_id"]
    with db.conn() as c:
        c.execute("INSERT INTO jobs (id, kind, status, stage, progress, stages, params, created, updated, org_id,"
                  " priority, attempts, cancel_requested) VALUES ('job_sugg1','train','succeeded','package',1,'[]','{}',"
                  "?,?,?,0,0,0)", (db.now(), db.now(), org))
    sig = np.zeros((4096, 3), np.float32)
    sig[2048:] = 1.0
    rec = recordings.create(org, None, "s", ["a", "b", "c"], 3200)
    recordings.store_array(rec, sig)
    out = client.post(f"/api/recordings/{rec['id']}/suggest", json={"job_id": "job_sugg1", "hop": 128}).json()
    labs = [(s["label"], s["start"], s["end"]) for s in out["spans"]]
    assert labs[0][0] == "normal" and labs[-1][0] == "outer_race" and labs[-1][2] == 4096
    assert 1900 < labs[-1][1] < 2200 and out["windows"] == (4096 - 256) // 128 + 1
    recordings.delete(rec["id"])


# ---- connectors ---------------------------------------------------------------------------------------------------
def test_sigv4_matches_the_aws_documented_example():
    # AWS docs, "Signature Calculations for the Authorization Header", example GET Bucket (List Objects)
    h = connectors.sigv4("GET", "https://examplebucket.s3.amazonaws.com/?max-keys=2&prefix=J", "us-east-1",
                         "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                         now=dt.datetime(2013, 5, 24, tzinfo=dt.timezone.utc))
    assert h["Authorization"].endswith("Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7")


S3_LIST = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>b</Name>
<Contents><Key>rigs/run1.csv</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified><Size>1234</Size></Contents>
<Contents><Key>rigs/run2.csv</Key><LastModified>2026-01-02T00:00:00.000Z</LastModified><Size>99</Size></Contents>
</ListBucketResult>"""


def test_connections_keep_secrets_and_import(client, monkeypatch):
    calls = []

    def fake_https(method, url, headers=None, body=None, limit=0):
        calls.append((method, url, headers))
        if "list-type=2" in url:
            return 200, {}, S3_LIST
        return 200, {}, _csv()

    monkeypatch.setattr(connectors, "https", fake_https)
    c = client.post("/api/connectors", json={"kind": "s3", "name": "rig bucket", "config": {
        "bucket": "acme-rigs", "region": "eu-west-1", "access_key": "AKIDEXAMPLE"},
        "secrets": {"secret_key": "very-secret"}}).json()
    assert c["secrets_set"] == ["secret_key"] and "very-secret" not in str(c)
    listed = client.get("/api/connectors").json()
    assert "very-secret" not in str(listed) and any(x["id"] == c["id"] for x in listed["connections"])
    from backend import db
    with db.conn() as con:
        stored = con.execute("SELECT secret FROM connections WHERE id=?", (c["id"],)).fetchone()[0]
    assert "very-secret" not in stored                                   # encrypted at rest
    items = client.get(f"/api/connectors/{c['id']}/browse").json()["items"]
    assert [x["key"] for x in items] == ["rigs/run1.csv", "rigs/run2.csv"]
    assert calls[-1][1].startswith("https://acme-rigs.s3.eu-west-1.amazonaws.com/?list-type=2")
    assert "Credential=AKIDEXAMPLE/" in calls[-1][2]["Authorization"]
    assert client.post(f"/api/connectors/{c['id']}/test").json()["ok"] is True
    ds = client.post(f"/api/connectors/{c['id']}/import", json={"key": "rigs/run1.csv", "template_id": "bearing-fault",
                                                                "licence": "proprietary"}).json()["dataset"]
    assert ds["source"] == "connector:s3" and ds["n"] > 10
    rec = client.post(f"/api/connectors/{c['id']}/import", json={"key": "rigs/run2.csv", "as": "recording"}).json()
    assert rec["recording"]["spans"][0]["label"] == "normal"
    # updating without a secret keeps the old one
    u = client.put(f"/api/connectors/{c['id']}", json={"config": {**c["config"], "prefix": "rigs/"}}).json()
    assert u["secrets_set"] == ["secret_key"] and u["config"]["prefix"] == "rigs/"
    assert client.delete(f"/api/connectors/{c['id']}").status_code == 200


def test_private_endpoints_are_refused_in_multi_tenant_mode(client, monkeypatch):
    monkeypatch.delenv("SAI_PLATFORM_ALLOW_PRIVATE_CONNECTORS", raising=False)
    c = client.post("/api/connectors", json={"kind": "s3", "name": "minio", "config": {
        "endpoint": "127.0.0.1", "bucket": "b", "path_style": True, "access_key": "k"},
        "secrets": {"secret_key": "s"}}).json()
    r = client.get(f"/api/connectors/{c['id']}/browse")
    assert r.status_code == 400 and "private" in r.text
    assert client.post(f"/api/connectors/{c['id']}/test").json()["ok"] is False


def test_database_queries_are_read_only_selects(client):
    c = client.post("/api/connectors", json={"kind": "postgres", "name": "plant db", "config": {
        "host": "db.example.com", "database": "plant", "user": "reader"}, "secrets": {"password": "pw"}}).json()
    for q in ("DELETE FROM t", "SELECT 1; DROP TABLE t", "update t set a=1"):
        r = client.post(f"/api/connectors/{c['id']}/import", json={"query": q, "template_id": "bearing-fault"})
        assert r.status_code == 400 and ("SELECT" in r.text or "one statement" in r.text)


def test_influx_annotated_csv_and_iso_times():
    raw = (b"#datatype,string,long,dateTime:RFC3339,double,double\n#group,false,false,false,false,false\n"
           b"#default,_result,,,,\n,result,table,_time,acc_x,acc_y\n"
           b",,0,2026-01-01T00:00:00Z,1.0,2.0\n,,0,2026-01-01T00:00:00.001Z,1.5,2.5\n"
           b",,0,2026-01-01T00:00:00.002Z,1.7,2.7\n")
    out = connectors.time_to_seconds(connectors.influx_csv(raw)).decode().splitlines()
    assert out[0] == "time,acc_x,acc_y" and out[2].startswith("0.001")
