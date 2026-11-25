"""Regression tests for the findings of the November 2026 code review (backend, firmware protocol, security)."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import zipfile

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))
os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import boards, catalog, data, db, hil, pipeline, trainer  # noqa: E402

db.init()


def _login(c, email=None, pw=None):
    r = c.post("/api/auth/login", json={"email": email or os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                        "password": pw or os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.mark.parametrize("url", ["http://example.com/a.csv", "https://127.0.0.1/a.csv", "https://169.254.169.254/latest",
                                 "https://10.0.0.5/x.csv", "https://localhost/x.csv", "https://example.com:8443/x.csv",
                                 "ftp://example.com/x.csv"])
def test_url_ingest_refuses_internal_targets(url):
    with pytest.raises(data.DataError):
        data.safe_fetch(url, 1000)


def test_npz_decompression_bomb_refused_before_loading():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        hdr = io.BytesIO()
        np.lib.format.write_array_header_1_0(hdr, {"descr": "<f4", "fortran_order": False, "shape": (600_000_000,)})
        z.writestr("X.npy", hdr.getvalue() + b"\0" * 64)
        z.writestr("y.npy", b"")
    with pytest.raises(data.DataError, match="too large"):
        data.from_npz_bytes(buf.getvalue(), catalog.builtin()["gas-sense"], "bomb.npz")


def test_non_finite_values_rejected():
    X = np.random.default_rng(0).normal(size=(40, 8, 32)).astype(np.float32)
    X[3, 2, 5] = np.inf
    buf = io.BytesIO()
    np.savez(buf, X=X, y=np.arange(40) % 5)
    with pytest.raises(data.DataError, match="NaN or infinite"):
        data.from_npz_bytes(buf.getvalue(), catalog.builtin()["gas-sense"], "inf.npz")


def test_split_never_puts_one_recording_in_two_splits():
    y = np.array([0, 1] * 60)                                  # every recording holds both classes
    g = np.repeat([f"r{i}" for i in range(12)], 10)
    idx = trainer.split(y, "classification", groups=g)
    s = {k: set(g[idx[k]].tolist()) for k in ("train", "val", "test")}
    assert not (s["train"] & s["test"]) and not (s["train"] & s["val"]) and not (s["val"] & s["test"])
    assert idx["split_mode"] == "recording"


def test_anomaly_metric_without_positives_is_none_and_json_safe():
    m = trainer.metrics("anomaly", np.zeros(20, int), np.zeros((20, 4)), ["normal", "anomaly"], None,
                        np.ones((20, 4)), 0.5)
    assert m["value"] is None and m["auc"] is None
    assert pipeline._finite({"a": float("nan"), "b": [1.0, float("inf")], "c": np.float32(2)}) == {"a": None, "b": [1.0, None], "c": 2.0}


def test_generated_header_escapes_labels_and_compiles(tmp_path):
    pre = {"mean": [25.0, 0.0], "std": [0.01, 1.0], "input": {"scale": 0.02, "zero_point": 3},
           "output": {"scale": 0.1, "zero_point": 0}, "task": "classification",
           "labels": ['say "hi"', "back\\slash", "new\nline", "café"]}
    h = pipeline._app_header(pre, {"labels": pre["labels"]}, True)
    assert "Q16" not in h.split("sai_app_quantize_f")[0] or "No Q16 integer path" in h   # 1/(0.01*0.02) overflows Q16
    (tmp_path / "sai_app.h").write_text(h)
    (tmp_path / "t.c").write_text('#include <string.h>\n#include "sai_app.h"\n'
                                  'int main(void){ return strcmp(sai_app_labels[0], "say \\"hi\\"") || '
                                  'strcmp(sai_app_labels[2], "new\\nline"); }\n')
    subprocess.run(["gcc", "-Wall", "-Werror", "-I", str(tmp_path), str(tmp_path / "t.c"), "-o", str(tmp_path / "t")], check=True)
    assert subprocess.run([str(tmp_path / "t")]).returncode == 0


def test_agent_cannot_take_over_boards_or_forge_results():
    boards.register("lab-a", [{"id": "BRD-A", "target": "cortex-m7"}])
    out = boards.register("lab-b", [{"id": "BRD-A", "target": "cortex-m7"}])
    assert out[0].get("error") and boards.board_agent("BRD-A") == "lab-a"
    run = boards.request_run("job_y", "cortex-m7", "BRD-A", {"expected_outputs": [[1, 2]]})
    with pytest.raises(PermissionError):                       # not claimed yet / not this agent's board
        boards.complete(run["id"], {"ok": True}, agent_id="lab-a")
    claimed = boards.claim("lab-a")
    assert claimed["id"] == run["id"]
    with pytest.raises(PermissionError):
        boards.complete(run["id"], {"ok": True}, agent_id="lab-b")
    r = boards.complete(run["id"], {"ok": True, "outputs_match": True, "cycles": 1}, agent_id="lab-a")   # no outputs
    assert r["status"] == "failed" and r["result"]["outputs_match"] is False
    with pytest.raises(PermissionError):                       # a finished run cannot be posted again
        boards.complete(run["id"], {"ok": True, "outputs": [[1, 2]]}, agent_id="lab-a")
    boards.register("lab-a", [{"id": "BRD-A", "target": "cortex-m7", "name": "renamed"}])    # re-register keeps state
    b = next(x for x in boards.list_boards() if x["id"] == "BRD-A")
    assert b["fail_count"] == 1 and b["name"] == "renamed"


def test_runs_past_their_lease_fail_and_free_the_board():
    boards.register("lab-l", [{"id": "BRD-L", "target": "md6605"}])
    run = boards.request_run("job_z", "md6605", "BRD-L", {})
    assert boards.claim("lab-l")["id"] == run["id"]
    with db.conn() as c:
        c.execute("UPDATE hil_runs SET updated=? WHERE id=?", (db.now() - boards.RUN_LEASE_S - 5, run["id"]))
    boards.reap_runs()
    assert boards.get_run(run["id"])["status"] == "failed"
    assert next(x for x in boards.list_boards() if x["id"] == "BRD-L")["status"] == "idle"


def test_plan_allowed_kinds_enforced_at_claim():
    boards.register("lab-p", [{"id": "BRD-P", "target": "md6605"}])                       # physical
    run = boards.request_run("job_k", "md6605", None, {"allowed_kinds": ["emulated", "simulated"]})
    assert boards.claim("lab-p") is None                         # a Standard-plan run never lands on a physical board
    with db.conn() as c:
        c.execute("UPDATE hil_runs SET status='cancelled' WHERE id=?", (run["id"],))


def test_queue_pause_is_shared_through_the_database():
    pipeline.pause_workers(True)
    pipeline._PAUSED.clear()                                     # another process: only the database knows
    assert pipeline.paused()
    pipeline.pause_workers(False)
    assert not pipeline.paused()


def test_hil_parse_is_strict():
    ok = "boot noise\nSAI-HIL 1 2 4 2\nCFG clock=emulated\nOUT 1 0102\nOUT 0 ff7f\nCYC 10 9 12\nSTACK 300\nDONE ok\n"
    r = hil.parse(ok)
    assert r["outputs"] == [[-1, 127], [1, 2]] and r["config"]["clock"] == "emulated"
    with pytest.raises(hil.HilError, match="1 of 2"):
        hil.parse(ok.replace("OUT 1 0102\n", ""))
    with pytest.raises(hil.HilError):
        hil.parse(ok.replace("OUT 0 ff7f", "OUT 0 ff"))
    with pytest.raises(hil.HilError):
        hil.write_vectors(__import__("pathlib").Path(tempfile.mkdtemp()) / "v.h", np.zeros((0, 4), np.int8))


def test_api_hardening():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        h = _login(c)
        assert "default-src 'self'" in c.get("/api/health").headers["content-security-policy"]
        # draft templates stay with research
        t = c.post("/api/catalog", headers=h, json={"name": "Secret draft", "task": "classification", "labels": ["a", "b"],
                                                    "input": {"channels": 1, "width": 64},
                                                    "variants": {"tiny": [{"op": "conv", "out": 4}, {"op": "gap"}]}}).json()
        bad = c.post("/api/catalog", headers=h, json={"name": "x", "labels": ["a", "b"], "input": {"channels": 1, "width": 64},
                                                      "variants": {"<img src=x>": [{"op": "gap"}]}})
        assert bad.status_code == 400
        org = c.post("/api/admin/orgs", json={"name": "Cyberdyne", "tier": "free"}, headers=h).json()
        c.post("/api/admin/users", headers=h, json={"email": "m@cyber.test", "name": "M", "role": "member",
                                                     "password": "password-1", "org_id": org["id"]})
        hm = _login(c, "m@cyber.test", "password-1")
        assert c.post("/api/datasets/demo", headers=hm, json={"template_id": t["id"]}).status_code == 404
        # password change signs out the other sessions
        h2 = _login(c, "m@cyber.test", "password-1")
        assert c.post("/api/me/password", headers=hm, json={"old": "password-1", "new": "password-2"}).status_code == 200
        assert c.get("/api/me", headers=h2).status_code == 401 and c.get("/api/me", headers=hm).status_code == 200
        # validation errors are readable, server paths are not exposed
        assert c.post("/api/projects", headers=hm, json={"name": 1}).status_code == 422
        for _ in range(10):
            c.post("/api/auth/login", json={"email": "m@cyber.test", "password": "nope"})
        r = c.post("/api/auth/login", json={"email": "m@cyber.test", "password": "password-2"})
        assert r.status_code == 401 and "too many" in r.json()["detail"]


# ---- follow-ups: cookies, retention, queued ingest ---------------------------------------------------------------
def test_session_cookie_secure_follows_https():
    from fastapi.testclient import TestClient
    from backend.app import app
    body = {"email": os.environ["SAI_PLATFORM_ADMIN_EMAIL"], "password": os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]}
    with TestClient(app) as c:
        r = c.post("/api/auth/login", json=body)
        assert "secure" not in r.headers["set-cookie"].lower()            # plain-http localhost keeps working
    with TestClient(app, base_url="https://platform.test") as c:
        r = c.post("/api/auth/login", json=body)
        assert "secure" in r.headers["set-cookie"].lower()
        assert "max-age" in c.get("/api/health").headers["strict-transport-security"]


def test_retention_sweep_keeps_packages_and_removes_rebuildables():
    from backend import config, retention
    old = __import__("time").time() - (retention.RETENTION_DAYS + 2) * 86400
    jobs = config.DATA_DIR / "jobs"
    for jid, status in (("job_ret_ok", "succeeded"), ("job_ret_bad", "failed")):
        d = jobs / jid
        (d / "build").mkdir(parents=True, exist_ok=True)
        (d / "package.zip").write_bytes(b"pkg")
        (d / "hil_build_md6605").mkdir(exist_ok=True)
        (d / "hil_build_md6605" / "x").write_bytes(b"0" * 1000)
        with db.conn() as c:
            c.execute("INSERT OR REPLACE INTO jobs (id, kind, status, created, updated) VALUES (?,?,?,?,?)",
                      (jid, "train", status, old, old))
    up_old, up_new = config.DATA_DIR / "uploads" / "old.onnx", config.DATA_DIR / "uploads" / "new.onnx"
    up_old.write_bytes(b"x" * 500)
    up_new.write_bytes(b"y")
    os.utime(up_old, (old, old))
    r = retention.sweep()
    assert not up_old.exists() and up_new.exists()
    assert (jobs / "job_ret_ok" / "package.zip").exists() and not (jobs / "job_ret_ok" / "hil_build_md6605").exists()
    assert not (jobs / "job_ret_bad").exists()
    assert r["freed_bytes"] >= 1500 and db.get_setting("retention_last")["freed_bytes"] == r["freed_bytes"]


def test_large_upload_becomes_a_background_import(monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from backend import config
    from backend.app import app
    monkeypatch.setattr(config, "INLINE_INGEST_BYTES", 1000)
    X = np.random.default_rng(3).normal(size=(60, 8, 32)).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, X=X, y=np.arange(60) % 5)
    with TestClient(app) as c:
        h = _login(c)
        r = c.post("/api/datasets/upload", headers=h, data={"template_id": "gas-sense", "licence": "proprietary"},
                   files={"file": ("big.npz", buf.getvalue(), "application/octet-stream")})
        assert r.status_code == 202, r.text
        jid = r.json()["job"]["id"]
        for _ in range(60):
            j = c.get(f"/api/jobs/{jid}", headers=h).json()
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.5)
        assert j["status"] == "succeeded", j["error"]
        d = c.get(f"/api/datasets/{j['result']['dataset_id']}", headers=h).json()
        assert d["n"] == 60 and d["licence"] == "proprietary"
        assert not list((config.DATA_DIR / "uploads").glob("ingest_*"))     # staged file removed
