"""Preprocessing stage: validation, and bit-exactness of the generated C (`sai_app_prepare`) against training."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

os.environ.setdefault("SAI_PLATFORM_DATA", tempfile.mkdtemp(prefix="saiplat-"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend import preprocess  # noqa: E402

META = {}


@pytest.mark.parametrize("cfg", [
    {"channels": [0, 2], "decimate": 2, "demean": True, "normalization": "zscore", "clip": 3.0},
    {"decimate": 4, "normalization": "robust"},
    {"normalization": "minmax"},
    {"demean": True, "normalization": "none", "clip": 50},
])
def test_generated_c_prepare_is_bit_exact(tmp_path, cfg):
    rng = np.random.default_rng(1)
    X = (rng.normal(0, 1, (40, 3, 1, 128)) * np.array([1, 5, 0.01]).reshape(1, 3, 1, 1) + 7).astype(np.float32)
    c = preprocess.normalize_cfg(cfg, META, X.shape[1:], "classification")
    Xw = preprocess.window_ops(X, c)
    off, sc = preprocess.fit_norm(Xw, c)
    qin = {"scale": 0.0371, "zero_point": -7}
    ref = preprocess.quantize(preprocess.apply_norm(Xw, off, sc, c["clip"]), qin["scale"], qin["zero_point"])
    (tmp_path / "sai_app_prepare.h").write_text(preprocess.c_prepare(c, off, sc, qin, list(X.shape[1:])))
    (tmp_path / "main.c").write_text("""
#include <stdio.h>
#include "sai_app_prepare.h"
int main(void) {
    static float raw[SAI_APP_RAW_CHANNELS * SAI_APP_RAW_SAMPLES];
    static int8_t q[SAI_APP_IN_CHANNELS * SAI_APP_IN_SAMPLES];
    while (fread(raw, sizeof raw, 1, stdin) == 1) {
        if (sai_app_prepare(raw, q)) return 1;
        fwrite(q, sizeof q, 1, stdout);
    }
    return 0;
}""")
    src = preprocess.SRC
    exe = tmp_path / "prep"
    subprocess.run(["gcc", "-O2", "-std=c11", "-I", str(tmp_path), "-I", str(src), str(tmp_path / "main.c"),
                    str(src / "sai_prep.c"), "-o", str(exe), "-lm"], check=True)
    out = subprocess.run([str(exe)], input=np.ascontiguousarray(X[:, :, 0, :]).tobytes(), capture_output=True, check=True)
    got = np.frombuffer(out.stdout, np.int8).reshape(ref.shape[0], -1)
    assert np.array_equal(got, ref.reshape(ref.shape[0], -1))


def test_prepare_compiles_for_both_mcus(tmp_path):
    c = preprocess.normalize_cfg({"channels": [1], "decimate": 2, "demean": True, "clip": 4}, META, (2, 1, 64), "classification")
    (tmp_path / "sai_app_prepare.h").write_text(preprocess.c_prepare(c, np.zeros(1), np.ones(1), {"scale": 0.1, "zero_point": 0}, [2, 1, 64]))
    (tmp_path / "u.c").write_text('#include "sai_app_prepare.h"\nint use(const float *r, int8_t *q) { return sai_app_prepare(r, q); }\n')
    rv = os.environ.get("SAI_MD6605_GCC") or shutil.which("riscv-none-elf-gcc")
    for cc, flags in (("arm-none-eabi-gcc", ["-mcpu=cortex-m7", "-mfpu=fpv5-d16", "-mfloat-abi=hard", "-mthumb"]),
                      (rv, ["-march=rv32imafc_zicsr", "-mabi=ilp32"])):
        if not cc or not shutil.which(cc) and not os.path.exists(cc):
            continue
        for f in ("u.c", str(preprocess.SRC / "sai_prep.c")):
            subprocess.run([cc, *flags, "-O2", "-Wall", "-Werror", "-c", "-I", str(tmp_path), "-I", str(preprocess.SRC),
                            str(tmp_path / f) if f == "u.c" else f, "-o", str(tmp_path / "o.o")], check=True)


def test_config_validation():
    shape = (3, 1, 128)
    with pytest.raises(preprocess.PrepError):
        preprocess.normalize_cfg({"channels": [5]}, META, shape, "classification")
    with pytest.raises(preprocess.PrepError):
        preprocess.normalize_cfg({"decimate": 32}, META, shape, "classification")     # < 8 samples left
    with pytest.raises(preprocess.PrepError):
        preprocess.normalize_cfg({"split": {"train": 0.9, "val": 0.05, "test": 0.0}}, META, shape, "classification")
    with pytest.raises(preprocess.PrepError):                                           # window steps on images
        preprocess.normalize_cfg({"demean": True}, META, (1, 24, 32), "classification")
    with pytest.raises(preprocess.PrepError):                                           # ... and on front-end features
        preprocess.normalize_cfg({"decimate": 2}, {"frontend": {"type": "mfcc"}}, (1, 1, 64), "classification")
    c = preprocess.normalize_cfg({"channels": [0, 1, 2]}, META, shape, "regression")
    assert c["channels"] is None and not c["window_ops"] and c["class_weights"] is False


def test_constant_channel_does_not_explode_and_preview():
    X = np.random.default_rng(2).normal(size=(30, 2, 1, 64)).astype(np.float32)
    X[:, 1] = 3.0                                                                       # dead sensor
    c = preprocess.normalize_cfg({"normalization": "zscore"}, META, X.shape[1:], "classification")
    off, sc = preprocess.fit_norm(X, c)
    assert sc[1] == 1.0 and np.isfinite(preprocess.apply_norm(X, off, sc, None)).all()
    pv = preprocess.preview(X, np.repeat([0, 1, 2], 10), ["a", "b", "c"], META,
                            preprocess.normalize_cfg({"decimate": 2, "demean": True}, META, X.shape[1:], "classification"),
                            "classification")
    assert pv["model_input_shape"] == [2, 1, 32] and len(pv["examples"]) == 3 and pv["after"][0]["mean"] == pytest.approx(0, abs=1e-5)


HAS_SAI = shutil.which(os.environ.get("SAI_BIN", "sai")) is not None


def _login(c, email=None, pw=None):
    r = c.post("/api/auth/login", json={"email": email or os.environ["SAI_PLATFORM_ADMIN_EMAIL"],
                                        "password": pw or os.environ["SAI_PLATFORM_ADMIN_PASSWORD"]})
    return {"Authorization": f"Bearer {r.json()['token']}"}


os.environ.setdefault("SAI_PLATFORM_ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("SAI_PLATFORM_ADMIN_PASSWORD", "admin-pass-123")


def test_data_page_endpoints_and_isolation():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        h = _login(c)
        ds = c.post("/api/datasets/demo", json={"template_id": "har-imu", "per_class": 30}, headers=h).json()
        st = c.get(f"/api/datasets/{ds['id']}/stats", headers=h).json()
        assert len(st["channels"]) == 6 and len(st["per_class"]) == 6 and st["per_class"][0]["recordings"]
        sm = c.get(f"/api/datasets/{ds['id']}/samples?label=1&offset=0&n=4", headers=h).json()
        assert sm["total"] == 30 and len(sm["items"]) == 4 and len(sm["items"][0]["series"]) == 6
        assert all(it["label"] == ds["labels"][1] for it in sm["items"])
        pv = c.post(f"/api/datasets/{ds['id']}/preprocess-preview", headers=h,
                    json={"preprocess": {"channels": [0, 1, 2], "decimate": 2, "demean": True}}).json()
        assert pv["model_input_shape"] == [3, 1, 64]
        bad = c.post(f"/api/datasets/{ds['id']}/preprocess-preview", headers=h, json={"preprocess": {"decimate": 64}})
        assert bad.status_code == 400
        org = c.post("/api/admin/orgs", json={"name": "Umbrella", "tier": "standard"}, headers=h).json()
        c.post("/api/admin/users", headers=h, json={"email": "u@umbrella.test", "name": "U", "role": "member",
                                                     "password": "password-1", "org_id": org["id"]})
        hu = _login(c, "u@umbrella.test", "password-1")
        for path in ("", "/stats", "/samples"):
            assert c.get(f"/api/datasets/{ds['id']}{path}", headers=hu).status_code == 404
        assert c.delete(f"/api/datasets/{ds['id']}", headers=hu).status_code == 404
        assert c.delete(f"/api/datasets/{ds['id']}", headers=h).json() == {"ok": True}
        assert c.get(f"/api/datasets/{ds['id']}", headers=h).status_code == 404


@pytest.mark.skipif(not HAS_SAI, reason="sai CLI not installed")
def test_project_with_window_steps_ships_prepare_and_stays_exact():
    import io
    import time
    import zipfile
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as c:
        c.headers.update(_login(c))
        ds = c.post("/api/datasets/demo", json={"template_id": "har-imu", "per_class": 40}).json()
        prep = {"channels": [0, 1, 2, 3], "decimate": 2, "demean": True, "normalization": "robust", "clip": 5,
                "augment": {"gain": 0.1, "noise": 0.02, "shift": 4}, "split": {"train": 0.6, "val": 0.2, "test": 0.2}}
        assert c.post("/api/projects", json={"name": "x", "template_id": "har-imu", "dataset_id": ds["id"],
                                             "preprocess": {"decimate": 3000}}).status_code == 400
        job = c.post("/api/projects", json={"name": "prep", "template_id": "har-imu", "dataset_id": ds["id"], "sizes": ["tiny"],
                                            "epochs": 3, "targets": ["cortex-m7"], "preprocess": prep}).json()["job"]
        for _ in range(300):
            j = c.get(f"/api/jobs/{job['id']}").json()
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(1)
        assert j["status"] == "succeeded", j["error"]
        r = j["result"]
        assert r["preprocessing"]["model_input_shape"] == [4, 1, 64] and r["preprocessing"]["raw_shape"] == [6, 1, 128]
        assert r["exact"]["value"] is not None
        z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{j['id']}/download/package.zip").content))
        names = set(z.namelist())
        assert {"sai_package/sai_app_prepare.h", "sai_package/sai_prep.c", "sai_package/sai_prep.h"} <= names
        hdr = z.read("sai_package/sai_app_prepare.h").decode()
        assert "SAI_APP_RAW_CHANNELS 6" in hdr and "SAI_APP_IN_CHANNELS 4" in hdr and "SAI_APP_IN_SAMPLES 64" in hdr
