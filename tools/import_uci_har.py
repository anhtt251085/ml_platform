# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""UCI HAR (Human Activity Recognition Using Smartphones) -> NPZ for the `har-imu` model type.

    python tools/import_uci_har.py "human+activity+recognition+using+smartphones.zip" har_uci.npz

Source: https://archive.ics.uci.edu/dataset/240 (CC BY 4.0). Citation required in publications:
  Anguita, Ghio, Oneto, Parra, Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using
  Smartphones. ESANN 2013.
30 subjects, Samsung Galaxy S II on the waist, 50 Hz, 2.56 s windows (128 samples, 50 % overlap).
Channels: total acceleration x/y/z (g) and gyroscope x/y/z (rad/s) - the raw signals an MCU next to the IMU sees.
`groups` = subject id, so the platform splits by person (no subject in both train and test).
"""

from __future__ import annotations

import io
import sys
import zipfile

import numpy as np

SIGNALS = ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
LABELS = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]


def _inner(z: zipfile.ZipFile) -> zipfile.ZipFile:
    names = [n for n in z.namelist() if n.endswith("UCI HAR Dataset.zip")]
    return zipfile.ZipFile(io.BytesIO(z.read(names[0]))) if names else z


def load(path: str):
    z = _inner(zipfile.ZipFile(path))
    root = next(n for n in z.namelist() if n.rstrip("/").endswith("UCI HAR Dataset")) if any(
        n.rstrip("/").endswith("UCI HAR Dataset") for n in z.namelist()) else "UCI HAR Dataset/"
    root = root if root.endswith("/") else root + "/"
    Xs, ys, gs = [], [], []
    for part in ("train", "test"):
        sig = [np.loadtxt(io.BytesIO(z.read(f"{root}{part}/Inertial Signals/{s}_{part}.txt")), dtype=np.float32)
               for s in SIGNALS]
        Xs.append(np.stack(sig, axis=1))                                   # [N, 6, 128]
        ys.append(np.loadtxt(io.BytesIO(z.read(f"{root}{part}/y_{part}.txt")), dtype=np.int64) - 1)
        gs.append(np.loadtxt(io.BytesIO(z.read(f"{root}{part}/subject_{part}.txt")), dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    X, y, g = load(src)
    np.savez_compressed(dst, X=X, y=y, labels=np.array(LABELS), groups=np.array([f"subject{s:02d}" for s in g]))
    print(f"{dst}: X {X.shape}, {len(set(g.tolist()))} subjects, classes {np.bincount(y).tolist()}")


if __name__ == "__main__":
    main()
