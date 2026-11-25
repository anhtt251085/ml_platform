# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Verify a SAI Platform package: every file's SHA-256 and the platform's Ed25519 signature.

    python tools/verify_package.py package.zip platform_public_key.pem
    curl https://<platform>/api/signing/public-key | python -c "import json,sys; print(json.load(sys.stdin)['public_key_pem'])" > key.pem

Exit 0 when the package is exactly what the platform built; prints whether it is a production or an
evaluation-only package.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.commercial import verify_package  # noqa: E402


def main():
    zip_path, key = sys.argv[1], sys.argv[2]
    r = verify_package(Path(zip_path).read_bytes(), Path(key).read_text())
    m = r["meta"]
    print(f"job {m.get('job')}  target {m.get('target')}  SAI {m.get('sai_version')}  key {m.get('key_id')}")
    print("production package" if m.get("deployable") else "EVALUATION-ONLY package")
    if r["ok"]:
        print("OK: signature valid, all files intact")
        return 0
    for p in r["problems"]:
        print("FAIL:", p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
