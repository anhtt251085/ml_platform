"""Commercial layer (M6): plan tiers and quotas, deployment rights, package signing, branding.

Tiers (orgs.tier) decide which tools an organisation can use; deployment rights (orgs.branding["deployment"], set
by SAI under contract: per-unit royalty or buyout) decide whether its packages are production packages. Without
deployment rights — or with data whose licence forbids commercial use — packages are evaluation-only.
Prices are set by SAI sales (roadmap decision #8) and are not part of the software.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import zipfile
from pathlib import Path

from fastapi import HTTPException

from . import config, db

PLANS = {
    "free": {"label": "Free", "summary": "Explore the pool, fit-check models, see SAI's plans and estimates.",
             "projects_per_month": 3, "packages": False, "farm": "none", "farm_runs_per_month": 0,
             "explorer": True, "priority": 0, "support": "community"},
    "standard": {"label": "Standard", "summary": "Train and download verified packages; emulated board farm.",
                 "projects_per_month": 50, "packages": True, "farm": "emulated", "farm_runs_per_month": 200,
                 "explorer": True, "priority": 0, "support": "email"},
    "professional": {"label": "Professional", "summary": "Everything, physical boards with measured latency, "
                     "priority queue.", "projects_per_month": None, "packages": True, "farm": "all",
                     "farm_runs_per_month": 2000, "explorer": True, "priority": 1, "support": "SLA"},
}
DEPLOYMENT_MODELS = ("none", "royalty", "buyout")


def _month_start() -> float:
    import calendar
    t = time.gmtime()
    return float(calendar.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0)))


def plan_of(u: dict) -> dict:
    tier = "professional" if u.get("staff") else (u.get("org") or {}).get("tier") or "free"
    return {"tier": tier, **PLANS[tier]}


def counts(org_id: str) -> dict:
    since = _month_start()
    with db.conn() as c:
        projects = c.execute("SELECT COUNT(*) FROM jobs WHERE org_id=? AND kind='train' AND created>=?",
                             (org_id, since)).fetchone()[0]
        runs = c.execute("SELECT COUNT(*) FROM hil_runs WHERE org_id=? AND created>=?", (org_id, since)).fetchone()[0]
    return {"projects": projects, "farm_runs": runs, "since": since}


def check_project(u: dict):
    plan = plan_of(u)
    if plan["projects_per_month"] is None or u.get("staff"):
        return plan
    if counts(u["org_id"])["projects"] >= plan["projects_per_month"]:
        raise HTTPException(402, f"the {plan['label']} plan includes {plan['projects_per_month']} projects per month; "
                                 "ask your SAI contact to upgrade")
    return plan


def check_package_download(u: dict):
    if not plan_of(u)["packages"]:
        raise HTTPException(402, "package downloads are part of the Standard and Professional plans; the Free plan "
                                 "shows the full report, plans and estimates")


def check_board_run(u: dict, board_kind: str | None):
    plan = plan_of(u)
    if plan["farm"] == "none":
        raise HTTPException(402, "board-farm runs are part of the Standard and Professional plans")
    if plan["farm"] == "emulated" and board_kind == "physical":
        raise HTTPException(402, "physical boards (measured latency) are part of the Professional plan; "
                                 "emulated boards are included in Standard")
    if not u.get("staff") and counts(u["org_id"])["farm_runs"] >= plan["farm_runs_per_month"]:
        raise HTTPException(402, f"monthly board-farm quota of the {plan['label']} plan reached")
    return plan


# ---- org settings: deployment rights and branding ------------------------------------------------------------
def settings(org_id: str) -> dict:
    with db.conn() as c:
        r = c.execute("SELECT branding FROM orgs WHERE id=?", (org_id,)).fetchone()
    s = db.uj(r[0], {}) if r else {}
    s.setdefault("deployment", {"model": "none", "units": None, "contract": None})
    s.setdefault("brand", {})
    return s


def save_settings(org_id: str, s: dict):
    with db.conn() as c:
        c.execute("UPDATE orgs SET branding=? WHERE id=?", (db.j(s), org_id))


def set_deployment(org_id: str, model: str, units: int | None = None, contract: str | None = None):
    if model not in DEPLOYMENT_MODELS:
        raise ValueError(f"deployment model must be one of {DEPLOYMENT_MODELS}")
    s = settings(org_id)
    s["deployment"] = {"model": model, "units": units, "contract": contract}
    save_settings(org_id, s)


def deployment_rights(org_id: str | None) -> bool:
    from . import auth
    if org_id in (None, auth.PLATFORM_ORG):
        return True
    return settings(org_id)["deployment"]["model"] != "none"


def set_brand(org_id: str, brand: dict):
    clean = {"product_name": str(brand.get("product_name") or "")[:40],
             "accent": brand.get("accent") if str(brand.get("accent") or "").startswith("#") and
             len(str(brand.get("accent"))) in (4, 7) else None,
             "logo_text": str(brand.get("logo_text") or "")[:2]}
    s = settings(org_id)
    s["brand"] = {k: v for k, v in clean.items() if v}
    save_settings(org_id, s)
    return s["brand"]


# ---- package signing -------------------------------------------------------------------------------------------
def _key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    d = config.DATA_DIR / "signing"
    d.mkdir(parents=True, exist_ok=True)
    path = Path(config.SIGNING_KEY) if config.SIGNING_KEY else d / "platform_ed25519.pem"
    if not path.exists():
        import os
        pem = Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                         serialization.NoEncryption())
        try:       # O_EXCL: two processes starting together keep one key; 0600 from the first byte
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(pem)
        except FileExistsError:
            time.sleep(0.2)                # the other process is writing it
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def public_key_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    return _key().public_key().public_bytes(serialization.Encoding.PEM,
                                            serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def key_id() -> str:
    return hashlib.sha256(public_key_pem().encode()).hexdigest()[:16]


def _canonical(files: dict, meta: dict) -> bytes:
    return json.dumps({"files": files, "meta": meta}, sort_keys=True, separators=(",", ":")).encode()


def sign_package(zip_path: Path, meta: dict):
    """Add SIGNATURE.json to the zip: SHA-256 of every file + Ed25519 signature over the canonical manifest."""
    with zipfile.ZipFile(zip_path) as z:
        files = {n: hashlib.sha256(z.read(n)).hexdigest() for n in sorted(z.namelist()) if not n.endswith("/")}
    meta = {**meta, "key_id": key_id(), "signed_at": int(time.time()), "algorithm": "Ed25519"}
    sig = base64.b64encode(_key().sign(_canonical(files, meta))).decode()
    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as z:
        z.writestr("sai_package/SIGNATURE.json", json.dumps({"files": files, "meta": meta, "signature": sig}, indent=1))
    return meta


def verify_package(zip_bytes: bytes, public_pem: str) -> dict:
    """Used by tools/verify_package.py and the tests: every file hashed, signature checked, nothing extra."""
    import io
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    allnames = [n for n in z.namelist() if not n.endswith("/")]
    if len(allnames) != len(set(allnames)):
        return {"ok": False, "problems": ["duplicate entry names (ambiguous archive)"], "meta": {}}
    if any(not n.startswith("sai_package/") or ".." in n.split("/") for n in allnames):
        return {"ok": False, "problems": ["entries outside sai_package/"], "meta": {}}
    sig = json.loads(z.read("sai_package/SIGNATURE.json"))
    pub = serialization.load_pem_public_key(public_pem.encode())
    problems = []
    try:
        pub.verify(base64.b64decode(sig["signature"]), _canonical(sig["files"], sig["meta"]))
    except InvalidSignature:
        problems.append("signature does not match the manifest")
    names = {n for n in z.namelist() if not n.endswith("/") and n != "sai_package/SIGNATURE.json"}
    if names != set(sig["files"]):
        problems.append(f"files added or removed: {sorted(names ^ set(sig['files']))}")
    for n in names & set(sig["files"]):
        if hashlib.sha256(z.read(n)).hexdigest() != sig["files"][n]:
            problems.append(f"modified: {n}")
    return {"ok": not problems, "problems": problems, "meta": sig["meta"]}
