# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Identity: organisations, users, roles, sessions, lab-agent tokens, audit log, usage metering.

Roles (every organisation has the same four; the SAI organisation's admins are the platform admins)
  engineer    projects, datasets, jobs, board runs; requests compiled-package downloads. No research pages.
  researcher  + Model Studio and Plan Explorer (fit checks, model analysis); in the SAI organisation also pool
              templates, drafts and publishing
  manager     + research pages; approves or rejects compiled-package downloads (never their own request)
  admin       everything in the organisation, including the Admin console (users, audit, branding); in the SAI
              organisation also every organisation, the board farm, lab agents, the job queue and retention
Legacy role names are accepted and mapped: platform_admin / org_admin -> admin, member / lab -> engineer.

Org tiers (commercial edition, M6): free (pool, fit check, estimates; no package download), standard, professional.
Passwords: scrypt. Sessions: random 32-byte tokens, only their SHA-256 is stored. OIDC plugs in by mapping an ID
token to a user and calling `new_session` (not wired in this build).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
import uuid

from fastapi import Depends, Header, HTTPException, Request

from . import config, db

ROLES = ("engineer", "researcher", "manager", "admin")
LEGACY_ROLES = {"platform_admin": "admin", "org_admin": "admin", "member": "engineer", "lab": "engineer"}
TIERS = ("free", "standard", "professional")
SESSION_S = 7 * 24 * 3600
PLATFORM_ORG = "org_sai"

# permission -> (roles that have it, only inside the SAI organisation?)
PERMS = {
    "research": (("researcher", "manager", "admin"), False),   # Model Studio + Plan Explorer pages, fit checks
    "studio": (("researcher", "admin"), True),                 # pool templates and drafts
    "publish": (("researcher", "admin"), True),                # publish to the pool, base weights, benchmarks
    "approve": (("manager", "admin"), False),                  # approve compiled-package downloads
    "org_admin": (("admin",), False),                          # Admin console of the organisation
    "farm_admin": (("admin",), True),                          # board farm, lab agents, regression
    "platform_admin": (("admin",), True),                      # every organisation, queue, retention
}


def norm_role(role: str) -> str:
    role = LEGACY_ROLES.get(role, role)
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    return role


def has(u: dict, perm: str) -> bool:
    roles, platform_only = PERMS[perm]
    return u.get("role") in roles and (not platform_only or u.get("org_id") == PLATFORM_ORG)


def perms(u: dict) -> dict:
    return {k: has(u, k) for k in PERMS}


def hash_pw(pw: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_pw(pw: str, stored: str | None) -> bool:
    try:
        _, salt, h = (stored or "").split("$")
    except ValueError:
        return False
    got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1, dklen=32)
    return hmac.compare_digest(got.hex(), h)


def _th(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ---- orgs & users ------------------------------------------------------------------------------------------------
def create_org(name: str, tier: str = "free", oid: str | None = None) -> dict:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    oid = oid or "org_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO orgs VALUES (?,?,?,?,?)", (oid, name, tier, db.now(), None))
    return org(oid)


def org(oid: str) -> dict | None:
    with db.conn() as c:
        r = db.row(c.execute("SELECT * FROM orgs WHERE id=?", (oid,)).fetchone())
    if r:
        r["branding"] = db.uj(r["branding"], {})
    return r


def orgs() -> list[dict]:
    with db.conn() as c:
        return [db.row(r) for r in c.execute("SELECT id, name, tier, created FROM orgs ORDER BY created")]


def set_tier(oid: str, tier: str):
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    with db.conn() as c:
        c.execute("UPDATE orgs SET tier=? WHERE id=?", (tier, oid))


def create_user(org_id: str, email: str, name: str, role: str, password: str) -> dict:
    role = norm_role(role)
    if len(password) < 8:
        raise ValueError("password must have at least 8 characters")
    uid = "usr_" + uuid.uuid4().hex[:10]
    try:
        with db.conn() as c:
            c.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,0)",
                      (uid, org_id, email.strip().lower(), name, role, hash_pw(password), db.now()))
    except Exception:
        raise ValueError("a user with this email exists") from None
    return user(uid)


def user(uid: str) -> dict | None:
    with db.conn() as c:
        r = db.row(c.execute("SELECT id, org_id, email, name, role, created, disabled FROM users WHERE id=?",
                             (uid,)).fetchone())
    return r


def users(org_id: str | None = None) -> list[dict]:
    with db.conn() as c:
        q = "SELECT id, org_id, email, name, role, created, disabled FROM users"
        return [db.row(r) for r in (c.execute(q + " WHERE org_id=? ORDER BY created", (org_id,)) if org_id
                                     else c.execute(q + " ORDER BY created"))]


_FAILS: dict[str, list[float]] = {}
FAIL_WINDOW_S, FAIL_MAX = 15 * 60, 10


def verify_password(email: str, password: str) -> dict:
    """Check credentials without creating a session. Throttled: FAIL_MAX failures per email per FAIL_WINDOW_S."""
    key = email.strip().lower()
    now = time.time()
    recent = [t for t in _FAILS.get(key, []) if now - t < FAIL_WINDOW_S]
    if len(recent) >= FAIL_MAX:
        raise PermissionError("too many failed sign-ins; try again in 15 minutes")
    with db.conn() as c:
        r = db.row(c.execute("SELECT * FROM users WHERE email=?", (key,)).fetchone())
    if not r or r["disabled"] or not check_pw(password, r["pw_hash"]):
        _FAILS[key] = recent + [now]
        raise PermissionError("wrong email or password")
    _FAILS.pop(key, None)
    return r


def login(email: str, password: str) -> tuple[str, dict]:
    r = verify_password(email, password)
    with db.conn() as c:
        c.execute("DELETE FROM sessions WHERE expires<?", (db.now(),))
    return new_session(r["id"]), user(r["id"])


def new_session(uid: str) -> str:
    token = secrets.token_urlsafe(32)
    with db.conn() as c:
        c.execute("INSERT INTO sessions VALUES (?,?,?,?)", (_th(token), uid, db.now(), db.now() + SESSION_S))
    return token


def logout(token: str):
    with db.conn() as c:
        c.execute("DELETE FROM sessions WHERE token_hash=?", (_th(token),))


def bootstrap():
    """First start: the SAI organisation and a platform admin. The password comes from SAI_PLATFORM_ADMIN_PASSWORD
    or is generated and written to <data>/ADMIN_PASSWORD (mode 600)."""
    if org(PLATFORM_ORG) is None:
        create_org("SAI", "professional", PLATFORM_ORG)
    if users():
        return
    email = os.environ.get("SAI_PLATFORM_ADMIN_EMAIL", "admin@sai.local")
    pw = os.environ.get("SAI_PLATFORM_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    create_user(PLATFORM_ORG, email, "Platform admin", "admin", pw)
    if not os.environ.get("SAI_PLATFORM_ADMIN_PASSWORD"):
        p = config.DATA_DIR / "ADMIN_PASSWORD"
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"{email}\n{pw}\n")
        print(f"[sai-platform] created {email}; password in {p}")


# ---- request context ---------------------------------------------------------------------------------------------
def _token(request: Request, authorization: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return request.cookies.get("sai_session")


def current(request: Request, authorization: str | None = Header(None)) -> dict:
    token = _token(request, authorization)
    if not token:
        raise HTTPException(401, "sign in required")
    with db.conn() as c:
        s = c.execute("SELECT user_id, expires FROM sessions WHERE token_hash=?", (_th(token),)).fetchone()
    if not s or s["expires"] < db.now():
        raise HTTPException(401, "session expired; sign in again")
    u = user(s["user_id"])
    if not u or u["disabled"]:
        raise HTTPException(401, "account disabled")
    o = org(u["org_id"]) or {}
    return {**u, "org": {"id": o.get("id"), "name": o.get("name"), "tier": o.get("tier")}, "token": token,
            "staff": u["org_id"] == PLATFORM_ORG}


def require(perm: str):
    def dep(u: dict = Depends(current)) -> dict:
        if not has(u, perm):
            raise HTTPException(403, f"your role ({u['role']}) cannot do this")
        return u
    return dep


def can_see(u: dict, org_id: str | None) -> bool:
    return has(u, "platform_admin") or org_id == u["org_id"] or (org_id is None and u["staff"])


def scope(u: dict) -> str | None:
    """org filter for list queries; None = all orgs (platform admin)."""
    return None if has(u, "platform_admin") else u["org_id"]


# ---- agents ------------------------------------------------------------------------------------------------------
def create_agent(name: str) -> tuple[dict, str]:
    aid = "agt_" + uuid.uuid4().hex[:8]
    token = "sai_agent_" + secrets.token_urlsafe(32)
    with db.conn() as c:
        c.execute("INSERT INTO agents VALUES (?,?,?,?,?,0)", (aid, name, _th(token), db.now(), None))
    return {"id": aid, "name": name}, token


def agents() -> list[dict]:
    with db.conn() as c:
        return [db.row(r) for r in c.execute("SELECT id, name, created, last_seen, disabled FROM agents ORDER BY created")]


def disable_agent(aid: str):
    with db.conn() as c:
        c.execute("UPDATE agents SET disabled=1 WHERE id=?", (aid,))


def agent_from_token(authorization: str | None) -> str:
    """Per-agent tokens; the shared dev token (SAI_PLATFORM_AGENT_TOKEN) is accepted only when explicitly enabled."""
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    with db.conn() as c:
        r = c.execute("SELECT id, disabled FROM agents WHERE token_hash=?", (_th(token),)).fetchone()
        if r and not r["disabled"]:
            c.execute("UPDATE agents SET last_seen=? WHERE id=?", (db.now(), r["id"]))
            return r["id"]
    if config.ALLOW_DEV_AGENT_TOKEN and token and hmac.compare_digest(token, config.AGENT_TOKEN):
        return "dev-agent"
    raise HTTPException(401, "bad agent token")


# ---- audit & usage -----------------------------------------------------------------------------------------------
def audit(u: dict | None, action: str, target: str = "", detail: dict | None = None):
    with db.conn() as c:
        c.execute("INSERT INTO audit VALUES (?,?,?,?,?,?)", (db.now(), (u or {}).get("org_id"), (u or {}).get("id"),
                                                             action, target, db.j(detail)))


def audit_log(org_id: str | None, limit: int = 200) -> list[dict]:
    with db.conn() as c:
        q = "SELECT a.*, u.email FROM audit a LEFT JOIN users u ON u.id = a.user_id"
        rows = (c.execute(q + " WHERE a.org_id=? ORDER BY ts DESC LIMIT ?", (org_id, limit)) if org_id
                else c.execute(q + " ORDER BY ts DESC LIMIT ?", (limit,)))
        out = [db.row(r) for r in rows]
    for r in out:
        r["detail"] = db.uj(r["detail"])
    return out


def meter(org_id: str | None, kind: str, qty: float, ref: str = ""):
    with db.conn() as c:
        c.execute("INSERT INTO usage VALUES (?,?,?,?,?)", (db.now(), org_id, kind, float(qty), ref))


def usage(org_id: str | None, since: float = 0) -> dict:
    with db.conn() as c:
        q = "SELECT kind, SUM(qty) AS qty, COUNT(*) AS n FROM usage WHERE ts>=?"
        rows = (c.execute(q + " AND org_id=? GROUP BY kind", (since, org_id)) if org_id
                else c.execute(q + " GROUP BY kind", (since,)))
        return {r["kind"]: {"qty": round(r["qty"], 3), "events": r["n"]} for r in rows}
