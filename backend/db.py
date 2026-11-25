"""Persistence: SQLite (WAL) with numbered migrations.

Tables are owned by the modules that use them; this module owns the connection and the schema history."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

_lock = threading.RLock()

MIGRATIONS: list[str] = [
    # 1: prototype v0.1
    """
    CREATE TABLE IF NOT EXISTS templates (id TEXT PRIMARY KEY, body TEXT, status TEXT, author TEXT, updated REAL);
    CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, body TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT, template_id TEXT, dataset_id TEXT,
                                         config TEXT, job_id TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, project_id TEXT, status TEXT, stage TEXT,
                                     progress REAL, stages TEXT, result TEXT, error TEXT, params TEXT,
                                     created REAL, updated REAL);
    CREATE TABLE IF NOT EXISTS job_logs (job_id TEXT, seq INTEGER, ts REAL, line TEXT);
    CREATE INDEX IF NOT EXISTS job_logs_i ON job_logs(job_id, seq);
    CREATE TABLE IF NOT EXISTS boards (id TEXT PRIMARY KEY, name TEXT, target TEXT, agent_id TEXT, status TEXT,
                                       last_seen REAL, simulated INTEGER, info TEXT);
    CREATE TABLE IF NOT EXISTS hil_runs (id TEXT PRIMARY KEY, job_id TEXT, board_id TEXT, target TEXT, status TEXT,
                                         created REAL, updated REAL, request TEXT, result TEXT);
    CREATE TABLE IF NOT EXISTS calibration (target TEXT, objective TEXT, macs INTEGER, cycles INTEGER, run_id TEXT,
                                            created REAL);
    """,
    # 2: M1 - organisations, users, sessions, agents, audit, usage; org ownership; DB-backed job queue
    """
    CREATE TABLE orgs (id TEXT PRIMARY KEY, name TEXT, tier TEXT, created REAL, branding TEXT);
    CREATE TABLE users (id TEXT PRIMARY KEY, org_id TEXT, email TEXT UNIQUE, name TEXT, role TEXT, pw_hash TEXT,
                        created REAL, disabled INTEGER DEFAULT 0);
    CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, user_id TEXT, created REAL, expires REAL);
    CREATE TABLE agents (id TEXT PRIMARY KEY, name TEXT, token_hash TEXT, created REAL, last_seen REAL,
                         disabled INTEGER DEFAULT 0);
    CREATE TABLE audit (ts REAL, org_id TEXT, user_id TEXT, action TEXT, target TEXT, detail TEXT);
    CREATE INDEX audit_i ON audit(org_id, ts);
    CREATE TABLE usage (ts REAL, org_id TEXT, kind TEXT, qty REAL, ref TEXT);
    CREATE INDEX usage_i ON usage(org_id, ts);
    ALTER TABLE templates ADD COLUMN org_id TEXT;
    ALTER TABLE datasets ADD COLUMN org_id TEXT;
    ALTER TABLE projects ADD COLUMN org_id TEXT;
    ALTER TABLE jobs ADD COLUMN org_id TEXT;
    ALTER TABLE jobs ADD COLUMN user_id TEXT;
    ALTER TABLE jobs ADD COLUMN priority INTEGER DEFAULT 0;
    ALTER TABLE jobs ADD COLUMN worker TEXT;
    ALTER TABLE jobs ADD COLUMN attempts INTEGER DEFAULT 0;
    ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER DEFAULT 0;
    ALTER TABLE hil_runs ADD COLUMN org_id TEXT;
    CREATE INDEX jobs_queue_i ON jobs(status, priority, created);
    """,
]


@contextmanager
def conn():
    with _lock:
        c = sqlite3.connect(config.DB_PATH, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()


def init():
    with conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        r = c.execute("SELECT version FROM schema_version").fetchone()
        have = r[0] if r else 0
        if not r:
            c.execute("INSERT INTO schema_version VALUES (0)")
        for v, sql in enumerate(MIGRATIONS, start=1):
            if v > have:
                c.executescript(sql)
                c.execute("UPDATE schema_version SET version=?", (v,))


def version() -> int:
    with conn() as c:
        return c.execute("SELECT version FROM schema_version").fetchone()[0]


def j(v):
    return None if v is None else json.dumps(v)


def uj(v, default=None):
    return default if v is None else json.loads(v)


def now():
    return time.time()


def row(r):
    return None if r is None else dict(r)
