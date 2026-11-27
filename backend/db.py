# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
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
    # 3: M3 - board-farm measurements of exact programs (plan digest), the platform's `measured` evidence
    """
    CREATE TABLE measurements (target TEXT, plan_digest TEXT, cycles INTEGER, stack_painted INTEGER, board TEXT,
                               run_id TEXT, created REAL);
    CREATE INDEX measurements_i ON measurements(target, plan_digest);
    """,
    # 4: M4 - board health, nightly regression results
    """
    ALTER TABLE boards ADD COLUMN fail_count INTEGER DEFAULT 0;
    ALTER TABLE boards ADD COLUMN health TEXT;
    CREATE TABLE regressions (run_id TEXT, template TEXT, size TEXT, target TEXT, ram INTEGER, flash INTEGER,
                              cycles INTEGER, feasible INTEGER, hil_match INTEGER, sai_version TEXT, created REAL);
    CREATE INDEX regressions_i ON regressions(template, size, target, created);
    """,
    # 5: review fixes - platform settings (queue pause shared by every worker process)
    """
    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
    """,
    # 6: training monitoring - per-epoch metrics of every training run (live curves)
    """
    CREATE TABLE job_metrics (job_id TEXT, seq INTEGER, ts REAL, series TEXT, data TEXT);
    CREATE INDEX job_metrics_i ON job_metrics(job_id, seq);
    """,
    # 7: four roles (engineer, researcher, manager, admin) and manager approval of compiled-package downloads
    """
    UPDATE users SET role='admin' WHERE role IN ('platform_admin', 'org_admin');
    UPDATE users SET role='engineer' WHERE role IN ('member', 'lab');
    CREATE TABLE package_approvals (job_id TEXT PRIMARY KEY, org_id TEXT, status TEXT, requested_by TEXT,
                                    requested_at REAL, request_note TEXT, decided_by TEXT, decided_at REAL,
                                    decision_note TEXT, downloads INTEGER DEFAULT 0, last_download REAL);
    CREATE INDEX package_approvals_i ON package_approvals(org_id, status);
    """,
    # 8: architecture library (custom entries; built-ins live in backend/archlib.py) and SAI verification results
    """
    CREATE TABLE architectures (id TEXT PRIMARY KEY, org_id TEXT, status TEXT, data TEXT, created_by TEXT,
                                created REAL, updated REAL);
    CREATE TABLE architecture_checks (arch_id TEXT, target TEXT, sai_version TEXT, ok INTEGER, fits INTEGER,
                                      ram INTEGER, flash INTEGER, latency_ms REAL, plan TEXT, detail TEXT,
                                      checked REAL);
    CREATE INDEX architecture_checks_i ON architecture_checks(arch_id, target, checked);
    """,
    # 9: recordings (continuous signals to label, captured live or imported) and data connections (cloud / DB)
    """
    CREATE TABLE recordings (id TEXT PRIMARY KEY, org_id TEXT, meta TEXT, status TEXT, token_hash TEXT,
                             created REAL, updated REAL);
    CREATE INDEX recordings_i ON recordings(org_id, created);
    CREATE TABLE connections (id TEXT PRIMARY KEY, org_id TEXT, kind TEXT, name TEXT, config TEXT, secret TEXT,
                              created_by TEXT, created REAL);
    CREATE INDEX connections_i ON connections(org_id);
    """,
    # 10: models SAI refused for a missing operator (compiler roadmap signal) and drift checks of projects
    """
    CREATE TABLE operator_gaps (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, org_id TEXT, job_id TEXT,
                                kind TEXT, source TEXT, op TEXT, target TEXT, capability TEXT, reason TEXT,
                                model_ops TEXT);
    CREATE INDEX operator_gaps_i ON operator_gaps(op, created);
    CREATE TABLE drift_checks (id TEXT PRIMARY KEY, job_id TEXT, org_id TEXT, created REAL, source TEXT,
                               status TEXT, result TEXT);
    CREATE INDEX drift_checks_i ON drift_checks(job_id, created);
    """,
    # 11: customer-VPC workers (data access inside the customer's network) and partner-hosted boards
    """
    CREATE TABLE vpc_workers (id TEXT PRIMARY KEY, org_id TEXT, name TEXT, token_hash TEXT, created REAL,
                              last_seen REAL, disabled INTEGER DEFAULT 0, info TEXT);
    CREATE TABLE vpc_tasks (id TEXT PRIMARY KEY, org_id TEXT, connection_id TEXT, kind TEXT, payload TEXT,
                            status TEXT, worker_id TEXT, result TEXT, error TEXT, created REAL, updated REAL);
    CREATE INDEX vpc_tasks_i ON vpc_tasks(org_id, status, created);
    ALTER TABLE agents ADD COLUMN partner TEXT;
    CREATE TABLE partner_boards (id TEXT PRIMARY KEY, partner TEXT, board TEXT, vendor TEXT, mcu TEXT, core TEXT,
                                 target TEXT, status TEXT, url TEXT, notes TEXT, created REAL);
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


def _statements(sql: str) -> list[str]:
    return [x.strip() for x in sql.split(";") if x.strip()]


def init():
    """Apply pending migrations under an exclusive lock, each migration and its version bump in one transaction
    (SQLite DDL is transactional): concurrent starters (API + workers) wait, a crash leaves no half schema."""
    with _lock:
        c = sqlite3.connect(config.DB_PATH, timeout=60, isolation_level=None)
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("BEGIN EXCLUSIVE")
            try:
                c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
                r = c.execute("SELECT version FROM schema_version").fetchone()
                have = r[0] if r else 0
                if not r:
                    c.execute("INSERT INTO schema_version VALUES (0)")
                for v, sql in enumerate(MIGRATIONS, start=1):
                    if v > have:
                        for stmt in _statements(sql):
                            c.execute(stmt)
                        c.execute("UPDATE schema_version SET version=?", (v,))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        finally:
            c.close()


def get_setting(key: str, default=None):
    with conn() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return uj(r[0]) if r else default


def set_setting(key: str, value):
    with conn() as c:
        c.execute("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, j(value)))


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
