"""SQLite persistence (one file, WAL). Swap for Postgres in production: the access layer is these functions only."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

_lock = threading.RLock()

SCHEMA = """
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
"""


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
        c.executescript(SCHEMA)


def j(v):
    return None if v is None else json.dumps(v)


def uj(v, default=None):
    return default if v is None else json.loads(v)


def now():
    return time.time()


def row(r):
    return None if r is None else dict(r)
