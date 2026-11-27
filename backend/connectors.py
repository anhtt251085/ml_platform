# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Cloud storage and database connectors: browse a customer's data where it lives and import it as a dataset or
as a recording to label.

Kinds
  s3        AWS S3 and S3-compatible stores (MinIO, Wasabi, Cloudflare R2, GCS through its HMAC interop endpoint
            storage.googleapis.com): ListObjectsV2 and GetObject signed with AWS Signature V4.
  azure     Azure Blob Storage with a container SAS token (read + list): List Blobs and Get Blob.
  influxdb  InfluxDB v2 / Cloud: a Flux query, annotated CSV back (pivot the fields into columns in the query).
  postgres  PostgreSQL / TimescaleDB: one read-only SELECT (READ ONLY transaction, statement timeout, row limit);
            needs the optional psycopg package.

Security
  * Secrets (keys, SAS tokens, passwords) are encrypted at rest with Fernet; the key is SAI_PLATFORM_SECRET_KEY or
    a key file generated in the data directory (0600). They are never returned by the API: the UI sees "set".
  * The platform only connects to public addresses (the address that was checked is the one connected to, no
    redirects), like URL import. Private endpoints (a MinIO in the customer's network, an internal database) need
    a single-customer install inside that network (SAI_PLATFORM_MODE=single) or SAI_PLATFORM_ALLOW_PRIVATE_CONNECTORS=1.
  * Connections belong to an organisation; only its members (and platform admins) can use them.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import ipaddress
import json
import os
import re
import socket
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlsplit

from . import config, data, db

KINDS = {
    "s3": {"label": "Amazon S3 / S3-compatible (MinIO, R2, GCS HMAC)",
           "fields": [("endpoint", "Endpoint", "s3.amazonaws.com"), ("region", "Region", "us-east-1"),
                      ("bucket", "Bucket", ""), ("prefix", "Prefix", ""), ("access_key", "Access key ID", ""),
                      ("path_style", "Path-style URLs (MinIO)", False)],
           "secrets": [("secret_key", "Secret access key")]},
    "azure": {"label": "Azure Blob Storage",
              "fields": [("account", "Storage account", ""), ("container", "Container", ""), ("prefix", "Prefix", "")],
              "secrets": [("sas", "SAS token (read + list)")]},
    "influxdb": {"label": "InfluxDB v2 / Cloud",
                 "fields": [("url", "URL", "https://eu-central-1-1.aws.cloud2.influxdata.com"), ("org", "Organisation", ""),
                            ("bucket", "Bucket", "")],
                 "secrets": [("token", "API token (read)")]},
    "postgres": {"label": "PostgreSQL / TimescaleDB",
                 "fields": [("host", "Host", ""), ("port", "Port", 5432), ("database", "Database", ""),
                            ("user", "User (read-only role recommended)", ""), ("sslmode", "SSL mode", "require")],
                 "secrets": [("password", "Password")]},
}
MAX_LIST = 500
MAX_ROWS = 2_000_000


class ConnectorError(data.DataError):
    pass


# ---- secrets -----------------------------------------------------------------------------------------------------
def _fernet():
    from cryptography.fernet import Fernet
    key = os.environ.get("SAI_PLATFORM_SECRET_KEY", "").encode()
    if not key:
        f = config.DATA_DIR / "secret.key"
        if not f.exists():
            f.write_bytes(Fernet.generate_key())
            os.chmod(f, 0o600)
        key = f.read_bytes().strip()
    return Fernet(key)


def _enc(d: dict) -> str:
    return _fernet().encrypt(json.dumps(d).encode()).decode()


def _dec(s: str | None) -> dict:
    return json.loads(_fernet().decrypt(s.encode())) if s else {}


# ---- storage -----------------------------------------------------------------------------------------------------
def _clean(kind: str, cfg: dict, secrets_in: dict | None, old_secret: dict | None = None) -> tuple[dict, dict]:
    if kind not in KINDS:
        raise ConnectorError(f"kind must be one of {', '.join(KINDS)}")
    spec = KINDS[kind]
    out = {}
    for key, _label, default in spec["fields"]:
        v = cfg.get(key, default)
        if isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, int):
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ConnectorError(f"{key} must be a number") from None
        else:
            v = str(v or "").strip()[:300]
        out[key] = v
    out["via_worker"] = bool(cfg.get("via_worker"))
    sec = dict(old_secret or {})
    for key, _label in spec["secrets"]:
        if (secrets_in or {}).get(key):
            sec[key] = str(secrets_in[key])
    if out["via_worker"]:
        sec = {}             # the customer-VPC worker holds the credentials; the platform never sees them
    if kind == "s3" and not out["bucket"]:
        raise ConnectorError("bucket is required")
    if kind == "azure" and not (re.fullmatch(r"[a-z0-9]{3,24}", out["account"]) and out["container"]):
        raise ConnectorError("give the storage account (3-24 lowercase letters/digits) and the container")
    if kind == "influxdb" and not ((out["url"].startswith("https://") or cfg.get("via_worker")) and out["org"]):
        raise ConnectorError("InfluxDB needs an https:// URL and the organisation")
    if kind == "postgres" and not (out["host"] and out["database"] and out["user"]):
        raise ConnectorError("host, database and user are required")
    return out, sec


def _public(r) -> dict:
    r = db.row(r)
    sec = _dec(r.pop("secret"))
    r["config"] = db.uj(r["config"], {})
    r["secrets_set"] = sorted(k for k, v in sec.items() if v)
    return r


def create(org_id: str, user_id: str, kind: str, name: str, cfg: dict, secrets_in: dict) -> dict:
    cfg, sec = _clean(kind, cfg, secrets_in)
    cid = "con_" + uuid.uuid4().hex[:10]
    with db.conn() as c:
        c.execute("INSERT INTO connections (id, org_id, kind, name, config, secret, created_by, created)"
                  " VALUES (?,?,?,?,?,?,?,?)", (cid, org_id, kind, (name or kind)[:80], db.j(cfg), _enc(sec),
                                                 user_id, db.now()))
    return get(cid)


def update(cid: str, name: str | None, cfg: dict, secrets_in: dict) -> dict:
    with db.conn() as c:
        r = c.execute("SELECT kind, secret, name FROM connections WHERE id=?", (cid,)).fetchone()
    cfg, sec = _clean(r[0], cfg, secrets_in, _dec(r[1]))
    with db.conn() as c:
        c.execute("UPDATE connections SET name=?, config=?, secret=? WHERE id=?",
                  ((name or r[2])[:80], db.j(cfg), _enc(sec), cid))
    return get(cid)


def get(cid: str) -> dict | None:
    with db.conn() as c:
        r = c.execute("SELECT * FROM connections WHERE id=?", (cid,)).fetchone()
    return _public(r) if r else None


def list_(org_id: str | None) -> list[dict]:
    q, a = "SELECT * FROM connections", ()
    if org_id:
        q, a = q + " WHERE org_id=?", (org_id,)
    with db.conn() as c:
        return [_public(r) for r in c.execute(q + " ORDER BY created DESC", a)]


def delete(cid: str):
    with db.conn() as c:
        c.execute("DELETE FROM connections WHERE id=?", (cid,))


def _full(cid: str) -> tuple[str, dict, dict]:
    with db.conn() as c:
        r = c.execute("SELECT kind, config, secret FROM connections WHERE id=?", (cid,)).fetchone()
    if not r:
        raise ConnectorError("unknown connection")
    return r[0], db.uj(r[1], {}), _dec(r[2])


# ---- network -----------------------------------------------------------------------------------------------------
def private_allowed() -> bool:
    return config.MODE == "single" or os.environ.get("SAI_PLATFORM_ALLOW_PRIVATE_CONNECTORS") == "1"


def _resolve(host: str, port: int) -> str:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ConnectorError(f"{host} does not resolve") from None
    addrs = sorted({i[4][0] for i in infos})
    if not private_allowed():
        for a in addrs:
            ip = ipaddress.ip_address(a)
            if not ip.is_global or ip.is_multicast:
                raise ConnectorError(f"{host} is a private or reserved address; private endpoints need an install "
                                     "inside that network (single-customer mode) or SAI_PLATFORM_ALLOW_PRIVATE_CONNECTORS=1")
    return addrs[0]


def https(method: str, url: str, headers: dict | None = None, body: bytes | None = None,
          limit: int = config.MAX_UPLOAD_BYTES) -> tuple[int, dict, bytes]:
    """One HTTPS request to the checked address (no redirects). Returns (status, headers, body)."""
    import http.client
    import ssl

    u = urlsplit(url)
    plain = u.scheme == "http" and private_allowed()          # inside the customer's network (MinIO on http)
    if (u.scheme != "https" and not plain) or not u.hostname:
        raise ConnectorError("only https:// endpoints are accepted")
    port = u.port or (80 if plain else 443)
    ip = _resolve(u.hostname, port)
    ctx = ssl.create_default_context()

    class _Pinned(http.client.HTTPSConnection):
        def connect(self):
            self.sock = ctx.wrap_socket(socket.create_connection((ip, port), timeout=60), server_hostname=u.hostname)

    class _PinnedPlain(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.create_connection((ip, port), timeout=60)

    conn = _PinnedPlain(u.hostname, port, timeout=120) if plain else _Pinned(u.hostname, port, timeout=120, context=ctx)
    try:
        conn.request(method, (u.path or "/") + (f"?{u.query}" if u.query else ""), body=body,
                     headers={"User-Agent": "sai-platform-connector/0.3", **(headers or {})})
        resp = conn.getresponse()
        raw = resp.read(limit + 1)
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
    except (OSError, http.client.HTTPException) as e:
        raise ConnectorError(f"{u.hostname}: {type(e).__name__}") from None
    finally:
        conn.close()
    if len(raw) > limit:
        raise ConnectorError("object too large")
    return resp.status, hdrs, raw


def _check(status: int, raw: bytes, what: str):
    if status in (301, 302, 307, 308):
        raise ConnectorError(f"{what}: redirected (wrong region or endpoint?)")
    if status >= 300:
        msg = re.sub(r"\s+", " ", raw[:400].decode(errors="replace"))
        m = re.search(r"<Message>(.*?)</Message>", msg) or re.search(r'"message"\s*:\s*"([^"]+)"', msg)
        raise ConnectorError(f"{what}: HTTP {status}" + (f" - {m.group(1)[:200]}" if m else ""))


# ---- S3 (SigV4) --------------------------------------------------------------------------------------------------
def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def sigv4(method: str, url: str, region: str, access_key: str, secret_key: str, payload: bytes = b"",
          now: dt.datetime | None = None, service: str = "s3") -> dict:
    """Headers for an AWS Signature Version 4 request (query parameters must already be in `url`)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    amz, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    u = urlsplit(url)
    host = u.hostname + (f":{u.port}" if u.port else "")
    phash = hashlib.sha256(payload).hexdigest()
    q = sorted((quote(k, safe="-_.~"), quote(v, safe="-_.~"))
               for k, v in (p.split("=", 1) if "=" in p else (p, "") for p in u.query.split("&") if p))
    cq = "&".join(f"{k}={v}" for k, v in q)
    path = quote(u.path or "/", safe="/-_.~%")
    canon = f"{method}\n{path}\n{cq}\nhost:{host}\nx-amz-content-sha256:{phash}\nx-amz-date:{amz}\n\n" \
            f"host;x-amz-content-sha256;x-amz-date\n{phash}"
    scope = f"{day}/{region}/{service}/aws4_request"
    sts = f"AWS4-HMAC-SHA256\n{amz}\n{scope}\n{hashlib.sha256(canon.encode()).hexdigest()}"
    k = _sign(_sign(_sign(_sign(("AWS4" + secret_key).encode(), day), region), service), "aws4_request")
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    return {"x-amz-date": amz, "x-amz-content-sha256": phash,
            "Authorization": f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
                             f"SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={sig}"}


def _s3_url(cfg: dict, key: str = "", query: str = "") -> str:
    scheme = "http" if cfg["endpoint"].startswith("http://") else "https"
    ep = cfg["endpoint"].replace("https://", "").replace("http://", "").strip("/") or "s3.amazonaws.com"
    if ep == "s3.amazonaws.com" and cfg["region"] not in ("", "us-east-1"):
        ep = f"s3.{cfg['region']}.amazonaws.com"
    k = quote(key, safe="/-_.~")
    if cfg.get("path_style"):
        base = f"{scheme}://{ep}/{cfg['bucket']}/{k}"
    else:
        base = f"{scheme}://{cfg['bucket']}.{ep}/{k}"
    return base + (f"?{query}" if query else "")


def _s3(cfg, sec, key="", query="", method="GET"):
    url = _s3_url(cfg, key, query)
    st, h, raw = https(method, url, sigv4(method, url, cfg["region"] or "us-east-1", cfg["access_key"],
                                          sec.get("secret_key", "")))
    _check(st, raw, "S3")
    return raw


def _xml_items(raw: bytes, item: str, fields: tuple) -> list[dict]:
    root = ET.fromstring(raw)
    for el in root.iter():                       # drop namespaces
        el.tag = el.tag.split("}")[-1]
    out = []
    for it in root.iter(item):
        out.append({f: (it.findtext(f) or it.findtext(f"Properties/{f}")) for f in fields})
    return out


# ---- operations ---------------------------------------------------------------------------------------------------
def browse(cid: str, prefix: str = "") -> list[dict]:
    return browse_with(*_full(cid), prefix)


def browse_with(kind: str, cfg: dict, sec: dict, prefix: str = "") -> list[dict]:
    prefix = (cfg.get("prefix") or "") + (prefix or "")
    if kind == "s3":
        raw = _s3(cfg, sec, "", "list-type=2&max-keys=%d&prefix=%s" % (MAX_LIST, quote(prefix, safe="")))
        return [{"key": x["Key"], "size": int(x["Size"] or 0), "modified": x["LastModified"]}
                for x in _xml_items(raw, "Contents", ("Key", "Size", "LastModified"))]
    if kind == "azure":
        url = (f"https://{cfg['account']}.blob.core.windows.net/{cfg['container']}?restype=container&comp=list"
               f"&maxresults={MAX_LIST}&prefix={quote(prefix, safe='')}&{sec.get('sas', '').lstrip('?')}")
        st, _, raw = https("GET", url, {"x-ms-version": "2021-08-06"})
        _check(st, raw, "Azure")
        return [{"key": x["Name"], "size": int(x["Content-Length"] or 0), "modified": x["Last-Modified"]}
                for x in _xml_items(raw, "Blob", ("Name", "Content-Length", "Last-Modified"))]
    if kind == "influxdb":
        csv_ = _influx(cfg, sec, 'import "influxdata/influxdb/schema"\nschema.measurements(bucket: "%s")'
                       % cfg["bucket"].replace('"', ""), limit=2_000_000)
        rows = [line.split(",") for line in csv_.decode().splitlines() if line and not line.startswith("#")]
        if not rows:
            return []
        hdr = rows[0]
        i = hdr.index("_value") if "_value" in hdr else -1
        return [{"key": r[i], "size": None, "modified": None, "kind": "measurement"} for r in rows[1:] if len(r) > i]
    raise ConnectorError("databases have no objects to browse: run a query")


def fetch(cid: str, key: str) -> tuple[bytes, str]:
    return fetch_with(*_full(cid), key)


def fetch_with(kind: str, cfg: dict, sec: dict, key: str) -> tuple[bytes, str]:
    if not key or ".." in key.split("/"):
        raise ConnectorError("give an object key")
    name = key.rsplit("/", 1)[-1]
    if kind == "s3":
        return _s3(cfg, sec, key), name
    if kind == "azure":
        url = (f"https://{cfg['account']}.blob.core.windows.net/{cfg['container']}/{quote(key, safe='/-_.~')}"
               f"?{sec.get('sas', '').lstrip('?')}")
        st, _, raw = https("GET", url, {"x-ms-version": "2021-08-06"})
        _check(st, raw, "Azure")
        return raw, name
    raise ConnectorError("this connection is queried, not browsed")


def _influx(cfg, sec, flux: str, limit: int = config.MAX_UPLOAD_BYTES) -> bytes:
    url = f"{cfg['url'].rstrip('/')}/api/v2/query?org={quote(cfg['org'], safe='')}"
    st, _, raw = https("POST", url, {"Authorization": f"Token {sec.get('token', '')}",
                                     "Content-Type": "application/vnd.flux", "Accept": "application/csv"},
                       flux.encode(), limit)
    _check(st, raw, "InfluxDB")
    return raw


SELECT_RE = re.compile(r"^\s*(select|with)\b", re.I)


def query(cid: str, q: str, max_rows: int = MAX_ROWS) -> bytes:
    return query_with(*_full(cid), q, max_rows)


def query_with(kind: str, cfg: dict, sec: dict, q: str, max_rows: int = MAX_ROWS) -> bytes:
    """Run a query; the result comes back as CSV text with a header row."""
    q = (q or "").strip()
    if not q:
        raise ConnectorError("write a query")
    if kind == "influxdb":
        raw = _influx(cfg, sec, q)
        return influx_csv(raw)
    if kind == "postgres":
        return _pg(cfg, sec, q, max_rows)
    raise ConnectorError("object stores are browsed, not queried")


def influx_csv(raw: bytes) -> bytes:
    """Annotated CSV (one block per table) -> one plain CSV of the first table schema; Influx bookkeeping
    columns (result, table, _start, _stop) are dropped."""
    out, hdr, keep = [], None, None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split(",")
        if hdr is None:
            hdr = cells
            keep = [i for i, h in enumerate(hdr) if h not in ("", "result", "table", "_start", "_stop")]
            out.append(",".join(hdr[i] for i in keep))
            continue
        if cells == hdr:
            continue
        out.append(",".join(cells[i] if i < len(cells) else "" for i in keep))
    if len(out) < 2:
        raise ConnectorError("the query returned no rows")
    return ("\n".join(out) + "\n").encode()


def _pg(cfg, sec, q, max_rows):
    if ";" in q.rstrip().rstrip(";"):
        raise ConnectorError("one statement only")
    if not SELECT_RE.match(q):
        raise ConnectorError("only SELECT (or WITH ... SELECT) queries")
    try:
        import psycopg
    except ImportError:
        raise ConnectorError("the PostgreSQL connector needs the psycopg package on the server "
                             "(pip install 'psycopg[binary]')") from None
    _resolve(cfg["host"], int(cfg["port"]))
    try:
        with psycopg.connect(host=cfg["host"], port=int(cfg["port"]), dbname=cfg["database"], user=cfg["user"],
                             password=sec.get("password", ""), sslmode=cfg.get("sslmode") or "require",
                             connect_timeout=15, options="-c statement_timeout=60000") as con:
            con.read_only = True
            with con.cursor() as cur:
                cur.execute(q.rstrip().rstrip(";"))
                cols = [d.name for d in cur.description or []]
                rows = cur.fetchmany(max_rows + 1)
    except psycopg.Error as e:
        raise ConnectorError(f"PostgreSQL: {str(e).splitlines()[0][:200]}") from None
    if len(rows) > max_rows:
        raise ConnectorError(f"more than {max_rows} rows; narrow the query (time range, LIMIT)")
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([v.isoformat() if hasattr(v, "isoformat") else v for v in r])
    return buf.getvalue().encode()


def test(cid: str) -> dict:
    return test_with(*_full(cid))


def test_with(kind: str, cfg: dict, sec: dict) -> dict:
    if kind in ("s3", "azure"):
        items = browse_with(kind, cfg, sec)
        return {"ok": True, "detail": f"listed {len(items)} object(s)"}
    if kind == "influxdb":
        items = browse_with(kind, cfg, sec)
        return {"ok": True, "detail": f"{len(items)} measurement(s) in {cfg['bucket'] or 'the bucket'}"}
    query_with(kind, cfg, sec, "SELECT 1 AS ok")
    return {"ok": True, "detail": "connected (read-only)"}


def time_to_seconds(csv_bytes: bytes) -> bytes:
    """ISO timestamps in a time column (Influx _time, SQL timestamps) -> seconds, so the sample rate can be read."""
    import csv
    import io
    rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8", errors="replace"))))
    if len(rows) < 2:
        return csv_bytes
    hdr = [h.strip() for h in rows[0]]
    ti = next((i for i, h in enumerate(hdr) if h.lower() in ("_time", "time", "timestamp", "ts")), None)
    if ti is None:
        return csv_bytes
    try:
        float(rows[1][ti])
        return csv_bytes
    except ValueError:
        pass
    t0 = None
    for r in rows[1:]:
        try:
            t = dt.datetime.fromisoformat(r[ti].replace("Z", "+00:00"))
        except ValueError:
            raise ConnectorError(f"cannot read time value '{r[ti][:40]}'") from None
        t0 = t if t0 is None else t0
        r[ti] = f"{(t - t0).total_seconds():.6f}"     # timedelta: exact to the microsecond
    hdr[ti] = "time"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(hdr)
    w.writerows(rows[1:])
    return buf.getvalue().encode()


def b64(s: bytes) -> str:
    return base64.b64encode(s).decode()
