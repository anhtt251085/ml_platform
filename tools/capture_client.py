#!/usr/bin/env python3
# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Stream sensor samples into a live capture on the SAI Platform.

Start a capture on the Data > Recordings page (or POST /api/recordings/capture); it shows the capture id and a
token. Then, on the machine the sensor is attached to:

    # a dev board printing "ax,ay,az" lines on its UART (needs: pip install pyserial)
    python tools/capture_client.py --server https://platform.example.com --id rec_ab12 --token cap_... \
        --serial /dev/ttyACM0 --baud 921600

    # anything that prints comma-separated lines
    my_logger | python tools/capture_client.py --server ... --id rec_ab12 --token cap_...

Lines that are not numbers (boot messages, prompts) are skipped; lines with the wrong number of values are
dropped by the server. Samples are sent in batches (--batch lines or every --interval seconds). Standard library
only (pyserial for --serial)."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request


def send(server: str, rid: str, token: str, lines: list[str]) -> int:
    req = urllib.request.Request(f"{server.rstrip('/')}/api/capture/{rid}/push", data="\n".join(lines).encode(),
                                 method="POST", headers={"X-Capture-Token": token, "Content-Type": "text/plain"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(__import__("json").loads(r.read())["samples"])
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 409, 413):
                sys.exit(f"server refused the samples: HTTP {e.code} {e.read()[:200].decode(errors='replace')}")
            time.sleep(2 ** attempt)
        except urllib.error.URLError:
            time.sleep(2 ** attempt)
    sys.exit("server unreachable")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--id", required=True, help="capture (recording) id")
    ap.add_argument("--token", required=True, help="capture token shown when the capture started")
    ap.add_argument("--serial", help="serial port (default: read stdin)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--batch", type=int, default=2000, help="lines per request")
    ap.add_argument("--interval", type=float, default=0.5, help="max seconds between requests")
    a = ap.parse_args()
    if a.serial:
        try:
            import serial
        except ImportError:
            sys.exit("--serial needs pyserial: pip install pyserial")
        port = serial.Serial(a.serial, a.baud, timeout=0.2)
        read = lambda: port.readline().decode("ascii", errors="replace")          # noqa: E731
    else:
        read = sys.stdin.readline
    buf, last, total = [], time.time(), 0
    try:
        while True:
            line = read()
            if not line and not a.serial:
                break
            line = line.strip()
            if line:
                buf.append(line)
            if len(buf) >= a.batch or (buf and time.time() - last > a.interval):
                total = send(a.server, a.id, a.token, buf)
                print(f"\r{total} samples", end="", file=sys.stderr, flush=True)
                buf, last = [], time.time()
    except KeyboardInterrupt:
        pass
    if buf:
        total = send(a.server, a.id, a.token, buf)
    print(f"\n{total} samples in capture {a.id}; stop it in the platform to label and use it", file=sys.stderr)


if __name__ == "__main__":
    main()
