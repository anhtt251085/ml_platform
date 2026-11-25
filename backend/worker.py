"""Standalone job worker: `python -m backend.worker [--kinds train,fitcheck] [--threads N]`.

Run train workers on GPU machines and compile/analyze workers next to the pinned toolchains; all of them share
the job queue in the database. Set SAI_PLATFORM_INPROC_WORKERS=0 on the API when workers run separately."""

from __future__ import annotations

import argparse
import os
import socket
import threading

from . import db, pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="", help=f"comma list of {','.join(pipeline.KINDS)} (default all)")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    db.init()
    kinds = [k for k in a.kinds.split(",") if k] or None
    pipeline.requeue_stale()
    threads = [threading.Thread(target=pipeline.work_forever,
                                args=(f"{socket.gethostname()}:{os.getpid()}:w{i}", kinds), daemon=True)
               for i in range(a.threads)]
    for t in threads:
        t.start()
    print(f"worker up: kinds={kinds or 'all'} threads={a.threads}")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
