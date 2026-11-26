# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Several models on one board: SAI's multi-model plugin (extensions/sai_multimodel of the SAI repository).

The customer picks finished projects (SAI Core builds) and a board. The platform writes the plugin's workload
manifest (models pinned by SHA-256, run one after another: `independent_sequential`), runs
`bin/sai-multimodel eval-bundle` in place from $SAI_REPO, and reports the device RAM of the shared build against
the same models deployed side by side, the flash, SAI's verification of every build, and what the plugin does not
claim. The plugin's own status is EVALUATION_ONLY (test signatures, no board run), so packages say so.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from . import config, sai_engine

LAUNCHER = str(config.ROOT / "bin" / "sai-multimodel")


def available() -> bool:
    return bool(config.SAI_REPO) and (Path(config.SAI_REPO) / "extensions" / "sai_multimodel" / "src").is_dir()


def why_unavailable() -> str:
    return ("several models on one board need SAI's multi-model plugin: set SAI_REPO to a checkout of the SAI "
            "repository that has extensions/sai_multimodel (branch product/sai-unified)")


def model_id(name: str, used: set) -> str:
    """A workload model id that is also a valid SAI symbol prefix: ^[a-z][a-z0-9_]{0,23}$, not sai / eai (SAI ABI
    namespacing), unique in the workload."""
    base = re.sub(r"[^a-z0-9_]+", "_", (name or "model").lower()).strip("_") or "model"
    if not base[:1].isalpha() or base in ("sai", "eai") or base.startswith(("sai_", "eai_")):
        base = "m_" + base
    base = base[:24].rstrip("_")
    mid, n = base, 2
    while mid in used:
        mid, n = f"{base[:24 - len(str(n)) - 1].rstrip('_')}_{n}", n + 1
    used.add(mid)
    return mid


def workload(name: str, models: list[dict], target: str, budgets: dict) -> str:
    import yaml
    wid = re.sub(r"[^a-z0-9_]+", "_", (name or "workload").lower()).strip("_")[:32] or "workload"
    wid = wid if wid[:1].isalpha() else "w_" + wid
    b = sai_engine.TARGETS[target]["budget"]
    cons = {"stack_budget_bytes": int(b.get("stack", 8192))}
    if budgets.get("max_ram"):
        cons["peak_total_ram_bytes"] = int(budgets["max_ram"])
    if budgets.get("max_flash"):
        cons["max_linked_flash_bytes"] = int(budgets["max_flash"])
    doc = {"schema_version": 1, "workload_id": wid, "mode": "independent_sequential", "target": target,
           "models": [{"id": m["id"], "source": f"models/{m['id']}.onnx", "sha256": m["sha256"]} for m in models],
           "execution": {"order": [m["id"] for m in models], "concurrent_calls": "forbidden"},
           "constraints": cons,
           "optimization": {"objective": "minimum_peak_ram",
                            "sharing": {"transient_workspace": True, "exact_constants": True,
                                        "kernel_implementations": True, "cross_model_features": False},
                            "max_candidate_plans_per_model": 24, "max_joint_search_states": 20000},
           "output": {"package_type": "static_library", "artifact_class": "EVALUATION_ONLY"}}
    return yaml.safe_dump(doc, sort_keys=False)


def run(cmd: list[str], cwd: Path, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run([LAUNCHER, *cmd], capture_output=True, text=True, timeout=timeout, cwd=cwd,
                          env=sai_engine._env())


def summarize(bundle: dict, models: list[dict], target: str, budgets: dict) -> dict:
    """Shared build vs the same models side by side, from the plugin's eval-bundle.json."""
    sel = (bundle.get("abi_search") or {}).get("selected") or {}
    abi = sel.get("abi") or {}
    builds = {(b["id"], b["abi"]): b for b in bundle.get("builds") or []}
    rows, sep_ws, sep_static, sep_io, stacks, flash = [], 0, 0, 0, [], 0
    for m in models:
        b10 = builds.get((m["id"], "1.0")) or {}
        bs = builds.get((m["id"], abi.get(m["id"], "1.0"))) or b10
        sep_ws += int(b10.get("workspace") or 0)
        sep_static += int(b10.get("static") or 0)
        sep_io += int(b10.get("input_bytes") or 0) + int(b10.get("output_bytes") or 0)
        stacks.append(int(b10.get("stack") or 0))
        flash += int(bs.get("linked_flash") or 0)
        rows.append({**{k: m[k] for k in ("id", "job_id", "project", "template_id")}, "abi": bs.get("abi"),
                     "workspace": bs.get("workspace"), "stack": bs.get("stack"), "static": bs.get("static"),
                     "ram_bound": bs.get("ram_bound"), "linked_flash": bs.get("linked_flash"),
                     "input_bytes": bs.get("input_bytes"), "output_bytes": bs.get("output_bytes")})
    separate = sep_ws + sep_static + max(stacks or [0]) + sep_io
    shared = int(sel.get("device_ram") or 0)
    ver = bundle.get("sai_verify") or {}
    passed = sum(1 for v in ver.values() if v.get("result") == "passed")
    b = sai_engine.TARGETS[target]["budget"]
    viol = []
    if sel.get("shared_workspace") is not None and int(sel["shared_workspace"]) + int(sel.get("static") or 0) > b.get("workspace", 1 << 62):
        viol.append(f"shared workspace {int(sel['shared_workspace']) + int(sel.get('static') or 0):,} B > board AI arena {b['workspace']:,} B")
    if sel.get("stack") and int(sel["stack"]) > b.get("stack", 1 << 62):
        viol.append(f"stack {sel['stack']:,} B > board stack reserve {b['stack']:,} B")
    if flash > b.get("flash", 1 << 62):
        viol.append(f"flash {flash:,} B > board AI ROM budget {b['flash']:,} B")
    if budgets.get("max_ram") and shared > float(budgets["max_ram"]):
        viol.append(f"RAM {shared:,} B > your limit {int(budgets['max_ram']):,} B")
    if budgets.get("max_flash") and flash > float(budgets["max_flash"]):
        viol.append(f"flash {flash:,} B > your limit {int(budgets['max_flash']):,} B")
    return {"models": rows, "target": target, "abi": abi, "shared": sel, "device_ram": shared,
            "separate_ram": separate, "saved_ram": separate - shared, "flash": flash,
            "verify": {"passed": passed, "total": len(ver), "checks": ver},
            "not_claimed": bundle.get("not_claimed") or [], "evidence": sel.get("evidence"),
            "fits": not viol, "violations": viol, "abi_table": (bundle.get("abi_search") or {}).get("table") or []}


def package(out: Path, bundle_dir: Path, report: dict, reason: str) -> Path:
    z_path = out / "package.zip"
    with zipfile.ZipFile(z_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle_dir.rglob("*")):
            if f.is_file():
                z.write(f, Path("sai_multimodel_package") / f.relative_to(bundle_dir))
        z.writestr("sai_multimodel_package/report.json", json.dumps(report, indent=1, default=str))
        z.writestr("sai_multimodel_package/EVALUATION_ONLY.txt", f"Evaluation only: {reason}. Not for deployment.\n")
    return z_path


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy_model(job_dir: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(job_dir / "model.onnx", dst)
