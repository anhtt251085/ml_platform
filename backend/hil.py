"""HIL firmware: build the SAI harness (harness/) around a SAI build for one board port, and run it under QEMU.

Ports (harness/ports/*.c + harness/ld/*.ld):
  qemu-mps2-an500   Cortex-M7 + FPU in QEMU (MPS2 AN500)     runs in CI and on the platform  -> `emulated`
  qemu-virt-rv32    RV32IMAFC in QEMU virt                    runs in CI and on the platform  -> `emulated`
  stm32h753-nucleo  NUCLEO-H753ZI, USART3 + DWT CYCCNT        lab agent (pyOCD flash)         -> `measured`
  md6605-evb        Sanken MD6605 EVB, mcycle (UART from SDK) lab agent (J-Link flash)        -> `measured`
The compile flags are SAI's own for each target, so the harness links the package library unchanged.
Emulated cycle counts follow QEMU's -icount instruction clock: they show the program runs and its outputs are
bit-exact on the target ISA; they are never latency evidence.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "harness"

M7_FLAGS = ["-mcpu=cortex-m7", "-mfpu=fpv5-d16", "-mfloat-abi=hard", "-mthumb", "-O2", "-std=gnu11",
            "-ffunction-sections", "-fdata-sections", "-fsigned-char"]
RV_FLAGS = ["-march=rv32imafc_zicsr", "-mabi=ilp32", "-O2", "-ffunction-sections", "-fdata-sections",
            "-fsigned-char", "-msmall-data-limit=8"]


def _rv_cc() -> str:
    return os.environ.get("SAI_MD6605_GCC") or shutil.which("riscv-none-elf-gcc") or "riscv-none-elf-gcc"


PORTS = {
    "qemu-mps2-an500": {"target": "cortex-m7", "isa": "arm", "flags": M7_FLAGS, "startup": "startup_cm7.c",
                        "emulator": ["qemu-system-arm", "-M", "mps2-an500", "-nographic", "-monitor", "none",
                                     "-semihosting-config", "enable=on,target=native", "-icount", "shift=0",
                                     "-kernel"], "clock_hz": None},
    "qemu-virt-rv32": {"target": "md6605", "isa": "riscv", "flags": RV_FLAGS, "startup": "startup_rv32.S",
                       "emulator": ["qemu-system-riscv32", "-M", "virt", "-bios", "none", "-nographic",
                                    "-monitor", "none", "-icount", "shift=0", "-kernel"], "clock_hz": None},
    "stm32h753-nucleo": {"target": "cortex-m7", "isa": "arm", "flags": M7_FLAGS, "startup": "startup_cm7.c",
                         "emulator": None, "clock_hz": 480_000_000},
    "md6605-evb": {"target": "md6605", "isa": "riscv", "flags": RV_FLAGS, "startup": "startup_rv32.S",
                   "emulator": None, "clock_hz": 66_666_666},
}


class HilError(RuntimeError):
    pass


def cc_for(port: str) -> str:
    return "arm-none-eabi-gcc" if PORTS[port]["isa"] == "arm" else _rv_cc()


def available(port: str) -> bool:
    p = PORTS[port]
    cc_ok = shutil.which(cc_for(port)) is not None or Path(cc_for(port)).exists()
    emu_ok = p["emulator"] is None or shutil.which(p["emulator"][0]) is not None
    return cc_ok and emu_ok


def write_vectors(path: Path, x: np.ndarray):
    if len(x) == 0:
        raise HilError("no test vectors")
    x = np.ascontiguousarray(x.reshape(len(x), -1), dtype=np.int8)
    rows = ",\n".join("  {" + ",".join(str(int(v)) for v in r) + "}" for r in x)
    path.write_text("/* Generated: SAI HIL test vectors (int8, model input layout). */\n#include <stdint.h>\n"
                    f"#define SAI_HIL_N {len(x)}u\n"
                    f"static const int8_t sai_hil_inputs[{len(x)}][{x.shape[1]}] = {{\n{rows}\n}};\n")


def build(sai_build: Path, port: str, vectors: np.ndarray, out: Path) -> dict:
    """Compile the harness + port + startup and link the build's lib/libsai_model.a. Returns elf path and sizes."""
    p = PORTS[port]
    out.mkdir(parents=True, exist_ok=True)
    write_vectors(out / "hil_vectors.h", vectors)
    elf = out / f"sai_hil_{port}.elf"
    cc = cc_for(port)
    srcs = [H / "sai_hil.c", H / "ports" / f"{port.replace('-', '_')}.c", H / p["startup"]]
    cmd = [cc, *p["flags"], "-I", str(out), "-I", str(sai_build / "model"), "-I", str(H), *map(str, srcs),
           str(sai_build / "lib" / "libsai_model.a"), "-nostartfiles", "--specs=nano.specs", "--specs=nosys.specs",
           "-Wl,--gc-sections", "-T", str(H / "ld" / f"{port.replace('-', '_')}.ld"), "-lm", "-lgcc", "-o", str(elf)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode:
        raise HilError(f"harness build for {port} failed: {proc.stderr[-1500:]}")
    size_tool = cc[:-3] + "size" if cc.endswith("gcc") else "size"
    sz = subprocess.run([size_tool, str(elf)], capture_output=True, text=True)
    text = data = bss = None
    m = re.search(r"^\s*(\d+)\s+(\d+)\s+(\d+)", sz.stdout, re.M)
    if m:
        text, data, bss = map(int, m.groups())
    return {"elf": str(elf), "port": port, "text": text, "data": data, "bss": bss}


LINE = re.compile(r"^(SAI-HIL|CFG|OUT|CYC|STACK|DONE)\b(.*)$")


def parse(stdout: str) -> dict:
    """Parse the harness line protocol strictly: lines before the SAI-HIL header are ignored (boot noise), outputs
    are placed by index, and the result must hold exactly the header's vector count of the header's size."""
    res = {"outputs": [], "cycles": None, "stack_painted": None, "status": None, "header": None, "config": {}}
    outs: dict[int, list] = {}
    try:
        for raw in stdout.replace("\r", "").splitlines():
            m = LINE.match(raw.strip())
            if not m:
                continue
            kind, rest = m.group(1), m.group(2).split()
            if kind == "SAI-HIL":
                res["header"], outs, res["status"] = [int(v) for v in rest[:4]], {}, None
            elif res["header"] is None:
                continue
            elif kind == "CFG":
                res["config"] = dict(kv.split("=", 1) for kv in rest if "=" in kv)
            elif kind == "OUT":
                i, hexs = int(rest[0]), rest[1]
                if res["header"] and len(hexs) != 2 * res["header"][3]:
                    raise HilError(f"OUT {i}: {len(hexs) // 2} bytes, expected {res['header'][3]}")
                outs[i] = np.frombuffer(bytes.fromhex(hexs), dtype=np.int8).astype(int).tolist()
            elif kind == "CYC":
                res["cycles"], res["cycles_min"], res["cycles_max"] = map(int, rest[:3])
            elif kind == "STACK":
                res["stack_painted"] = int(rest[0])
            elif kind == "DONE":
                res["status"] = rest[0] if rest else "?"
    except (ValueError, IndexError) as e:
        raise HilError(f"garbled harness output: {e}") from None
    if res["status"] == "ok":
        n = res["header"][1] if res["header"] else 0
        if sorted(outs) != list(range(n)):
            raise HilError(f"harness reported {len(outs)} of {n} outputs (lost UART lines?)")
    res["outputs"] = [outs[i] for i in sorted(outs)]
    return res


def run_emulated(elf: str, port: str, timeout: int = 120) -> dict:
    p = PORTS[port]
    if not p["emulator"]:
        raise HilError(f"{port} is a physical board; a lab agent flashes it")
    stderr = ""
    try:
        proc = subprocess.run([*p["emulator"], elf], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = f"timeout after {timeout} s"
    res = parse(stdout)
    if res["status"] != "ok":
        raise HilError(f"{port}: harness did not finish ({res['status']}); output tail: {stdout[-300:]!r}; "
                       f"qemu: {stderr.strip()[-300:]!r}")
    return res
