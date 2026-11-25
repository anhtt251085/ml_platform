#!/usr/bin/env bash
# Reproduce the SAI findings of issue #9 against an installed `sai` (run from the repository root).
#   tools/sai_findings_repro.sh /path/to/mcu_edge_ai
# Each case prints what SAI returned; the expected behaviour is in the comment above it.
set -uo pipefail
SAI_SRC="${1:?path to a checkout of mcu_edge_ai}"
FC1D="$SAI_SRC/tests/data/onnx_ops/fc_1d_onnx.onnx"
OUT=$(mktemp -d)

echo "== 1. explore --one-axis on a 1-D graph with +fcp plans"
# expected: an explore report; observed at 6e2cf61: schema.invalid (internal_error), the +fcp names do not match
# the plan-name pattern of explore-report-1d.schema.json
sai explore "$FC1D" --target md6605 --one-axis --format json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("error", {}).get("code") or "ok: report with %d ranked plans" % len(d.get("ranking", [])))'

echo "== 2. RISC-V GCC without newlib-nano"
# expected: a structured toolchain_missing error; observed: internal_error + RuntimeError traceback (sai_link)
mkdir -p "$OUT/nonano" && cat > "$OUT/nonano/riscv-none-elf-gcc" <<'G'
#!/bin/sh
for a in "$@"; do case "$a" in --specs=nano.specs) echo "riscv-none-elf-gcc: fatal error: cannot read spec file 'nano.specs': No such file or directory" >&2; exit 1;; esac; done
exec riscv-none-elf-gcc "$@"
G
chmod +x "$OUT/nonano/riscv-none-elf-gcc"
SAI_MD6605_GCC="$OUT/nonano/riscv-none-elf-gcc" sai estimate "$FC1D" --target md6605 --objective fast --format json 2>&1 | head -3

echo "== 3. strided 1-D graph lowers on the 2-D path (documentation)"
python3 - "$OUT" <<'P'
import sys, numpy as np
sys.path.insert(0, ".")
from backend import archspec, qexport
s = archspec.normalize({"input": {"channels": 2, "width": 64}, "layers": [{"op": "conv", "out": 4, "k": 3, "stride": 2}, {"op": "gap"}], "outputs": 2})
n = archspec.SpecNet(s).eval(); n.fold_bn()
qexport.export_qdq(n, np.random.randn(16, 2, 1, 64).astype(np.float32), sys.argv[1] + "/strided.onnx")
P
sai explore "$OUT/strided.onnx" --target cortex-m7 --one-axis --format json | python3 -c 'import json,sys; print(json.load(sys.stdin).get("error", {}).get("code"))'
rm -rf "$OUT"
